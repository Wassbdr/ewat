"""Tests du gate brut de collecte v5 (v5/collect/run_episode.py).

L'ancien gate (`traces>0 and logs>0 and prom>0`) laissait passer des épisodes
inexploitables : 12 traces, aucun service profond tracé, faute jamais appliquée
(~450 épisodes vides les 17-20/07). Ces tests verrouillent les contrôles qui le
remplacent — en particulier la détection de faute PAR SERVICE : le chaos ne frappe
qu'un service parmi 41, donc un p99 global le dilue jusqu'à l'invisibilité.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# v5/ n'est pas un package installé : la campagne tourne avec `cwd=v5`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v5"))

from collect.run_episode import (  # noqa: E402
    FAULT_MIN_RATIO,
    _fault_visible,
    _p99,
    _span_stats,
    _traced_services,
)

T0 = 1_000_000.0  # t_start de référence (secondes epoch)
BOUNDARIES = {"baseline_start": 0.0, "injection_start": 100.0, "injection_end": 200.0}
# _fault_visible ne juge que le DERNIER TIERS de l'injection (intensité au pic).
PEAK_START = 100.0 + (200.0 - 100.0) * 2 / 3  # ≈ 166.7


def _span(start_s: float, duration_ms: float, error: bool = False,
          pid: str = "p0") -> dict:
    """Span au format du dump Jaeger (startTime/duration en microsecondes)."""
    return {
        "startTime": int(start_s * 1e6),
        "duration": int(duration_ms * 1000),
        "processID": pid,
        "tags": [{"key": "error", "value": True}] if error else [],
    }


def _trace(services: list[str], spans: list[dict]) -> dict:
    return {
        "processes": {f"p{i}": {"serviceName": s} for i, s in enumerate(services)},
        "spans": spans,
    }


class TestTracedServices:
    def test_compte_les_services_porteurs_de_spans(self):
        traces = [_trace(["ts-order", "ts-travel"], []), _trace(["ts-order"], [])]
        assert _traced_services(traces) == {"ts-order", "ts-travel"}

    def test_ignore_les_services_non_tt(self):
        """Jaeger connaît aussi des services d'infra : seuls les `ts-*` comptent."""
        traces = [_trace(["ts-order", "jaeger-query", "istio-proxy"], [])]
        assert _traced_services(traces) == {"ts-order"}

    def test_dump_vide(self):
        assert _traced_services([]) == set()


class TestSpanStats:
    def test_ne_retient_que_la_fenetre_demandee(self):
        traces = [_trace(["ts-a"], [_span(50, 10), _span(150, 20), _span(250, 30)])]
        stats = _span_stats(traces, 100, 200)  # [100, 200[
        assert stats["ts-a"]["dur"] == [20.0]

    def test_separe_les_services(self):
        """Le cœur du correctif : ne pas mélanger les spans de services différents."""
        spans = [_span(10, 5, pid="p0"), _span(11, 500, pid="p1")]
        stats = _span_stats([_trace(["ts-sain", "ts-fautif"], spans)], 0, 100)
        assert stats["ts-sain"]["dur"] == [5.0]
        assert stats["ts-fautif"]["dur"] == [500.0]

    def test_compte_les_erreurs(self):
        spans = [_span(10, 5, error=True), _span(11, 5), _span(12, 5), _span(13, 5)]
        stats = _span_stats([_trace(["ts-a"], spans)], 0, 100)
        assert stats["ts-a"]["n"] == 4
        assert stats["ts-a"]["err"] == 1

    def test_fenetre_vide(self):
        assert _span_stats([_trace(["ts-a"], [_span(10, 5)])], 500, 600) == {}


class TestFaultVisible:
    def _traces(self, base_ms: float, peak_ms: float, n: int = 40,
                base_err: bool = False, peak_err: bool = False,
                pid: str = "p0", services: list[str] | None = None) -> list[dict]:
        spans = [_span(T0 + 10 + i * 0.5, base_ms, base_err, pid) for i in range(n)]
        spans += [_span(T0 + PEAK_START + 5 + i * 0.5, peak_ms, peak_err, pid)
                  for i in range(n)]
        return [_trace(services or ["ts-a"], spans)]

    def test_faute_latence_detectee(self):
        visible, score = _fault_visible(self._traces(10, 100), T0, BOUNDARIES)
        assert visible and score > FAULT_MIN_RATIO

    def test_chaos_jamais_applique_rejete(self):
        """Le mode d'échec réel : injection sans effet → rapport ~1.0."""
        visible, score = _fault_visible(self._traces(10, 10), T0, BOUNDARIES)
        assert not visible
        assert score == pytest.approx(1.0, abs=0.05)

    def test_faute_visible_par_les_erreurs_seules(self):
        """Une panne franche peut ne pas ralentir mais faire échouer les spans."""
        traces = self._traces(10, 10, base_err=False, peak_err=True)
        visible, score = _fault_visible(traces, T0, BOUNDARIES)
        assert visible and score > FAULT_MIN_RATIO

    def test_un_seul_service_touche_parmi_beaucoup(self):
        """Régression du 2026-07-27 : le p99 global diluait la faute jusqu'à 1.00.

        Un service ralenti ×20 au milieu de 3 services sains doit être détecté.
        """
        spans = []
        for i in range(40):  # 3 services sains, stables
            for pid in ("p0", "p1", "p2"):
                spans.append(_span(T0 + 10 + i * 0.5, 10, pid=pid))
                spans.append(_span(T0 + PEAK_START + 5 + i * 0.5, 10, pid=pid))
        for i in range(40):  # le service fautif : 10 ms → 200 ms
            spans.append(_span(T0 + 10 + i * 0.5, 10, pid="p3"))
            spans.append(_span(T0 + PEAK_START + 5 + i * 0.5, 200, pid="p3"))
        traces = [_trace(["ts-a", "ts-b", "ts-c", "ts-fautif"], spans)]
        visible, score = _fault_visible(traces, T0, BOUNDARIES)
        assert visible
        assert score > 10  # porté par le seul service touché

    def test_trop_peu_de_spans_ne_bloque_pas(self):
        """Incertitude ⇒ on laisse passer : jamais de rejet à l'aveugle."""
        visible, score = _fault_visible(self._traces(10, 100, n=5), T0, BOUNDARIES)
        assert visible
        assert score != score  # NaN

    def test_aucune_trace(self):
        visible, _ = _fault_visible([], T0, BOUNDARIES)
        assert visible

    def test_spans_hors_fenetre_de_pic_ignores(self):
        """Le ramp monte progressivement : seul le dernier tiers fait foi."""
        spans = [_span(T0 + 10 + i * 0.5, 10) for i in range(40)]
        # injection présente mais AVANT le pic → pas de données au pic
        spans += [_span(T0 + 105 + i * 0.5, 500) for i in range(40)]
        visible, score = _fault_visible([_trace(["ts-a"], spans)], T0, BOUNDARIES)
        assert visible
        assert score != score  # NaN : rien à comparer au pic


def test_p99_borne_superieure():
    assert _p99([1.0]) == 1.0
    assert _p99(list(range(100))) == 99
