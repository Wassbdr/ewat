"""Tests du gate brut de collecte v5 (v5/collect/run_episode.py).

L'ancien gate (`traces>0 and logs>0 and prom>0`) laissait passer des épisodes
inexploitables : 12 traces, aucun service profond tracé, faute jamais appliquée
(~450 épisodes vides les 17-20/07). Ces tests verrouillent les trois contrôles qui
le remplacent : couverture des services tracés, et manifestation de la faute.
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


def _span(start_s: float, duration_ms: float, error: bool = False) -> dict:
    """Span au format du dump Jaeger (startTime/duration en microsecondes)."""
    return {
        "startTime": int(start_s * 1e6),
        "duration": int(duration_ms * 1000),
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
        durs, _ = _span_stats(traces, T0 - T0 + 100, 200)  # [100, 200[
        assert durs == [20.0]

    def test_taux_erreur(self):
        spans = [_span(10, 5, error=True), _span(11, 5), _span(12, 5), _span(13, 5)]
        _, err_rate = _span_stats([_trace(["ts-a"], spans)], 0, 100)
        assert err_rate == pytest.approx(0.25)

    def test_fenetre_vide(self):
        durs, err = _span_stats([_trace(["ts-a"], [_span(10, 5)])], 500, 600)
        assert durs == [] and err == 0.0


class TestFaultVisible:
    def _traces(self, base_ms: float, inj_ms: float, n: int = 50,
                base_err: bool = False, inj_err: bool = False) -> list[dict]:
        spans = [_span(T0 + 10 + i * 0.1, base_ms, base_err) for i in range(n)]
        spans += [_span(T0 + 110 + i * 0.1, inj_ms, inj_err) for i in range(n)]
        return [_trace(["ts-a"], spans)]

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
        traces = self._traces(10, 10, base_err=False, inj_err=True)
        visible, score = _fault_visible(traces, T0, BOUNDARIES)
        assert visible and score > FAULT_MIN_RATIO

    def test_trop_peu_de_spans_ne_bloque_pas(self):
        """Incertitude ⇒ on laisse passer : le gate ne doit jamais rejeter à l'aveugle."""
        visible, score = _fault_visible(self._traces(10, 100, n=5), T0, BOUNDARIES)
        assert visible
        assert score != score  # NaN

    def test_aucune_trace(self):
        visible, _ = _fault_visible([], T0, BOUNDARIES)
        assert visible


def test_p99_borne_superieure():
    assert _p99([1.0]) == 1.0
    assert _p99(list(range(100))) == 99
