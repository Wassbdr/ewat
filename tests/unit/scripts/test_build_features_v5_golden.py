"""Golden test du pipeline v5 (D6, audit 2026-06).

Fabrique un mini-dump Train Ticket synthétique (3 services, 8 steps) et
vérifie bout-en-bout :

1. ``collect.build_features_v5.build_episode`` — contrat complet : shapes
   (T, 3, 18), labels de régime, imputation, schéma de features = registre,
   propagation held_out/bug, graphe non vide, mapping features → valeurs.
2. ``scripts.validate_v5`` — détection d'un N non conforme (3 ≠ 41).
3. ``scripts.assemble_dataset`` — routage held-out test-only + erreur si un
   scénario est absent du test (D4/D5).

Aucun cluster ni modèle externe requis (semantic désactivé).
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "v5")):
    if p not in sys.path:
        sys.path.insert(0, p)

from collect.build_features_v5 import build_episode  # noqa: E402
from scripts.validate_v5 import _check_episode  # noqa: E402
from telemetry.feature_names import SCHEMA_V5_1, get_schema  # noqa: E402

SERVICES = ["ts-alpha", "ts-beta", "ts-gamma"]
T0 = 1_750_000_000.0
STEP = 30
N_BINS = 8  # grille [T0, T0+240], 8 bins de 30 s

V5_NAMES = get_schema(SCHEMA_V5_1)
FI = {name: i for i, name in enumerate(V5_NAMES)}


# ---------------------------------------------------------------------------
# Fabrication du mini-dump
# ---------------------------------------------------------------------------


def _series(svc: str, values: list[float]) -> dict:
    """Une série Prometheus pod-level (le builder strippe -xxxxx-yyyyy)."""
    return {
        "metric": {"pod": f"{svc}-abc12-x9y8z"},
        "values": [[T0 + STEP * i, str(v)] for i, v in enumerate(values)],
    }


def _write_prometheus(dump: Path) -> None:
    const = lambda v: [v] * (N_BINS + 1)  # noqa: E731
    # cpu de ts-beta monte pendant l'injection (bins 3-5)
    beta_cpu = [0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.3, 0.2, 0.2]
    prom = {
        "cpu": [_series("ts-alpha", const(0.2)),
                _series("ts-beta", beta_cpu),
                _series("ts-gamma", const(0.3))],
        "ram": [_series(s, const(100e6)) for s in SERVICES],
        "mem_limit": [_series(s, const(500e6)) for s in SERVICES],
        "net_rx": [_series(s, const(1e3)) for s in SERVICES],
        "net_tx": [_series(s, const(1e3)) for s in SERVICES],
        "fs_reads": [_series(s, const(10.0)) for s in SERVICES],
        "fs_writes": [_series(s, const(10.0)) for s in SERVICES],
        "restarts": [_series(s, const(0.0)) for s in SERVICES],
        "jvm_heap_used": [_series(s, const(50e6)) for s in SERVICES],
        "jvm_heap_max": [_series(s, const(100e6)) for s in SERVICES],
        "jvm_gc_sum": [_series(s, const(0.01)) for s in SERVICES],
        "jvm_threads_blocked": [_series(s, const(0.0)) for s in SERVICES],
    }
    with gzip.open(dump / "prometheus.json.gz", "wt") as f:
        json.dump(prom, f)


def _write_jaeger(dump: Path) -> None:
    """Un appel ts-alpha → ts-beta (+ → ts-gamma 1 bin sur 2) par bin."""
    traces = []
    for i in range(N_BINS):
        start_us = int((T0 + STEP * i + 5) * 1e6)
        spans = [
            {"spanID": f"root{i}", "operationName": "GET /api",
             "startTime": start_us, "duration": 5000,
             "references": [], "processID": "p1", "tags": []},
            {"spanID": f"child{i}", "operationName": "callBeta",
             "startTime": start_us + 1000, "duration": 3000,
             "references": [{"refType": "CHILD_OF", "spanID": f"root{i}"}],
             "processID": "p2", "tags": []},
        ]
        if i % 2 == 0:
            spans.append(
                {"spanID": f"leaf{i}", "operationName": "callGamma",
                 "startTime": start_us + 2000, "duration": 1000,
                 "references": [{"refType": "CHILD_OF", "spanID": f"child{i}"}],
                 "processID": "p3", "tags": []})
        traces.append({
            "traceID": f"trace{i}",
            "spans": spans,
            "processes": {"p1": {"serviceName": "ts-alpha"},
                          "p2": {"serviceName": "ts-beta"},
                          "p3": {"serviceName": "ts-gamma"}},
        })
    with gzip.open(dump / "jaeger.json.gz", "wt") as f:
        json.dump({"traces": traces}, f)


def _write_loki(dump: Path) -> None:
    streams = []
    for svc in SERVICES:
        values = []
        for i in range(N_BINS):
            ts_ns = str(int((T0 + STEP * i + 2) * 1e9))
            values.append([ts_ns, "INFO request handled ok"])
            # ts-beta logge des erreurs pendant l'injection (bins 3-5)
            if svc == "ts-beta" and 3 <= i <= 5:
                values.append([ts_ns, "ERROR boom request failed"])
        streams.append({"stream": {"app": svc}, "values": values})
    with gzip.open(dump / "loki.json.gz", "wt") as f:
        json.dump({"streams": streams}, f)


def _make_dump(tmp_path: Path, *, held_out: bool = False,
               is_bug: bool = False, bug_id: str | None = None) -> Path:
    ep = tmp_path / "episode_cpu_stress_000_test"
    ep.mkdir(parents=True, exist_ok=True)
    _write_prometheus(ep)
    _write_jaeger(ep)
    _write_loki(ep)
    meta = {
        "episode_id": ep.name,
        "scenario": "cpu_stress",
        "category": "contention",
        "targets": ["ts-beta"],
        "chaos_resource": "chaos/cpu_stress.yaml",
        "boundaries_rel": {"injection_start": 90, "injection_end": 180,
                           "recovery_end": 240},
        "ramp_s": 0,
        "step": STEP,
        "is_bug": is_bug,
        "bug_id": bug_id,
        "held_out": held_out,
    }
    (ep / "episode_meta.json").write_text(json.dumps(meta))
    return ep


# ---------------------------------------------------------------------------
# 1. build_features_v5 — contrat golden
# ---------------------------------------------------------------------------


@pytest.fixture()
def built_episode(tmp_path: Path) -> Path:
    ep = _make_dump(tmp_path)
    build_episode(ep, services=SERVICES, step=STEP, with_semantic=False)
    return ep


def test_build_writes_full_contract(built_episode: Path) -> None:
    for artefact in ("signal.npz", "signal_raw.npz", "signal_mask.npz",
                     "adjacency.npz", "labels.parquet", "services.json",
                     "metadata.json", "feature_provenance.json"):
        assert (built_episode / artefact).exists(), f"artefact manquant: {artefact}"


def test_build_shapes_and_schema(built_episode: Path) -> None:
    sig = np.load(built_episode / "signal.npz")["signal"]
    raw = np.load(built_episode / "signal_raw.npz")["signal_raw"]
    mask = np.load(built_episode / "signal_mask.npz")["missing_mask"]
    adj = np.load(built_episode / "adjacency.npz")["adjacency"]
    assert sig.shape == (N_BINS, 3, 18)
    assert raw.shape == sig.shape and mask.shape == sig.shape
    assert adj.shape == (N_BINS, 3, 3, 3)

    meta = json.loads((built_episode / "metadata.json").read_text())
    assert meta["signal_feature_names"] == V5_NAMES
    assert meta["dataset_schema_version"] == SCHEMA_V5_1
    assert meta["grid_step_s"] == STEP


def test_build_imputation_and_mask(built_episode: Path) -> None:
    sig = np.load(built_episode / "signal.npz")["signal"]
    raw = np.load(built_episode / "signal_raw.npz")["signal_raw"]
    mask = np.load(built_episode / "signal_mask.npz")["missing_mask"]
    assert not np.isnan(sig).any(), "signal imputé contient des NaN"
    # semantic désactivé → la feature est NaN dans le brut, le masque le trace
    assert np.isnan(raw[:, :, FI["semantic_anomaly"]]).all()
    np.testing.assert_array_equal(mask, np.isnan(raw))


def test_build_regime_labels(built_episode: Path) -> None:
    labels = pd.read_parquet(built_episode / "labels.parquet")
    assert len(labels) == N_BINS
    # rel ∈ {0,30,...,210} ; injection [90,180) → bins 3-5 ; recovery ≥180 → 6-7
    assert list(labels["regime"]) == (
        ["normal"] * 3 + ["injection"] * 3 + ["recovery"] * 2
    )
    assert not labels["held_out_flag"].any()
    assert (labels["fault_type"] == "chaos").all()
    assert labels["scenario"].iloc[0] == "cpu_stress"


def test_build_feature_values_mapped(built_episode: Path) -> None:
    sig = np.load(built_episode / "signal.npz")["signal"]
    beta = SERVICES.index("ts-beta")
    # cpu de ts-beta monte à 0.9 pendant l'injection (mapping nom → indice)
    assert sig[3, beta, FI["cpu_util"]] == pytest.approx(0.9, abs=1e-5)
    assert sig[0, beta, FI["cpu_util"]] == pytest.approx(0.2, abs=1e-5)
    # mem_limit_ratio = 100e6/500e6 ; jvm_heap_ratio = 50/100
    assert sig[0, beta, FI["mem_limit_ratio"]] == pytest.approx(0.2, abs=1e-5)
    assert sig[0, beta, FI["jvm_heap_ratio"]] == pytest.approx(0.5, abs=1e-5)
    # log_error_rate de ts-beta vivant pendant l'injection, nul avant
    assert sig[4, beta, FI["log_error_rate"]] > 0.0
    assert sig[0, beta, FI["log_error_rate"]] == pytest.approx(0.0, abs=1e-6)


def test_build_graph_alpha_to_beta(built_episode: Path) -> None:
    adj = np.load(built_episode / "adjacency.npz")["adjacency"]
    a, b = SERVICES.index("ts-alpha"), SERVICES.index("ts-beta")
    # volume (canal 0) sur l'arête alpha→beta à chaque bin
    assert (adj[:, a, b, 0] > 0).all(), "arête alpha→beta absente"
    assert (adj[:, :, :, 0] > 0).any(axis=(1, 2)).all(), "bins sans arête"


def test_build_heldout_bug_propagation(tmp_path: Path) -> None:
    ep = _make_dump(tmp_path / "bug", held_out=True, is_bug=True, bug_id="F3")
    build_episode(ep, services=SERVICES, step=STEP, with_semantic=False)
    labels = pd.read_parquet(ep / "labels.parquet")
    assert labels["held_out_flag"].all()
    assert (labels["fault_type"] == "bug").all()
    assert (labels["bug_id"] == "F3").all()


# ---------------------------------------------------------------------------
# 2. validate_v5 — la porte détecte les violations de contrat
# ---------------------------------------------------------------------------


def test_validate_v5_rejects_wrong_n(built_episode: Path) -> None:
    res = _check_episode(built_episode, max_raw_nan=0.50, trace_floor=2,
                         min_graph_fraction=0.10)
    assert not res["pass"]
    assert any("(N,F)" in f for f in res["failures"]), res["failures"]
    # mais le schéma de features, lui, est conforme au registre
    assert not any("registre" in f for f in res["failures"]), res["failures"]


# ---------------------------------------------------------------------------
# 3. assemble_dataset — held-out test-only + garde scénario manquant
# ---------------------------------------------------------------------------


def _fake_featured_episode(root: Path, eid: str, scenario: str,
                           t_start: float, held_out: bool = False) -> None:
    d = root / eid
    d.mkdir(parents=True)
    sig = np.zeros((10, 3, 17), dtype=np.float32)
    np.savez_compressed(d / "signal.npz", signal=sig)
    (d / "services.json").write_text(json.dumps(["a", "b", "c"]))
    meta = {
        "episode_id": eid,
        "scenario": {"name": scenario, "category": "anomaly",
                     "targets": ["a"], "file": "x.yaml"},
        "boundaries": {"baseline_start": t_start, "recovery_end": t_start + 600},
        "quality_snapshot": {"signal_nan_ratio": 0.0, "metrics_nan_ratio": 0.0,
                             "traces_nan_ratio": 0.0, "logs_nan_ratio": 0.0},
    }
    (d / "metadata.json").write_text(json.dumps(meta))
    pd.DataFrame({"regime": ["normal"] * 10,
                  "held_out_flag": [held_out] * 10}).to_parquet(d / "labels.parquet")


def _run_assemble(root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.assemble_dataset",
         "--features-root", str(root), "--output", str(out),
         "--symlink-episodes", *extra],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_assemble_routes_heldout_test_only(tmp_path: Path) -> None:
    root = tmp_path / "features"
    for i in range(5):
        _fake_featured_episode(root, f"ep_cpu_{i:03d}", "cpu_stress",
                               t_start=1_700_000_000.0 + 1000 * i)
    _fake_featured_episode(root, "ep_bug_f3_000", "bug_f3",
                           t_start=1_700_010_000.0, held_out=True)

    out = tmp_path / "dataset"
    proc = _run_assemble(root, out)
    assert proc.returncode == 0, proc.stderr

    split = json.loads((out / "split.json").read_text())
    assert "ep_bug_f3_000" in split["test"]
    assert "ep_bug_f3_000" not in split["train"] + split["val"]
    # le scénario splittable garde sa couverture train/val/test
    assert len(split["train"]) >= 1 and len(split["val"]) >= 1
    assert any(e.startswith("ep_cpu") for e in split["test"])

    manifest = json.loads((out / "dataset.json").read_text())
    assert manifest["held_out"]["episode_ids"] == ["ep_bug_f3_000"]
    assert manifest["held_out"]["scenarios"] == ["bug_f3"]

    index = pd.read_parquet(out / "index.parquet")
    assert bool(index.set_index("episode_id").loc["ep_bug_f3_000", "held_out"])
    assert not index.set_index("episode_id").loc["ep_cpu_000", "held_out"]


def test_assemble_errors_on_missing_test_scenario(tmp_path: Path) -> None:
    root = tmp_path / "features"
    for i in range(5):
        _fake_featured_episode(root, f"ep_cpu_{i:03d}", "cpu_stress",
                               t_start=1_700_000_000.0 + 1000 * i)
    # scénario à 1 épisode → quota test impossible → doit être une ERREUR
    _fake_featured_episode(root, "ep_solo_000", "solo_scenario",
                           t_start=1_700_020_000.0)

    out = tmp_path / "dataset"
    proc = _run_assemble(root, out)
    assert proc.returncode != 0
    assert "MISSING" in proc.stderr

    # l'override documenté dégrade l'erreur en warning
    proc2 = _run_assemble(root, tmp_path / "dataset2",
                          "--allow-missing-test-scenarios")
    assert proc2.returncode == 0, proc2.stderr
