"""Tests for AlertAssembler."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ewat.alerts.alert import Alert
from ewat.alerts.assembler import AlertAssembler
from ewat.drift.detector import DriftDetector, DriftResult
from ewat.encoder.stgcn import STGCNEncoder
from ewat.precursor.model import PrecursorClassifier
from ewat.typing.siamese import SiameseTyper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_NODES = 6
D_FEAT = 17
D_HIDDEN = 16
D_EMBED = 16
D_PROJ = 8
N_CLUSTERS = 3


def _make_typer() -> SiameseTyper:
    encoder = STGCNEncoder(d_feat=D_FEAT, n_nodes=N_NODES, d_hidden=D_HIDDEN, d_embed=D_EMBED)
    return SiameseTyper(encoder, d_proj=D_PROJ)


def _make_classifiers(
    n_clusters: int = N_CLUSTERS, d: int = D_PROJ
) -> dict[int, PrecursorClassifier]:
    rng = np.random.default_rng(0)
    classifiers: dict[int, PrecursorClassifier] = {}
    for c in range(n_clusters):
        clf = PrecursorClassifier(n_clusters=n_clusters)
        z = rng.normal(0, 1, (30, d)).astype(np.float32)
        labels = np.array([i % n_clusters for i in range(30)], dtype=int)
        clf.fit(z, labels)
        classifiers[c] = clf
    return classifiers


def _make_assembler(threshold: float = 0.0) -> AlertAssembler:
    typer = _make_typer()
    classifiers = _make_classifiers()
    k_optimal = {c: 2 for c in range(N_CLUSTERS)}
    fiches = {c: {"cluster": c, "top_feature": "cpu_util"} for c in range(N_CLUSTERS)}
    return AlertAssembler(
        typer=typer,
        classifiers=classifiers,
        k_optimal=k_optimal,
        fiches=fiches,
        threshold=threshold,
        device=torch.device("cpu"),
    )


def _make_signal(t: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 1, (t, N_NODES, D_FEAT)).astype(np.float32)
    adjacency = rng.uniform(0, 1, (t, N_NODES, N_NODES, 3)).astype(np.float32)
    return signal, adjacency


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_returns_list():
    assembler = _make_assembler()
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    assert isinstance(alerts, list)


def test_predict_threshold_zero_returns_all_clusters():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    assert len(alerts) == N_CLUSTERS


def test_predict_threshold_one_returns_empty():
    assembler = _make_assembler(threshold=1.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    assert len(alerts) == 0


def test_predict_alerts_sorted_by_probability_desc():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    probs = [a.probability for a in alerts]
    assert probs == sorted(probs, reverse=True)


def test_predict_alert_types():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    for a in alerts:
        assert isinstance(a, Alert)


def test_predict_probability_in_range():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    for a in alerts:
        assert 0.0 <= a.probability <= 1.0


def test_predict_horizon_steps_matches_k_optimal():
    k_optimal = {0: 4, 1: 6, 2: 2}
    typer = _make_typer()
    classifiers = _make_classifiers()
    fiches: dict[int, dict] = {}
    assembler = AlertAssembler(typer, classifiers, k_optimal, fiches, threshold=0.0)
    signal, adjacency = _make_signal(t=10)
    alerts = assembler.predict(signal, adjacency)
    for a in alerts:
        assert a.horizon_steps == k_optimal[a.cluster_id]


def test_predict_horizon_seconds_30_per_step():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    for a in alerts:
        assert a.horizon_seconds == pytest.approx(a.horizon_steps * 30.0)


def test_predict_fiche_attached():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    for a in alerts:
        assert "cluster" in a.fiche


def test_predict_timestamp_propagated():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency, timestamp=12345.0)
    for a in alerts:
        assert a.timestamp == pytest.approx(12345.0)


def test_predict_episode_id_propagated():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency, episode_id="ep_test")
    for a in alerts:
        assert a.episode_id == "ep_test"


def test_predict_short_signal_left_pads():
    """Signal shorter than k* → must left-pad and not crash."""
    k_optimal = {0: 6, 1: 6, 2: 6}
    typer = _make_typer()
    classifiers = _make_classifiers()
    assembler = AlertAssembler(typer, classifiers, k_optimal, {}, threshold=0.0)
    signal, adjacency = _make_signal(t=3)  # shorter than k=6
    alerts = assembler.predict(signal, adjacency)
    assert len(alerts) == N_CLUSTERS  # should not crash and return all


def test_predict_no_classifiers_returns_empty():
    typer = _make_typer()
    assembler = AlertAssembler(typer, {}, {}, {}, threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    assert alerts == []


def test_predict_cluster_ids_match_classifiers():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    returned_ids = {a.cluster_id for a in alerts}
    assert returned_ids == set(assembler.classifiers.keys())


# ---------------------------------------------------------------------------
# Drift detector integration
# ---------------------------------------------------------------------------


def _make_mock_detector(flag: bool) -> DriftDetector:
    """Return a DriftDetector whose update() always returns the given flag."""
    detector = MagicMock(spec=DriftDetector)
    regime = "drift" if flag else "normal"
    detector.update.return_value = DriftResult(flag=flag, mmd2=0.0, regime=regime)
    return detector


def test_drift_flag_passes_through_with_drift_flag():
    """Look-through: drift does NOT suppress alerts; they carry drift_flag=True."""
    assembler = _make_assembler(threshold=0.0)
    assembler.drift_detector = _make_mock_detector(flag=True)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    assert len(alerts) == N_CLUSTERS
    assert all(a.drift_flag for a in alerts)


def test_no_drift_flag_produces_clean_alerts():
    """When drift_detector.flag is False, alerts are produced with drift_flag=False."""
    assembler = _make_assembler(threshold=0.0)
    assembler.drift_detector = _make_mock_detector(flag=False)
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    assert len(alerts) == N_CLUSTERS
    assert not any(a.drift_flag for a in alerts)


def test_detector_reset_on_episode_change():
    """Detector must be reset when episode_id changes, not on same episode."""
    assembler = _make_assembler(threshold=0.0)
    detector = _make_mock_detector(flag=False)
    assembler.drift_detector = detector
    assembler._last_episode_id = "ep1"  # prime state so first call is same-episode
    signal, adjacency = _make_signal()

    assembler.predict(signal, adjacency, episode_id="ep1")
    detector.reset.assert_not_called()

    assembler.predict(signal, adjacency, episode_id="ep2")
    detector.reset.assert_called_once()


def test_no_drift_detector_produces_alerts():
    """Without a drift_detector, predict() works normally."""
    assembler = _make_assembler(threshold=0.0)
    assert assembler.drift_detector is None
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    assert len(alerts) == N_CLUSTERS


# ---------------------------------------------------------------------------
# New behaviour: episode_id=None resets, grouping by k*, step_seconds
# ---------------------------------------------------------------------------


def test_episode_id_none_always_resets_drift_detector():
    """episode_id=None must trigger a reset on every call (defensive default)."""
    assembler = _make_assembler(threshold=0.0)
    detector = _make_mock_detector(flag=False)
    assembler.drift_detector = detector
    signal, adjacency = _make_signal()

    assembler.predict(signal, adjacency, episode_id=None)
    assert detector.reset.call_count == 1
    assembler.predict(signal, adjacency, episode_id=None)
    assert detector.reset.call_count == 2


def test_first_call_resets_when_no_prior_episode():
    """The first call must reset, fixing the empty-string-default bug."""
    assembler = _make_assembler(threshold=0.0)
    detector = _make_mock_detector(flag=False)
    assembler.drift_detector = detector
    signal, adjacency = _make_signal()

    assembler.predict(signal, adjacency, episode_id="")
    detector.reset.assert_called_once()


def test_grouping_runs_one_encoder_pass_per_distinct_k():
    """Multiple clusters with the same k* should share one encoder forward."""
    typer = _make_typer()
    classifiers = _make_classifiers()
    # All three clusters share k*=4 → one encoder forward.
    k_optimal = {0: 4, 1: 4, 2: 4}
    assembler = AlertAssembler(typer, classifiers, k_optimal, {}, threshold=0.0)

    embed_calls = {"n": 0}
    real_embed = typer.embed

    def counting_embed(sig: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        embed_calls["n"] += 1
        return real_embed(sig, adj)

    typer.embed = counting_embed  # type: ignore[assignment]
    signal, adjacency = _make_signal(t=8)
    assembler.predict(signal, adjacency)
    assert embed_calls["n"] == 1


def test_grouping_two_distinct_k_values():
    """Two distinct k* values → two encoder forwards."""
    typer = _make_typer()
    classifiers = _make_classifiers()
    k_optimal = {0: 2, 1: 4, 2: 4}
    assembler = AlertAssembler(typer, classifiers, k_optimal, {}, threshold=0.0)

    embed_calls = {"n": 0}
    real_embed = typer.embed

    def counting_embed(sig: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        embed_calls["n"] += 1
        return real_embed(sig, adj)

    typer.embed = counting_embed  # type: ignore[assignment]
    signal, adjacency = _make_signal(t=8)
    assembler.predict(signal, adjacency)
    assert embed_calls["n"] == 2  # one per distinct k*


def test_regime_mask_filters_window():
    """When regime_mask is provided, only normal-regime steps feed the encoder."""
    typer = _make_typer()
    classifiers = _make_classifiers()
    k_optimal = {c: 3 for c in range(N_CLUSTERS)}
    assembler = AlertAssembler(typer, classifiers, k_optimal, {}, threshold=0.0)
    signal, adjacency = _make_signal(t=10)
    mask = np.zeros(10, dtype=bool)
    mask[2:6] = True  # only 4 normal steps in the middle
    alerts = assembler.predict(signal, adjacency, regime_mask=mask)
    assert len(alerts) == N_CLUSTERS


def test_regime_mask_wrong_length_raises():
    assembler = _make_assembler(threshold=0.0)
    signal, adjacency = _make_signal(t=8)
    with pytest.raises(ValueError):
        assembler.predict(signal, adjacency, regime_mask=np.ones(3, dtype=bool))


def test_step_seconds_overrides_default():
    typer = _make_typer()
    classifiers = _make_classifiers()
    k_optimal = {c: 3 for c in range(N_CLUSTERS)}
    assembler = AlertAssembler(
        typer, classifiers, k_optimal, {}, threshold=0.0, step_seconds=15.0,
    )
    signal, adjacency = _make_signal()
    alerts = assembler.predict(signal, adjacency)
    for a in alerts:
        assert a.horizon_seconds == pytest.approx(a.horizon_steps * 15.0)
