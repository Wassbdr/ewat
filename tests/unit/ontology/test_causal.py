"""Tests for KSG transfer entropy and causal relation computation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ewat.ontology.causal import (
    _ksg_mi,
    _transfer_entropy,
    _total_te,
    compute_causal_relations,
)


# ---------------------------------------------------------------------------
# Unit tests for KSG estimators
# ---------------------------------------------------------------------------

def test_ksg_mi_independent_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = rng.normal(0, 1, 200)
    mi = _ksg_mi(x, y, k=5)
    assert mi >= 0.0
    assert mi < 0.5  # independent → near 0


def test_ksg_mi_identical_positive():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200)
    mi = _ksg_mi(x, x, k=5)
    assert mi > 0.1  # identical → high MI


def test_ksg_mi_correlated_greater_than_independent():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 200)
    y_dep = x + 0.2 * rng.normal(0, 1, 200)
    y_ind = rng.normal(0, 1, 200)
    mi_dep = _ksg_mi(x, y_dep, k=5)
    mi_ind = _ksg_mi(x, y_ind, k=5)
    assert mi_dep > mi_ind


def test_ksg_mi_nonnegative():
    rng = np.random.default_rng(3)
    for _ in range(10):
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        assert _ksg_mi(x, y, k=5) >= 0.0


def test_ksg_mi_too_short_returns_zero():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    assert _ksg_mi(x, y, k=5) == 0.0


def test_transfer_entropy_independent_near_zero():
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 200)
    y = rng.normal(0, 1, 200)
    te = _transfer_entropy(x, y, lag=1, k=5)
    assert te >= 0.0
    assert te < 0.3


def test_transfer_entropy_causal_positive():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 300)
    # y is driven by x with lag 1
    y = np.zeros(300)
    y[1:] = 0.9 * x[:-1] + 0.1 * rng.normal(0, 1, 299)
    te = _transfer_entropy(x, y, lag=1, k=5)
    assert te > 0.0


def test_transfer_entropy_nonnegative():
    rng = np.random.default_rng(12)
    for _ in range(5):
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        assert _transfer_entropy(x, y, lag=1, k=5) >= 0.0


def test_transfer_entropy_too_short_returns_zero():
    x = np.zeros(5)
    y = np.zeros(5)
    assert _transfer_entropy(x, y, lag=1, k=5) == 0.0


# ---------------------------------------------------------------------------
# compute_causal_relations
# ---------------------------------------------------------------------------

def _make_signal_store(tmpdir: Path, n_ep: int = 12, T: int = 60) -> dict[str, dict]:
    """Create synthetic signal.npz files and return a cluster manifest."""
    manifest = {}
    rng = np.random.default_rng(99)
    for i in range(n_ep):
        cluster = i % 3
        ep_id = f"ep_{i:03d}"
        ep_dir = tmpdir / ep_id
        ep_dir.mkdir()
        signal = rng.normal(0, 1, (T, 6, 17)).astype(np.float32)
        np.savez(ep_dir / "signal.npz", signal=signal)
        manifest[ep_id] = {"cluster": cluster, "split": "train", "scenario": f"sc_{cluster}"}
    return manifest


def test_causal_returns_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_signal_store(Path(tmpdir))
        rels = compute_causal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            n_clusters=3,
            n_permutations=5,
            min_support=2,
            min_series_length=10,
        )
    assert isinstance(rels, list)


def test_causal_relation_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_signal_store(Path(tmpdir))
        rels = compute_causal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            n_clusters=3,
            n_permutations=5,
            min_support=2,
            min_series_length=10,
        )
    for r in rels:
        assert r.relation_type == "causal"


def test_causal_strength_nonnegative():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_signal_store(Path(tmpdir))
        rels = compute_causal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            n_clusters=3,
            n_permutations=5,
            min_support=2,
            min_series_length=10,
        )
    for r in rels:
        assert r.strength >= 0.0


def test_causal_p_value_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_signal_store(Path(tmpdir))
        rels = compute_causal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            n_clusters=3,
            n_permutations=10,
            p_threshold=1.0,  # accept all
            min_support=2,
            min_series_length=10,
        )
    for r in rels:
        assert r.p_value is not None
        assert 0.0 <= r.p_value <= 1.0


def test_causal_insufficient_support_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_signal_store(Path(tmpdir), n_ep=3)  # 1 ep per cluster
        rels = compute_causal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            n_clusters=3,
            n_permutations=5,
            min_support=10,  # impossible
            min_series_length=5,
        )
    assert rels == []


def test_causal_series_too_short_skips():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_signal_store(Path(tmpdir), T=5)  # very short
        rels = compute_causal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            n_clusters=3,
            n_permutations=5,
            min_support=2,
            min_series_length=50,  # longer than data
        )
    assert rels == []
