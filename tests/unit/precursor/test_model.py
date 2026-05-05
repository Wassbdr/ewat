"""Tests for PrecursorClassifier and find_optimal_k."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ewat.precursor.model import PrecursorClassifier, baseline_auroc, find_optimal_k

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs(n_clusters: int = 3, n_per_cluster: int = 30, d: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, (n_clusters, d))
    z = np.vstack([rng.normal(centres[c], 0.5, (n_per_cluster, d)) for c in range(n_clusters)])
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return z.astype(np.float32), labels


# ---------------------------------------------------------------------------
# PrecursorClassifier
# ---------------------------------------------------------------------------

def test_fit_does_not_raise():
    z, labels = _make_blobs()
    clf = PrecursorClassifier(n_clusters=3)
    clf.fit(z, labels)


def test_predict_proba_shape():
    z, labels = _make_blobs()
    clf = PrecursorClassifier(n_clusters=3)
    clf.fit(z, labels)
    proba = clf.predict_proba(z)
    assert proba.shape == (len(z), 3)


def test_predict_proba_values_in_zero_one():
    z, labels = _make_blobs()
    clf = PrecursorClassifier(n_clusters=3)
    clf.fit(z, labels)
    proba = clf.predict_proba(z)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_auroc_separable_clusters_above_baseline():
    z, labels = _make_blobs(n_clusters=3, n_per_cluster=50)
    clf = PrecursorClassifier(n_clusters=3)
    clf.fit(z, labels)
    auroc = clf.auroc_per_type(z, labels)
    for c in range(3):
        assert auroc[c] > baseline_auroc(3), f"Type {c} AUROC={auroc[c]:.3f} ≤ 0.5"


def test_auroc_keys_cover_all_clusters():
    z, labels = _make_blobs(n_clusters=4)
    clf = PrecursorClassifier(n_clusters=4)
    clf.fit(z, labels)
    auroc = clf.auroc_per_type(z, labels)
    assert set(auroc.keys()) == {0, 1, 2, 3}


def test_auroc_nan_for_single_class():
    z = np.random.default_rng(5).normal(0, 1, (20, 4)).astype(np.float32)
    labels = np.zeros(20, dtype=int)   # only cluster 0
    clf = PrecursorClassifier(n_clusters=3)
    clf.fit(z, labels)
    auroc = clf.auroc_per_type(z, labels)
    # Clusters 1 and 2 have no test examples → NaN
    assert np.isnan(auroc[1])
    assert np.isnan(auroc[2])


def test_degenerate_cluster_predict_proba_returns_05():
    """Cluster with no training examples → predict_proba returns 0.5."""
    z, labels = _make_blobs(n_clusters=2)
    clf = PrecursorClassifier(n_clusters=3)  # 3 clusters but only 2 in labels
    clf.fit(z, labels)
    proba = clf.predict_proba(z)
    # Column 2 should be 0.5 (degenerate)
    np.testing.assert_allclose(proba[:, 2], 0.5)


def test_save_load_roundtrip():
    z, labels = _make_blobs()
    clf = PrecursorClassifier(n_clusters=3)
    clf.fit(z, labels)
    proba_before = clf.predict_proba(z)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "clf.pkl"
        clf.save(path)
        clf2 = PrecursorClassifier.load(path)

    proba_after = clf2.predict_proba(z)
    np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5)


# ---------------------------------------------------------------------------
# find_optimal_k
# ---------------------------------------------------------------------------

def test_find_optimal_k_returns_argmax():
    auroc_table = {
        2: {0: 0.55, 1: 0.60, 2: 0.80},
        4: {0: 0.70, 1: 0.55, 2: 0.75},
        6: {0: 0.65, 1: 0.72, 2: 0.70},
    }
    k_opt = find_optimal_k(auroc_table, n_clusters=3)
    assert k_opt[0] == 4   # best AUROC for type 0 at k=4
    assert k_opt[1] == 6   # best AUROC for type 1 at k=6
    assert k_opt[2] == 2   # best AUROC for type 2 at k=2


def test_find_optimal_k_covers_all_clusters():
    auroc_table = {2: {0: 0.6, 1: 0.7}, 4: {0: 0.65, 1: 0.68}}
    k_opt = find_optimal_k(auroc_table, n_clusters=2)
    assert set(k_opt.keys()) == {0, 1}


def test_find_optimal_k_handles_nan():
    auroc_table = {
        2: {0: float("nan"), 1: 0.6},
        4: {0: 0.7, 1: float("nan")},
    }
    k_opt = find_optimal_k(auroc_table, n_clusters=2)
    assert k_opt[0] == 4   # only valid at k=4
    assert k_opt[1] == 2   # only valid at k=2


# ---------------------------------------------------------------------------
# baseline_auroc
# ---------------------------------------------------------------------------

def test_baseline_auroc_is_half():
    assert baseline_auroc(10) == pytest.approx(0.5)
