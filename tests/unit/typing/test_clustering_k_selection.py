"""Tests for cluster_embeddings k_selection_method.

Covers Step 6 fix 6.4 (audit 2026-05-26): Tibshirani gap rule + SE reporting.
"""

import numpy as np
import pytest

from ewat.typing.clustering import (
    cluster_embeddings,
    _tibshirani_k_selection,
)


def _make_blobs(k_true: int = 3, n_per: int = 30, d: int = 8, seed: int = 0):
    """Generate well-separated isotropic blobs in d-dim space."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5.0, size=(k_true, d))
    points = []
    for c in centers:
        points.append(c + rng.normal(0, 0.5, size=(n_per, d)))
    return np.vstack(points).astype(np.float32)


def test_silhouette_argmax_still_default():
    z = _make_blobs(k_true=3, n_per=20)
    result = cluster_embeddings(z, k_range=range(2, 8), n_gap_refs=3)
    assert result.k_selection_method == "silhouette"
    assert result.k_optimal == max(result.silhouette_scores,
                                   key=result.silhouette_scores.__getitem__)


def test_gap_tibshirani_selects_k_for_clear_blobs():
    z = _make_blobs(k_true=3, n_per=30)
    result = cluster_embeddings(
        z, k_range=range(2, 8), n_gap_refs=5,
        k_selection_method="gap_tibshirani",
    )
    assert result.k_selection_method == "gap_tibshirani"
    # For well-separated 3 blobs, gap-Tibshirani should pick K in [2, 4]
    # (some sampling variance); silhouette is the fallback if rule fails.
    assert result.k_optimal in (2, 3, 4)


def test_gap_se_is_populated():
    z = _make_blobs(k_true=3, n_per=20)
    result = cluster_embeddings(z, k_range=range(2, 5), n_gap_refs=5)
    assert set(result.gap_se.keys()) == set(result.gap_stats.keys())
    for k, se in result.gap_se.items():
        assert se >= 0.0, f"gap SE must be non-negative, got {se} for K={k}"
    # The Tibshirani correction (1 + 1/B) should make SE >= raw std
    # so values are >= 0, finite
    assert all(np.isfinite(v) for v in result.gap_se.values())


def test_tibshirani_helper_returns_smallest_satisfying_k():
    """Synthetic gap curve : K=3 satisfies the rule, K=4 also; expect 3."""
    gap_stats = {2: 0.5, 3: 0.9, 4: 0.95, 5: 0.96}
    gap_se = {2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05}
    # gap(3)=0.9 >= gap(4) - s(4) = 0.95 - 0.05 = 0.90 → True
    assert _tibshirani_k_selection(gap_stats, gap_se) == 3


def test_tibshirani_helper_returns_none_when_no_k_satisfies():
    """Monotonically increasing gap with tiny SE → no K satisfies the rule."""
    gap_stats = {2: 0.1, 3: 0.5, 4: 1.0, 5: 1.5}
    gap_se = {2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01}
    # gap(2)=0.1 >= 0.5-0.01=0.49 → False
    # gap(3)=0.5 >= 1.0-0.01=0.99 → False
    # gap(4)=1.0 >= 1.5-0.01=1.49 → False
    assert _tibshirani_k_selection(gap_stats, gap_se) is None


def test_invalid_k_selection_method_raises():
    z = _make_blobs(k_true=2, n_per=15)
    with pytest.raises(ValueError, match="k_selection_method"):
        cluster_embeddings(z, k_range=range(2, 5), n_gap_refs=3,
                           k_selection_method="bogus")


def test_gap_tibshirani_always_returns_valid_k():
    """Regardless of whether the rule fires or falls back, k_optimal is valid."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, size=(60, 6))
    result = cluster_embeddings(
        z, k_range=range(2, 5), n_gap_refs=3,
        k_selection_method="gap_tibshirani",
    )
    # k_optimal must be one of the evaluated K values
    assert result.k_optimal in result.silhouette_scores
    assert result.k_selection_method == "gap_tibshirani"
