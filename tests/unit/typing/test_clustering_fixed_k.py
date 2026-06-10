"""Tests fixed_k + HDBSCAN dans cluster_embeddings (M8/T3, audit 2026-06)."""

from __future__ import annotations

import numpy as np
import pytest

from ewat.typing.clustering import cluster_embeddings


def _blobs(k: int = 3, n_per: int = 20, d: int = 8, seed: int = 0) -> np.ndarray:
    """k blobs gaussiens bien séparés, L2-normalisés (géométrie projet)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 10.0
    z = np.concatenate(
        [centers[i] + rng.normal(scale=0.3, size=(n_per, d)) for i in range(k)]
    )
    return (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)


def test_fixed_k_short_circuits_selection():
    z = _blobs(k=3)
    result = cluster_embeddings(z, fixed_k=5, linkage="average", metric="cosine")
    assert result.k_optimal == 5
    assert result.k_selection_method == "fixed"
    assert len(set(result.labels)) == 5
    # pas de sélection → pas de gap, une seule silhouette
    assert result.gap_stats == {} and list(result.silhouette_scores) == [5]


def test_fixed_k_matches_true_structure():
    z = _blobs(k=4, seed=1)
    result = cluster_embeddings(z, fixed_k=4, linkage="average", metric="cosine")
    assert result.silhouette_scores[4] > 0.5  # blobs bien séparés


def test_fixed_k_out_of_range_raises():
    z = _blobs(k=2, n_per=5)
    with pytest.raises(ValueError, match="fixed_k"):
        cluster_embeddings(z, fixed_k=1)
    with pytest.raises(ValueError, match="fixed_k"):
        cluster_embeddings(z, fixed_k=len(z))


def test_fixed_k_is_deterministic():
    z = _blobs(k=3, seed=2)
    r1 = cluster_embeddings(z, fixed_k=3, linkage="average", metric="cosine")
    r2 = cluster_embeddings(z, fixed_k=3, linkage="average", metric="cosine")
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_hdbscan_finds_blobs_and_assigns_all():
    z = _blobs(k=3, n_per=30, seed=3)
    result = cluster_embeddings(z, k_selection_method="hdbscan", metric="cosine")
    assert result.k_selection_method == "hdbscan"
    assert result.k_optimal == 3
    # le bruit est réassigné : aucun label −1
    assert (result.labels >= 0).all()
    assert len(result.labels) == len(z)


def test_hdbscan_all_noise_degrades_gracefully():
    # données uniformes très éparses → HDBSCAN peut tout marquer bruit
    rng = np.random.default_rng(4)
    z = rng.uniform(size=(12, 6)).astype(np.float32)
    result = cluster_embeddings(z, k_selection_method="hdbscan",
                                hdbscan_min_cluster_size=10)
    assert (result.labels >= 0).all()
    assert result.k_optimal >= 1


def test_unknown_method_still_raises():
    z = _blobs()
    with pytest.raises(ValueError, match="k_selection_method"):
        cluster_embeddings(z, k_selection_method="kmeans")
