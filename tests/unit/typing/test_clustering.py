"""Unit tests for src/ewat/typing/clustering.py."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs(k: int = 3, n_per_cluster: int = 20, d: int = 8, seed: int = 0):
    """Synthetic embeddings with k well-separated clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(k, d))
    X = np.vstack([
        centers[i] + rng.normal(scale=0.3, size=(n_per_cluster, d))
        for i in range(k)
    ]).astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cluster_labels_shape():
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=3)
    result = cluster_embeddings(X, k_range=range(2, 6), n_gap_refs=2)
    assert result.labels.shape == (len(X),), \
        f"Expected ({len(X)},), got {result.labels.shape}"


def test_cluster_labels_in_range():
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=4)
    result = cluster_embeddings(X, k_range=range(2, 7), n_gap_refs=2)
    k = result.k_optimal
    assert set(result.labels).issubset(set(range(k))), \
        f"Labels out of range [0, {k-1}]: {set(result.labels)}"


def test_three_clear_blobs_k_optimal_is_3():
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=3, n_per_cluster=30, d=16, seed=7)
    result = cluster_embeddings(X, k_range=range(2, 8), n_gap_refs=2)
    assert result.k_optimal == 3, \
        f"Expected K=3 for 3 clear clusters, got K={result.k_optimal}"


def test_silhouette_positive_on_clear_clusters():
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=3, n_per_cluster=30, d=8)
    result = cluster_embeddings(X, k_range=range(2, 6), n_gap_refs=2)
    best_sil = result.silhouette_scores[result.k_optimal]
    assert best_sil > 0.3, \
        f"Silhouette too low for clear clusters: {best_sil:.3f}"


def test_silhouette_scores_dict_populated():
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=2)
    result = cluster_embeddings(X, k_range=range(2, 5), n_gap_refs=2)
    assert len(result.silhouette_scores) >= 2
    for k, score in result.silhouette_scores.items():
        assert isinstance(k, int)
        assert isinstance(score, float)


def test_gap_stats_computed():
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=3)
    result = cluster_embeddings(X, k_range=range(2, 5), n_gap_refs=3)
    assert len(result.gap_stats) > 0
    for k, gap in result.gap_stats.items():
        assert isinstance(gap, float)


def test_result_dataclass_fields():
    from ewat.typing.clustering import cluster_embeddings, ClusterResult
    X = _make_blobs(k=2)
    result = cluster_embeddings(X, k_range=range(2, 4), n_gap_refs=2)
    assert isinstance(result, ClusterResult)
    assert hasattr(result, "labels")
    assert hasattr(result, "k_optimal")
    assert hasattr(result, "silhouette_scores")
    assert hasattr(result, "gap_stats")


def test_k_range_clip_small_dataset():
    """When N < max(k_range), k_range must be clipped without error."""
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=2, n_per_cluster=4)  # N=8
    result = cluster_embeddings(X, k_range=range(2, 20), n_gap_refs=2)
    assert result.k_optimal <= len(X) - 1


def test_deterministic_given_same_seed():
    from ewat.typing.clustering import cluster_embeddings
    X = _make_blobs(k=3)
    r1 = cluster_embeddings(X, k_range=range(2, 5), n_gap_refs=3, random_state=0)
    r2 = cluster_embeddings(X, k_range=range(2, 5), n_gap_refs=3, random_state=0)
    assert r1.k_optimal == r2.k_optimal
    np.testing.assert_array_equal(r1.labels, r2.labels)


# ---------------------------------------------------------------------------
# Linkage / metric variants
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def test_unknown_linkage_raises():
    from ewat.typing.clustering import cluster_embeddings
    with pytest.raises(ValueError):
        cluster_embeddings(_make_blobs(k=2), linkage="bogus")


def test_unknown_metric_raises():
    from ewat.typing.clustering import cluster_embeddings
    with pytest.raises(ValueError):
        cluster_embeddings(_make_blobs(k=2), metric="bogus")


def test_ward_with_cosine_raises():
    from ewat.typing.clustering import cluster_embeddings
    with pytest.raises(ValueError):
        cluster_embeddings(_make_blobs(k=2), linkage="ward", metric="cosine")


def test_cosine_clustering_runs_on_normalised_embeddings():
    from ewat.typing.clustering import cluster_embeddings
    X = _l2_normalize(_make_blobs(k=3, n_per_cluster=20, d=8, seed=11))
    result = cluster_embeddings(
        X, k_range=range(2, 6), n_gap_refs=2,
        linkage="average", metric="cosine",
    )
    assert result.linkage == "average"
    assert result.metric == "cosine"
    assert result.k_optimal == 3


def test_compare_linkages_runs_multiple_methods():
    from ewat.typing.clustering import compare_linkages
    X = _l2_normalize(_make_blobs(k=4, n_per_cluster=15, d=8, seed=23))
    results = compare_linkages(
        X,
        k_range=range(2, 7),
        methods=(
            ("ward", "euclidean"),
            ("average", "cosine"),
            ("complete", "cosine"),
        ),
        n_gap_refs=2,
    )
    assert set(results.keys()) == {
        "ward__euclidean", "average__cosine", "complete__cosine",
    }
    for r in results.values():
        assert r.k_optimal in range(2, 7)


def test_compare_linkages_skips_invalid_methods_silently():
    from ewat.typing.clustering import compare_linkages
    X = _l2_normalize(_make_blobs(k=2, n_per_cluster=10, d=4))
    results = compare_linkages(
        X,
        k_range=range(2, 5),
        methods=(("ward", "cosine"), ("average", "cosine")),
        n_gap_refs=2,
    )
    assert "ward__cosine" not in results
    assert "average__cosine" in results
