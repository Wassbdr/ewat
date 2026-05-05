"""Agglomerative clustering of episode embeddings.

Used after siamese fine-tuning to discover the empirical anomaly type
ontology C = {C_1, ..., C_K} from the z_e embedding space.

K selection
-----------
1. **Silhouette score** (Kaufman & Rousseeuw 1990) — argmax over k_range.
   Threshold for H1 falsification: silhouette < 0.3 on held-out split.
2. **Gap statistic** (Tibshirani et al. 2001) — computed for validation,
   not used for K selection to keep the pipeline simple.

References
----------
- Kaufman & Rousseeuw (1990) — Silhouette threshold justification.
- Tibshirani, Walther & Hastie (2001) — Gap statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    """Output of :func:`cluster_embeddings`.

    Attributes
    ----------
    labels:           (N_ep,) cluster assignments ∈ [0, k_optimal-1].
    k_optimal:        Number of clusters chosen by silhouette maximisation.
    silhouette_scores: {k → silhouette_score} for each k in k_range.
    gap_stats:        {k → gap_value} for each k in k_range (validation only).
    """

    labels: np.ndarray
    k_optimal: int
    silhouette_scores: dict[int, float] = field(default_factory=dict)
    gap_stats: dict[int, float] = field(default_factory=dict)


def cluster_embeddings(
    z: np.ndarray,
    k_range: range = range(2, 16),
    n_gap_refs: int = 10,
    random_state: int = 42,
) -> ClusterResult:
    """Agglomerative clustering (Ward linkage) with automatic K selection.

    Parameters
    ----------
    z:           (N_ep, d_embed) — episode embeddings (numpy, float32/64).
    k_range:     Range of K values to evaluate.
    n_gap_refs:  Number of reference datasets for gap statistic.
    random_state: Random seed for gap statistic reference sampling.

    Returns
    -------
    ClusterResult with optimal K, labels, and diagnostics.

    Notes
    -----
    If N_ep < max(k_range), k_range is automatically clipped to N_ep-1.
    """
    n = len(z)
    if n < 2:
        raise ValueError(f"Need at least 2 samples; got {n}")

    # Clip k_range to valid values
    max_k = min(max(k_range), n - 1)
    min_k = max(min(k_range), 2)
    valid_k = [k for k in k_range if min_k <= k <= max_k]
    if not valid_k:
        raise ValueError(f"No valid K in k_range={k_range} for n={n} samples")

    silhouette_scores: dict[int, float] = {}
    all_labels: dict[int, np.ndarray] = {}

    for k in valid_k:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(z)
        all_labels[k] = labels
        # silhouette_score requires at least 2 distinct labels
        n_distinct = len(set(labels))
        if n_distinct >= 2:
            silhouette_scores[k] = float(silhouette_score(z, labels))
        else:
            silhouette_scores[k] = -1.0

    k_optimal = max(silhouette_scores, key=silhouette_scores.__getitem__)
    best_labels = all_labels[k_optimal]

    # Gap statistic (reference = uniform in bounding box of z)
    gap_stats = _gap_statistic(z, valid_k, n_gap_refs, random_state)

    return ClusterResult(
        labels=best_labels,
        k_optimal=k_optimal,
        silhouette_scores=silhouette_scores,
        gap_stats=gap_stats,
    )


def _inertia(z: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squared distances to centroid."""
    total = 0.0
    for k in set(labels):
        cluster = z[labels == k]
        centroid = cluster.mean(axis=0)
        total += float(np.sum((cluster - centroid) ** 2))
    return total


def _gap_statistic(
    z: np.ndarray,
    valid_k: list[int],
    n_refs: int,
    random_state: int,
) -> dict[int, float]:
    """Compute gap statistic for each K.

    gap(K) = E[log(W_ref(K))] − log(W(K))

    A larger gap is better.  K at which gap first exceeds gap(K+1) - s(K+1)
    is the Tibshirani et al. selection rule (not used here; stored for inspection).
    """
    rng = np.random.default_rng(random_state)
    z_min = z.min(axis=0)
    z_max = z.max(axis=0)

    gap_stats: dict[int, float] = {}
    for k in valid_k:
        # Observed inertia
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_obs = model.fit_predict(z)
        w_obs = _inertia(z, labels_obs)

        # Reference inertia (average over n_refs uniform samples)
        w_refs = []
        for _ in range(n_refs):
            z_ref = rng.uniform(z_min, z_max, size=z.shape)
            labels_ref = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z_ref)
            w_refs.append(_inertia(z_ref, labels_ref))

        log_w_ref = float(np.mean(np.log(np.maximum(w_refs, 1e-10))))
        log_w_obs = float(np.log(max(w_obs, 1e-10)))
        gap_stats[k] = log_w_ref - log_w_obs

    return gap_stats
