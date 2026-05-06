"""Agglomerative clustering of episode embeddings.

Used after siamese fine-tuning to discover the empirical anomaly type
ontology C = {C_1, ..., C_K} from the z_e embedding space.

K selection
-----------
1. **Silhouette score** (Kaufman & Rousseeuw 1990) — argmax over k_range.
   Threshold for H1 falsification: silhouette < 0.3 on held-out split.
2. **Gap statistic** (Tibshirani et al. 2001) — computed for validation,
   not used for K selection to keep the pipeline simple.

Linkage / metric
----------------
The default ``("ward", "euclidean")`` setting matches the original EWAT
pipeline. Spherical / cosine variants are also supported because the
:class:`SiameseTyper` projects embeddings onto the unit sphere — Euclidean
distance on unit vectors is monotonically related to cosine distance, but
``("average", "cosine")`` is the standard recipe for spherical clustering
and produces tighter angular clusters when the embedding manifold is
genuinely curved.

The :func:`compare_linkages` helper runs the same pipeline for multiple
(linkage, metric) choices and returns silhouette / gap statistics per K so
H1 conclusions can be cross-checked.

References
----------
- Kaufman & Rousseeuw (1990) — Silhouette threshold justification.
- Tibshirani, Walther & Hastie (2001) — Gap statistic.
- Dhillon & Modha (2001) — Spherical k-means.
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
    labels:            (N_ep,) cluster assignments ∈ [0, k_optimal-1].
    k_optimal:         Number of clusters chosen by silhouette maximisation.
    silhouette_scores: {k → silhouette_score} for each k in k_range.
    gap_stats:         {k → gap_value} for each k in k_range (validation only).
    linkage:           Linkage criterion used.
    metric:            Distance metric used.
    """

    labels: np.ndarray
    k_optimal: int
    silhouette_scores: dict[int, float] = field(default_factory=dict)
    gap_stats: dict[int, float] = field(default_factory=dict)
    linkage: str = "ward"
    metric: str = "euclidean"


_VALID_LINKAGES: tuple[str, ...] = ("ward", "average", "complete", "single")
_VALID_METRICS: tuple[str, ...] = ("euclidean", "cosine", "manhattan", "l1", "l2")


def _validate_linkage_metric(linkage: str, metric: str) -> None:
    if linkage not in _VALID_LINKAGES:
        raise ValueError(
            f"unknown linkage {linkage!r}; expected one of {_VALID_LINKAGES}"
        )
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {_VALID_METRICS}"
        )
    if linkage == "ward" and metric != "euclidean":
        raise ValueError(
            "Ward linkage only supports Euclidean distance "
            "(scikit-learn restriction). Use linkage='average' for cosine."
        )


def _build_clusterer(
    n_clusters: int, linkage: str, metric: str,
) -> AgglomerativeClustering:
    """Sklearn ≥1.4 deprecated ``affinity`` in favour of ``metric``."""
    if linkage == "ward":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage, metric=metric,
    )


def _silhouette_metric_arg(metric: str) -> str:
    """Map our metric names to silhouette_score's accepted strings."""
    if metric in ("l2", "euclidean"):
        return "euclidean"
    if metric == "l1":
        return "manhattan"
    return metric


def cluster_embeddings(
    z: np.ndarray,
    k_range: range = range(2, 16),
    n_gap_refs: int = 10,
    random_state: int = 42,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> ClusterResult:
    """Agglomerative clustering with automatic K selection.

    Parameters
    ----------
    z:            ``(N_ep, d_embed)`` — episode embeddings.
    k_range:      Range of K values to evaluate.
    n_gap_refs:   Number of reference datasets for gap statistic.
    random_state: Random seed for gap statistic reference sampling.
    linkage:      Linkage criterion. ``"ward"``, ``"average"``,
                  ``"complete"`` or ``"single"``.
    metric:       Distance metric. ``"euclidean"`` (default) or
                  ``"cosine"``. Ward only accepts Euclidean.

    Returns
    -------
    ClusterResult with optimal K, labels, silhouette/gap diagnostics, and
    the (linkage, metric) pair used.

    Notes
    -----
    If ``N_ep < max(k_range)``, k_range is automatically clipped to
    ``N_ep - 1``.
    """
    _validate_linkage_metric(linkage, metric)
    n = len(z)
    if n < 2:
        raise ValueError(f"Need at least 2 samples; got {n}")

    max_k = min(max(k_range), n - 1)
    min_k = max(min(k_range), 2)
    valid_k = [k for k in k_range if min_k <= k <= max_k]
    if not valid_k:
        raise ValueError(f"No valid K in k_range={k_range} for n={n} samples")

    silhouette_scores: dict[int, float] = {}
    all_labels: dict[int, np.ndarray] = {}
    sil_metric = _silhouette_metric_arg(metric)

    for k in valid_k:
        model = _build_clusterer(k, linkage=linkage, metric=metric)
        labels = model.fit_predict(z)
        all_labels[k] = labels
        n_distinct = len(set(labels))
        if n_distinct >= 2:
            silhouette_scores[k] = float(
                silhouette_score(z, labels, metric=sil_metric)
            )
        else:
            silhouette_scores[k] = -1.0

    k_optimal = max(silhouette_scores, key=silhouette_scores.__getitem__)
    best_labels = all_labels[k_optimal]

    gap_stats = _gap_statistic(
        z, valid_k, n_gap_refs, random_state, linkage=linkage, metric=metric,
    )

    return ClusterResult(
        labels=best_labels,
        k_optimal=k_optimal,
        silhouette_scores=silhouette_scores,
        gap_stats=gap_stats,
        linkage=linkage,
        metric=metric,
    )


def compare_linkages(
    z: np.ndarray,
    k_range: range = range(2, 16),
    methods: tuple[tuple[str, str], ...] = (
        ("ward", "euclidean"),
        ("average", "cosine"),
        ("complete", "cosine"),
    ),
    n_gap_refs: int = 5,
    random_state: int = 42,
) -> dict[str, ClusterResult]:
    """Run :func:`cluster_embeddings` for several (linkage, metric) pairs.

    Returns a dict ``{"<linkage>__<metric>" → ClusterResult}``. Useful to
    cross-validate the optimal K* and silhouette geometry across methods,
    especially on L2-normalised embeddings where Euclidean and cosine are
    equivalent up to a monotonic transform.

    A flat side-effect is provided: the function never raises if a
    particular method is invalid; it is silently skipped. Pass an explicit
    ``methods`` tuple to control the comparison.
    """
    results: dict[str, ClusterResult] = {}
    for linkage, metric in methods:
        try:
            _validate_linkage_metric(linkage, metric)
        except ValueError:
            continue
        key = f"{linkage}__{metric}"
        results[key] = cluster_embeddings(
            z,
            k_range=k_range,
            n_gap_refs=n_gap_refs,
            random_state=random_state,
            linkage=linkage,
            metric=metric,
        )
    return results


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
    linkage: str = "ward",
    metric: str = "euclidean",
) -> dict[int, float]:
    """Compute gap statistic for each K.

    gap(K) = E[log(W_ref(K))] − log(W(K))

    A larger gap is better.  K at which gap first exceeds gap(K+1) - s(K+1)
    is the Tibshirani et al. selection rule (not used here; stored for
    inspection). The reference distribution is uniform over the bounding
    box of ``z`` for both Euclidean and cosine pipelines.
    """
    rng = np.random.default_rng(random_state)
    z_min = z.min(axis=0)
    z_max = z.max(axis=0)

    gap_stats: dict[int, float] = {}
    for k in valid_k:
        model = _build_clusterer(k, linkage=linkage, metric=metric)
        labels_obs = model.fit_predict(z)
        w_obs = _inertia(z, labels_obs)

        w_refs = []
        for _ in range(n_refs):
            z_ref = rng.uniform(z_min, z_max, size=z.shape)
            labels_ref = _build_clusterer(
                k, linkage=linkage, metric=metric,
            ).fit_predict(z_ref)
            w_refs.append(_inertia(z_ref, labels_ref))

        log_w_ref = float(np.mean(np.log(np.maximum(w_refs, 1e-10))))
        log_w_obs = float(np.log(max(w_obs, 1e-10)))
        gap_stats[k] = log_w_ref - log_w_obs

    return gap_stats
