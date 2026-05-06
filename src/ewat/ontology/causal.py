"""Causal relations via Transfer Entropy (KSG estimator).

Kraskov, Stögbauer & Grassberger (2004) — estimator 1 (KSG-1).

TE(X → Y, lag=1) = CMI(Y_{t+1}; X_t | Y_t)
                 = MI(Y_{t+1}; (X_t, Y_t)) − MI(Y_{t+1}; Y_t)

Both MI terms are estimated via KSG-1 using the Chebyshev (L∞) metric in
joint space and marginal ball counts.

TE aggregation across the 17 features
======================================

Two methods are supported via the ``te_method`` argument:

- ``"univariate_sum"`` (default, kept for backward compatibility) — TE is
  computed feature-by-feature and **summed**.
  *Bias warning*: a sum of marginal TE estimates ignores synergy and may
  double-count shared information. It is fast (n_features × KSG-1) but is
  **not** a multivariate causal information measure.
- ``"multivariate"`` — proper KSG-1 CMI on the 17-D joint state,
  TE(X → Y) = MI(Y_{t+lag}; (X_t, Y_t)) − MI(Y_{t+lag}; Y_t) where every
  vector lives in ℝ^{17}. Slower but theoretically sound.

Episode aggregation
===================

The current implementation **averages episodes per cluster** before estimating
TE on the mean trajectory. This collapses inter-episode variance and is a
known source of ecological bias. A future iteration should switch to a
pooled / hierarchical estimator (e.g. averaging TE across episode pairs,
or stacking episodes with KSG-on-blocks).

Statistical inference
=====================

Permutation null is built by shuffling the time axis of X (or Y for the
reverse direction). The p-value uses the standard biased estimator with the
``+1`` correction recommended by Phipson & Smyth (2010):

    p_hat = (1 + #{T_perm ≥ T_obs}) / (1 + M)

so that observed permutations cannot yield p = 0 exactly. Multiple
testing (K × (K − 1) directed pair-tests) is corrected via Holm or BH-FDR
through the ``correction`` argument; ``p_value`` on returned relations is
the *adjusted* p-value.

References
==========
- Kraskov, Stögbauer & Grassberger (2004) — Estimating mutual information.
- Schreiber (2000) — Measuring information transfer.
- Phipson & Smyth (2010) — Permutation p-values should never be zero.
- Benjamini & Hochberg (1995) — Controlling FDR.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import digamma  # type: ignore[import-untyped]
from sklearn.neighbors import KDTree

from ewat.ontology.cooccurrence import benjamini_hochberg, holm_bonferroni
from ewat.ontology.graph import OntologyRelation

# ---------------------------------------------------------------------------
# KSG-1 mutual information estimator
# ---------------------------------------------------------------------------

def _ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """KSG estimator 1 for I(X; Y).

    Parameters
    ----------
    x, y: 1-D or 2-D arrays of shape (n,) or (n, d). Must have equal length.
    k:    Number of nearest neighbours.

    Returns
    -------
    Non-negative float — estimated mutual information in nats.
    """
    n = len(x)
    if n < k + 2:
        return 0.0

    x = np.atleast_2d(x).T if x.ndim == 1 else np.asarray(x, dtype=float)
    y = np.atleast_2d(y).T if y.ndim == 1 else np.asarray(y, dtype=float)

    xy = np.hstack([x, y])

    tree_xy = KDTree(xy, metric="chebyshev")
    dist_k = tree_xy.query(xy, k=k + 1, return_distance=True)[0][:, -1]

    tree_x = KDTree(x, metric="chebyshev")
    tree_y = KDTree(y, metric="chebyshev")
    nx = tree_x.query_radius(x, r=dist_k, count_only=True) - 1
    ny = tree_y.query_radius(y, r=dist_k, count_only=True) - 1

    mi = float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return max(0.0, mi)


def _transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1, k: int = 5) -> float:
    """Transfer Entropy TE(X → Y, lag) via KSG-1 CMI for univariate signals.

    TE = MI(Y_{t+lag}; (X_t, Y_t)) − MI(Y_{t+lag}; Y_t)
    """
    n = len(x)
    if n - lag < k + 2:
        return 0.0

    y_fut = y[lag:]
    y_past = y[:-lag]
    x_past = x[:-lag]

    xy_past = np.column_stack([x_past, y_past])
    mi_full = _ksg_mi(y_fut, xy_past, k=k)
    mi_past = _ksg_mi(y_fut, y_past, k=k)
    return max(0.0, mi_full - mi_past)


def _multivariate_te(
    x_mat: np.ndarray, y_mat: np.ndarray, lag: int = 1, k: int = 5
) -> float:
    """Proper multivariate Transfer Entropy TE(X → Y) on full ℝ^d signals.

    x_mat, y_mat: shape (T, d). All d features are treated jointly.
    """
    n = x_mat.shape[0]
    if n - lag < k + 2:
        return 0.0

    y_fut = y_mat[lag:]
    y_past = y_mat[:-lag]
    x_past = x_mat[:-lag]

    xy_past = np.hstack([x_past, y_past])
    mi_full = _ksg_mi(y_fut, xy_past, k=k)
    mi_past = _ksg_mi(y_fut, y_past, k=k)
    return max(0.0, mi_full - mi_past)


def _total_te(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lag: int = 1,
    k: int = 5,
    method: Literal["univariate_sum", "multivariate"] = "univariate_sum",
) -> float:
    """Aggregate TE over the feature dimension.

    See module docstring for the bias warnings on each method.
    """
    if method == "univariate_sum":
        return sum(
            _transfer_entropy(x_mat[:, f], y_mat[:, f], lag=lag, k=k)
            for f in range(x_mat.shape[1])
        )
    if method == "multivariate":
        return _multivariate_te(x_mat, y_mat, lag=lag, k=k)
    raise ValueError(f"unknown te_method: {method!r}")


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def _load_mean_signal(features_root: Path, episode_id: str) -> np.ndarray:
    """Load signal.npz and return spatial mean → (T, 17) float32."""
    sig = np.load(features_root / episode_id / "signal.npz")["signal"].astype(np.float32)
    sig = np.nan_to_num(sig, nan=0.0)
    return sig.mean(axis=1)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _permutation_p_value(
    observed: float,
    perm_stats: list[float],
) -> float:
    """Phipson–Smyth (2010) corrected p-value for a one-sided permutation test.

        p_hat = (1 + #{T_perm ≥ T_obs}) / (1 + M)
    """
    m = len(perm_stats)
    if m == 0:
        return 1.0
    geq = sum(1 for s in perm_stats if s >= observed)
    return float((1 + geq) / (1 + m))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_causal_relations(
    cluster_manifest: dict[str, dict],
    features_root: Path,
    n_clusters: int,
    lag: int = 1,
    k_knn: int = 5,
    n_permutations: int = 100,
    p_threshold: float = 0.05,
    min_support: int = 5,
    max_episodes_per_cluster: int = 20,
    min_series_length: int = 30,
    seed: int = 42,
    te_method: Literal["univariate_sum", "multivariate"] = "univariate_sum",
    correction: Literal["holm", "bh", "none"] = "bh",
) -> list[OntologyRelation]:
    """Compute TE-KSG causal relations between cluster type pairs.

    Parameters
    ----------
    cluster_manifest:
        ``{episode_id → {"cluster": int, ...}}``.
    features_root:
        Root of the feature store.
    n_clusters:
        Total number of cluster types.
    lag:
        TE lag in timesteps (default 1).
    k_knn:
        KSG nearest-neighbour count.
    n_permutations:
        Permutation test iterations.
    p_threshold:
        Maximum *adjusted* p-value to emit a relation.
    min_support:
        Minimum episodes per cluster to attempt TE.
    max_episodes_per_cluster:
        Cap episodes sampled per cluster (for speed).
    min_series_length:
        Minimum time-series length for KSG (n_min=30 in formalisation.md).
    seed:
        RNG seed (controls subsampling and permutation shuffles).
    te_method:
        ``"univariate_sum"`` (fast, biased) or ``"multivariate"`` (KSG-1 in ℝ^d).
    correction:
        Multiple-testing correction across the K(K−1) directed pair-tests.
        ``"bh"`` (default), ``"holm"``, or ``"none"``.

    Returns
    -------
    List of ``OntologyRelation`` with ``relation_type="causal"``. The
    ``p_value`` field is the *adjusted* p-value.
    """
    if correction not in ("holm", "bh", "none"):
        raise ValueError(f"unknown correction: {correction!r}")
    if te_method not in ("univariate_sum", "multivariate"):
        raise ValueError(f"unknown te_method: {te_method!r}")

    features_root = Path(features_root)
    rng = np.random.default_rng(seed)

    cluster_eps: dict[int, list[str]] = {c: [] for c in range(n_clusters)}
    for ep_id, info in cluster_manifest.items():
        cluster_eps[int(info["cluster"])].append(ep_id)

    signal_cache: dict[str, np.ndarray | None] = {}

    def get_signal(ep_id: str) -> np.ndarray | None:
        if ep_id not in signal_cache:
            try:
                signal_cache[ep_id] = _load_mean_signal(features_root, ep_id)
            except Exception:
                signal_cache[ep_id] = None
        return signal_cache[ep_id]

    candidates: list[tuple[int, int, float, int]] = []
    raw_pvals: list[float] = []

    for src, tgt in combinations(range(n_clusters), 2):
        eps_src = cluster_eps[src]
        eps_tgt = cluster_eps[tgt]

        if len(eps_src) < min_support or len(eps_tgt) < min_support:
            continue

        if len(eps_src) > max_episodes_per_cluster:
            eps_src = rng.choice(
                eps_src, size=max_episodes_per_cluster, replace=False
            ).tolist()
        if len(eps_tgt) > max_episodes_per_cluster:
            eps_tgt = rng.choice(
                eps_tgt, size=max_episodes_per_cluster, replace=False
            ).tolist()

        sigs_src = [s for ep in eps_src if (s := get_signal(ep)) is not None]
        sigs_tgt = [s for ep in eps_tgt if (s := get_signal(ep)) is not None]

        if not sigs_src or not sigs_tgt:
            continue

        min_t = min(min(s.shape[0] for s in sigs_src), min(s.shape[0] for s in sigs_tgt))
        if min_t - lag < min_series_length:
            continue

        # NOTE: averaging trajectories across episodes is an ecological bias
        # source. Documented at module level; future work should switch to a
        # hierarchical / pooled estimator.
        x_mat = np.stack([s[:min_t] for s in sigs_src]).mean(axis=0)
        y_mat = np.stack([s[:min_t] for s in sigs_tgt]).mean(axis=0)

        observed_te = _total_te(x_mat, y_mat, lag=lag, k=k_knn, method=te_method)
        observed_te_rev = _total_te(y_mat, x_mat, lag=lag, k=k_knn, method=te_method)

        # Permutation null
        perm_te: list[float] = []
        perm_te_rev: list[float] = []
        for _ in range(n_permutations):
            x_perm = rng.permutation(x_mat)
            perm_te.append(_total_te(x_perm, y_mat, lag=lag, k=k_knn, method=te_method))
            y_perm = rng.permutation(y_mat)
            perm_te_rev.append(
                _total_te(y_perm, x_mat, lag=lag, k=k_knn, method=te_method)
            )

        support = len(eps_src) + len(eps_tgt)

        # Skip pairs with strictly zero observed TE (numerical zero)
        if observed_te > 0.0:
            p_raw = _permutation_p_value(observed_te, perm_te)
            candidates.append((src, tgt, float(observed_te), support))
            raw_pvals.append(p_raw)
        if observed_te_rev > 0.0:
            p_raw_rev = _permutation_p_value(observed_te_rev, perm_te_rev)
            candidates.append((tgt, src, float(observed_te_rev), support))
            raw_pvals.append(p_raw_rev)

    if not candidates:
        return []

    if correction == "holm":
        adj_pvals = holm_bonferroni(raw_pvals)
    elif correction == "bh":
        adj_pvals = benjamini_hochberg(raw_pvals)
    else:
        adj_pvals = list(raw_pvals)

    relations: list[OntologyRelation] = []
    for (src, tgt, te_val, support), p_adj in zip(candidates, adj_pvals):
        if not np.isnan(p_adj) and p_adj < p_threshold:
            relations.append(
                OntologyRelation(
                    source=src,
                    target=tgt,
                    relation_type="causal",
                    strength=te_val,
                    p_value=float(p_adj),
                    support=support,
                )
            )
    return relations
