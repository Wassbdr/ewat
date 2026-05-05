"""Causal relations via Transfer Entropy (KSG estimator).

Kraskov, Stögbauer & Grassberger (2004) — estimator 1 (KSG-1).

TE(X → Y, lag=1) = CMI(Y_{t+1}; X_t | Y_t)
                 = MI(Y_{t+1}; (X_t, Y_t)) − MI(Y_{t+1}; Y_t)

Both MI terms are estimated via KSG-1 using the Chebyshev (L∞) metric in
joint space and marginal ball counts.

Algorithm for each cluster pair (i, j)
---------------------------------------
1. Load signal.npz for sampled episodes from clusters i and j.
2. Compute spatial mean over nodes → (T, 17) per episode.
3. Truncate pairs to min(T_i, T_j) so series have equal length.
4. For each of the 17 features: compute TE(x_f → y_f, lag=1).
5. Strength = sum of per-feature TE values.
6. Permutation test: shuffle x 100 times → p-value.
7. Emit OntologyRelation if p < p_threshold and n_episodes ≥ min_support.

Note: n_min=30 (formalisation.md) applies to the time-series length used for
KSG estimation. We enforce this via the min_series_length parameter.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree

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

    # k-th NN distances in joint space (Chebyshev / L∞)
    tree_xy = KDTree(xy, metric="chebyshev")
    dist_k = tree_xy.query(xy, k=k + 1, return_distance=True)[0][:, -1]

    # Marginal ball counts (strict, so subtract self)
    tree_x = KDTree(x, metric="chebyshev")
    tree_y = KDTree(y, metric="chebyshev")
    nx = tree_x.query_radius(x, r=dist_k, count_only=True) - 1
    ny = tree_y.query_radius(y, r=dist_k, count_only=True) - 1

    mi = float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return max(0.0, mi)


def _transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1, k: int = 5) -> float:
    """Transfer Entropy TE(X → Y, lag) via KSG-1 CMI.

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


def _total_te(x_mat: np.ndarray, y_mat: np.ndarray, lag: int = 1, k: int = 5) -> float:
    """Sum of TE over all 17 features. x_mat, y_mat: (T, 17)."""
    return sum(
        _transfer_entropy(x_mat[:, f], y_mat[:, f], lag=lag, k=k)
        for f in range(x_mat.shape[1])
    )


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def _load_mean_signal(features_root: Path, episode_id: str) -> np.ndarray:
    """Load signal.npz and return spatial mean → (T, 17) float32."""
    sig = np.load(features_root / episode_id / "signal.npz")["signal"].astype(np.float32)
    # (T, N, 17) → mean over N → (T, 17)
    sig = np.nan_to_num(sig, nan=0.0)
    return sig.mean(axis=1)


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
) -> list[OntologyRelation]:
    """Compute TE-KSG causal relations between cluster type pairs.

    Parameters
    ----------
    cluster_manifest:        {episode_id → {"cluster": int, ...}}
    features_root:           Root of feature store.
    n_clusters:              Total number of cluster types.
    lag:                     TE lag in timesteps (default 1).
    k_knn:                   KSG nearest-neighbour count.
    n_permutations:          Permutation test iterations.
    p_threshold:             Maximum p-value to emit a relation.
    min_support:             Minimum episodes per cluster to attempt TE.
    max_episodes_per_cluster: Cap episodes sampled per cluster (speed).
    min_series_length:       Minimum time-series length for KSG (n_min=30).
    seed:                    RNG seed.

    Returns
    -------
    List of OntologyRelation with relation_type="causal".
    """
    features_root = Path(features_root)
    rng = np.random.default_rng(seed)

    # Group episodes by cluster
    cluster_eps: dict[int, list[str]] = {c: [] for c in range(n_clusters)}
    for ep_id, info in cluster_manifest.items():
        cluster_eps[int(info["cluster"])].append(ep_id)

    # Precompute mean signals (cached)
    signal_cache: dict[str, np.ndarray] = {}

    def get_signal(ep_id: str) -> np.ndarray | None:
        if ep_id not in signal_cache:
            try:
                signal_cache[ep_id] = _load_mean_signal(features_root, ep_id)
            except Exception:
                signal_cache[ep_id] = None  # type: ignore[assignment]
        return signal_cache[ep_id]

    relations: list[OntologyRelation] = []

    for src, tgt in combinations(range(n_clusters), 2):
        eps_src = cluster_eps[src]
        eps_tgt = cluster_eps[tgt]

        if len(eps_src) < min_support or len(eps_tgt) < min_support:
            continue

        # Subsample
        if len(eps_src) > max_episodes_per_cluster:
            eps_src = rng.choice(eps_src, size=max_episodes_per_cluster, replace=False).tolist()
        if len(eps_tgt) > max_episodes_per_cluster:
            eps_tgt = rng.choice(eps_tgt, size=max_episodes_per_cluster, replace=False).tolist()

        # Build mean trajectory for each cluster (truncate to min_T)
        sigs_src = [s for ep in eps_src if (s := get_signal(ep)) is not None]
        sigs_tgt = [s for ep in eps_tgt if (s := get_signal(ep)) is not None]

        if not sigs_src or not sigs_tgt:
            continue

        min_T = min(min(s.shape[0] for s in sigs_src), min(s.shape[0] for s in sigs_tgt))
        if min_T - lag < min_series_length:
            continue

        x_mat = np.stack([s[:min_T] for s in sigs_src]).mean(axis=0)  # (min_T, 17)
        y_mat = np.stack([s[:min_T] for s in sigs_tgt]).mean(axis=0)  # (min_T, 17)

        observed_te = _total_te(x_mat, y_mat, lag=lag, k=k_knn)

        if observed_te == 0.0:
            continue

        # Permutation test (shuffle x features independently)
        perm_count = 0
        for _ in range(n_permutations):
            x_perm = rng.permutation(x_mat)   # shuffle rows (time)
            perm_te = _total_te(x_perm, y_mat, lag=lag, k=k_knn)
            if perm_te >= observed_te:
                perm_count += 1
        p_val = perm_count / n_permutations

        # Also compute reverse TE
        observed_te_rev = _total_te(y_mat, x_mat, lag=lag, k=k_knn)
        perm_count_rev = 0
        for _ in range(n_permutations):
            y_perm = rng.permutation(y_mat)
            if _total_te(y_perm, x_mat, lag=lag, k=k_knn) >= observed_te_rev:
                perm_count_rev += 1
        p_val_rev = perm_count_rev / n_permutations

        support = len(eps_src) + len(eps_tgt)
        if p_val < p_threshold:
            relations.append(OntologyRelation(
                source=src, target=tgt,
                relation_type="causal",
                strength=float(observed_te),
                p_value=float(p_val),
                support=support,
            ))
        if p_val_rev < p_threshold:
            relations.append(OntologyRelation(
                source=tgt, target=src,
                relation_type="causal",
                strength=float(observed_te_rev),
                p_value=float(p_val_rev),
                support=support,
            ))

    return relations
