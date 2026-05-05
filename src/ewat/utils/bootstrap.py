"""Bootstrap confidence intervals for scalar metrics.

All functions return a CI dataclass with estimate, lo, hi.

Usage
-----
    from ewat.utils.bootstrap import bootstrap_ci, bootstrap_proportion_ci, bootstrap_auroc_ci

    ci = bootstrap_proportion_ci(successes=33, total=45)
    print(ci)  # "0.7333 [0.5867, 0.8667]"

    ci = bootstrap_auroc_ci(y_true, y_score, n=1000)
    print(f"AUROC = {ci.estimate:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass(frozen=True)
class CI:
    """Percentile bootstrap confidence interval."""

    estimate: float
    lo: float
    hi: float
    n_bootstrap: int
    alpha: float

    def __str__(self) -> str:
        if any(np.isnan(v) for v in (self.estimate, self.lo, self.hi)):
            return "NaN"
        return f"{self.estimate:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
        }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Percentile bootstrap CI for a generic scalar metric.

    Parameters
    ----------
    metric_fn:
        Function (array → scalar). Must handle arbitrary resamples.
    data:
        1-D array of observations (will be flattened if needed).
    n:
        Number of bootstrap resamples.
    alpha:
        Significance level: coverage = 1 - alpha (default → 95% CI).
    rng:
        Optional RNG for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data, dtype=float).ravel()
    if len(data) == 0:
        nan = float("nan")
        return CI(estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha)

    estimate = float(metric_fn(data))
    idx = rng.integers(0, len(data), size=(n, len(data)))
    boot_stats = np.array([metric_fn(data[idx[i]]) for i in range(n)])

    lo = float(np.nanpercentile(boot_stats, 100.0 * alpha / 2))
    hi = float(np.nanpercentile(boot_stats, 100.0 * (1.0 - alpha / 2)))
    return CI(estimate=estimate, lo=lo, hi=hi, n_bootstrap=n, alpha=alpha)


def bootstrap_mean_ci(
    data: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Percentile bootstrap CI for the mean of data."""
    return bootstrap_ci(np.mean, data, n=n, alpha=alpha, rng=rng)


def bootstrap_proportion_ci(
    successes: int,
    total: int,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Percentile bootstrap CI for a proportion (binomial).

    Much faster than the generic bootstrap_ci because the resampling
    is vectorised over all n bootstraps at once.

    Parameters
    ----------
    successes:
        Number of successes out of total trials.
    total:
        Total number of trials.
    """
    if rng is None:
        rng = np.random.default_rng()
    if total == 0:
        nan = float("nan")
        return CI(estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha)

    estimate = successes / total
    data = np.zeros(total, dtype=np.float64)
    data[:successes] = 1.0

    # Fully vectorised resampling: (n, total) indices → (n,) means
    idx = rng.integers(0, total, size=(n, total))
    boot_props = data[idx].mean(axis=1)  # (n,)

    lo = float(np.percentile(boot_props, 100.0 * alpha / 2))
    hi = float(np.percentile(boot_props, 100.0 * (1.0 - alpha / 2)))
    return CI(estimate=float(estimate), lo=lo, hi=hi, n_bootstrap=n, alpha=alpha)


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Percentile bootstrap CI for AUROC.

    Resamples (y_true, y_score) pairs jointly. Bootstrap resamples that
    contain only one class are skipped (to avoid degenerate AUC).

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0/1).
    y_score:
        Predicted scores / probabilities for class 1.
    """
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng()
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    nan = float("nan")
    if len(np.unique(y_true)) < 2:
        return CI(estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha)

    estimate = float(roc_auc_score(y_true, y_score))
    m = len(y_true)
    boot_stats: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_stats.append(float(roc_auc_score(yt, y_score[idx])))

    if not boot_stats:
        return CI(estimate=estimate, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha)

    arr = np.array(boot_stats)
    lo = float(np.percentile(arr, 100.0 * alpha / 2))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2)))
    return CI(estimate=estimate, lo=lo, hi=hi, n_bootstrap=n, alpha=alpha)


def bootstrap_silhouette_ci(
    z: np.ndarray,
    labels: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Percentile bootstrap CI for silhouette score.

    Resamples episodes (rows of z and corresponding labels) jointly.
    Skips resamples with fewer than 2 distinct labels.

    Parameters
    ----------
    z:
        (N, d) embedding matrix.
    labels:
        (N,) integer cluster labels.
    """
    from sklearn.metrics import silhouette_score

    if rng is None:
        rng = np.random.default_rng()
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels, dtype=int)
    m = len(z)

    nan = float("nan")
    if len(np.unique(labels)) < 2 or m < 2:
        return CI(estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha)

    estimate = float(silhouette_score(z, labels))
    boot_stats: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        z_s = z[idx]
        l_s = labels[idx]
        if len(np.unique(l_s)) < 2:
            continue
        boot_stats.append(float(silhouette_score(z_s, l_s)))

    if not boot_stats:
        return CI(estimate=estimate, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha)

    arr = np.array(boot_stats)
    lo = float(np.percentile(arr, 100.0 * alpha / 2))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2)))
    return CI(estimate=estimate, lo=lo, hi=hi, n_bootstrap=n, alpha=alpha)
