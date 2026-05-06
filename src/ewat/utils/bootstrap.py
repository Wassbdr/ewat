"""Bootstrap confidence intervals for scalar metrics.

All functions return a ``CI`` dataclass with estimate, lo, hi.

Two interval families are supported:

- ``method="percentile"`` — classic percentile bootstrap (default; fast).
- ``method="bca"`` — bias-corrected and accelerated bootstrap (Efron 1987).
  Recommended when the statistic is **skewed** or near a boundary
  (e.g. AUROC close to 1.0 or 0.0).

Reproducibility
---------------

A bootstrap CI without a fixed RNG is **not reproducible**. The functions
emit a ``RuntimeWarning`` when called with ``rng=None`` so that downstream
experiments (multi-seed runs, MLflow logs) can detect non-deterministic CI
inputs early. Pass an explicit ``np.random.Generator`` to silence the
warning.

Usage
-----

.. code-block:: python

    from ewat.utils.bootstrap import bootstrap_auroc_ci

    rng = np.random.default_rng(42)
    ci = bootstrap_auroc_ci(y_true, y_score, n=1000, rng=rng, method="bca")
    print(f"AUROC = {ci.estimate:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]")

References
----------
- Efron (1987) — Better bootstrap confidence intervals.
- Davison & Hinkley (1997) — Bootstrap Methods and Their Application.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

CIMethod = Literal["percentile", "bca"]


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return ``rng`` if provided, else a fresh ``default_rng()`` plus a warning."""
    if rng is None:
        warnings.warn(
            "bootstrap CI called without an explicit `rng` — results will be "
            "non-reproducible. Pass `rng=np.random.default_rng(seed)` to fix this.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.random.default_rng()
    return rng


@dataclasses.dataclass(frozen=True)
class CI:
    """Bootstrap confidence interval (percentile or BCa)."""

    estimate: float
    lo: float
    hi: float
    n_bootstrap: int
    alpha: float
    method: CIMethod = "percentile"
    n_effective: int | None = None

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
            "method": self.method,
            "n_effective": self.n_effective,
        }


def _percentile_interval(
    boot_stats: np.ndarray, alpha: float
) -> tuple[float, float]:
    lo = float(np.nanpercentile(boot_stats, 100.0 * alpha / 2))
    hi = float(np.nanpercentile(boot_stats, 100.0 * (1.0 - alpha / 2)))
    return lo, hi


def _bca_interval(
    estimate: float,
    boot_stats: np.ndarray,
    jackknife_stats: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval.

    Falls back to percentile bounds when the BCa correction degenerates
    (e.g. all jackknife replicates equal, or ``z0`` infinite).
    """
    boot_stats = boot_stats[~np.isnan(boot_stats)]
    if boot_stats.size == 0:
        nan = float("nan")
        return nan, nan

    proportion_below = float(np.mean(boot_stats < estimate))
    if proportion_below in (0.0, 1.0):
        return _percentile_interval(boot_stats, alpha)

    z0 = float(norm.ppf(proportion_below))

    jk_mean = float(np.mean(jackknife_stats))
    diffs = jk_mean - jackknife_stats
    num = float(np.sum(diffs**3))
    den = 6.0 * float(np.sum(diffs**2)) ** 1.5
    if den == 0.0 or not np.isfinite(num) or not np.isfinite(den):
        return _percentile_interval(boot_stats, alpha)
    a = num / den

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1.0 - alpha / 2)

    def _adjust(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if denom == 0.0:
            return float("nan")
        return float(norm.cdf(z0 + (z0 + z) / denom))

    p_lo = _adjust(z_lo)
    p_hi = _adjust(z_hi)
    if not (np.isfinite(p_lo) and np.isfinite(p_hi)):
        return _percentile_interval(boot_stats, alpha)
    p_lo = float(np.clip(p_lo, 0.0, 1.0))
    p_hi = float(np.clip(p_hi, 0.0, 1.0))
    lo = float(np.nanpercentile(boot_stats, 100.0 * p_lo))
    hi = float(np.nanpercentile(boot_stats, 100.0 * p_hi))
    return lo, hi


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    method: CIMethod = "percentile",
) -> CI:
    """Bootstrap CI for a generic scalar metric.

    Parameters
    ----------
    metric_fn:
        Function (array → scalar). Must handle arbitrary resamples.
    data:
        1-D array of observations (will be flattened if needed).
    n:
        Number of bootstrap resamples.
    alpha:
        Significance level: coverage = 1 − alpha (default → 95% CI).
    rng:
        Optional RNG. Required for reproducibility (warns if ``None``).
    method:
        ``"percentile"`` (default) or ``"bca"``.
    """
    rng = _resolve_rng(rng)
    data = np.asarray(data, dtype=float).ravel()
    if len(data) == 0:
        nan = float("nan")
        return CI(
            estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha, method=method,
            n_effective=0,
        )

    estimate = float(metric_fn(data))
    idx = rng.integers(0, len(data), size=(n, len(data)))
    boot_stats = np.array([metric_fn(data[idx[i]]) for i in range(n)])

    if method == "bca":
        m = len(data)
        jk = np.array([metric_fn(np.delete(data, i)) for i in range(m)])
        lo, hi = _bca_interval(estimate, boot_stats, jk, alpha)
    else:
        lo, hi = _percentile_interval(boot_stats, alpha)

    return CI(
        estimate=estimate, lo=lo, hi=hi, n_bootstrap=n, alpha=alpha, method=method,
        n_effective=int(np.sum(~np.isnan(boot_stats))),
    )


def bootstrap_mean_ci(
    data: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    method: CIMethod = "percentile",
) -> CI:
    """Bootstrap CI for the mean of data."""
    return bootstrap_ci(np.mean, data, n=n, alpha=alpha, rng=rng, method=method)


def bootstrap_proportion_ci(
    successes: int,
    total: int,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    method: CIMethod = "percentile",
) -> CI:
    """Bootstrap CI for a proportion (binomial).

    Vectorised resampling for speed.
    """
    rng = _resolve_rng(rng)
    if total == 0:
        nan = float("nan")
        return CI(
            estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha, method=method,
            n_effective=0,
        )

    estimate = successes / total
    data = np.zeros(total, dtype=np.float64)
    data[:successes] = 1.0

    idx = rng.integers(0, total, size=(n, total))
    boot_props = data[idx].mean(axis=1)

    if method == "bca":
        jk = np.array(
            [np.delete(data, i).mean() if total > 1 else estimate for i in range(total)]
        )
        lo, hi = _bca_interval(float(estimate), boot_props, jk, alpha)
    else:
        lo, hi = _percentile_interval(boot_props, alpha)
    return CI(
        estimate=float(estimate), lo=lo, hi=hi, n_bootstrap=n, alpha=alpha,
        method=method, n_effective=n,
    )


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    method: CIMethod = "percentile",
) -> CI:
    """Bootstrap CI for AUROC.

    Resamples ``(y_true, y_score)`` pairs jointly. Bootstrap resamples that
    contain only one class are skipped (to avoid degenerate AUC). The number
    of *effective* (non-skipped) draws is reported on the returned ``CI``.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0/1).
    y_score:
        Predicted scores / probabilities for class 1.
    method:
        ``"percentile"`` (default) or ``"bca"``. BCa is recommended when
        AUROC is close to 0 or 1.
    """
    from sklearn.metrics import roc_auc_score

    rng = _resolve_rng(rng)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    nan = float("nan")
    if len(np.unique(y_true)) < 2:
        return CI(
            estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha, method=method,
            n_effective=0,
        )

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
        return CI(
            estimate=estimate, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha,
            method=method, n_effective=0,
        )

    arr = np.array(boot_stats)
    if method == "bca":
        # Jackknife on observations: drop one (y_true, y_score) at a time.
        jk: list[float] = []
        for i in range(m):
            yt = np.delete(y_true, i)
            ys = np.delete(y_score, i)
            if len(np.unique(yt)) < 2:
                continue
            jk.append(float(roc_auc_score(yt, ys)))
        if jk:
            lo, hi = _bca_interval(estimate, arr, np.asarray(jk), alpha)
        else:
            lo, hi = _percentile_interval(arr, alpha)
    else:
        lo, hi = _percentile_interval(arr, alpha)
    return CI(
        estimate=estimate, lo=lo, hi=hi, n_bootstrap=n, alpha=alpha, method=method,
        n_effective=len(boot_stats),
    )


def bootstrap_silhouette_ci(
    z: np.ndarray,
    labels: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    method: CIMethod = "percentile",
) -> CI:
    """Bootstrap CI for silhouette score.

    Resamples episodes (rows of ``z`` and corresponding ``labels``) jointly.
    Skips resamples with fewer than 2 distinct labels.
    """
    from sklearn.metrics import silhouette_score

    rng = _resolve_rng(rng)
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels, dtype=int)
    m = len(z)

    nan = float("nan")
    if len(np.unique(labels)) < 2 or m < 2:
        return CI(
            estimate=nan, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha, method=method,
            n_effective=0,
        )

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
        return CI(
            estimate=estimate, lo=nan, hi=nan, n_bootstrap=n, alpha=alpha,
            method=method, n_effective=0,
        )

    arr = np.array(boot_stats)
    if method == "bca":
        jk: list[float] = []
        for i in range(m):
            z_jk = np.delete(z, i, axis=0)
            l_jk = np.delete(labels, i)
            if len(np.unique(l_jk)) < 2 or z_jk.shape[0] < 2:
                continue
            jk.append(float(silhouette_score(z_jk, l_jk)))
        if jk:
            lo, hi = _bca_interval(estimate, arr, np.asarray(jk), alpha)
        else:
            lo, hi = _percentile_interval(arr, alpha)
    else:
        lo, hi = _percentile_interval(arr, alpha)
    return CI(
        estimate=estimate, lo=lo, hi=hi, n_bootstrap=n, alpha=alpha, method=method,
        n_effective=len(boot_stats),
    )
