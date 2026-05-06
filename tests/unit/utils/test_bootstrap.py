"""Tests for ``ewat.utils.bootstrap``."""

from __future__ import annotations

import numpy as np
import pytest

from ewat.utils.bootstrap import (
    CI,
    bootstrap_auroc_ci,
    bootstrap_ci,
    bootstrap_mean_ci,
    bootstrap_proportion_ci,
    bootstrap_silhouette_ci,
)


# ---------------------------------------------------------------------------
# CI dataclass
# ---------------------------------------------------------------------------


def test_ci_str_and_dict():
    ci = CI(estimate=0.5, lo=0.4, hi=0.6, n_bootstrap=100, alpha=0.05)
    s = str(ci)
    assert "0.5000" in s
    assert "0.4000" in s
    d = ci.as_dict()
    assert d["estimate"] == 0.5
    assert d["ci_lo"] == 0.4
    assert d["ci_hi"] == 0.6
    assert d["method"] == "percentile"


def test_ci_str_handles_nan():
    nan = float("nan")
    ci = CI(estimate=nan, lo=nan, hi=nan, n_bootstrap=100, alpha=0.05)
    assert str(ci) == "NaN"


# ---------------------------------------------------------------------------
# Reproducibility / RNG warning
# ---------------------------------------------------------------------------


def test_warns_when_no_rng():
    with pytest.warns(RuntimeWarning):
        bootstrap_proportion_ci(10, 30, n=20)


def test_no_warning_with_rng(recwarn):
    rng = np.random.default_rng(0)
    bootstrap_proportion_ci(10, 30, n=20, rng=rng)
    runtime = [w for w in recwarn.list if w.category is RuntimeWarning]
    assert runtime == []


def test_same_seed_gives_same_ci():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    ci_a = bootstrap_proportion_ci(20, 50, n=200, rng=rng_a)
    ci_b = bootstrap_proportion_ci(20, 50, n=200, rng=rng_b)
    assert ci_a.lo == ci_b.lo
    assert ci_a.hi == ci_b.hi


# ---------------------------------------------------------------------------
# Generic / mean CI
# ---------------------------------------------------------------------------


def test_bootstrap_mean_covers_population_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 0.1, size=200)
    ci = bootstrap_mean_ci(data, n=500, rng=np.random.default_rng(1))
    assert ci.lo <= 0.5 <= ci.hi
    assert ci.estimate == pytest.approx(float(data.mean()), rel=1e-6)


def test_bootstrap_ci_handles_empty_data():
    ci = bootstrap_ci(np.mean, np.array([]), n=10, rng=np.random.default_rng(0))
    assert np.isnan(ci.estimate)
    assert ci.n_effective == 0


def test_bootstrap_lo_le_hi():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, size=80)
    ci = bootstrap_mean_ci(data, n=300, rng=np.random.default_rng(3))
    assert ci.lo <= ci.hi
    in_bounds = ci.lo <= ci.estimate <= ci.hi
    on_lower = np.isclose(ci.lo, ci.estimate)
    on_upper = np.isclose(ci.hi, ci.estimate)
    assert in_bounds or on_lower or on_upper


# ---------------------------------------------------------------------------
# Proportion
# ---------------------------------------------------------------------------


def test_proportion_ci_total_zero():
    ci = bootstrap_proportion_ci(0, 0, n=50, rng=np.random.default_rng(0))
    assert np.isnan(ci.estimate)


def test_proportion_ci_estimate_matches():
    ci = bootstrap_proportion_ci(33, 45, n=300, rng=np.random.default_rng(0))
    assert ci.estimate == pytest.approx(33 / 45, rel=1e-6)
    assert 0.0 <= ci.lo <= ci.hi <= 1.0


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def test_auroc_ci_single_class_returns_nan():
    y_true = np.zeros(20, dtype=int)
    y_score = np.linspace(0, 1, 20)
    ci = bootstrap_auroc_ci(y_true, y_score, n=20, rng=np.random.default_rng(0))
    assert np.isnan(ci.estimate)


def test_auroc_ci_perfect_separation():
    rng = np.random.default_rng(3)
    y_true = np.array([0] * 25 + [1] * 25)
    y_score = np.concatenate([rng.normal(0, 0.1, 25), rng.normal(2, 0.1, 25)])
    ci = bootstrap_auroc_ci(y_true, y_score, n=200, rng=np.random.default_rng(4))
    assert ci.estimate >= 0.95
    assert ci.lo >= 0.7


def test_auroc_ci_reports_n_effective():
    rng = np.random.default_rng(5)
    y_true = np.array([0] * 20 + [1] * 20)
    y_score = rng.normal(size=40)
    ci = bootstrap_auroc_ci(y_true, y_score, n=100, rng=np.random.default_rng(6))
    assert ci.n_effective is not None
    assert ci.n_effective <= 100


# ---------------------------------------------------------------------------
# BCa
# ---------------------------------------------------------------------------


def test_bca_returns_finite_for_skewed_metric():
    rng = np.random.default_rng(0)
    data = rng.exponential(scale=1.0, size=80)
    ci = bootstrap_ci(
        np.mean, data, n=500, rng=np.random.default_rng(1), method="bca"
    )
    assert ci.method == "bca"
    assert np.isfinite(ci.lo)
    assert np.isfinite(ci.hi)
    assert ci.lo <= ci.hi


def test_bca_close_to_percentile_for_symmetric():
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, size=200)
    ci_p = bootstrap_ci(
        np.mean, data, n=500, rng=np.random.default_rng(8), method="percentile",
    )
    ci_b = bootstrap_ci(
        np.mean, data, n=500, rng=np.random.default_rng(8), method="bca",
    )
    # For a symmetric statistic, BCa and percentile bounds should be close.
    assert abs(ci_p.lo - ci_b.lo) < 0.1
    assert abs(ci_p.hi - ci_b.hi) < 0.1


def test_bca_auroc_finite():
    rng = np.random.default_rng(9)
    y_true = np.array([0] * 30 + [1] * 30)
    y_score = np.concatenate([rng.normal(0, 1, 30), rng.normal(1, 1, 30)])
    ci = bootstrap_auroc_ci(
        y_true, y_score, n=200, rng=np.random.default_rng(10), method="bca",
    )
    assert np.isfinite(ci.lo)
    assert np.isfinite(ci.hi)


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------


def test_silhouette_ci_two_clusters():
    rng = np.random.default_rng(11)
    z = np.vstack([rng.normal(0, 0.1, (15, 2)), rng.normal(3, 0.1, (15, 2))])
    labels = np.array([0] * 15 + [1] * 15)
    ci = bootstrap_silhouette_ci(z, labels, n=100, rng=np.random.default_rng(12))
    assert ci.estimate > 0.5
    assert ci.lo <= ci.hi


def test_silhouette_ci_single_cluster_returns_nan():
    rng = np.random.default_rng(13)
    z = rng.normal(size=(20, 3))
    labels = np.zeros(20, dtype=int)
    ci = bootstrap_silhouette_ci(z, labels, n=50, rng=np.random.default_rng(14))
    assert np.isnan(ci.estimate)
