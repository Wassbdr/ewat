"""Tests for Step 9 audit fixes.

- 9.1: OpenMax.tail_size_ratio adaptive
- 9.2: MahalanobisOOD as drop-in alternative to OpenMax
"""

import numpy as np
import pytest

from ewat.openset import MahalanobisOOD, OpenMax


def _separated_blobs(n_classes=4, n_per=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    pts, labels = [], []
    for c in range(n_classes):
        center = np.zeros(d)
        center[c % d] = 5.0
        pts.append(center + rng.normal(0, 0.5, size=(n_per, d)))
        labels.append(np.full(n_per, c, dtype=int))
    return np.vstack(pts), np.concatenate(labels)


# -- 9.1 OpenMax.tail_size_ratio -------------------------------------------

def test_tail_size_ratio_validates_range():
    with pytest.raises(ValueError, match="tail_size_ratio"):
        OpenMax(n_classes=4, tail_size_ratio=0.0)
    with pytest.raises(ValueError, match="tail_size_ratio"):
        OpenMax(n_classes=4, tail_size_ratio=1.5)


def test_tail_size_ratio_adapts_per_class():
    """tail_size = max(3, int(n_class * ratio))."""
    X, y = _separated_blobs(n_classes=3, n_per=30, d=4)
    o = OpenMax(n_classes=3, tail_size_ratio=0.25).fit(X, y)
    # n_per=30, ratio=0.25 → tail=7
    for c in range(3):
        assert o._effective_tail_size[c] == max(3, int(30 * 0.25))


def test_tail_size_ratio_min_floor_at_3():
    """Even with tiny n_class, effective tail stays >= 3."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (10, 4))
    y = np.concatenate([np.zeros(8, dtype=int), np.ones(2, dtype=int)])
    o = OpenMax(n_classes=2, tail_size_ratio=0.1).fit(X, y)
    # n=2 → 2*0.1=0.2 → floor at 3 (but n_class < 3 so degenerate path)
    assert o._effective_tail_size[1] >= 3


def test_tail_size_ratio_overrides_tail_size():
    X, y = _separated_blobs(n_classes=3, n_per=20)
    o_static = OpenMax(n_classes=3, tail_size=15).fit(X, y)
    o_ratio = OpenMax(n_classes=3, tail_size=15, tail_size_ratio=0.5).fit(X, y)
    # static uses 15 directly, ratio uses 10 (=20*0.5)
    assert o_static._effective_tail_size[0] == 15
    assert o_ratio._effective_tail_size[0] == 10


# -- 9.2 MahalanobisOOD ----------------------------------------------------

def test_mahalanobis_fit_and_predict_shapes():
    X, y = _separated_blobs(n_classes=4, n_per=30, d=8)
    m = MahalanobisOOD(n_classes=4).fit(X, y)
    assert m.fitted_
    assert m.class_means_.shape == (4, 8)
    assert m.precision_.shape == (8, 8)
    p = m.predict_proba(X[:5])
    assert p.shape == (5, 5)   # K + 1
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_mahalanobis_known_input_low_unknown_score():
    """A known training input should produce low unknown score."""
    X, y = _separated_blobs(n_classes=3, n_per=30, d=8, seed=0)
    m = MahalanobisOOD(n_classes=3).fit(X, y)
    u = m.unknown_score(X)
    # Most training points should have unknown < 0.5
    assert (u < 0.5).sum() >= 0.7 * len(u)


def test_mahalanobis_far_input_high_unknown_score():
    """A point far from every class mean should have high unknown score."""
    X, y = _separated_blobs(n_classes=3, n_per=30, d=8)
    m = MahalanobisOOD(n_classes=3).fit(X, y)
    rng = np.random.default_rng(42)
    far = rng.normal(50.0, 1.0, size=(20, 8))
    u = m.unknown_score(far)
    # At least most should have unknown > 0.5
    assert (u > 0.5).sum() >= 10


def test_mahalanobis_invalid_shrinkage_raises():
    with pytest.raises(ValueError, match="shrinkage"):
        MahalanobisOOD(n_classes=3, shrinkage=-0.1)
    with pytest.raises(ValueError, match="shrinkage"):
        MahalanobisOOD(n_classes=3, shrinkage=1.5)


def test_mahalanobis_predict_before_fit_raises():
    m = MahalanobisOOD(n_classes=3)
    with pytest.raises(RuntimeError, match="must be fit"):
        m.predict_proba(np.random.randn(5, 8))
    with pytest.raises(RuntimeError, match="must be fit"):
        m.unknown_score(np.random.randn(5, 8))


def test_mahalanobis_handles_small_class():
    """A class with only 1 sample should not crash the fit (tied covariance
    uses pooled residuals from other classes)."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0, 1, (50, 4)),
        rng.normal(5, 1, (1, 4)),
    ])
    y = np.concatenate([np.zeros(50, dtype=int), np.ones(1, dtype=int)])
    m = MahalanobisOOD(n_classes=2).fit(X, y)
    p = m.predict_proba(X[:5])
    assert p.shape == (5, 3)


def test_mahalanobis_shape_validation_at_predict():
    X, y = _separated_blobs(n_classes=3, n_per=20, d=8)
    m = MahalanobisOOD(n_classes=3).fit(X, y)
    with pytest.raises(ValueError, match="fit dim"):
        m.predict_proba(np.random.randn(5, 4))


def test_set_threshold_changes_mode():
    m = MahalanobisOOD(n_classes=3)
    assert m.threshold_mode == "none"
    m.set_threshold(2.5)
    assert m.threshold_mode == "calibrated"
    assert m._threshold == 2.5
