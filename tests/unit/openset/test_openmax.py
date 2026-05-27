"""Tests for OpenMax open-set recognition."""

import numpy as np
import pytest

from ewat.openset.openmax import OpenMax


def _make_separated_data(n_classes=5, n_per_class=50, d=8, seed=0):
    """Make well-separated class clusters in d-dim space."""
    rng = np.random.default_rng(seed)
    activations, labels = [], []
    for c in range(n_classes):
        center = np.zeros(d)
        center[c % d] = 5.0
        pts = center + rng.normal(0, 0.5, size=(n_per_class, d))
        activations.append(pts)
        labels.append(np.full(n_per_class, c, dtype=int))
    return np.vstack(activations), np.concatenate(labels)


def test_openmax_fit_shape():
    X, y = _make_separated_data()
    openmax = OpenMax(n_classes=5, tail_size=10)
    openmax.fit(X, y)
    assert openmax.fitted_
    assert openmax.class_means_.shape == (5, 8)
    assert len(openmax.weibulls_) == 5


def test_predict_proba_shape_includes_unknown():
    X, y = _make_separated_data()
    openmax = OpenMax(n_classes=5, tail_size=10).fit(X, y)
    p = openmax.predict_proba(X[:10])
    assert p.shape == (10, 6)   # K+1
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_known_inputs_classified_to_known_class():
    X, y = _make_separated_data()
    openmax = OpenMax(n_classes=5, tail_size=10).fit(X, y)
    pred = openmax.predict(X)
    # The model should classify most train samples to their true class (or at
    # least to a known class — not unknown).
    n_unknown = (pred == 5).sum()
    assert n_unknown < 0.2 * len(pred), \
        f"Too many train samples classified unknown: {n_unknown}/{len(pred)}"


def test_far_inputs_trigger_unknown():
    """Inputs FAR from all class means should get high unknown probability."""
    X, y = _make_separated_data(d=8)
    openmax = OpenMax(n_classes=5, tail_size=10).fit(X, y)
    # Generate inputs far from any class center (which are at canonical axes)
    rng = np.random.default_rng(42)
    far_inputs = rng.normal(50.0, 1.0, size=(20, 8))   # far from any class
    p = openmax.predict_proba(far_inputs)
    unknown_probs = p[:, -1]
    # At least most samples should have unknown > 0.5
    assert (unknown_probs > 0.5).sum() >= 10, \
        f"Far inputs not flagged unknown: {(unknown_probs > 0.5).sum()}/20"


def test_invalid_tail_size_raises():
    with pytest.raises(ValueError, match="tail_size"):
        OpenMax(n_classes=5, tail_size=2)


def test_invalid_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        OpenMax(n_classes=5, metric="bogus")


def test_predict_before_fit_raises():
    openmax = OpenMax(n_classes=5)
    with pytest.raises(RuntimeError, match="must be fit"):
        openmax.predict_proba(np.random.randn(10, 8))


def test_unknown_score_shape():
    X, y = _make_separated_data()
    openmax = OpenMax(n_classes=5, tail_size=10).fit(X, y)
    s = openmax.unknown_score(X[:10])
    assert s.shape == (10,)
    assert ((0 <= s) & (s <= 1)).all()


def test_cosine_metric_works():
    X, y = _make_separated_data()
    openmax = OpenMax(n_classes=5, tail_size=10, metric="cosine").fit(X, y)
    p = openmax.predict_proba(X[:10])
    assert p.shape == (10, 6)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_handles_small_class_count():
    """If a class has fewer than tail_size samples, fitting should not crash."""
    rng = np.random.default_rng(0)
    X = np.concatenate([
        rng.normal(0, 1, (50, 4)),
        rng.normal(5, 1, (3, 4)),   # tiny class
    ])
    y = np.concatenate([np.zeros(50, dtype=int), np.ones(3, dtype=int)])
    openmax = OpenMax(n_classes=2, tail_size=10).fit(X, y)
    assert openmax.fitted_
    p = openmax.predict_proba(X)
    assert p.shape == (53, 3)


def test_2d_input_required():
    openmax = OpenMax(n_classes=5)
    with pytest.raises(ValueError, match="2D"):
        openmax.fit(np.random.randn(50), np.zeros(50))
