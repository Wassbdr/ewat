"""Tests empreinte scaler (M15, audit 2026-06)."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from ewat.utils.fingerprint import scaler_fingerprint


def _fit(data: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(data)


def test_same_data_same_fingerprint():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 17))
    assert scaler_fingerprint(_fit(data)) == scaler_fingerprint(_fit(data.copy()))


def test_different_data_different_fingerprint():
    rng = np.random.default_rng(0)
    a = _fit(rng.normal(size=(100, 17)))
    b = _fit(rng.normal(loc=5.0, size=(100, 17)))
    assert scaler_fingerprint(a) != scaler_fingerprint(b)


def test_unfitted_scaler_does_not_crash():
    fp = scaler_fingerprint(StandardScaler())
    assert isinstance(fp, str) and len(fp) == 64


def test_fingerprint_covers_scale_not_only_mean():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(100, 4))
    s1 = _fit(data)
    s2 = _fit(data)
    s2.scale_ = s2.scale_ * 2.0  # même mean_, scale_ différent
    assert scaler_fingerprint(s1) != scaler_fingerprint(s2)
