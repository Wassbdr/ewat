"""Unit tests for src/ewat/drift/mmd.py."""

import numpy as np
import pytest

from src.ewat.drift.mmd import RFFKernel, _impute_column_mean


class TestRFFKernelPhi:
    def test_phi_shape(self):
        k = RFFKernel(sigma=1.0, rff_dim=64, seed=0)
        X = np.random.default_rng(0).standard_normal((50, 10))
        phi = k.phi(X)
        assert phi.shape == (50, 64)

    def test_phi_normalisation_scale(self):
        """Each row of phi should have norm close to 1 in expectation (not guaranteed per row)."""
        k = RFFKernel(sigma=1.0, rff_dim=1024, seed=0)
        X = np.zeros((1, 5))
        phi = k.phi(X)
        assert phi.shape == (1, 1024)
        # cos values are in [-1,1], scaled by sqrt(2/D) so element magnitude ≤ sqrt(2/D)
        assert np.all(np.abs(phi) <= np.sqrt(2.0 / 1024) + 1e-9)

    def test_phi_requires_sigma(self):
        k = RFFKernel(sigma=None, rff_dim=32)
        X = np.ones((5, 3))
        with pytest.raises(RuntimeError, match="sigma"):
            k.phi(X)

    def test_phi_cached_projection(self):
        """Calling phi twice on different inputs should use the same W, b."""
        k = RFFKernel(sigma=1.0, rff_dim=32, seed=0)
        X1 = np.zeros((10, 4))
        X2 = np.ones((10, 4))
        k.phi(X1)
        W1 = k._W.copy()
        k.phi(X2)
        np.testing.assert_array_equal(k._W, W1)


class TestRFFKernelFitSigma:
    def test_fit_sigma_positive(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        k = RFFKernel(seed=0)
        k.fit_sigma(X)
        assert k.sigma is not None
        assert k.sigma > 0

    def test_fit_sigma_invalidates_projection(self):
        k = RFFKernel(sigma=1.0, rff_dim=16, seed=0)
        X = np.ones((10, 3))
        k.phi(X)  # initialise W
        assert k._W is not None
        k.fit_sigma(X)
        assert k._W is None  # must be cleared


class TestMMDSquared:
    def test_mmd2_zero_identical(self):
        """MMD² must be 0 when X_ref == X_cur (same distribution)."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 8))
        k = RFFKernel(sigma=1.0, rff_dim=512, seed=0)
        mmd2 = k.mmd_squared(X, X)
        assert mmd2 == pytest.approx(0.0, abs=1e-12)

    def test_mmd2_symmetry(self):
        """MMD² should be symmetric."""
        rng = np.random.default_rng(3)
        A = rng.standard_normal((80, 6))
        B = rng.standard_normal((80, 6)) + 2.0
        k = RFFKernel(sigma=1.0, rff_dim=256, seed=0)
        assert k.mmd_squared(A, B) == pytest.approx(k.mmd_squared(B, A), rel=1e-9)

    def test_mmd2_detects_shift(self):
        """MMD² should be larger when distributions differ."""
        rng = np.random.default_rng(5)
        X_ref = rng.standard_normal((200, 10))
        X_same = rng.standard_normal((60, 10))
        X_diff = rng.standard_normal((60, 10)) + 5.0

        k = RFFKernel(seed=0)
        mmd2_same = k.mmd_squared(X_ref, X_same)
        k2 = RFFKernel(seed=0)
        mmd2_diff = k2.mmd_squared(X_ref, X_diff)
        assert mmd2_diff > mmd2_same

    def test_mmd2_nonneg(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((50, 4))
        B = rng.standard_normal((50, 4))
        k = RFFKernel(sigma=1.0, rff_dim=128, seed=0)
        assert k.mmd_squared(A, B) >= 0.0

    def test_mmd2_handles_empty(self):
        k = RFFKernel(sigma=1.0, rff_dim=32)
        A = np.zeros((0, 3))
        B = np.ones((10, 3))
        assert k.mmd_squared(A, B) == 0.0

    def test_mmd2_auto_sigma(self):
        """If sigma is None, mmd_squared should calibrate it from X_ref."""
        rng = np.random.default_rng(11)
        X = rng.standard_normal((100, 5))
        k = RFFKernel(sigma=None, rff_dim=64, seed=0)
        mmd2 = k.mmd_squared(X, X)
        assert k.sigma is not None and k.sigma > 0
        assert mmd2 == pytest.approx(0.0, abs=1e-12)

    def test_mmd2_with_nans(self):
        """NaN features should be handled without raising."""
        rng = np.random.default_rng(13)
        X = rng.standard_normal((60, 6))
        X[:, 2] = np.nan  # entire column NaN → dropped by valid_cols mask
        k = RFFKernel(sigma=1.0, rff_dim=64, seed=0)
        mmd2 = k.mmd_squared(X, X)
        assert np.isfinite(mmd2)


class TestImputeColumnMean:
    def test_no_nans_unchanged(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _impute_column_mean(X)
        np.testing.assert_array_equal(out, X)

    def test_fills_nans(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        out = _impute_column_mean(X)
        assert out[0, 1] == pytest.approx(4.0)  # mean of [nan, 4.0] = 4.0

    def test_all_nan_column_fills_zero(self):
        X = np.array([[np.nan], [np.nan]])
        out = _impute_column_mean(X)
        np.testing.assert_array_equal(out, np.zeros_like(X))
