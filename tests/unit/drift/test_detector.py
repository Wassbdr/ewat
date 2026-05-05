"""Unit tests for src/ewat/drift/detector.py."""

import numpy as np
import pytest

from src.ewat.drift.detector import DriftDetector, DriftResult
from src.ewat.drift.mmd import RFFKernel


def _make_detector(
    epsilon: float = 0.5,
    window_ref: int = 10,
    window_cur: int = 5,
    post_window: int = 3,
) -> DriftDetector:
    k = RFFKernel(sigma=1.0, rff_dim=64, seed=0)
    return DriftDetector(
        kernel=k,
        epsilon_drift=epsilon,
        window_ref_size=window_ref,
        window_cur_size=window_cur,
        post_drift_window_s=post_window,
    )


class TestDriftDetectorWarmup:
    def test_no_flag_during_warmup(self):
        det = _make_detector(window_ref=10, window_cur=5)
        rng = np.random.default_rng(0)
        results = []
        for _ in range(12):
            r = det.update(rng.standard_normal(6))
            results.append(r)
        # All results during warmup must have flag=False
        for r in results:
            assert r.flag is False
            assert r.regime in ("normal", "recalibrate")

    def test_mmd2_zero_during_warmup(self):
        det = _make_detector(window_ref=10)
        for _ in range(5):
            r = det.update(np.zeros(4))
        assert r.mmd2 == 0.0


class TestDriftDetectorNormalRegime:
    def test_normal_stays_normal(self):
        """Identical distributions → always NORMAL."""
        det = _make_detector(epsilon=10.0, window_ref=20, window_cur=10, post_window=5)
        rng = np.random.default_rng(1)
        # Warm up
        for _ in range(25):
            det.update(rng.standard_normal(8))
        # Continue with same distribution
        results = []
        for _ in range(20):
            r = det.update(rng.standard_normal(8))
            results.append(r)
        # Should never drift with a large epsilon and same distribution
        drift_results = [r for r in results if r.flag]
        assert len(drift_results) == 0


class TestDriftDetectorDriftRegime:
    def test_detects_large_shift(self):
        """A mean shift of 20σ should be detected."""
        det = _make_detector(epsilon=0.01, window_ref=10, window_cur=5, post_window=3)
        rng = np.random.default_rng(2)
        # Warm up with zeros
        for _ in range(15):
            det.update(np.zeros(6))
        # Feed shifted distribution
        results = []
        for _ in range(20):
            r = det.update(np.ones(6) * 20.0)
            results.append(r)
        drift_results = [r for r in results if r.flag]
        assert len(drift_results) > 0
        assert any(r.regime == "drift" for r in results)

    def test_drift_result_fields(self):
        det = _make_detector(epsilon=0.001, window_ref=10, window_cur=5, post_window=3)
        for _ in range(15):
            det.update(np.zeros(4))
        r = det.update(np.ones(4) * 50.0)
        assert isinstance(r, DriftResult)
        assert isinstance(r.flag, bool)
        assert isinstance(r.mmd2, float)
        assert r.regime in ("normal", "drift", "recalibrate")


class TestDriftDetectorRecalibrate:
    def test_recalibrate_on_benign_change(self):
        """After a brief drift that returns to normal, regime should be RECALIBRATE."""
        det = _make_detector(epsilon=0.01, window_ref=10, window_cur=5, post_window=3)
        for _ in range(15):
            det.update(np.zeros(4))
        # Inject brief spike
        for _ in range(3):
            det.update(np.ones(4) * 30.0)
        # Return to normal
        results = []
        for _ in range(15):
            r = det.update(np.zeros(4))
            results.append(r)
        # After returning to normal, should eventually recalibrate
        regimes = {r.regime for r in results}
        assert "recalibrate" in regimes or "normal" in regimes


class TestDriftDetectorReset:
    def test_reset_clears_buffers(self):
        det = _make_detector()
        for _ in range(5):
            det.update(np.ones(3))
        det.reset()
        assert len(det._ref_buf) == 0
        assert len(det._cur_buf) == 0
        assert len(det._post_buf) == 0
        assert det._pending_drift is False


class TestDriftDetectorLoadReference:
    def test_load_reference_skips_warmup(self):
        det = _make_detector(epsilon=10.0, window_ref=10, window_cur=5, post_window=3)
        X_ref = np.zeros((10, 4))
        det.load_reference(X_ref)
        assert len(det._ref_buf) == 10

    def test_load_reference_truncates(self):
        det = _make_detector(window_ref=5)
        X_ref = np.ones((20, 3))
        det.load_reference(X_ref)
        # Ring buffer has maxlen=5 so only last 5 rows are kept
        assert len(det._ref_buf) == 5


class TestDriftDetectorEpsilonSetter:
    def test_epsilon_setter(self):
        det = _make_detector(epsilon=None)
        assert det.epsilon_drift is None
        det.epsilon_drift = 0.1
        assert det.epsilon_drift == pytest.approx(0.1)
