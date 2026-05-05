"""Unit tests for src/ewat/drift/calibration.py."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.ewat.drift.calibration import (
    _episode_mmd2_sequence,
    calibrate_epsilon,
    save_calibration,
)
from src.ewat.drift.mmd import RFFKernel


def _make_signal(T: int, N: int = 2, d: int = 4, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((T, N, d)) + shift
    return sig.astype(np.float64)


def _make_drift_signal(
    T: int, N: int = 2, d: int = 4, switch_at: int = 20, shift: float = 50.0, seed: int = 0
):
    """Signal that starts normal then drifts at switch_at — ref window is normal, cur is drifted."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((T, N, d))
    sig[switch_at:] += shift
    return sig.astype(np.float64)


class TestEpisodeMMD2Sequence:
    def test_empty_for_short_episode(self):
        sig = _make_signal(T=5, N=2, d=3)
        k = RFFKernel(sigma=1.0, rff_dim=32, seed=0)
        result = _episode_mmd2_sequence(sig, k, window_ref_size=4, window_cur_size=4)
        assert result == []

    def test_nonneg_values(self):
        sig = _make_signal(T=50, N=2, d=4)
        k = RFFKernel(sigma=1.0, rff_dim=64, seed=0)
        result = _episode_mmd2_sequence(sig, k, window_ref_size=20, window_cur_size=10)
        assert len(result) > 0
        assert all(v >= 0.0 for v in result)

    def test_length_correct(self):
        T, ref, cur = 60, 20, 10
        sig = _make_signal(T=T)
        k = RFFKernel(sigma=1.0, rff_dim=32, seed=0)
        result = _episode_mmd2_sequence(sig, k, window_ref_size=ref, window_cur_size=cur)
        # Steps from t=ref to T-cur inclusive: T - ref - cur + 1
        assert len(result) == T - ref - cur + 1


class TestCalibrateEpsilon:
    def test_epsilon_positive(self):
        rng = np.random.default_rng(42)
        drift_sigs = [_make_signal(T=60, shift=10.0, seed=i) for i in range(3)]
        normal_sigs = [_make_signal(T=60, shift=0.0, seed=i + 100) for i in range(3)]
        k = RFFKernel(sigma=1.0, rff_dim=64, seed=0)
        eps = calibrate_epsilon(
            drift_sigs, normal_sigs, k,
            window_ref_size=20, window_cur_size=10,
        )
        assert eps > 0.0

    def test_drift_epsilon_greater_than_normal_mmd2(self):
        """ε_drift should exceed the max normal MMD² (separability invariant).

        Drift signals start normal (first 20 timesteps) then shift by 50σ so
        that the reference window is clean while the current window is drifted.
        Uses the same kernel throughout for comparable MMD² scales.
        """
        # switch_at must equal window_ref_size so ref=normal, cur=drifted
        drift_sigs = [_make_drift_signal(T=80, switch_at=20, shift=50.0, seed=i) for i in range(5)]
        normal_sigs = [_make_signal(T=80, shift=0.0, seed=i + 50) for i in range(5)]
        # Calibrate sigma on the normal reference window
        k = RFFKernel(seed=0)
        ref = normal_sigs[0][:20].reshape(20, -1).astype(np.float64)
        k.fit_sigma(ref)

        eps = calibrate_epsilon(
            drift_sigs, normal_sigs, k,
            window_ref_size=20, window_cur_size=10,
            percentile=95,
        )
        normal_mmd2s: list[float] = []
        for sig in normal_sigs:
            normal_mmd2s.extend(
                _episode_mmd2_sequence(sig, k, window_ref_size=20, window_cur_size=10)
            )
        assert eps > max(normal_mmd2s)

    def test_raises_when_no_drift_values(self):
        k = RFFKernel(sigma=1.0, rff_dim=32)
        # Too-short signals → no MMD² computed
        drift_sigs = [_make_signal(T=3)]
        with pytest.raises(ValueError, match="no drift MMD"):
            calibrate_epsilon(drift_sigs, [], k, window_ref_size=20, window_cur_size=10)


class TestSaveCalibration:
    def test_saves_json(self, tmp_path):
        out = tmp_path / "eps.json"
        save_calibration(0.042, out, extra={"percentile": 95, "n_values": 100})
        data = json.loads(out.read_text())
        assert data["epsilon_drift"] == pytest.approx(0.042)
        assert data["percentile"] == 95

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "eps.json"
        save_calibration(0.1, out)
        assert out.exists()
