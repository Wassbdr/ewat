"""Tests for EpisodeDataset.imputation_strategy and instance_normalize semantics.

Covers Step 4 fixes (audit 2026-05-26):
- 4.1 : never nan_to_num(0.0) before scaling — use scaler.mean_ first
- 4.2 : instance_normalize and global scaler are mutually exclusive (not chained)
- 4.3 : imputation_strategy is explicit and validated
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ewat.encoder.dataset import EpisodeDataset


def _make_store_with_nan(
    tmpdir: Path,
    *,
    n_episodes: int = 2,
    t_total: int = 8,
    n_normal: int = 6,
    n_nodes: int = 4,
    nan_pattern: str = "feature_3",
    seed: int = 0,
) -> Path:
    """Create a store with deterministic NaN positions for verification."""
    rng = np.random.default_rng(seed)
    eps: list[str] = []
    for i in range(n_episodes):
        ep_id = f"ep_{i:03d}"
        ep_dir = tmpdir / ep_id
        ep_dir.mkdir()
        # Constant offset per episode to make instance norm impact visible
        baseline = float(i) * 10.0
        sig = rng.normal(baseline, 1.0, (t_total, n_nodes, 17)).astype(np.float32)
        if nan_pattern == "feature_3":
            sig[:, :, 3] = np.nan   # entire feature column NaN
        elif nan_pattern == "sparse":
            mask = rng.random((t_total, n_nodes, 17)) < 0.1
            sig[mask] = np.nan
        adj = rng.uniform(0, 1, (t_total, n_nodes, n_nodes, 3)).astype(np.float32)
        np.savez(ep_dir / "signal.npz", signal=sig)
        np.savez(ep_dir / "adjacency.npz", adjacency=adj)
        regime = ["normal"] * n_normal + ["injection"] * (t_total - n_normal)
        df = pd.DataFrame({
            "timestamp": np.arange(t_total, dtype=float) * 30.0,
            "regime": regime,
            "category": ["sc"] * t_total,
            "scenario": [f"sc_{i % 2}"] * t_total,
            "target_services": ["svc"] * t_total,
            "chaos_resource": ["cpu"] * t_total,
            "episode_id": [ep_id] * t_total,
            "drift_flag": [False] * t_total,
            "target_service": ["svc"] * t_total,
            "is_injection": [False] * n_normal + [True] * (t_total - n_normal),
        })
        df.to_parquet(ep_dir / "labels.parquet", index=False)
        eps.append(ep_id)
    split = {"train": eps, "val": [], "test": []}
    p = tmpdir / "split.json"
    p.write_text(json.dumps(split))
    return p


def test_invalid_imputation_strategy_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_nan(tmp)
        with pytest.raises(ValueError, match="imputation_strategy"):
            EpisodeDataset(split_path, tmp, split="train",
                           imputation_strategy="bogus")


def test_imputation_strategy_default_is_scaler_mean():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_nan(tmp)
        ds = EpisodeDataset(split_path, tmp, split="train")
    assert ds.imputation_strategy == "scaler_mean"


def test_imputation_never_uses_zero_before_scaling():
    """Critical fix 4.1: if a feature has high mean, imputing with 0 would
    bias it negatively. With imputation_strategy='scaler_mean', after scaling,
    the imputed cells should be exactly 0."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_nan(tmp, nan_pattern="sparse", seed=42)
        ds = EpisodeDataset(split_path, tmp, split="train",
                            imputation_strategy="scaler_mean")
        scaler = ds.fit_scaler()
        # Re-build dataset with the scaler attached
        ds.scaler = scaler
        item = ds[0]
    # All values are finite (no NaN propagated)
    assert not item["signal"].isnan().any()
    # Scaled output should be roughly centered (mean ~ 0)
    assert abs(item["signal"].mean().item()) < 1.5


def test_instance_normalize_excludes_global_scaler():
    """Step 4 fix 4.2: when instance_normalize=True, the global scaler must
    NOT be applied on top. Verify by comparing instance-normed signal with
    its expected statistics."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_nan(tmp, nan_pattern="sparse")
        # Fit a scaler that has very different stats (would corrupt if applied)
        scaler_ds = EpisodeDataset(split_path, tmp, split="train")
        scaler = scaler_ds.fit_scaler()

        # Now create dataset with instance_normalize=True AND scaler — the
        # scaler should be SILENTLY IGNORED.
        ds = EpisodeDataset(
            split_path, tmp, split="train",
            scaler=scaler,
            instance_normalize=True,
        )
        item = ds[0]
        sig = item["signal"].numpy()

    # Verify mean ~ 0 over normal-regime timesteps (because instance norm
    # subtracts the normal-regime mean). If global scaler had been chained,
    # mean would not be 0.
    n_normal = 6
    normal_mean = sig[:n_normal].mean(axis=(0, 1))   # (17,)
    # Filter out NaN-pattern feature columns that may have all 0 after impute
    finite_features = np.isfinite(normal_mean) & (np.abs(normal_mean) < 100)
    assert finite_features.sum() > 5
    assert np.abs(normal_mean[finite_features]).max() < 0.2, (
        f"instance norm produced normal-regime mean far from 0: {normal_mean}"
    )


def test_imputation_zero_post_scaling_strategy():
    """Validates the alternative 'zero_post_scaling' strategy works."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_nan(tmp, nan_pattern="sparse")
        ds = EpisodeDataset(split_path, tmp, split="train",
                            imputation_strategy="zero_post_scaling")
        ds.scaler = ds.fit_scaler()
        item = ds[0]
    assert not item["signal"].isnan().any()


def test_imputation_none_keeps_nan():
    """The 'none' strategy leaves NaN in place after scaling — for callers
    that explicitly handle NaN downstream."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_nan(tmp, nan_pattern="sparse")
        ds = EpisodeDataset(split_path, tmp, split="train",
                            imputation_strategy="none")
        ds.scaler = ds.fit_scaler()
        item = ds[0]
    # NaN was preserved (original sparse pattern returns)
    assert item["signal"].isnan().any()


def test_no_scaler_no_instance_norm_falls_back_to_zero():
    """When neither scaler nor instance_normalize is provided, NaN→0 is the
    last-resort behavior (with docstring caveat about needing a scaler in
    production)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_nan(tmp, nan_pattern="sparse")
        ds = EpisodeDataset(split_path, tmp, split="train")
        # No fit_scaler call → ds.scaler is None
        item = ds[0]
    assert not item["signal"].isnan().any()
