"""Tests for EpisodeDataset.fit_scaler NaN-aware behaviour.

Covers Step 2 fix 2.3 (audit 2026-05-26): per-feature scaler fit uses every
valid observation, instead of discarding any row containing a single NaN.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ewat.encoder.dataset import EpisodeDataset


def _make_store_with_partial_nan(
    tmpdir: Path,
    *,
    n_episodes: int = 4,
    t_total: int = 12,
    n_normal: int = 8,
    n_nodes: int = 6,
    nan_feature: int = 5,
    nan_ratio: float = 0.5,
    seed: int = 0,
) -> Path:
    """Create episodes with a chosen fraction of NaN in a single feature dim.

    Critical case: previously, fit_scaler() rejected the ROW if any feature
    was NaN. So if feature 5 has 50% NaN, 50% of rows were discarded for
    *all* features, biasing means of other features.
    """
    rng = np.random.default_rng(seed)
    eps: list[str] = []
    for i in range(n_episodes):
        ep_id = f"ep_{i:03d}"
        ep_dir = tmpdir / ep_id
        ep_dir.mkdir()
        sig = rng.normal(0, 1, (t_total, n_nodes, 17)).astype(np.float32)
        # Set 50% of values in feature ``nan_feature`` to NaN
        nan_mask = rng.random((t_total, n_nodes)) < nan_ratio
        sig[nan_mask, nan_feature] = np.nan
        adj = rng.uniform(0, 1, (t_total, n_nodes, n_nodes, 3)).astype(np.float32)
        np.savez(ep_dir / "signal.npz", signal=sig)
        np.savez(ep_dir / "adjacency.npz", adjacency=adj)
        regime = ["normal"] * n_normal + ["injection"] * (t_total - n_normal)
        df = pd.DataFrame({
            "timestamp": np.arange(t_total, dtype=float) * 30.0,
            "regime": regime,
            "category": ["sc"] * t_total,
            "scenario": ["sc"] * t_total,
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


def test_fit_scaler_uses_all_valid_rows_per_feature():
    """A feature with 50% NaN should still see ~half the rows as input;
    other features should see 100% of the rows (not just the rows where the
    NaN feature is valid)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_partial_nan(tmp, nan_feature=5, nan_ratio=0.5)
        ds = EpisodeDataset(split_path, tmp, split="train")
        scaler = ds.fit_scaler()
    # Mean and scale should be finite for all 17 features (no all-NaN columns)
    assert np.all(np.isfinite(scaler.mean_)), "fit_scaler produced NaN means"
    assert np.all(scaler.scale_ > 0), "fit_scaler produced zero scale"
    # For feature 5 (50% NaN), mean should still be ~0 (data was N(0,1))
    assert abs(scaler.mean_[5]) < 0.5
    # For features 0-4 and 6+, also ~0 mean (full data)
    for f in [0, 1, 6, 10, 16]:
        assert abs(scaler.mean_[f]) < 0.3, f"feature {f} mean drift: {scaler.mean_[f]}"


def test_fit_scaler_handles_all_nan_column():
    """A feature that is 100% NaN across all episodes should still produce
    finite mean=0, scale=1 (passthrough)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_partial_nan(tmp, nan_feature=3, nan_ratio=1.0)
        ds = EpisodeDataset(split_path, tmp, split="train")
        scaler = ds.fit_scaler()
    assert np.isfinite(scaler.mean_[3])
    assert scaler.mean_[3] == 0.0
    assert scaler.scale_[3] == 1.0


def test_fit_scaler_transform_works_with_imputation():
    """Verify scaler.transform produces 0 when input == scaler.mean_."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = _make_store_with_partial_nan(tmp)
        ds = EpisodeDataset(split_path, tmp, split="train")
        scaler = ds.fit_scaler()
    sample = scaler.mean_.reshape(1, -1)
    out = scaler.transform(sample)
    np.testing.assert_allclose(out, np.zeros((1, 17)), atol=1e-6)


def test_fit_scaler_uses_more_data_than_old_strategy():
    """Sanity check: NaN-aware fit must use more rows than the previous
    row-rejection strategy."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 50% NaN in 1 feature → old strategy rejects 50% of rows for ALL features
        split_path = _make_store_with_partial_nan(tmp, nan_feature=10, nan_ratio=0.5)
        ds = EpisodeDataset(split_path, tmp, split="train")
        scaler = ds.fit_scaler()
    # Each feature except #10 should have seen all rows. We can't directly
    # introspect via StandardScaler, but n_samples_seen_ reflects the max
    # across features (≥ 50% more than the old strategy on this dataset).
    # Old: ~50% rows. New: max feature sees 100% of rows.
    # 4 episodes × 12 timesteps × 6 nodes = 288 rows
    expected_max = 4 * 12 * 6
    assert scaler.n_samples_seen_ >= int(0.99 * expected_max), (
        f"NaN-aware fit_scaler should use ~all {expected_max} rows for non-NaN "
        f"features, got n_samples_seen_={scaler.n_samples_seen_}"
    )
