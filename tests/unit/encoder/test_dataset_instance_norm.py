"""Tests for EpisodeDataset.instance_normalize."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ewat.encoder.dataset import EpisodeDataset


def _make_synthetic_store(
    tmpdir: Path,
    n_episodes: int = 4,
    t_total: int = 12,
    n_normal: int = 8,
    n_nodes: int = 6,
    baseline_offset: float = 5.0,
) -> Path:
    """Create a feature store where each episode has a different absolute
    baseline. instance_normalize should remove the baseline difference."""
    rng = np.random.default_rng(0)
    episode_ids = []
    for i in range(n_episodes):
        ep_id = f"ep_{i:03d}"
        ep_dir = tmpdir / ep_id
        ep_dir.mkdir()
        # Add a per-episode baseline shift to feature 0
        base_signal = rng.normal(0, 1, (t_total, n_nodes, 17)).astype(np.float32)
        base_signal[..., 0] += baseline_offset * i  # 0, 5, 10, 15 baseline shift
        adj = rng.uniform(0, 1, (t_total, n_nodes, n_nodes, 3)).astype(np.float32)
        np.savez(ep_dir / "signal.npz", signal=base_signal)
        np.savez(ep_dir / "adjacency.npz", adjacency=adj)
        regime = ["normal"] * n_normal + ["injection"] * (t_total - n_normal)
        df = pd.DataFrame({
            "timestamp": np.arange(t_total, dtype=float) * 30.0,
            "regime": regime,
            "category": ["sc"] * t_total,
            "scenario": [f"sc_{i % 3}"] * t_total,
            "target_services": ["svc"] * t_total,
            "chaos_resource": ["cpu"] * t_total,
            "episode_id": [ep_id] * t_total,
            "drift_flag": [False] * t_total,
            "target_service": ["svc"] * t_total,
            "is_injection": [False] * n_normal + [True] * (t_total - n_normal),
        })
        df.to_parquet(ep_dir / "labels.parquet", index=False)
        episode_ids.append(ep_id)

    split = {"train": episode_ids[:2], "val": episode_ids[2:3], "test": episode_ids[3:]}
    split_path = tmpdir / "split.json"
    split_path.write_text(json.dumps(split))
    return split_path


def test_instance_normalize_removes_baseline_shift():
    """After instance_normalize, episodes with different baselines should have
    similar mean feature 0 over the normal regime (close to 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        split_path = _make_synthetic_store(tmpdir, baseline_offset=10.0)
        ds_no = EpisodeDataset(split_path, tmpdir, split="train", instance_normalize=False)
        ds_yes = EpisodeDataset(split_path, tmpdir, split="train", instance_normalize=True)
        sig_no_0 = ds_no[0]["signal"].numpy()[:8, :, 0].mean()
        sig_no_1 = ds_no[1]["signal"].numpy()[:8, :, 0].mean()
        sig_yes_0 = ds_yes[0]["signal"].numpy()[:8, :, 0].mean()
        sig_yes_1 = ds_yes[1]["signal"].numpy()[:8, :, 0].mean()
    # Without instance norm: baselines very different (~0 vs ~10)
    assert abs(sig_no_0 - sig_no_1) > 5.0
    # With instance norm: both ~0 (mean of normal regime is removed)
    assert abs(sig_yes_0) < 0.1
    assert abs(sig_yes_1) < 0.1


def test_instance_normalize_preserves_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        split_path = _make_synthetic_store(tmpdir)
        ds = EpisodeDataset(split_path, tmpdir, split="train", instance_normalize=True)
        item = ds[0]
    assert item["signal"].shape == (12, 6, 17)
    assert item["adjacency"].shape == (12, 6, 6, 3)


def test_instance_normalize_default_is_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        split_path = _make_synthetic_store(tmpdir, baseline_offset=10.0)
        ds = EpisodeDataset(split_path, tmpdir, split="train")
        ds_explicit = EpisodeDataset(split_path, tmpdir, split="train", instance_normalize=False)
        assert (ds[0]["signal"] == ds_explicit[0]["signal"]).all()


def test_no_nans_after_instance_normalize():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        split_path = _make_synthetic_store(tmpdir)
        ds = EpisodeDataset(split_path, tmpdir, split="train", instance_normalize=True)
        for i in range(len(ds)):
            assert not ds[i]["signal"].isnan().any()
