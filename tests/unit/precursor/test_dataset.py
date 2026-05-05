"""Tests for PrecursorDataset."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ewat.precursor.dataset import PrecursorDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_feature_store(
    tmpdir: Path,
    n_episodes: int = 6,
    t_total: int = 20,
    n_normal: int = 13,
    n_nodes: int = 6,
) -> dict[str, dict]:
    """Create synthetic feature store and return a cluster manifest."""
    manifest = {}
    rng = np.random.default_rng(0)
    splits = ["train"] * 4 + ["val"] + ["test"]

    for i in range(n_episodes):
        ep_id = f"ep_{i:03d}"
        cluster = i % 3
        split = splits[i % len(splits)]

        ep_dir = tmpdir / ep_id
        ep_dir.mkdir()

        signal = rng.normal(0, 1, (t_total, n_nodes, 17)).astype(np.float32)
        adjacency = rng.uniform(0, 1, (t_total, n_nodes, n_nodes, 3)).astype(np.float32)
        np.savez(ep_dir / "signal.npz", signal=signal)
        np.savez(ep_dir / "adjacency.npz", adjacency=adjacency)

        # labels.parquet: first n_normal steps are 'normal', rest are 'injection'
        regime = ["normal"] * n_normal + ["injection"] * (t_total - n_normal)
        df = pd.DataFrame({
            "timestamp": np.arange(t_total, dtype=float) * 30.0,
            "regime": regime,
            "category": [f"sc_{cluster}"] * t_total,
            "scenario": [f"sc_{cluster}"] * t_total,
            "target_services": ["svc"] * t_total,
            "chaos_resource": ["cpu"] * t_total,
            "episode_id": [ep_id] * t_total,
            "drift_flag": [False] * t_total,
            "target_service": ["svc"] * t_total,
            "is_injection": [False] * n_normal + [True] * (t_total - n_normal),
        })
        df.to_parquet(ep_dir / "labels.parquet", index=False)

        manifest[ep_id] = {"cluster": cluster, "split": split, "scenario": f"sc_{cluster}"}

    return manifest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dataset_length_train_split():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_feature_store(Path(tmpdir))
        ds = PrecursorDataset(manifest, Path(tmpdir), k=6, split="train")
    assert len(ds) == 4   # 4 train episodes


def test_dataset_length_no_split_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_feature_store(Path(tmpdir), n_episodes=6)
        ds = PrecursorDataset(manifest, Path(tmpdir), k=6, split=None)
    assert len(ds) == 6


def test_signal_shape_equals_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_feature_store(Path(tmpdir))
        k = 6
        ds = PrecursorDataset(manifest, Path(tmpdir), k=k, split=None)
        item = ds[0]
    assert item["signal"].shape == (k, 6, 17)


def test_adjacency_shape_equals_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_feature_store(Path(tmpdir))
        k = 4
        ds = PrecursorDataset(manifest, Path(tmpdir), k=k, split=None)
        item = ds[0]
    assert item["adjacency"].shape == (k, 6, 6, 3)


def test_cluster_label_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_feature_store(Path(tmpdir))
        ds = PrecursorDataset(manifest, Path(tmpdir), k=6, split=None)
        for i in range(len(ds)):
            assert 0 <= ds[i]["cluster"] < 3


def test_episode_id_is_string():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_feature_store(Path(tmpdir))
        ds = PrecursorDataset(manifest, Path(tmpdir), k=6, split=None)
        assert isinstance(ds[0]["episode_id"], str)


def test_left_padding_when_warmup_shorter_than_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        # n_normal=3, k=8 → warmup shorter → must left-pad
        manifest = _make_feature_store(Path(tmpdir), n_normal=3)
        k = 8
        ds = PrecursorDataset(manifest, Path(tmpdir), k=k, split=None)
        item = ds[0]
    assert item["signal"].shape[0] == k
    # First (8-3)=5 rows should be zeros (left-pad)
    assert item["signal"][:5].abs().sum().item() == pytest.approx(0.0)


def test_no_nans_in_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = _make_feature_store(Path(tmpdir))
        ds = PrecursorDataset(manifest, Path(tmpdir), k=6, split=None)
        for i in range(len(ds)):
            assert not ds[i]["signal"].isnan().any()
            assert not ds[i]["adjacency"].isnan().any()


def test_large_k_truncates_to_available():
    with tempfile.TemporaryDirectory() as tmpdir:
        # warmup=13, k=20 → use all 13 warmup + pad 7
        manifest = _make_feature_store(Path(tmpdir), n_normal=13)
        k = 20
        ds = PrecursorDataset(manifest, Path(tmpdir), k=k, split=None)
        item = ds[0]
    assert item["signal"].shape[0] == k
