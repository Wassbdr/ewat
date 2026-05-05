"""PrecursorDataset — pre-injection windows for typed precursor training.

For each episode we extract the last k timesteps of the pre-injection
(regime == 'normal') window.  This is the signal available right before the
anomaly starts, which the precursor model must classify.

If the warmup has fewer than k steps, the window is left-padded with zeros.

Cluster labels come from the cluster_manifest produced by
experiments/typing/train.py (cluster_artifacts/cluster_manifest.json).

Usage
-----
>>> ds = PrecursorDataset(cluster_manifest, features_root, k=6, split="train")
>>> item = ds[0]
>>> item["signal"].shape   # (k, N, 17)
>>> item["cluster"]        # int — cluster type of this episode
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class PrecursorDataset(Dataset):
    """Pre-injection windows for precursor classification.

    Parameters
    ----------
    cluster_manifest: {episode_id → {"cluster": int, "split": str, "scenario": str}}
    features_root:    Root of the feature store (data/features/v3/).
    k:                Window length in timesteps (signal steps, not minutes).
    scaler:           Pre-fitted StandardScaler (optional).
    split:            Filter by split ("train", "val", "test", or None for all).
    """

    def __init__(
        self,
        cluster_manifest: dict[str, dict],
        features_root: Path,
        k: int,
        scaler: StandardScaler | None = None,
        split: str | None = None,
    ) -> None:
        self.features_root = Path(features_root)
        self.k = k
        self.scaler = scaler

        self.episodes: list[tuple[str, int]] = []   # (episode_id, cluster_label)
        for ep_id, info in cluster_manifest.items():
            if split is not None and info["split"] != split:
                continue
            self.episodes.append((ep_id, int(info["cluster"])))

    def load_scaler(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep_id, cluster_label = self.episodes[idx]
        ep_dir = self.features_root / ep_id

        signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)   # (T, N, 17)
        adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)  # (T, N, N, 3)

        # Find pre-injection (normal) timestep indices
        labels_df = pd.read_parquet(ep_dir / "labels.parquet", columns=["regime", "timestamp"])
        normal_mask = (labels_df["regime"] == "normal").values   # bool array length T

        normal_indices = np.where(normal_mask)[0]

        # Extract last k steps from the pre-injection window
        if len(normal_indices) == 0:
            # No normal steps — use first k steps of episode
            normal_indices = np.arange(min(self.k, len(labels_df)))

        last_k_idx = normal_indices[-self.k:]   # at most k indices
        actual_len = len(last_k_idx)

        sig_window = signal[last_k_idx]        # (actual_len, N, 17)
        adj_window = adjacency[last_k_idx]     # (actual_len, N, N, 3)

        # Normalise
        if self.scaler is not None:
            T, N, d = sig_window.shape
            flat = sig_window.reshape(-1, d)
            flat = np.where(np.isnan(flat), 0.0, flat)
            flat = self.scaler.transform(flat).astype(np.float32)
            sig_window = flat.reshape(T, N, d)
        else:
            sig_window = np.nan_to_num(sig_window, nan=0.0)

        adjacency = np.nan_to_num(adj_window, nan=0.0)

        # Left-pad to k if window is shorter
        if actual_len < self.k:
            pad = self.k - actual_len
            sig_window = np.concatenate(
                [np.zeros((pad, *sig_window.shape[1:]), dtype=np.float32), sig_window], axis=0
            )
            adjacency = np.concatenate(
                [np.zeros((pad, *adjacency.shape[1:]), dtype=np.float32), adjacency], axis=0
            )

        return {
            "signal": torch.from_numpy(sig_window),      # (k, N, 17)
            "adjacency": torch.from_numpy(adjacency),    # (k, N, N, 3)
            "cluster": cluster_label,
            "episode_id": ep_id,
        }
