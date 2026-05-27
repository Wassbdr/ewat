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
    window_position:  Which slice of the pre-injection (normal) window to use.
                      - "last" (default, status quo): normal_indices[-k:] — right
                        before injection. Used by H3 precursor training.
                      - "first": normal_indices[:k] — beginning of the normal
                        regime, maximally distant from injection. Used by the
                        distant-window stress test (A1) to detect static
                        scenario signature leakage.
                      - "middle": k indices centered in the middle of the
                        normal window.
    """

    def __init__(
        self,
        cluster_manifest: dict[str, dict],
        features_root: Path,
        k: int,
        scaler: StandardScaler | None = None,
        split: str | None = None,
        window_position: str = "last",
    ) -> None:
        if window_position not in ("last", "first", "middle"):
            raise ValueError(
                f"window_position must be one of 'last', 'first', 'middle' — got {window_position!r}"
            )
        self.features_root = Path(features_root)
        self.k = k
        self.scaler = scaler
        self.window_position = window_position

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
        adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)

        # Find pre-injection (normal) timestep indices
        labels_df = pd.read_parquet(ep_dir / "labels.parquet", columns=["regime", "timestamp"])
        normal_mask = (labels_df["regime"] == "normal").values   # bool array length T

        normal_indices = np.where(normal_mask)[0]

        # Extract k steps from the pre-injection window — slice position depends on window_position
        if len(normal_indices) == 0:
            # No normal steps — use first k steps of episode
            normal_indices = np.arange(min(self.k, len(labels_df)))

        n_normal = len(normal_indices)
        if self.window_position == "last":
            last_k_idx = normal_indices[-self.k:]
        elif self.window_position == "first":
            last_k_idx = normal_indices[: self.k]
        else:  # "middle"
            if n_normal <= self.k:
                last_k_idx = normal_indices
            else:
                start = (n_normal - self.k) // 2
                last_k_idx = normal_indices[start : start + self.k]
        actual_len = len(last_k_idx)

        sig_window = signal[last_k_idx]        # (actual_len, N, 17)
        adj_window = adjacency[last_k_idx]     # (actual_len, N, N, 3)

        # Normalise — Step 4 fix 4.1 (audit 2026-05-26): consistent NaN imputation
        # with EpisodeDataset.imputation_strategy="scaler_mean". Never use
        # nan_to_num(0.0) BEFORE scaling, because (0 - mean) / std biases.
        if self.scaler is not None:
            t_len, n_nodes, d = sig_window.shape
            flat = sig_window.reshape(-1, d)
            nan_mask = np.isnan(flat)
            # Impute NaN with scaler.mean_ → after transform, becomes 0 (neutral).
            flat = np.where(nan_mask, self.scaler.mean_, flat)
            flat = self.scaler.transform(flat).astype(np.float32)
            sig_window = flat.reshape(t_len, n_nodes, d)
        else:
            # No scaler: last-resort NaN→0 with a docstring caveat.
            # Production code should always pass a fitted scaler.
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
