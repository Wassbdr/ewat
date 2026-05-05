"""EpisodeDataset — PyTorch Dataset for EWAT episodes.

Loads signal.npz + adjacency.npz + labels.parquet from the feature store,
applies optional per-feature StandardScaler, and exposes them as tensors.

Usage
-----
>>> ds = EpisodeDataset(split_json, features_root, split="train")
>>> ds.fit_scaler(scaler_path)          # fit on train split, save to disk
>>> ds.load_scaler(scaler_path)         # load in val/test splits
>>> loader = DataLoader(ds, batch_size=32, collate_fn=collate_episodes)
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


class EpisodeDataset(Dataset):
    """Dataset over ewat_v3 episodes.

    Parameters
    ----------
    split_json:    Path to split.json (keys: train/val/test → list of episode_ids).
    features_root: Root of the feature store (data/features/v3/).
    split:         Which split to load ("train", "val", "test").
    scaler:        Pre-fitted StandardScaler.  None = no normalisation.
    """

    FEATURE_NAMES: list[str] = [
        "cpu_util", "ram_util", "latency_p99", "error_rate_http",
        "net_sat", "disk_io", "queue_depth",
        "span_dur_median", "abnormal_span_rate", "trace_depth", "fan_out",
        "retry_rate", "latency_cv",
        "log_error_rate", "log_warn_rate", "semantic_anomaly", "lexical_entropy",
    ]

    def __init__(
        self,
        split_json: Path,
        features_root: Path,
        split: str = "train",
        scaler: StandardScaler | None = None,
    ) -> None:
        self.features_root = Path(features_root)
        self.split = split
        self.scaler = scaler

        index = json.loads(Path(split_json).read_text())
        if split not in index:
            raise ValueError(f"Unknown split '{split}'; available: {list(index)}")
        self.episode_ids: list[str] = index[split]

    # ------------------------------------------------------------------
    # Scaler helpers
    # ------------------------------------------------------------------

    def fit_scaler(self, save_path: Path | None = None) -> StandardScaler:
        """Fit StandardScaler on this split and optionally save to disk."""
        all_values: list[np.ndarray] = []
        for ep_id in self.episode_ids:
            sig = np.load(self.features_root / ep_id / "signal.npz")["signal"]  # (T,N,17)
            flat = sig.reshape(-1, sig.shape[-1])  # (T*N, 17)
            # keep only non-NaN rows
            mask = ~np.isnan(flat).any(axis=1)
            if mask.any():
                all_values.append(flat[mask])

        X = np.concatenate(all_values, axis=0)
        self.scaler = StandardScaler().fit(X)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(self.scaler, f)

        return self.scaler

    def load_scaler(self, path: Path) -> None:
        """Load a previously fitted scaler from disk."""
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, idx: int) -> dict:
        ep_id = self.episode_ids[idx]
        ep_dir = self.features_root / ep_id

        # Load signal (T, N, 17) and adjacency (T, N, N, 3)
        signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)

        # Normalise (scaler operates on 17-dim feature axis)
        if self.scaler is not None:
            T, N, d = signal.shape
            flat = signal.reshape(-1, d)               # (T*N, 17)
            nan_mask = np.isnan(flat)
            flat = np.where(nan_mask, 0.0, flat)       # impute before scaling
            flat = self.scaler.transform(flat).astype(np.float32)
            signal = flat.reshape(T, N, d)
        else:
            # Replace NaN with 0 even without scaler
            signal = np.nan_to_num(signal, nan=0.0)

        # Replace any remaining NaN in adjacency
        adjacency = np.nan_to_num(adjacency, nan=0.0)

        # Load scenario label (majority vote over timesteps)
        labels_df = pd.read_parquet(ep_dir / "labels.parquet")
        scenario: str = labels_df["scenario"].iloc[0]

        return {
            "signal": torch.from_numpy(signal),        # (T, N, 17)
            "adjacency": torch.from_numpy(adjacency),  # (T, N, N, 3)
            "scenario": scenario,
            "episode_id": ep_id,
            "T": signal.shape[0],
        }


def collate_episodes(batch: list[dict]) -> dict:
    """Pad variable-T episodes to the longest T in the batch.

    Padding is done with zeros (signal and adjacency).
    """
    max_T = max(item["T"] for item in batch)

    signals, adjacencies = [], []
    for item in batch:
        sig = item["signal"]      # (T, N, 17)
        adj = item["adjacency"]   # (T, N, N, 3)
        pad_T = max_T - sig.shape[0]
        if pad_T > 0:
            sig = torch.cat([sig, torch.zeros(pad_T, *sig.shape[1:])], dim=0)
            adj = torch.cat([adj, torch.zeros(pad_T, *adj.shape[1:])], dim=0)
        signals.append(sig)
        adjacencies.append(adj)

    return {
        "signal": torch.stack(signals),       # (B, max_T, N, 17)
        "adjacency": torch.stack(adjacencies),  # (B, max_T, N, N, 3)
        "scenario": [item["scenario"] for item in batch],
        "episode_id": [item["episode_id"] for item in batch],
        "T": torch.tensor([item["T"] for item in batch], dtype=torch.long),
    }
