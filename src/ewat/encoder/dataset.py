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
    split_json:        Path to split.json (keys: train/val/test → list of episode_ids).
    features_root:     Root of the feature store (data/features/v3/).
    split:             Which split to load ("train", "val", "test").
    scaler:            Pre-fitted StandardScaler.  None = no normalisation.
    instance_normalize:
        If ``True``, z-score each episode using its own *normal-regime* mean
        and std per feature (across nodes and normal-regime timesteps). When
        ``True``, the global ``scaler`` is **not** applied on top — the two
        normalisations are mutually exclusive (Step 4 fix 4.2, audit 2026-05-26).
        Instance normalization removes absolute baseline differences between
        services / scenarios, leaving only relative dynamics. Set ``True`` for
        the architecture v2 (Chaos Mesh target) pipeline; ``False`` preserves
        v3 backward compatibility with the global scaler.
    imputation_strategy:
        How to replace NaN values before normalisation. One of:

        - ``"scaler_mean"`` (default): replace NaN with the train-set mean of
          each feature (from ``scaler.mean_``). After scaling, imputed cells
          become 0 — the neutral value. **Never use 0.0 as the raw imputation**
          (Step 4 fix 4.1) because ``(0 - mean) / std`` introduces bias.
        - ``"zero_post_scaling"``: pre-scale with whatever scaler, then replace
          remaining NaN with 0 — only safe if the scaler already centred the
          data. Discouraged.
        - ``"none"``: leave NaN in place. Caller must handle NaN downstream.

        Saved in checkpoint metadata for reproducibility at inference time.
    """

    FEATURE_NAMES: list[str] = [
        "cpu_util", "ram_util", "latency_p99", "error_rate_http",
        "net_sat", "disk_io", "queue_depth",
        "span_dur_p99", "abnormal_span_rate", "trace_depth", "fan_out",
        "retry_rate", "latency_cv",
        "log_error_rate", "log_warn_rate", "semantic_anomaly", "lexical_entropy",
    ]

    # Step 4 fix 4.1 (audit 2026-05-26): allowed imputation strategies.
    _VALID_IMPUTATION = ("scaler_mean", "zero_post_scaling", "none")

    def __init__(
        self,
        split_json: Path,
        features_root: Path,
        split: str = "train",
        scaler: StandardScaler | None = None,
        instance_normalize: bool = False,
        imputation_strategy: str = "scaler_mean",
    ) -> None:
        if imputation_strategy not in self._VALID_IMPUTATION:
            raise ValueError(
                f"imputation_strategy must be one of {self._VALID_IMPUTATION}, "
                f"got {imputation_strategy!r}"
            )
        self.features_root = Path(features_root)
        self.split = split
        self.scaler = scaler
        self.instance_normalize = instance_normalize
        self.imputation_strategy = imputation_strategy

        index = json.loads(Path(split_json).read_text())
        if split not in index:
            raise ValueError(f"Unknown split '{split}'; available: {list(index)}")
        self.episode_ids: list[str] = index[split]

    # ------------------------------------------------------------------
    # Scaler helpers
    # ------------------------------------------------------------------

    def fit_scaler(self, save_path: Path | None = None) -> StandardScaler:
        """Fit StandardScaler on this split and optionally save to disk.

        Step 2 fix 2.3 (audit 2026-05-26):
        The previous implementation discarded any row containing a single
        NaN feature, which could remove up to 50% of the training data when
        modalities have heterogeneous NaN ratios (e.g. T ~50% NaN on v3).

        New behaviour: fit *per-feature* on non-NaN values. Each of the 17
        features computes its own mean/std from whatever rows have a valid
        observation for that feature. This is mathematically equivalent to
        marginal nan-aware z-scoring and uses 100% of the available signal.

        The resulting StandardScaler has ``mean_`` and ``scale_`` arrays that
        can still be applied row-wise at inference via ``scaler.transform``,
        provided NaN are imputed beforehand (the dataset does this via
        ``flat = np.where(nan_mask, self.scaler.mean_, flat)`` in ``__getitem__``).
        """
        # Gather all rows from all train episodes, keeping NaN in place.
        all_rows: list[np.ndarray] = []
        for ep_id in self.episode_ids:
            sig = np.load(self.features_root / ep_id / "signal.npz")["signal"]
            flat = sig.reshape(-1, sig.shape[-1])  # (T*N, 17)
            all_rows.append(flat)
        X = np.concatenate(all_rows, axis=0).astype(np.float64)
        # Per-feature mean/std with NaN-aware reduction
        n_features = X.shape[1]
        mean = np.zeros(n_features, dtype=np.float64)
        scale = np.ones(n_features, dtype=np.float64)
        n_seen = np.zeros(n_features, dtype=np.int64)
        for f in range(n_features):
            col = X[:, f]
            valid = col[~np.isnan(col)]
            n_seen[f] = valid.size
            if valid.size >= 2:
                mean[f] = float(valid.mean())
                std = float(valid.std(ddof=0))
                scale[f] = std if std > 1e-12 else 1.0
            elif valid.size == 1:
                mean[f] = float(valid[0])
                scale[f] = 1.0
            # else: all NaN → keep mean=0, scale=1 (will pass through unchanged)
        # Sanity warning if some features have very few observations
        sparse = [(i, int(n_seen[i])) for i in range(n_features) if n_seen[i] < 10]
        if sparse:
            import logging
            logging.getLogger(__name__).warning(
                "fit_scaler: features with <10 valid observations: %s. "
                "Their mean/std are defaults and may bias scaling.", sparse,
            )
        # Hydrate a real StandardScaler so transform()/inverse_transform() work
        self.scaler = StandardScaler()
        self.scaler.mean_ = mean
        self.scaler.scale_ = scale
        self.scaler.var_ = scale ** 2
        self.scaler.n_features_in_ = n_features
        self.scaler.n_samples_seen_ = int(n_seen.max())

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

        # Step 4 fix 4.2 (audit 2026-05-26): instance_normalize and global scaler
        # are MUTUALLY EXCLUSIVE — they are two competing normalisations.
        # Previous behaviour chained them (instance → global), which is
        # conceptually weird (z-scoring already-z-scored data).
        if self.instance_normalize:
            labels_df = pd.read_parquet(ep_dir / "labels.parquet", columns=["regime"])
            normal_mask = (labels_df["regime"] == "normal").values
            if normal_mask.sum() >= 2:
                ref = signal[normal_mask]                                     # (n_normal, N, 17)
                mu = np.nanmean(ref, axis=(0, 1), keepdims=True)              # (1, 1, 17)
                sd = np.nanstd(ref, axis=(0, 1), keepdims=True) + 1e-6
                signal = ((signal - mu) / sd).astype(np.float32)
            # Step 4 fix 4.1: still need to handle NaN after instance norm
            # (mean/std were nan-aware via nanmean/nanstd, so NaN stays NaN).
            # Impute with 0.0 because we're already in z-scored space (mean=0).
            signal = np.nan_to_num(signal, nan=0.0)
        elif self.scaler is not None:
            T, N, d = signal.shape
            flat = signal.reshape(-1, d)               # (T*N, 17)
            nan_mask = np.isnan(flat)
            # Step 4 fix 4.1: never impute with 0.0 BEFORE scaling.
            # `(0 - scaler.mean_) / scaler.scale_` would bias features away from 0.
            if self.imputation_strategy == "scaler_mean":
                # Replace NaN with train mean → after transform, becomes 0 (neutral).
                flat = np.where(nan_mask, self.scaler.mean_, flat)
                flat = self.scaler.transform(flat).astype(np.float32)
            elif self.imputation_strategy == "zero_post_scaling":
                # Pre-scale (NaN propagates through transform), then impute 0.
                # Only correct if mean ≈ 0 already, otherwise biased.
                flat = self.scaler.transform(np.where(nan_mask, 0.0, flat)).astype(np.float32)
                flat = np.where(np.isnan(flat), 0.0, flat)
            elif self.imputation_strategy == "none":
                flat = self.scaler.transform(np.where(nan_mask, self.scaler.mean_, flat)).astype(np.float32)
                # Reinject NaN where original was NaN
                flat = np.where(nan_mask, np.nan, flat)
            signal = flat.reshape(T, N, d)
        else:
            # No scaler, no instance norm: raw signal. Last-resort NaN→0 since
            # most downstream operators (conv, GCN) don't accept NaN.
            # Caller is responsible for fitting a scaler in production.
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
