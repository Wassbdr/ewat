"""E10 (audit 2026-06) — cas limites de production jamais testés.

Épisode tout-NaN, graphe sans arêtes sur tout l'épisode, feature à variance
nulle, T=1, dataset à N−1 services. Chaque cas doit soit fonctionner, soit
échouer avec une erreur explicite — jamais produire des NaN silencieux.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from ewat.encoder.dataset import EpisodeDataset
from ewat.encoder.stgcn import STGCNEncoder


# ---------------------------------------------------------------------------
# Helpers — fabrique un feature store minimal
# ---------------------------------------------------------------------------

def _write_episode(root, ep_id, signal):
    d = root / ep_id
    d.mkdir(parents=True)
    T, N, _ = signal.shape
    np.savez_compressed(d / "signal.npz", signal=signal.astype(np.float32))
    np.savez_compressed(d / "adjacency.npz",
                        adjacency=np.random.rand(T, N, N, 3).astype(np.float32))
    pd.DataFrame({
        "regime": ["normal"] * max(T // 2, 1) + ["injection"] * (T - max(T // 2, 1)),
        "scenario": ["edge_case"] * T,
    }).to_parquet(d / "labels.parquet")


def _make_store(tmp_path, signals: dict):
    root = tmp_path / "features"
    for ep_id, sig in signals.items():
        _write_episode(root, ep_id, sig)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": list(signals), "val": [], "test": []}))
    return split, root


# ---------------------------------------------------------------------------
# Épisode tout-NaN
# ---------------------------------------------------------------------------

def test_all_nan_episode_scaler_and_getitem(tmp_path):
    rng = np.random.default_rng(0)
    signals = {
        "ep_ok": rng.normal(size=(10, 3, 17)),
        "ep_nan": np.full((10, 3, 17), np.nan),
    }
    split, root = _make_store(tmp_path, signals)
    ds = EpisodeDataset(split, root, split="train")
    scaler = ds.fit_scaler()  # NaN-aware : ne doit pas crasher
    assert not np.isnan(scaler.mean_).any()
    for i in range(len(ds)):
        item = ds[i]
        assert torch.isfinite(item["signal"]).all(), \
            "NaN résiduel après imputation scaler_mean"


def test_all_nan_feature_column_gets_default_stats(tmp_path):
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(10, 3, 17))
    sig[:, :, 5] = np.nan  # disk_io entièrement NaN (cas v3 réel)
    split, root = _make_store(tmp_path, {"ep": sig})
    ds = EpisodeDataset(split, root, split="train")
    scaler = ds.fit_scaler()
    assert scaler.mean_[5] == 0.0 and scaler.scale_[5] == 1.0
    assert torch.isfinite(ds[0]["signal"]).all()


# ---------------------------------------------------------------------------
# Variance nulle
# ---------------------------------------------------------------------------

def test_constant_feature_no_nan_after_scaling(tmp_path):
    rng = np.random.default_rng(2)
    sig = rng.normal(size=(10, 3, 17))
    sig[:, :, 4] = 7.0  # variance exactement nulle
    split, root = _make_store(tmp_path, {"ep": sig})
    ds = EpisodeDataset(split, root, split="train")
    scaler = ds.fit_scaler()
    assert scaler.scale_[4] == 1.0, "std=0 doit retomber sur scale=1 (pas /0)"
    assert torch.isfinite(ds[0]["signal"]).all()


# ---------------------------------------------------------------------------
# Graphe sans arêtes / T=1
# ---------------------------------------------------------------------------

def test_encoder_empty_graph_whole_episode():
    enc = STGCNEncoder(use_self_loops=True)
    sig = torch.randn(2, 8, 6, 17)
    adj = torch.zeros(2, 8, 6, 6, 3)  # aucune arête sur tout l'épisode
    z = enc(sig, adj)
    assert z.shape == (2, 64) and torch.isfinite(z).all()


def test_encoder_empty_graph_without_self_loops_still_finite():
    enc = STGCNEncoder(use_self_loops=False)  # comportement legacy
    sig = torch.randn(1, 8, 6, 17)
    adj = torch.zeros(1, 8, 6, 6, 3)
    z = enc(sig, adj)
    assert torch.isfinite(z).all()


def test_encoder_single_timestep():
    enc = STGCNEncoder(use_self_loops=True)
    sig = torch.randn(2, 1, 6, 17)
    adj = torch.rand(2, 1, 6, 6, 3)
    z = enc(sig, adj, lengths=torch.tensor([1, 1]))
    assert z.shape == (2, 64) and torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# N−1 services (épisode v5 où un service a disparu)
# ---------------------------------------------------------------------------

def test_mixed_n_services_fails_loud_not_silent(tmp_path):
    rng = np.random.default_rng(3)
    signals = {
        "ep_full": rng.normal(size=(10, 3, 17)),
        "ep_short": rng.normal(size=(10, 2, 17)),  # un service manquant
    }
    split, root = _make_store(tmp_path, signals)
    ds = EpisodeDataset(split, root, split="train")
    items = [ds[i] for i in range(len(ds))]
    # le N hétérogène doit casser au stack du collate, pas silencieusement après
    from ewat.encoder.dataset import collate_episodes
    with pytest.raises((RuntimeError, ValueError)):
        collate_episodes(items)


def test_mixed_feature_dim_raises_explicit(tmp_path):
    rng = np.random.default_rng(4)
    signals = {
        "ep_v4": rng.normal(size=(10, 3, 17)),
        "ep_v5": rng.normal(size=(10, 3, 18)),  # store mixte v4/v5
    }
    split, root = _make_store(tmp_path, signals)
    ds = EpisodeDataset(split, root, split="train")
    with pytest.raises(ValueError, match="features"):
        _ = [ds[i] for i in range(len(ds))]
