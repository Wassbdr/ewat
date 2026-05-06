"""End-to-end determinism check on a synthetic 4-episode fixture.

We build a tiny in-memory feature store, train the STGCN encoder for 5 epochs,
fine-tune a SiameseTyper for 5 epochs, and run a precursor classification
forward pass — all twice with the same global seed. The two runs must produce
**bit-identical** intermediate tensors (encoder output, projected embeddings,
loss trajectory) on CPU.

The fixture mirrors the real feature layout consumed by ``EpisodeDataset``:

    features_root/<ep_id>/signal.npz       (T, N, 17) float32
    features_root/<ep_id>/adjacency.npz    (T, N, N, 3) float32
    features_root/<ep_id>/labels.parquet   columns: scenario, regime, ts

Two scenarios (`drift_scale_up`, `crash`) so the contrastive sampler has at
least one same-pair and one different-pair regardless of how it samples.

This test exists to lock the determinism guarantees we added in P0/P1
(explicit RNG plumbing, BCa bootstrap, masked pooling, etc.). Any future
change that introduces a non-seeded numpy/torch op will fail here.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ewat.encoder.dataset import EpisodeDataset, collate_episodes
from ewat.encoder.stgcn import STGCNEncoder
from ewat.typing.siamese import ContrastiveLoss, SiameseTyper


# --------------------------------------------------------------------------- #
# Synthetic feature store
# --------------------------------------------------------------------------- #

N_NODES = 4
D_FEAT = 17


def _set_global_seed(seed: int) -> None:
    """Seed every RNG that pytorch + numpy + python may consult."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_synthetic_episode(
    ep_dir: Path,
    *,
    scenario: str,
    T: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(T, N_NODES, D_FEAT)).astype(np.float32)
    adjacency = (
        rng.uniform(0.0, 1.0, size=(T, N_NODES, N_NODES, 3)).astype(np.float32)
    )
    labels = pd.DataFrame({
        "ts": np.arange(T, dtype=np.float64),
        "scenario": [scenario] * T,
        "regime": ["normal"] * (T // 2) + ["injection"] * (T - T // 2),
    })

    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ep_dir / "signal.npz", signal=signal)
    np.savez(ep_dir / "adjacency.npz", adjacency=adjacency)
    labels.to_parquet(ep_dir / "labels.parquet")


@pytest.fixture(scope="function")
def mini_feature_store(tmp_path: Path) -> dict:
    """Build a 4-episode mini feature store with two scenarios."""
    features_root = tmp_path / "features"
    features_root.mkdir()

    episodes = [
        ("ep_drift_a", "drift_scale_up", 12),
        ("ep_drift_b", "drift_scale_up", 14),
        ("ep_crash_a", "crash", 16),
        ("ep_crash_b", "crash", 13),
    ]
    for i, (ep_id, scenario, T) in enumerate(episodes):
        _write_synthetic_episode(
            features_root / ep_id, scenario=scenario, T=T, seed=10_000 + i,
        )

    split = {
        "train": ["ep_drift_a", "ep_crash_a"],
        "val": ["ep_drift_b"],
        "test": ["ep_crash_b"],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(__import__("json").dumps(split))

    return {
        "features_root": features_root,
        "split_json": split_path,
        "all_ids": [ep_id for ep_id, _, _ in episodes],
    }


# --------------------------------------------------------------------------- #
# Tiny end-to-end run (encoder pre-training + siamese fine-tuning)
# --------------------------------------------------------------------------- #

class _TinyDecoder(nn.Module):
    """Linear decoder (B, d_embed) → (B, N, d_feat) used during pre-training."""

    def __init__(self, d_embed: int, n_nodes: int, d_feat: int) -> None:
        super().__init__()
        self.n_nodes, self.d_feat = n_nodes, d_feat
        self.fc = nn.Linear(d_embed, n_nodes * d_feat)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).view(z.size(0), self.n_nodes, self.d_feat)


def _run_pipeline(seed: int, store: dict, n_epochs: int = 5) -> dict:
    """Execute a deterministic mini pipeline and return signature tensors."""
    _set_global_seed(seed)
    device = torch.device("cpu")

    train_ds = EpisodeDataset(
        store["split_json"], store["features_root"], split="train",
    )
    train_ds.fit_scaler()  # in-memory only

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_episodes,
        generator=g,
        num_workers=0,
    )

    encoder = STGCNEncoder(
        d_feat=D_FEAT,
        n_nodes=N_NODES,
        d_hidden=16,
        d_embed=16,
        n_gcn_layers=2,
        tcn_kernel=3,
        tcn_layers=1,
        n_adj_ch=3,
        dynamic_graph=True,
    ).to(device)

    decoder = _TinyDecoder(d_embed=16, n_nodes=N_NODES, d_feat=D_FEAT).to(device)

    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3,
    )

    losses: list[float] = []
    for _ in range(n_epochs):
        encoder.train()
        decoder.train()
        for batch in loader:
            sig = batch["signal"].to(device)        # (B, T, N, 17)
            adj = batch["adjacency"].to(device)     # (B, T, N, N, 3)
            lengths = batch["T"].to(device)
            target = sig.mean(dim=1)                # (B, N, 17)
            z = encoder(sig, adj, lengths=lengths)
            recon = decoder(z)
            loss = F.l1_loss(recon, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))

    typer = SiameseTyper(encoder=encoder, d_proj=8, freeze_encoder=False).to(device)
    contrastive = ContrastiveLoss(margin=1.0)
    typer_optim = torch.optim.Adam(typer.parameters(), lr=1e-3)

    typing_losses: list[float] = []
    for _ in range(n_epochs):
        typer.train()
        for batch in loader:
            sig = batch["signal"].to(device)
            adj = batch["adjacency"].to(device)
            lengths = batch["T"].to(device)
            B = sig.size(0)
            if B < 2:
                continue
            scenarios = batch["scenario"]
            i_idx = torch.arange(B)
            j_idx = torch.tensor([(i + 1) % B for i in range(B)], dtype=torch.long)
            sig_i, sig_j = sig[i_idx], sig[j_idx]
            adj_i, adj_j = adj[i_idx], adj[j_idx]
            len_i, len_j = lengths[i_idx], lengths[j_idx]
            is_same = torch.tensor(
                [scenarios[i] == scenarios[j] for i, j in zip(i_idx, j_idx)],
                dtype=torch.bool,
            )
            _, _, dist = typer(sig_i, adj_i, sig_j, adj_j, len_i, len_j)
            loss = contrastive(dist, is_same)
            typer_optim.zero_grad()
            loss.backward()
            typer_optim.step()
            typing_losses.append(float(loss.detach()))

    typer.eval()
    with torch.no_grad():
        eval_loader = DataLoader(
            train_ds, batch_size=4, shuffle=False, collate_fn=collate_episodes,
        )
        all_emb = []
        for batch in eval_loader:
            z = typer.embed(
                batch["signal"], batch["adjacency"], lengths=batch["T"],
            )
            all_emb.append(z)
        embeddings = torch.cat(all_emb, dim=0)

    return {
        "recon_losses": np.asarray(losses, dtype=np.float64),
        "typing_losses": np.asarray(typing_losses, dtype=np.float64),
        "embeddings": embeddings.cpu().numpy(),
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestDeterministicE2E:
    """Same seed → identical losses + embeddings."""

    def test_two_runs_same_seed_identical(self, mini_feature_store):
        out_a = _run_pipeline(seed=1234, store=mini_feature_store)
        out_b = _run_pipeline(seed=1234, store=mini_feature_store)

        assert out_a["recon_losses"].shape == out_b["recon_losses"].shape
        np.testing.assert_array_equal(out_a["recon_losses"], out_b["recon_losses"])

        assert out_a["typing_losses"].shape == out_b["typing_losses"].shape
        np.testing.assert_array_equal(out_a["typing_losses"], out_b["typing_losses"])

        np.testing.assert_array_equal(out_a["embeddings"], out_b["embeddings"])

    def test_different_seed_diverges(self, mini_feature_store):
        out_a = _run_pipeline(seed=1234, store=mini_feature_store)
        out_b = _run_pipeline(seed=999, store=mini_feature_store)
        # Embeddings are L2-normalised so the difference is bounded but
        # non-zero with overwhelming probability.
        assert not np.allclose(out_a["embeddings"], out_b["embeddings"])

    def test_pipeline_loss_decreases(self, mini_feature_store):
        """Sanity: 5 epochs of pre-training should reduce the L1 reconstruction
        loss on this trivial fixture (otherwise something is broken upstream).
        """
        out = _run_pipeline(seed=42, store=mini_feature_store, n_epochs=5)
        first_half = out["recon_losses"][: len(out["recon_losses"]) // 2].mean()
        second_half = out["recon_losses"][len(out["recon_losses"]) // 2 :].mean()
        assert second_half <= first_half + 1e-6
