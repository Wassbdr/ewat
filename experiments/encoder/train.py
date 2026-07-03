"""Encoder pre-training — reconstruction-based self-supervised learning.

Trains STGCNEncoder with a lightweight reconstruction objective:
    target = signal.mean(dim=T)    # (B, N, 17) — time-averaged signal
    pred   = Decoder(encoder(s,a)) # (B, N, 17)
    loss   = L1(pred, target)

No scenario labels are needed.  The learned z_e will be fine-tuned in the
siamese typing step (experiments/typing/train.py).

Usage
-----
    python -m experiments.encoder.train \\
        --dataset data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --output experiments/encoder \\
        [--epochs 100] [--lr 1e-3] [--batch-size 32] [--patience 15]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ewat.encoder.dataset import EpisodeDataset, collate_episodes
from ewat.encoder.factory import build_encoder
from ewat.encoder.stgcn import STGCNEncoder
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Simple linear decoder (not part of the library — only used for pre-training)
# ---------------------------------------------------------------------------

class _ReconstructionDecoder(nn.Module):
    """Maps z_e ∈ ℝ^{d_embed} → (N * d_feat) → reshape to (N, d_feat)."""

    def __init__(self, d_embed: int, n_nodes: int, d_feat: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.d_feat = d_feat
        self.fc = nn.Linear(d_embed, n_nodes * d_feat)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_embed) → (B, N, d_feat)"""
        return self.fc(z).view(z.size(0), self.n_nodes, self.d_feat)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _epoch(
    encoder: STGCNEncoder,
    decoder: _ReconstructionDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)

    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            signal = batch["signal"].to(device)      # (B, T, N, 17)
            adjacency = batch["adjacency"].to(device)  # (B, T, N, N, 3)

            target = signal.mean(dim=1)              # (B, N, 17) — time mean

            z = encoder(signal, adjacency)           # (B, d_embed)
            pred = decoder(z)                        # (B, N, 17)
            loss = functional.l1_loss(pred, target)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-train STGCNEncoder (reconstruction)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to dataset dir (contains split.json)")
    parser.add_argument("--features-root", type=Path, required=True,
                        help="Path to feature store root (data/features/v3/)")
    parser.add_argument("--output", type=Path, default=Path("experiments/encoder"),
                        help="Output directory for checkpoints and scaler")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (val loss epochs)")
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--d-embed", type=int, default=64)
    parser.add_argument("--encoder-arch", choices=["stgcn", "stgat"], default="stgcn")
    # Step 5 fix 5.3 (audit 2026-05-26): expose use_layer_norm for new runs.
    # Default True since the audit recommended it for new training (LN
    # stabilises gradients on the 8-layer GCN+TCN stack). Set --no-layer-norm
    # to opt out for backward compatibility with v3 checkpoints.
    parser.add_argument("--use-layer-norm", dest="use_layer_norm",
                        action="store_true", default=True,
                        help="Use LayerNorm in TCN blocks (default: True)")
    parser.add_argument("--no-layer-norm", dest="use_layer_norm",
                        action="store_false",
                        help="Disable LayerNorm (v3 backward-compat)")
    # M4 (audit 2026-06): self-loops Â=D̃^{-1/2}(A+I)D̃^{-1/2} dans la conv
    # spatiale (nœuds isolés des fenêtres G(t) creuses). Défaut True pour les
    # nouveaux runs ; le flag est enregistré dans arch (pas de trace
    # state_dict) et les anciens checkpoints restent à False via la factory.
    parser.add_argument("--use-self-loops", dest="use_self_loops",
                        action="store_true", default=True,
                        help="Self-loops A+I dans la conv spatiale (défaut: True)")
    parser.add_argument("--no-self-loops", dest="use_self_loops",
                        action="store_false",
                        help="Désactive les self-loops (comportement pré-audit)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> None:
    """Pre-train the STGCN encoder. Pure function — no argv parsing.

    Used both by the legacy argparse ``main()`` and by the Hydra entry
    point in ``experiments.encoder.train_hydra``.
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    scaler_path = args.output / "scaler.pkl"
    split_json = args.dataset / "split.json"

    # --- Datasets ---
    train_ds = EpisodeDataset(split_json, args.features_root, split="train")
    val_ds = EpisodeDataset(split_json, args.features_root, split="val")

    print("Fitting scaler on train set …")
    train_ds.fit_scaler(scaler_path)
    val_ds.load_scaler(scaler_path)
    print(f"Scaler saved to {scaler_path}")
    # M15 (audit 2026-06): empreinte du scaler embarquée dans le checkpoint —
    # l'AlertAssembler la recompare au chargement (cohérence train/inférence).
    from ewat.utils.fingerprint import scaler_fingerprint
    scaler_sha = scaler_fingerprint(train_ds.scaler)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_episodes, num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_episodes, num_workers=0, pin_memory=pin,
    )

    # --- Model ---
    # D1 (audit 2026-06): dimensions résolues depuis les données (schéma v4=17
    # ou v5.1=18 via metadata, N depuis le 1er épisode) au lieu de 6/17 en dur.
    d_feat = len(train_ds.feature_names)
    n_nodes = train_ds[0]["signal"].shape[1]
    print(f"Data dims: N={n_nodes} services, d_feat={d_feat} features")
    arch_name = getattr(args, "encoder_arch", "stgcn")
    arch_kwargs = {}
    if arch_name == "stgcn":
        # Flags STGCN-only (STGAT ne les accepte pas — TypeError latent corrigé)
        arch_kwargs = {
            "use_layer_norm": getattr(args, "use_layer_norm", True),
            "use_self_loops": getattr(args, "use_self_loops", True),
        }
    encoder = build_encoder(
        arch_name,
        d_feat=d_feat, n_nodes=n_nodes,
        d_hidden=args.d_hidden, d_embed=args.d_embed,
        **arch_kwargs,
    ).to(device)
    decoder = _ReconstructionDecoder(args.d_embed, n_nodes, d_feat).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {n_enc:,} | Decoder params: {n_dec:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", str(args.output / "mlruns"))
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        mlflow.set_experiment("ewat")
    except Exception:
        pass

    try:
        run = mlflow.start_run(run_name="encoder_pretrain")
        mlflow.log_params({
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "d_embed": args.d_embed, "d_hidden": args.d_hidden,
            "patience": args.patience, "seed": args.seed,
        })
    except Exception:
        run = None

    for epoch in range(1, args.epochs + 1):
        train_loss = _epoch(encoder, decoder, train_loader, optimizer, device)
        val_loss = _epoch(encoder, decoder, val_loader, None, device)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if run is not None:
            try:
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            except Exception:
                pass

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = {
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "scaler_path": str(scaler_path),
                "scaler_sha256": scaler_sha,  # M15: cohérence train/inférence
                "args": vars(args),
                "arch": {
                    "architecture": getattr(args, "encoder_arch", "stgcn"),
                    "d_feat": d_feat,
                    "n_nodes": n_nodes,
                    "d_hidden": args.d_hidden,
                    "d_embed": args.d_embed,
                    "n_gcn_layers": 2,
                    "tcn_kernel": 3,
                    "tcn_layers": 2,
                    "n_adj_ch": 3,
                    "step_seconds": 30.0,
                    # M4: requis par build_encoder_from_checkpoint (pas de
                    # trace state_dict)
                    "use_self_loops": arch_kwargs.get("use_self_loops", False),
                },
            }
            torch.save(ckpt, ckpt_dir / "best_encoder.pt")
            print(f"  ✓ Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final checkpoint (last epoch)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "scaler_path": str(scaler_path),
        "scaler_sha256": scaler_sha,  # M15: cohérence train/inférence
        "args": vars(args),
        "arch": {
            "architecture": arch_name,
            "d_feat": d_feat,
            "n_nodes": n_nodes,
            "d_hidden": args.d_hidden,
            "d_embed": args.d_embed,
            "n_gcn_layers": 2,
            "tcn_kernel": 3,
            "tcn_layers": 2,
            "n_adj_ch": 3,
            "step_seconds": 30.0,
            "use_self_loops": arch_kwargs.get("use_self_loops", False),
        },
    }, ckpt_dir / "last_encoder.pt")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best_encoder.pt'}")

    if run is not None:
        try:
            mlflow.log_metrics({"best_val_loss": best_val_loss})
            mlflow.end_run()
        except Exception:
            pass

    # Save training summary
    summary = {
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch,
        "scaler_path": str(scaler_path),
        "checkpoint": str(ckpt_dir / "best_encoder.pt"),
    }
    (args.output / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {args.output / 'train_summary.json'}")


def main() -> None:
    """Argparse entry point (legacy)."""
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
