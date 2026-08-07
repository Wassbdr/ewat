"""SimCLR pre-training of the STGCN encoder.

This is an alternative to ``experiments.encoder.train`` (reconstruction
objective). The output checkpoint has the same shape and metadata so it
can be plugged into ``experiments.typing.train`` without any change.

Pipeline
--------
1. Load EpisodeDataset (train split) and fit StandardScaler.
2. Build STGCNEncoder + SimCLRHead.
3. For each epoch, the trainer applies stochastic temporal augmentations
   to every episode to produce two views and minimises NT-Xent loss.
4. Save the encoder checkpoint with the ``arch`` metadata expected by
   downstream pipelines.

Usage
-----
    python -m experiments.encoder.simclr_train \\
        --dataset data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --output experiments/encoder_simclr \\
        --epochs 50

The output layout matches ``experiments/encoder/train.py``:

    <output>/
      checkpoints/best_encoder.pt    # encoder_state + arch metadata
      checkpoints/last_encoder.pt
      scaler.pkl
      train_summary.json

so you can drop-in as ``--encoder-checkpoint`` for typing fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import torch

import mlflow
from ewat.encoder.dataset import EpisodeDataset, collate_episodes
from ewat.encoder.simclr import (
    AugmentationConfig,
    SimCLRHead,
    SimCLRTrainer,
)
from ewat.encoder.stgcn import STGCNEncoder
from utils.seeding import seed_everything

D_FEAT = 17
N_NODES = 6


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SimCLR pre-training (STGCN encoder)")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("experiments/encoder_simclr"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--d-embed", type=int, default=64)
    p.add_argument("--d-proj", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    # Augmentation knobs
    p.add_argument("--crop-min", type=float, default=0.6)
    p.add_argument("--crop-max", type=float, default=1.0)
    p.add_argument("--jitter-std", type=float, default=0.05)
    p.add_argument("--mask-ratio", type=float, default=0.15)
    return p


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    scaler_path = args.output / "scaler.pkl"
    split_json = args.dataset / "split.json"

    train_ds = EpisodeDataset(split_json, args.features_root, split="train")
    print("Fitting scaler on train set …")
    train_ds.fit_scaler(scaler_path)
    print(f"Scaler saved to {scaler_path}")

    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_episodes,
        num_workers=0,
        pin_memory=pin,
    )

    encoder = STGCNEncoder(
        d_feat=D_FEAT, n_nodes=N_NODES,
        d_hidden=args.d_hidden, d_embed=args.d_embed,
    ).to(device)
    head = SimCLRHead(d_in=args.d_embed, d_proj=args.d_proj).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    aug_cfg = AugmentationConfig(
        crop_min=args.crop_min,
        crop_max=args.crop_max,
        jitter_std=args.jitter_std,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
    )
    trainer = SimCLRTrainer(
        encoder, head, optimizer,
        aug_cfg=aug_cfg, temperature=args.temperature,
        device=device, seed=args.seed,
    )

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", str(args.output / "mlruns"))
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        mlflow.set_experiment("ewat")
        run = mlflow.start_run(run_name="encoder_simclr")
        mlflow.log_params({
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "d_embed": args.d_embed, "d_hidden": args.d_hidden,
            "d_proj": args.d_proj, "temperature": args.temperature,
            "seed": args.seed,
            "crop_min": args.crop_min, "crop_max": args.crop_max,
            "jitter_std": args.jitter_std, "mask_ratio": args.mask_ratio,
        })
    except Exception:
        run = None

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        state = trainer.run_epoch(train_loader, epoch=epoch)
        print(f"Epoch {epoch:03d}/{args.epochs}  ntxent_loss={state.train_loss:.4f}")
        if run is not None:
            try:
                mlflow.log_metrics({"ntxent_loss": state.train_loss}, step=epoch)
            except Exception:
                pass

        if state.train_loss < best_loss:
            best_loss = state.train_loss
            torch.save({
                "encoder_state": encoder.state_dict(),
                "head_state": head.state_dict(),
                "epoch": epoch,
                "train_loss": state.train_loss,
                "scaler_path": str(scaler_path),
                "args": vars(args),
                "arch": {
                    "d_feat": D_FEAT,
                    "n_nodes": N_NODES,
                    "d_hidden": args.d_hidden,
                    "d_embed": args.d_embed,
                    "n_gcn_layers": 2,
                    "tcn_kernel": 3,
                    "tcn_layers": 2,
                    "n_adj_ch": 3,
                    "step_seconds": 30.0,
                },
            }, ckpt_dir / "best_encoder.pt")

    torch.save({
        "encoder_state": encoder.state_dict(),
        "head_state": head.state_dict(),
        "epoch": args.epochs,
        "train_loss": state.train_loss,
        "scaler_path": str(scaler_path),
        "args": vars(args),
        "arch": {
            "d_feat": D_FEAT,
            "n_nodes": N_NODES,
            "d_hidden": args.d_hidden,
            "d_embed": args.d_embed,
            "n_gcn_layers": 2,
            "tcn_kernel": 3,
            "tcn_layers": 2,
            "n_adj_ch": 3,
            "step_seconds": 30.0,
        },
    }, ckpt_dir / "last_encoder.pt")

    summary = {
        "best_loss": best_loss,
        "epochs_trained": args.epochs,
        "scaler_path": str(scaler_path),
        "checkpoint": str(ckpt_dir / "best_encoder.pt"),
        "history": list(trainer.history),
    }
    (args.output / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {args.output / 'train_summary.json'}")

    if run is not None:
        try:
            mlflow.log_metrics({"best_ntxent_loss": best_loss})
            mlflow.end_run()
        except Exception:
            pass


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
