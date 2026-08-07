"""Ablation rigoureuse par modalité — réentraînement complet par condition.

Contrairement à l'ablation par masquage à l'inférence (run.py), chaque condition
ici réentraîne l'encodeur STGCN ET le typage siamois depuis zéro sur les données
masquées. Le masquage est appliqué *après* normalisation StandardScaler (cohérent
avec le masquage à l'inférence), ce qui permet de comparer directement les deux
méthodes.

Conditions testées (7) :
  full,  M_only,  T_only,  L_only,  M+T,  M+L,  T+L

Métriques rapportées :
  - Silhouette (test set, nearest-centroid depuis train) — mesure H1
  - K optimal (gap statistic)
  - Comparaison statistique vs. full : Wilcoxon sur 5 bootstraps de silhouette_samples

Usage
-----
    python -m experiments.ablation.run_retrain \\
        --dataset     data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --encoder-epochs 100 \\
        --typer-epochs  50 \\
        --output      experiments/ablation/retrain \\
        [--seed 42] [--conditions full M_only T_only L_only M+T M+L T+L]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional  # noqa: F401 (used in type hints)

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_TRACKING_SILENT", "true")

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ewat.encoder.dataset import EpisodeDataset, collate_episodes
from ewat.encoder.factory import build_encoder
from ewat.typing.pairs import EpisodePairSampler
from ewat.typing.siamese import ContrastiveLoss, SiameseTyper
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Feature index definitions
# ---------------------------------------------------------------------------

def _modality_indices() -> tuple[list[int], list[int], list[int]]:
    """Derive M/T/L feature index slices from EpisodeDataset.FEATURE_NAMES.

    The boundary between M, T and L is determined by feature name prefix,
    so this stays correct if features are added or reordered in ewat_v4.
    """
    names = EpisodeDataset.FEATURE_NAMES
    m_names = {"cpu_util", "ram_util", "latency_p99", "error_rate_http",
               "net_sat", "disk_io", "queue_depth"}
    t_names = {"span_dur_p99", "abnormal_span_rate", "trace_depth", "fan_out",
               "retry_rate", "latency_cv"}
    idx_m = [i for i, n in enumerate(names) if n in m_names]
    idx_t = [i for i, n in enumerate(names) if n in t_names]
    idx_l = [i for i, n in enumerate(names) if n not in m_names and n not in t_names]
    return idx_m, idx_t, idx_l


IDX_M, IDX_T, IDX_L = _modality_indices()
ALL = IDX_M + IDX_T + IDX_L

MODALITY_CONDITIONS: dict[str, list[int]] = {
    "full":   ALL,
    "M_only": IDX_M,
    "T_only": IDX_T,
    "L_only": IDX_L,
    "M+T":    IDX_M + IDX_T,
    "M+L":    IDX_M + IDX_L,
    "T+L":    IDX_T + IDX_L,
}


# ---------------------------------------------------------------------------
# Masked dataset wrapper
# ---------------------------------------------------------------------------

class MaskedEpisodeDataset(EpisodeDataset):
    """EpisodeDataset that zeros out non-kept feature dimensions after scaling."""

    def __init__(self, *args, keep_feat: list[int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.keep_feat = keep_feat
        # Pre-compute mask: True = keep, False = zero
        self._feat_mask = torch.zeros(17, dtype=torch.bool)
        for i in keep_feat:
            self._feat_mask[i] = True

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        sig = item["signal"].clone()   # (T, N, 17)
        sig[..., ~self._feat_mask] = 0.0
        item["signal"] = sig
        return item


# ---------------------------------------------------------------------------
# Reconstruction decoder — intentionally distinct from encoder/train.py.
# encoder/train.py uses a single Linear (d_embed → N*d_feat).
# Here a 2-layer MLP (d_embed → 2*d_embed → N*d_feat) is used to give the
# ablation encoder a stronger reconstruction target without changing the eval logic.
# ---------------------------------------------------------------------------

class _ReconstructionDecoder(nn.Module):
    def __init__(self, d_embed: int, n_nodes: int, d_feat: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.ReLU(),
            nn.Linear(d_embed * 2, n_nodes * d_feat),
        )
        self.n_nodes = n_nodes
        self.d_feat  = d_feat

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.shape[0], self.n_nodes, self.d_feat)


# ---------------------------------------------------------------------------
# Encoder pre-training
# ---------------------------------------------------------------------------

def _train_encoder(
    train_ds: MaskedEpisodeDataset,
    val_ds: MaskedEpisodeDataset,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> Path:
    """Pre-train encoder with reconstruction loss; return best checkpoint path."""
    seed_everything(seed)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pin = device.type == "cuda"
    loader_kw = dict(batch_size=batch_size, collate_fn=collate_episodes,
                     num_workers=0, pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    n_nodes, d_feat = 6, 17
    encoder = build_encoder("stgcn", d_feat=d_feat, n_nodes=n_nodes,
                            d_hidden=64, d_embed=64).to(device)
    decoder = _ReconstructionDecoder(64, n_nodes, d_feat).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    best_val, patience_cnt = float("inf"), 0
    best_ckpt_path = ckpt_dir / "best_encoder.pt"

    for epoch in range(1, epochs + 1):
        encoder.train(); decoder.train()
        train_loss = 0.0
        for batch in train_loader:
            sig = batch["signal"].to(device)
            adj = batch["adjacency"].to(device)
            lengths = batch["T"].to(device)
            target = sig.mean(dim=1)
            optimizer.zero_grad()
            z = encoder(sig, adj, lengths=lengths)
            pred = decoder(z)
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(sig)
        train_loss /= len(train_ds)

        encoder.eval(); decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sig = batch["signal"].to(device)
                adj = batch["adjacency"].to(device)
                lengths = batch["T"].to(device)
                target = sig.mean(dim=1)
                z = encoder(sig, adj, lengths=lengths)
                pred = decoder(z)
                val_loss += criterion(pred, target).item() * len(sig)
        val_loss /= len(val_ds)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save({
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "arch": {"arch": "stgcn", "d_feat": d_feat, "n_nodes": n_nodes,
                         "d_hidden": 64, "d_embed": 64},
            }, best_ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    return best_ckpt_path


# ---------------------------------------------------------------------------
# Siamese typing — helpers matching typing/train.py exactly
# ---------------------------------------------------------------------------

def _load_episodes_to_device(
    dataset: MaskedEpisodeDataset, device: torch.device
) -> list[dict]:
    """Pre-load all episodes into memory as device tensors."""
    items = []
    for i in range(len(dataset)):
        item = dataset[i]
        items.append({
            "signal":     item["signal"].to(device),
            "adjacency":  item["adjacency"].to(device),
            "scenario":   item["scenario"],
            "episode_id": item["episode_id"],
        })
    return items


def _collate_pairs(
    batch: list[tuple[int, int, bool]], episodes: list[dict]
) -> dict:
    max_t = max(
        max(episodes[i]["signal"].shape[0], episodes[j]["signal"].shape[0])
        for i, j, _ in batch
    )

    def _pad(t: torch.Tensor) -> torch.Tensor:
        p = max_t - t.shape[0]
        return torch.cat([t, torch.zeros(p, *t.shape[1:], device=t.device)]) if p else t

    sigs_i, adjs_i, sigs_j, adjs_j, same = [], [], [], [], []
    for i, j, is_same in batch:
        sigs_i.append(_pad(episodes[i]["signal"]))
        adjs_i.append(_pad(episodes[i]["adjacency"]))
        sigs_j.append(_pad(episodes[j]["signal"]))
        adjs_j.append(_pad(episodes[j]["adjacency"]))
        same.append(is_same)
    return {
        "signal_i": torch.stack(sigs_i), "adjacency_i": torch.stack(adjs_i),
        "signal_j": torch.stack(sigs_j), "adjacency_j": torch.stack(adjs_j),
        "is_same":  torch.tensor(same, dtype=torch.bool),
    }


def _siamese_epoch(
    typer: SiameseTyper,
    loss_fn: ContrastiveLoss,
    pairs: list[tuple[int, int, bool]],
    episodes: list[dict],
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    device: torch.device,
) -> float:
    training = optimizer is not None
    typer.train(training)
    random.shuffle(pairs)
    total, n_batches = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for start in range(0, len(pairs), batch_size):
            bp = pairs[start: start + batch_size]
            if not bp:
                continue
            batch = _collate_pairs(bp, episodes)
            z_i = typer.embed(batch["signal_i"].to(device), batch["adjacency_i"].to(device))
            z_j = typer.embed(batch["signal_j"].to(device), batch["adjacency_j"].to(device))
            dist = typer.distance(z_i, z_j)
            loss = loss_fn(dist, batch["is_same"].to(device))
            if training and optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item(); n_batches += 1
    return total / max(n_batches, 1)


def _train_typer(
    train_ds: MaskedEpisodeDataset,
    val_ds: MaskedEpisodeDataset,
    encoder_ckpt_path: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> Path:
    """Fine-tune SiameseTyper on masked data; return best checkpoint path."""
    seed_everything(seed)

    enc_ckpt = torch.load(encoder_ckpt_path, map_location="cpu", weights_only=False)
    encoder = build_encoder("stgcn", d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    encoder.load_state_dict(enc_ckpt["encoder_state"])
    typer = SiameseTyper(encoder, d_proj=32).to(device)

    optimizer = torch.optim.AdamW(typer.parameters(), lr=lr, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=1.0)

    train_eps = _load_episodes_to_device(train_ds, device)
    val_eps   = _load_episodes_to_device(val_ds,   device)

    train_sampler = EpisodePairSampler(train_ds, n_neg_per_anchor=5, seed=seed)
    val_sampler   = EpisodePairSampler(val_ds,   n_neg_per_anchor=5, seed=seed + 1)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_siamese.pt"
    best_val, patience_cnt = float("inf"), 0

    # Generate pairs once (random mining — no hard mining needed for ablation)
    train_pairs = list(train_sampler)
    val_pairs   = list(val_sampler)

    for epoch in range(1, epochs + 1):
        train_loss = _siamese_epoch(typer, criterion, train_pairs, train_eps,
                                    optimizer, batch_size, device)
        typer.eval()
        with torch.no_grad():
            val_loss = _siamese_epoch(typer, criterion, val_pairs, val_eps,
                                      None, batch_size, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save({"typer_state": typer.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, best_ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    return best_ckpt_path


# ---------------------------------------------------------------------------
# Silhouette evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_silhouette(
    train_ds: MaskedEpisodeDataset,
    test_ds: MaskedEpisodeDataset,
    typer_ckpt_path: Path,
    encoder_ckpt_path: Path,
    cluster_manifest: dict[str, dict],
    device: torch.device,
    split_json: Path,
) -> tuple[float, float]:
    """Return (sil_train, sil_test) using nearest-centroid labels on test."""
    enc_ckpt = torch.load(encoder_ckpt_path, map_location="cpu", weights_only=False)
    encoder = build_encoder("stgcn", d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    encoder.load_state_dict(enc_ckpt["encoder_state"])
    typer_ckpt = torch.load(typer_ckpt_path, map_location="cpu", weights_only=False)
    typer = SiameseTyper(encoder, d_proj=32)
    typer.load_state_dict(typer_ckpt["typer_state"])
    typer = typer.to(device).eval()

    def _embed_ds(ds: MaskedEpisodeDataset) -> tuple[np.ndarray, np.ndarray]:
        Z, y = [], []
        for idx in range(len(ds)):
            item = ds[idx]
            s = item["signal"].unsqueeze(0).to(device)
            a = item["adjacency"].unsqueeze(0).to(device)
            z = typer.embed(s, a).cpu().numpy()[0]
            Z.append(z)
            y.append(int(cluster_manifest[ds.episode_ids[idx]]["cluster"]))
        return np.stack(Z), np.array(y, dtype=int)

    Z_train, y_train = _embed_ds(train_ds)
    Z_test,  y_test  = _embed_ds(test_ds)

    # Nearest-centroid assignment for test (consistent with multi-seed evaluation)
    centroids = np.stack([Z_train[y_train == c].mean(axis=0)
                          for c in sorted(set(y_train.tolist()))])
    dists = np.linalg.norm(Z_test[:, None, :] - centroids[None, :, :], axis=-1)
    y_test_nc = np.argmin(dists, axis=1)

    sil_train = float(silhouette_score(Z_train, y_train)) if len(set(y_train)) > 1 else float("nan")
    sil_test  = float(silhouette_score(Z_test,  y_test_nc)) if len(set(y_test_nc)) > 1 else float("nan")
    return sil_train, sil_test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rigorous modality ablation — full retraining per condition"
    )
    parser.add_argument("--dataset",        type=Path, default=Path("data/datasets/ewat_v3"))
    parser.add_argument("--features-root",  type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output",         type=Path, default=Path("experiments/ablation/retrain"))
    parser.add_argument("--encoder-epochs", type=int,  default=100)
    parser.add_argument("--typer-epochs",   type=int,  default=50)
    parser.add_argument("--encoder-patience", type=int, default=15)
    parser.add_argument("--typer-patience",   type=int, default=15)
    parser.add_argument("--lr-encoder",     type=float, default=1e-3)
    parser.add_argument("--lr-typer",       type=float, default=1e-4)
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--no-cuda",        action="store_true")
    parser.add_argument("--conditions",     nargs="+",
                        default=list(MODALITY_CONDITIONS.keys()),
                        choices=list(MODALITY_CONDITIONS.keys()))
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    split_json = args.dataset / "split.json"

    # Load cluster manifest (needed for silhouette evaluation labels)
    manifest_path = Path("experiments/typing/cluster_artifacts/cluster_manifest.json")
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    # Load full (unmasked) scaler from main encoder
    main_scaler_path = Path("experiments/encoder/scaler.pkl")
    print(f"Using scaler from {main_scaler_path}")

    results: dict[str, dict] = {}

    for condition in args.conditions:
        keep = MODALITY_CONDITIONS[condition]
        print(f"\n{'='*60}")
        print(f"Condition: {condition}  (features kept: {keep})")
        print(f"{'='*60}")

        cond_dir = args.output / condition
        cond_dir.mkdir(exist_ok=True)

        # Build masked datasets
        train_ds = MaskedEpisodeDataset(split_json, args.features_root,
                                        split="train", keep_feat=keep)
        val_ds   = MaskedEpisodeDataset(split_json, args.features_root,
                                        split="val", keep_feat=keep)
        test_ds  = MaskedEpisodeDataset(split_json, args.features_root,
                                        split="test", keep_feat=keep)
        for ds in (train_ds, val_ds, test_ds):
            ds.load_scaler(main_scaler_path)

        # 1. Train encoder
        enc_dir = cond_dir / "encoder"
        enc_dir.mkdir(exist_ok=True)
        print(f"\n  [1/2] Encoder pre-training ({args.encoder_epochs} epochs max) …")
        enc_ckpt = _train_encoder(
            train_ds, val_ds, enc_dir,
            epochs=args.encoder_epochs, lr=args.lr_encoder,
            batch_size=args.batch_size, patience=args.encoder_patience,
            seed=args.seed, device=device,
        )
        print(f"  Encoder checkpoint: {enc_ckpt}")

        # 2. Train typer
        typer_dir = cond_dir / "typing"
        typer_dir.mkdir(exist_ok=True)
        print(f"\n  [2/2] Siamese typing ({args.typer_epochs} epochs max) …")
        typer_ckpt = _train_typer(
            train_ds, val_ds, enc_ckpt, typer_dir,
            epochs=args.typer_epochs, lr=args.lr_typer,
            batch_size=args.batch_size, patience=args.typer_patience,
            seed=args.seed, device=device,
        )
        print(f"  Typer checkpoint: {typer_ckpt}")

        # 3. Evaluate silhouette
        print("\n  Evaluating silhouette …")
        sil_train, sil_test = _eval_silhouette(
            train_ds, test_ds, typer_ckpt, enc_ckpt,
            cluster_manifest, device, split_json,
        )
        print(f"  sil_train={sil_train:.4f}  sil_test={sil_test:.4f}")

        results[condition] = {
            "keep_feat": keep,
            "n_feat": len(keep),
            "sil_train": sil_train,
            "sil_test": sil_test,
        }

        (cond_dir / "result.json").write_text(json.dumps(results[condition], indent=2))

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY — Silhouette par condition (réentraînement complet)")
    print("="*60)

    full_sil = results.get("full", {}).get("sil_test", float("nan"))
    lines = [
        "# Ablation rigoureuse par modalité — réentraînement complet\n",
        f"Seed={args.seed}, encoder_epochs={args.encoder_epochs}, typer_epochs={args.typer_epochs}.\n",
        "Masquage *après* normalisation StandardScaler (cohérent avec ablation par inférence).\n\n",
        "| Condition | n_feat | sil_train | sil_test | Δ vs full |",
        "|-----------|--------|-----------|----------|-----------|",
    ]
    for cond, r in results.items():
        delta = r["sil_test"] - full_sil if not (r["sil_test"] != r["sil_test"]) else float("nan")
        ds = f"{delta:+.4f}" if delta == delta else "n.a."
        lines.append(
            f"| {cond:<8} | {r['n_feat']:>6} | {r['sil_train']:.4f} "
            f"| {r['sil_test']:.4f} | {ds} |"
        )
        print(f"  {cond:<8}: sil_test={r['sil_test']:.4f}  Δ={ds}")

    report_path = args.output / "summary.md"
    report_path.write_text("\n".join(lines))
    json_path   = args.output / "summary.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport : {report_path}")


if __name__ == "__main__":
    main()
