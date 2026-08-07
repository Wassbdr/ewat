"""C1 — Train STGCN encoder + 15-way classifier on Chaos Mesh scenario labels.

Objectif (Plan unifié — Phase C1)
---------------------------------
Éliminer la circularité d'évaluation H3 à la source en remplaçant la cible
(cluster EWAT auto-référent) par les 15 scénarios Chaos Mesh, qui sont une
vérité terrain indépendante du pipeline.

Pipeline
--------
    S(t) ∈ ℝ^{T×N×17}
        ↓ instance normalize (per-episode z-score on normal regime)
        ↓ STGCN encoder → z_e ∈ ℝ^{d_embed}
        ↓ linear head 15-way
        ↓ softmax  → CE loss

Evaluation
----------
- macro-AUROC test (OvR sur les 15 scénarios) + IC bootstrap.
- per-scenario AUROC.
- LOSO-CV (retrain encodeur 15 fois, hold-out un scénario à chaque fois) — option
  ``--loso``, lent (~15 × time of single training).

Usage
-----
    python -m experiments.architecture_v2.train_chaos_mesh \\
        --dataset data/datasets/ewat_v4_strat \\
        --features-root data/features/v4 \\
        --output experiments/architecture_v2/chaos_mesh \\
        [--epochs 80] [--seed 42] [--no-instance-norm]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ewat.encoder.dataset import EpisodeDataset, collate_episodes
from ewat.encoder.stgcn import STGCNEncoder
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Model: STGCN + linear classifier head
# ---------------------------------------------------------------------------

class STGCNClassifier(nn.Module):
    """STGCN encoder + linear classifier on top.

    Outputs raw logits (B, n_classes). Use CE loss on these.
    """

    def __init__(
        self,
        n_classes: int,
        d_feat: int = 17,
        n_nodes: int = 6,
        d_hidden: int = 64,
        d_embed: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = STGCNEncoder(
            d_feat=d_feat, n_nodes=n_nodes,
            d_hidden=d_hidden, d_embed=d_embed,
            dropout=dropout, use_layer_norm=True,
            use_self_loops=True,  # M4 audit 2026-06 (rerun C1 post-fixes)
        )
        self.classifier = nn.Linear(d_embed, n_classes)

    def forward(
        self,
        signal: torch.Tensor,
        adjacency: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(signal, adjacency, lengths)   # (B, d_embed)
        logits = self.classifier(z)                     # (B, n_classes)
        return logits, z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_scenarios(dataset_dir: Path) -> list[str]:
    df = pd.read_parquet(dataset_dir / "index.parquet")
    return sorted(df["scenario"].unique().tolist())


def _scenario_to_int(scenarios: list[str], classes: list[str]) -> np.ndarray:
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    return np.array([cls_to_idx[s] for s in scenarios], dtype=int)


def _macro_auroc(y: np.ndarray, p: np.ndarray, n_classes: int) -> tuple[float, dict]:
    per_class = {}
    aurocs = []
    for c in range(n_classes):
        y_bin = (y == c).astype(int)
        if y_bin.sum() < 1 or y_bin.sum() == len(y_bin):
            per_class[c] = float("nan")
            continue
        try:
            auc = float(roc_auc_score(y_bin, p[:, c]))
        except ValueError:
            auc = float("nan")
        per_class[c] = auc
        if not np.isnan(auc):
            aurocs.append(auc)
    return (float(np.mean(aurocs)) if aurocs else float("nan")), per_class


@torch.no_grad()
def _collect_logits(
    model: STGCNClassifier,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on a loader, return (probas, y_true) in canonical class order."""
    model.eval()
    all_probas, all_y = [], []
    for batch in loader:
        sig = batch["signal"].to(device)
        adj = batch["adjacency"].to(device)
        lengths = batch.get("T")
        if lengths is not None:
            lengths = lengths.to(device)
        logits, _ = model(sig, adj, lengths)
        probas = F.softmax(logits, dim=-1).cpu().numpy()
        y_true = _scenario_to_int(batch["scenario"], classes)
        all_probas.append(probas)
        all_y.append(y_true)
    return np.concatenate(all_probas, axis=0), np.concatenate(all_y, axis=0)


def _train_one_run(
    args, classes: list[str], device: torch.device,
    train_ep_filter: list[str] | None = None,
    log_prefix: str = "",
) -> tuple[STGCNClassifier, dict]:
    """Train encoder + classifier on (split.json train), optionally filtering
    episode_ids (used for LOSO retraining). Returns model + result dict.
    """
    split_json = args.dataset / "split.json"

    # Build train dataset (optionally restricted)
    ds_train = EpisodeDataset(
        split_json, args.features_root, split="train",
        instance_normalize=not args.no_instance_norm,
    )
    if train_ep_filter is not None:
        keep = set(train_ep_filter)
        ds_train.episode_ids = [e for e in ds_train.episode_ids if e in keep]
    ds_val = EpisodeDataset(
        split_json, args.features_root, split="val",
        instance_normalize=not args.no_instance_norm,
    )

    # Fit scaler on filtered train (or full train if no filter), reuse on val/test
    scaler = ds_train.fit_scaler()
    ds_val.scaler = scaler

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_episodes, num_workers=0,
    )
    loader_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_episodes, num_workers=0,
    )

    model = STGCNClassifier(
        n_classes=len(classes), d_feat=17, n_nodes=6,
        d_hidden=args.d_hidden, d_embed=args.d_embed, dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val_auroc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for batch in loader_train:
            sig = batch["signal"].to(device)
            adj = batch["adjacency"].to(device)
            lengths = batch["T"].to(device)
            y = torch.from_numpy(_scenario_to_int(batch["scenario"], classes)).long().to(device)
            logits, _ = model(sig, adj, lengths)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        train_loss = total_loss / max(n, 1)

        probas_val, y_val = _collect_logits(model, loader_val, classes, device)
        macro_val, _ = _macro_auroc(y_val, probas_val, len(classes))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{log_prefix}epoch {epoch+1:3d} | loss={train_loss:.4f} | "
                  f"val macro-AUROC={macro_val:.3f}")
        if macro_val > best_val_auroc:
            best_val_auroc = macro_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auroc": best_val_auroc, "scaler": scaler}


def _bootstrap_macro_ci(
    y: np.ndarray, p: np.ndarray, n_classes: int,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m, _ = _macro_auroc(y[idx], p[idx], n_classes)
        if not np.isnan(m):
            boots.append(m)
    if not boots:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(boots)),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C1 — STGCN Chaos Mesh trainer")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/architecture_v2/chaos_mesh"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--d-embed", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-instance-norm", action="store_true",
                   help="Disable per-episode instance normalization (default: enabled).")
    p.add_argument("--loso", action="store_true",
                   help="Run LOSO-CV after stratified training (15× retraining).")
    return p


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.output.mkdir(parents=True, exist_ok=True)

    classes = _scan_scenarios(args.dataset)
    n_classes = len(classes)
    print(f"Dataset: {args.dataset} | {n_classes} classes | "
          f"instance_norm={'on' if not args.no_instance_norm else 'OFF'}")

    # ---- Stratified training ----
    print("\n=== Stratified training ===")
    model, info = _train_one_run(args, classes, device, log_prefix="[strat] ")

    # Eval on test
    split_json = args.dataset / "split.json"
    ds_test = EpisodeDataset(
        split_json, args.features_root, split="test",
        instance_normalize=not args.no_instance_norm,
    )
    ds_test.scaler = info["scaler"]
    loader_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_episodes,
    )
    probas_test, y_test = _collect_logits(model, loader_test, classes, device)
    macro_test, per_class = _macro_auroc(y_test, probas_test, n_classes)
    rng = np.random.default_rng(args.seed)
    boot_mean, ci_lo, ci_hi = _bootstrap_macro_ci(
        y_test, probas_test, n_classes, args.n_bootstrap, rng
    )

    print(f"\n** TEST macro-AUROC = {macro_test:.3f} **")
    print(f"   Bootstrap mean = {boot_mean:.3f}  95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]")

    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "classifier_state": model.classifier.state_dict(),
        "classes": classes,
        "args": vars(args).copy() | {
            "dataset": str(args.dataset),
            "features_root": str(args.features_root),
            "output": str(args.output),
        },
    }, ckpt_dir / "best_model.pt")
    with open(ckpt_dir / "scaler.pkl", "wb") as f:
        pickle.dump(info["scaler"], f)
    print(f"Saved checkpoint to {ckpt_dir / 'best_model.pt'}")

    # Save test predictions + summary
    np.savez(args.output / "test_predictions.npz",
             probas=probas_test, y_true=y_test, classes=np.array(classes))
    summary = {
        "dataset": str(args.dataset),
        "classes": classes,
        "n_classes": n_classes,
        "stratified": {
            "best_val_auroc": info["best_val_auroc"],
            "test_macro_auroc": macro_test,
            "test_ci_95": [ci_lo, ci_hi],
            "test_bootstrap_mean": boot_mean,
            "per_scenario_auroc": {
                classes[c]: per_class.get(c, float("nan"))
                for c in range(n_classes)
            },
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "d_hidden": args.d_hidden, "d_embed": args.d_embed,
            "instance_norm": not args.no_instance_norm, "seed": args.seed,
        },
    }

    # ---- LOSO (optional, heavy) ----
    if args.loso:
        print(f"\n=== LOSO-CV ({len(classes)} folds) ===")
        loso_results = {}
        for s_idx, s in enumerate(classes):
            print(f"\n--- LOSO fold {s_idx+1}/{n_classes}: hold out {s} ---")
            # Filter train: keep episodes whose scenario != s
            df = pd.read_parquet(args.dataset / "index.parquet")
            train_keep = df[(df["split"] == "train") & (df["scenario"] != s)]
            train_ids = train_keep["episode_id"].tolist()
            model_loso, info_loso = _train_one_run(
                args, classes, device, train_ep_filter=train_ids,
                log_prefix=f"[LOSO-{s}] ",
            )
            ds_test_loso = EpisodeDataset(
                split_json, args.features_root, split="test",
                instance_normalize=not args.no_instance_norm,
            )
            ds_test_loso.scaler = info_loso["scaler"]
            loader_test_loso = DataLoader(
                ds_test_loso, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_episodes,
            )
            probas_loso, y_loso = _collect_logits(
                model_loso, loader_test_loso, classes, device
            )
            macro_loso, _ = _macro_auroc(y_loso, probas_loso, n_classes)
            # Top-1 sur épisodes held-out
            s_idx_canonical = classes.index(s)
            s_mask = (y_loso == s_idx_canonical)
            top1 = float("nan")
            if s_mask.any():
                argmax = np.argmax(probas_loso[s_mask], axis=1)
                top1 = float(np.mean(argmax == y_loso[s_mask]))
            loso_results[s] = {
                "macro_auroc_full_test": macro_loso, "top1_held_out": top1,
            }
            print(f"  macro(full)={macro_loso:.3f} | top1(held-out)={top1:.3f}")
        loso_macros = np.array([r["macro_auroc_full_test"]
                                for r in loso_results.values()])
        loso_top1 = np.array([r["top1_held_out"]
                              for r in loso_results.values()])
        summary["loso"] = {
            "macro_auroc_full_test_mean": float(np.mean(loso_macros)),
            "macro_auroc_full_test_std": float(np.std(loso_macros)),
            "top1_held_out_mean": float(np.nanmean(loso_top1)),
            "top1_held_out_std": float(np.nanstd(loso_top1)),
            "per_scenario": loso_results,
        }
        print(f"\n** LOSO macro = {summary['loso']['macro_auroc_full_test_mean']:.3f} "
              f"± {summary['loso']['macro_auroc_full_test_std']:.3f} **")
        print(f"   top1 held-out = {summary['loso']['top1_held_out_mean']:.3f} "
              f"± {summary['loso']['top1_held_out_std']:.3f}")

    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    lines = [
        "# C1 — STGCN Chaos Mesh Classifier",
        "",
        f"Dataset : `{args.dataset}` | classes : {n_classes} | instance_norm : "
        f"{'on' if not args.no_instance_norm else 'OFF'}",
        "",
        "## Stratified",
        "",
        f"- **Test macro-AUROC = {macro_test:.3f}**  "
        f"[95% CI {ci_lo:.3f}, {ci_hi:.3f}]",
        f"- Bootstrap mean = {boot_mean:.3f}",
        f"- Best val macro-AUROC = {info['best_val_auroc']:.3f}",
        "",
        "### Per-scenario AUROC (test set)",
        "",
        "| scenario | AUROC |",
        "|---|---|",
    ]
    for c in range(n_classes):
        v = per_class.get(c, float("nan"))
        cls = classes[c]
        lines.append(f"| {cls} | {v:.3f} |" if not np.isnan(v) else f"| {cls} | NaN |")
    if args.loso and "loso" in summary:
        lines += [
            "",
            "## LOSO-CV (full retrain per held-out scenario)",
            "",
            f"- macro-AUROC (full test) = {summary['loso']['macro_auroc_full_test_mean']:.3f}"
            f" ± {summary['loso']['macro_auroc_full_test_std']:.3f}",
            f"- top-1 acc held-out = {summary['loso']['top1_held_out_mean']:.3f}"
            f" ± {summary['loso']['top1_held_out_std']:.3f}",
        ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
