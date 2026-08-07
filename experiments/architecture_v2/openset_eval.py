"""C3 — Open-set LOSO evaluation with OpenMax.

For each held-out scenario s ∈ 15 Chaos Mesh scenarios:
  1. Retrain STGCN+15-way classifier on 14 scenarios (no fold for s).
  2. Compute embeddings for the test set.
  3. Fit OpenMax on the TRAIN-of-14 embeddings (Weibull on distances to class
     means).
  4. At inference on the test set:
       - p_open ∈ ℝ^16 = (15 known classes + 1 unknown)
       - For test episodes from scenario s: count how often argmax = unknown
       - For test episodes from known scenarios: macro-AUROC and degradation
         vs closed-set.

Metrics
-------
- **Unknown AUROC** : binary discrimination (known vs unknown) using
  ``p_open[..., -1]`` as the unknown score. Computed over all test episodes
  labeled known / unknown depending on whether their scenario is the held-out s.
- **Open-set top-1** : argmax over (15 + 1) classes — measures the fraction of
  held-out episodes correctly tagged as "unknown".
- **Closed-set macro-AUROC** : 15-way OvR macro-AUROC computed on the *known*
  classes after OpenMax revision. Must not degrade much vs closed-set.

Usage
-----
    python -m experiments.architecture_v2.openset_eval \\
        --dataset data/datasets/ewat_v4_strat \\
        --features-root data/features/v4 \\
        --output experiments/architecture_v2/openset \\
        [--epochs 80] [--tail-size 20] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ewat.encoder.dataset import EpisodeDataset, collate_episodes
from ewat.openset.openmax import OpenMax
from experiments.architecture_v2.train_chaos_mesh import (
    STGCNClassifier,
    _macro_auroc,
    _scan_scenarios,
    _scenario_to_int,
    _train_one_run,
)
from utils.seeding import seed_everything


@torch.no_grad()
def _collect_logits_and_embeddings(
    model: STGCNClassifier,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logits, embeddings z_e, y_true)."""
    model.eval()
    all_logits, all_z, all_y = [], [], []
    for batch in loader:
        sig = batch["signal"].to(device)
        adj = batch["adjacency"].to(device)
        lengths = batch["T"].to(device)
        logits, z = model(sig, adj, lengths)
        all_logits.append(logits.cpu().numpy())
        all_z.append(z.cpu().numpy())
        all_y.append(_scenario_to_int(batch["scenario"], classes))
    return (np.concatenate(all_logits, axis=0),
            np.concatenate(all_z, axis=0),
            np.concatenate(all_y, axis=0))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C3 — OpenMax LOSO open-set evaluation")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/architecture_v2/openset"))
    p.add_argument("--epochs", type=int, default=60,
                   help="Epochs per fold. 60 should suffice (val AUROC plateaus ~50).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--d-embed", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tail-size", type=int, default=20)
    p.add_argument("--alpha-rank", type=int, default=None,
                   help="OpenMax alpha_rank. Default = n_classes")
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-instance-norm", action="store_true")
    p.add_argument("--scenarios", type=str, nargs="+", default=None,
                   help="Subset of scenarios to hold out (default: all 15).")
    return p


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.output.mkdir(parents=True, exist_ok=True)

    classes = _scan_scenarios(args.dataset)
    n_classes = len(classes)
    print(f"Dataset: {args.dataset} | {n_classes} classes")

    df = pd.read_parquet(args.dataset / "index.parquet")
    scenarios_to_test = args.scenarios if args.scenarios else classes
    print(f"LOSO folds: {len(scenarios_to_test)}")

    per_scenario: dict[str, dict] = {}
    unknown_aurocs, closed_aurocs, top1_unknown_rates = [], [], []

    for fold_idx, held_s in enumerate(scenarios_to_test):
        print(f"\n=== LOSO fold {fold_idx+1}/{len(scenarios_to_test)}: hold out '{held_s}' ===")
        train_ids = df[(df["split"] == "train") & (df["scenario"] != held_s)]["episode_id"].tolist()

        model, info = _train_one_run(
            args, classes, device, train_ep_filter=train_ids,
            log_prefix=f"[LOSO-{held_s}] ",
        )

        # Build train loader (full v4_strat train minus held_s) for OpenMax fit
        split_json = args.dataset / "split.json"
        ds_train_loso = EpisodeDataset(
            split_json, args.features_root, split="train",
            instance_normalize=not args.no_instance_norm,
        )
        ds_train_loso.episode_ids = [
            e for e in ds_train_loso.episode_ids if e in set(train_ids)
        ]
        ds_train_loso.scaler = info["scaler"]
        loader_train_loso = DataLoader(
            ds_train_loso, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_episodes,
        )
        train_logits, train_z, train_y = _collect_logits_and_embeddings(
            model, loader_train_loso, classes, device
        )

        # OpenMax fit on training embeddings (Weibull on per-class distances)
        openmax = OpenMax(
            n_classes=n_classes, tail_size=args.tail_size,
            alpha_rank=args.alpha_rank, metric=args.metric,
        ).fit(train_z, train_y)

        # Eval on test set
        ds_test = EpisodeDataset(
            split_json, args.features_root, split="test",
            instance_normalize=not args.no_instance_norm,
        )
        ds_test.scaler = info["scaler"]
        loader_test = DataLoader(
            ds_test, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_episodes,
        )
        test_logits, test_z, test_y = _collect_logits_and_embeddings(
            model, loader_test, classes, device
        )

        # OpenMax prediction with logit-revision mode
        p_open = openmax.predict_proba(test_z, logits=test_logits)   # (n, K+1)
        unknown_score = p_open[:, -1]

        held_s_idx = classes.index(held_s)
        unknown_label = (test_y == held_s_idx).astype(int)
        # Unknown AUROC: discrimination known vs unknown using unknown_score
        if unknown_label.sum() > 0 and unknown_label.sum() < len(unknown_label):
            try:
                unknown_auroc = float(roc_auc_score(unknown_label, unknown_score))
            except ValueError:
                unknown_auroc = float("nan")
        else:
            unknown_auroc = float("nan")

        # Top-1 unknown rate on held-out test episodes
        held_mask = (test_y == held_s_idx)
        if held_mask.any():
            argmax_class = p_open[held_mask].argmax(axis=1)
            top1_unknown = float(np.mean(argmax_class == n_classes))
        else:
            top1_unknown = float("nan")

        # Closed-set macro-AUROC after OpenMax revision (only on known classes)
        known_mask = ~held_mask
        if known_mask.sum() >= 2:
            known_y = test_y[known_mask]
            known_probs = p_open[known_mask, :n_classes]
            # Renormalize (drop unknown column)
            known_probs = known_probs / (known_probs.sum(axis=1, keepdims=True) + 1e-12)
            macro_after, _ = _macro_auroc(known_y, known_probs, n_classes)
        else:
            macro_after = float("nan")

        # Closed-set macro-AUROC BEFORE OpenMax (softmax of original logits)
        softmax = F.softmax(torch.from_numpy(test_logits[known_mask]), dim=-1).numpy()
        macro_before, _ = _macro_auroc(test_y[known_mask], softmax, n_classes)

        unknown_aurocs.append(unknown_auroc)
        top1_unknown_rates.append(top1_unknown)
        closed_aurocs.append(macro_after)
        per_scenario[held_s] = {
            "unknown_auroc": unknown_auroc,
            "top1_unknown_rate_on_held": top1_unknown,
            "closed_macro_auroc_after_openmax": macro_after,
            "closed_macro_auroc_before_openmax": macro_before,
            "n_held_in_test": int(held_mask.sum()),
            "n_known_in_test": int(known_mask.sum()),
        }
        print(f"  unknown-AUROC = {unknown_auroc:.3f}  |  top1(unknown) on held = "
              f"{top1_unknown:.3f}")
        print(f"  closed AUROC : before={macro_before:.3f}  after={macro_after:.3f}")

    # Aggregate
    unknown_arr = np.array([x for x in unknown_aurocs if not np.isnan(x)])
    closed_arr = np.array([x for x in closed_aurocs if not np.isnan(x)])
    top1_arr = np.array([x for x in top1_unknown_rates if not np.isnan(x)])
    summary = {
        "dataset": str(args.dataset),
        "n_classes": n_classes,
        "tail_size": args.tail_size, "metric": args.metric, "seed": args.seed,
        "epochs": args.epochs,
        "n_folds": len(scenarios_to_test),
        "unknown_auroc_mean": float(np.mean(unknown_arr)),
        "unknown_auroc_std": float(np.std(unknown_arr)),
        "top1_unknown_rate_mean": float(np.mean(top1_arr)),
        "top1_unknown_rate_std": float(np.std(top1_arr)),
        "closed_macro_auroc_after_openmax_mean": float(np.mean(closed_arr)),
        "closed_macro_auroc_after_openmax_std": float(np.std(closed_arr)),
        "per_scenario": per_scenario,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== AGGREGATE ({len(unknown_arr)}/{len(scenarios_to_test)}) ===")
    print(f"  Unknown AUROC = {summary['unknown_auroc_mean']:.3f} ± "
          f"{summary['unknown_auroc_std']:.3f}")
    print(f"  Top1 unknown on held = {summary['top1_unknown_rate_mean']:.3f} ± "
          f"{summary['top1_unknown_rate_std']:.3f}")
    print(f"  Closed macro-AUROC (after OpenMax) = "
          f"{summary['closed_macro_auroc_after_openmax_mean']:.3f}")

    # Markdown
    lines = [
        "# C3 — OpenMax LOSO open-set evaluation",
        "",
        f"Dataset : `{args.dataset}` | classes : {n_classes} | tail = "
        f"{args.tail_size} | metric = {args.metric}",
        "",
        "## Aggregate",
        "",
        f"- **Unknown AUROC = {summary['unknown_auroc_mean']:.3f} ± "
        f"{summary['unknown_auroc_std']:.3f}**",
        f"- Top1(unknown) rate on held-out = "
        f"{summary['top1_unknown_rate_mean']:.3f} ± "
        f"{summary['top1_unknown_rate_std']:.3f}",
        f"- Closed macro-AUROC after OpenMax = "
        f"{summary['closed_macro_auroc_after_openmax_mean']:.3f}",
        "",
        "## Per held-out scenario",
        "",
        "| held-out | n_held | unknown-AUROC | top1(unknown) | closed before | closed after |",
        "|---|---|---|---|---|---|",
    ]
    for s, r in per_scenario.items():
        lines.append(
            f"| {s} | {r['n_held_in_test']} | {r['unknown_auroc']:.3f} | "
            f"{r['top1_unknown_rate_on_held']:.3f} | "
            f"{r['closed_macro_auroc_before_openmax']:.3f} | "
            f"{r['closed_macro_auroc_after_openmax']:.3f} |"
        )
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
