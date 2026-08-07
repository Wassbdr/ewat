"""A2 — Leave-One-Scenario-Out cross-validation (precursor only).

Question
--------
The stratified split keeps the *same scenarios* in train, val and test (just
different repetitions). Does EWAT generalize to a scenario it never saw at
training time?

Method
------
For each scenario s ∈ {15 Chaos Mesh scenarios}:
  1. Keep encoder + SiameseTyper *as is* (trained on all 15 scenarios).
  2. Refit the PrecursorClassifier on train embeddings, EXCLUDING all episodes
     of scenario s.
  3. Evaluate on the held-out scenario's test episodes only.
  4. Report macro-AUROC and top-1 cluster accuracy.

This is a *light* LOSO (precursor-only). A full LOSO (retraining the encoder
and siamois too) would require ~15 × 50 min of compute — possible but
expensive. Given A1 already shows scenario-signature leakage, this lighter
test answers the complementary question: "can the precursor predict the cluster
of a held-out scenario's episode using embeddings produced by the all-scenarios
siamois?".

Interpretation
--------------
- If macro-AUROC_LOSO ≈ macro-AUROC_stratified → the precursor learns
  cluster-discriminating features from embeddings, not scenario identity.
- If macro-AUROC_LOSO ≪ macro-AUROC_stratified → the precursor memorizes
  scenario-specific signatures it has seen at training (consistent with A1).

Usage
-----
    python -m experiments.h3_robustness.loso_cv \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        [--k 6] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ewat.precursor.dataset import PrecursorDataset
from ewat.precursor.model import PrecursorClassifier
from experiments.h3_robustness.distant_window import _load_typer
from utils.seeding import seed_everything


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A2 — Leave-One-Scenario-Out CV")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/h3_robustness/loso_cv"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--classifier-type", default="lr_tuned",
                        choices=["lr", "lr_tuned", "rf", "svc"])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _slice_embeddings(
    z: np.ndarray, y: np.ndarray, ep_ids: list[str],
    scenarios: list[str], exclude_scenario: str | None = None,
    include_scenario: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(z), dtype=bool)
    if exclude_scenario is not None:
        mask &= np.array([s != exclude_scenario for s in scenarios])
    if include_scenario is not None:
        mask &= np.array([s == include_scenario for s in scenarios])
    return z[mask], y[mask]


@torch.no_grad()
def _embed_with_scenarios(
    typer, dataset: PrecursorDataset, device: torch.device,
    cluster_manifest: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Like _embed_dataset but also returns the scenario list aligned with z."""
    typer.eval()
    embeddings, labels, scenarios = [], [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        sig = item["signal"].unsqueeze(0).to(device)
        adj = item["adjacency"].unsqueeze(0).to(device)
        z = typer.embed(sig, adj).cpu().numpy()
        embeddings.append(z[0])
        labels.append(item["cluster"])
        scenarios.append(cluster_manifest[item["episode_id"]]["scenario"])
    return np.stack(embeddings), np.array(labels, dtype=int), scenarios


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    encoder_dir = args.encoder_dir if args.encoder_dir is not None else (
        args.typing_dir.parent / "encoder"
    )

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest = json.loads(manifest_path.read_text())
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1
    all_scenarios = sorted({v["scenario"] for v in cluster_manifest.values()})
    print(f"Manifest: {len(cluster_manifest)} ep | {n_clusters} clusters | "
          f"{len(all_scenarios)} scenarios")

    typer, scaler_path = _load_typer(args.typing_dir, encoder_dir, device)

    # Embed everything once (default window_position="last" — same as H3 status quo)
    ds_train = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="train")
    ds_test = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="test")
    if scaler_path.exists():
        ds_train.load_scaler(scaler_path)
        ds_test.load_scaler(scaler_path)

    print(f"Embedding train ({len(ds_train)}) …")
    z_train, y_train, sc_train = _embed_with_scenarios(
        typer, ds_train, device, cluster_manifest
    )
    print(f"Embedding test ({len(ds_test)}) …")
    z_test, y_test, sc_test = _embed_with_scenarios(
        typer, ds_test, device, cluster_manifest
    )

    rng = np.random.default_rng(args.seed)
    per_scenario_results: dict[str, dict] = {}
    macro_aurocs, top1_accs = [], []

    for s in all_scenarios:
        # LOSO: remove scenario s from TRAINING only.
        z_tr_loso, y_tr_loso = _slice_embeddings(
            z_train, y_train, [], sc_train, exclude_scenario=s
        )
        # Evaluate on the FULL test set (45 episodes) — AUROC defined on real OvR
        # task. Held-out scenario's clusters are still in the negative class for
        # other-cluster classifiers.
        z_te_full, y_te_full = z_test, y_test

        # Subset of held-out scenario s in test set (for top-1 unseen-scenario acc)
        s_mask = np.array([sc == s for sc in sc_test])
        z_te_s, y_te_s = z_test[s_mask], y_test[s_mask]

        train_clusters = set(y_tr_loso.tolist())
        clusters_in_held_out = sorted(set(y_te_s.tolist()))
        held_out_clusters_zero_train = [c for c in clusters_in_held_out
                                        if c not in train_clusters]

        clf = PrecursorClassifier(
            n_clusters=n_clusters, reg_c=args.reg_c, max_iter=args.max_iter,
            classifier_type=args.classifier_type,
        )
        clf.fit(z_tr_loso, y_tr_loso)

        # macro-AUROC on FULL test set
        auroc_per_cluster = clf.auroc_per_type(z_te_full, y_te_full)
        valid_aurocs = [v for v in auroc_per_cluster.values() if not np.isnan(v)]
        macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

        # Top-1 cluster accuracy on held-out scenario's test episodes only
        if len(z_te_s) > 0:
            proba_s = clf.predict_proba(z_te_s)   # (n_s, n_clusters)
            argmax_pred = np.argmax(proba_s, axis=1)
            top1_acc_s = float(np.mean(argmax_pred == y_te_s))
        else:
            top1_acc_s = float("nan")

        macro_aurocs.append(macro_auroc)
        top1_accs.append(top1_acc_s)
        per_scenario_results[s] = {
            "n_test_full": int(len(y_te_full)),
            "n_test_held_out": int(len(y_te_s)),
            "n_train_loso": int(len(y_tr_loso)),
            "clusters_in_held_out": clusters_in_held_out,
            "clusters_missing_in_train": held_out_clusters_zero_train,
            "macro_auroc_full_test": macro_auroc,
            "top1_acc_held_out": top1_acc_s,
            "per_cluster_auroc": {str(c): float(v)
                                  for c, v in auroc_per_cluster.items()
                                  if not np.isnan(v)},
        }
        print(f"  {s:<28s} | macro-AUROC(full)={macro_auroc:.3f} | "
              f"top1(s)={top1_acc_s:.3f} | missing={held_out_clusters_zero_train}")

    # Aggregates (ignoring NaNs)
    macro_arr = np.array([x for x in macro_aurocs if not np.isnan(x)])
    top1_arr = np.array([x for x in top1_accs if not np.isnan(x)])
    summary = {
        "k": args.k,
        "n_clusters": n_clusters,
        "n_scenarios": len(all_scenarios),
        "classifier_type": args.classifier_type,
        "seed": args.seed,
        "loso_macro_auroc_full_mean": float(np.mean(macro_arr)) if len(macro_arr) else float("nan"),
        "loso_macro_auroc_full_std": float(np.std(macro_arr)) if len(macro_arr) else float("nan"),
        "loso_top1_acc_held_out_mean": float(np.mean(top1_arr)) if len(top1_arr) else float("nan"),
        "loso_top1_acc_held_out_std": float(np.std(top1_arr)) if len(top1_arr) else float("nan"),
        "per_scenario": per_scenario_results,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\nLOSO aggregate (over {len(macro_arr)}/{len(all_scenarios)} scenarios):")
    print(f"  macro-AUROC (full test) = {summary['loso_macro_auroc_full_mean']:.3f} ± "
          f"{summary['loso_macro_auroc_full_std']:.3f}")
    print(f"  top-1 acc (held-out s)  = {summary['loso_top1_acc_held_out_mean']:.3f} ± "
          f"{summary['loso_top1_acc_held_out_std']:.3f}")

    # Markdown
    lines = [
        "# A2 — Leave-One-Scenario-Out (precursor-only)",
        "",
        f"k = {args.k} | classifier = {args.classifier_type} | seed = {args.seed} "
        f"| {len(all_scenarios)} scenarios",
        "",
        "## Aggregate (over scenarios with ≥1 valid cluster AUROC)",
        "",
        f"- **LOSO macro-AUROC (full test, 45 ep) = "
        f"{summary['loso_macro_auroc_full_mean']:.3f} ± "
        f"{summary['loso_macro_auroc_full_std']:.3f}**",
        f"- LOSO top-1 acc on held-out scenario (3 ep) = "
        f"{summary['loso_top1_acc_held_out_mean']:.3f} ± "
        f"{summary['loso_top1_acc_held_out_std']:.3f}",
        "",
        "Reference: stratified macro-AUROC (status quo H3, same encoder/siamois) "
        "≈ 0.904 with this setup (lr_tuned, k=6).",
        "",
        "## Per held-out scenario",
        "",
        "| held-out scenario | macro-AUROC (full test) | top-1 acc (held-out) | missing train clusters |",
        "|---|---|---|---|",
    ]
    for s in all_scenarios:
        r = per_scenario_results[s]
        mc = ",".join(str(c) for c in r["clusters_missing_in_train"]) or "—"
        lines.append(
            f"| {s} | {r['macro_auroc_full_test']:.3f} | "
            f"{r['top1_acc_held_out']:.3f} | {mc} |"
        )
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
