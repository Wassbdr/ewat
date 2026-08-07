"""A3 — Permutation test on cluster labels (null distribution).

Question
--------
What macro-AUROC do we get when the cluster label is *randomly permuted* across
train episodes? This gives an empirical null distribution: any AUROC at or near
the null is non-informative.

Method
------
1. Embed train + test with the trained encoder + siamois (same as H3).
2. For each of `n_permutations`:
     - Randomly shuffle ``y_train`` (preserving the label distribution).
     - Fit PrecursorClassifier on (z_train, y_train_shuffled).
     - Compute macro-AUROC on the (unshuffled) test set.
3. Compare the true AUROC against the null distribution → empirical p-value.

Interpretation
--------------
- p_value = (# null ≥ observed) / n_permutations
- If p_value < 0.05 → the model learns something better than chance from the
  embedding (which doesn't say *what* it learns; could still be scenario
  signature — see A1).
- If p_value > 0.05 → the precursor task is indistinguishable from random
  permutations (catastrophic).

Usage
-----
    python -m experiments.h3_robustness.permutation_test \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        [--k 6] [--n-permutations 100] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ewat.precursor.dataset import PrecursorDataset
from ewat.precursor.model import PrecursorClassifier
from experiments.h3_robustness.distant_window import _embed_dataset, _load_typer
from utils.seeding import seed_everything


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A3 — Permutation null distribution")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/h3_robustness/permutation_test"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--classifier-type", default="lr",
                        choices=["lr", "lr_tuned", "rf", "svc"],
                        help="lr (faster) recommended for permutation null")
    # Step 10 fix 10.7 (audit 2026-05-26): n=100 gives p-value resolution
    # ±0.01 (too coarse near α=0.05). Default raised to 500 for tighter SE.
    parser.add_argument("--n-permutations", type=int, default=500)
    # Step 10 fix 10.7: stratified permutation shuffles labels WITHIN each
    # scenario to preserve scenario marginal distributions. Without it,
    # a permutation can collapse all "drift" labels to "anomaly" and
    # vice-versa, breaking the null hypothesis interpretation.
    parser.add_argument("--stratified-permutation", action="store_true",
                        help="Permute labels within each scenario rather than "
                             "across the full train set (preserves marginals)")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _macro_auroc(clf: PrecursorClassifier, z: np.ndarray, y: np.ndarray) -> float:
    auroc = clf.auroc_per_type(z, y)
    valid = [a for a in auroc.values() if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


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
    print(f"Manifest: {len(cluster_manifest)} ep | {n_clusters} clusters")

    typer, scaler_path = _load_typer(args.typing_dir, encoder_dir, device)

    ds_train = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="train")
    ds_test = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="test")
    if scaler_path.exists():
        ds_train.load_scaler(scaler_path)
        ds_test.load_scaler(scaler_path)

    print("Embedding train / test once …")
    z_train, y_train = _embed_dataset(typer, ds_train, device)
    z_test, y_test = _embed_dataset(typer, ds_test, device)

    # Observed (un-permuted) AUROC for reference
    clf = PrecursorClassifier(
        n_clusters=n_clusters, reg_c=args.reg_c, max_iter=args.max_iter,
        classifier_type=args.classifier_type,
    )
    clf.fit(z_train, y_train)
    observed_macro = _macro_auroc(clf, z_test, y_test)
    print(f"Observed macro-AUROC (true labels) = {observed_macro:.3f}")

    rng = np.random.default_rng(args.seed)

    # Step 10 fix 10.7: precompute scenario groups for stratified permutation.
    train_scenarios = None
    if args.stratified_permutation:
        # Load scenarios for the train episodes (same order as z_train, y_train)
        train_scenarios = np.array(
            [info["scenario"] for ep, info in cluster_manifest.items()
             if info["split"] == "train"]
        )
        if len(train_scenarios) != len(y_train):
            raise RuntimeError(
                f"stratified permutation: scenario list size "
                f"({len(train_scenarios)}) != y_train ({len(y_train)})"
            )
        print(f"Stratified permutation across "
              f"{len(np.unique(train_scenarios))} scenario groups.")

    null_aurocs: list[float] = []
    for i in range(args.n_permutations):
        if args.stratified_permutation:
            # Shuffle labels within each scenario group
            y_perm = y_train.copy()
            for sc in np.unique(train_scenarios):
                idx = np.where(train_scenarios == sc)[0]
                if len(idx) > 1:
                    y_perm[idx] = rng.permutation(y_train[idx])
        else:
            y_perm = rng.permutation(y_train)
        clf_p = PrecursorClassifier(
            n_clusters=n_clusters, reg_c=args.reg_c, max_iter=args.max_iter,
            classifier_type=args.classifier_type,
        )
        clf_p.fit(z_train, y_perm)
        macro = _macro_auroc(clf_p, z_test, y_test)
        null_aurocs.append(macro)
        if (i + 1) % 10 == 0:
            print(f"  perm {i + 1:3d}/{args.n_permutations}: macro = {macro:.3f}")

    null_arr = np.array(null_aurocs)
    null_arr_clean = null_arr[~np.isnan(null_arr)]
    p_emp = float(np.mean(null_arr_clean >= observed_macro))
    summary = {
        "k": args.k,
        "classifier_type": args.classifier_type,
        "n_permutations": args.n_permutations,
        "n_clusters": n_clusters,
        "seed": args.seed,
        "observed_macro_auroc": observed_macro,
        "null_macro_auroc_mean": float(np.mean(null_arr_clean)),
        "null_macro_auroc_std": float(np.std(null_arr_clean)),
        "null_macro_auroc_p95": float(np.quantile(null_arr_clean, 0.95)),
        "p_value_empirical": p_emp,
        "null_distribution": null_arr.tolist(),
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nNull distribution: mean={summary['null_macro_auroc_mean']:.3f} "
          f"± {summary['null_macro_auroc_std']:.3f} "
          f"(p95={summary['null_macro_auroc_p95']:.3f})")
    print(f"Observed = {observed_macro:.3f}  →  empirical p = {p_emp:.3f}")

    # Markdown
    lines = [
        "# A3 — Permutation null distribution",
        "",
        f"k = {args.k} | classifier = {args.classifier_type} "
        f"| n_permutations = {args.n_permutations} | seed = {args.seed}",
        "",
        f"- **Observed macro-AUROC** = {observed_macro:.3f}",
        f"- **Null distribution** = {summary['null_macro_auroc_mean']:.3f} ± "
        f"{summary['null_macro_auroc_std']:.3f} (p95 = {summary['null_macro_auroc_p95']:.3f})",
        f"- **Empirical p-value** = {p_emp:.3f}",
        "",
        "**Interpretation**",
        "",
        "- p < 0.05 → the model extracts label-aligned signal from embeddings "
        "(may still be scenario signature — see A1).",
        "- p ≥ 0.05 → indistinguishable from chance, catastrophic.",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
