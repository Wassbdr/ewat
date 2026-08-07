"""Precursor training — Étape 3.

For each cluster type C_i and each horizon k (in timesteps), trains a binary
classifier f_i(z_pre(k)) → p̂_i ∈ [0,1] predicting whether anomaly type C_i
is about to occur.

z_pre(k) is produced by passing the pre-injection window of length k through
the trained SiameseTyper to get an L2-normalised embedding in ℝ^{d_proj}.

Pipeline
--------
1. Load SiameseTyper checkpoint from experiments/typing/.
2. Load cluster manifest from experiments/typing/cluster_artifacts/.
3. For each k in k_values:
   a. Build PrecursorDataset (pre-injection windows of length k).
   b. Embed windows → z_pre (N_ep, d_proj) via SiameseTyper.
   c. Fit PrecursorClassifier on train embeddings.
   d. Evaluate AUROC on val + test sets.
4. Find k*_i = argmax_k AUROC_i(k) per type (H3 evaluation).
5. Save best classifiers, AUROC table, results.md.

Usage
-----
    python -m experiments.precursor.train \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        [--output experiments/precursor] \\
        [--k-values 2 4 6 8 10 12] [--reg-c 1.0]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import numpy as np
import torch

import mlflow
from ewat.precursor.dataset import PrecursorDataset
from ewat.precursor.model import PrecursorClassifier, baseline_auroc, find_optimal_k
from ewat.typing.siamese import SiameseTyper
from ewat.utils.bootstrap import bootstrap_auroc_ci
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_dataset(
    typer: SiameseTyper,
    dataset: PrecursorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed all episodes in dataset; return (embeddings, labels)."""
    typer.eval()
    embeddings, labels = [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        sig = item["signal"].unsqueeze(0).to(device)   # (1, k, N, 17)
        adj = item["adjacency"].unsqueeze(0).to(device)  # (1, k, N, N, 3)
        z = typer.embed(sig, adj).cpu().numpy()          # (1, d_proj)
        embeddings.append(z[0])
        labels.append(item["cluster"])
    return np.stack(embeddings), np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precursor training (Étape 3)")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"),
                        help="Output dir from experiments/typing/train.py")
    parser.add_argument("--encoder-dir", type=Path, default=None,
                        help="Encoder dir (default: typing_dir.parent/encoder)")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
                        help="Horizon lengths in timesteps (30s each)")
    parser.add_argument("--reg-c", type=float, default=1.0,
                        help="Logistic regression regularisation (inverse C, lr only)")
    parser.add_argument("--classifier-type", default="lr",
                        choices=["lr", "lr_tuned", "rf", "svc"],
                        help="Binary classifier type per cluster type")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for AUROC CIs (0 = skip)")
    # Step 8 fix 8.4 (audit 2026-05-26): BCa is more accurate near AUROC=0/1
    # (Efron 1987). Percentile remains default for backward compat with tests.
    parser.add_argument("--ci-method", choices=["percentile", "bca"],
                        default="bca",
                        help="Bootstrap CI method. 'bca' is more accurate "
                             "near AUROC=0/1 (default since audit 2026-05-26)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> None:
    """Run the per-cluster precursor training. Pure function — no argv parsing.

    Used both by the legacy argparse ``main()`` and by the Hydra entry point
    in ``experiments.precursor.train_hydra``.
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output.mkdir(parents=True, exist_ok=True)

    # --- Load cluster manifest ---
    artifacts_dir = args.typing_dir / "cluster_artifacts"
    manifest_path = artifacts_dir / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cluster manifest not found at {manifest_path}. "
            "Run experiments/typing/train.py --eval-only first."
        )
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1
    print(f"Manifest: {len(cluster_manifest)} episodes, {n_clusters} clusters")

    # --- Load SiameseTyper ---
    encoder_dir = args.encoder_dir if args.encoder_dir is not None else (
        args.typing_dir.parent / "encoder"
    )
    ckpt_path = args.typing_dir / "checkpoints" / "best_siamese.pt"
    enc_ckpt_path = encoder_dir / "checkpoints" / "best_encoder.pt"
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
    # Step 5 fix 5.3 (audit 2026-05-26): auto-detect use_layer_norm from
    # state_dict keys via build_encoder_from_checkpoint helper.
    from ewat.encoder.factory import build_encoder_from_checkpoint
    encoder = build_encoder_from_checkpoint(enc_ckpt)
    encoder.load_state_dict(enc_ckpt["encoder_state"])
    typer_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d_proj = int(typer_ckpt.get("d_proj", 32))
    typer = SiameseTyper(encoder, d_proj=d_proj)
    typer.load_state_dict(typer_ckpt["typer_state"])
    typer = typer.to(device).eval()
    print(f"SiameseTyper loaded from {ckpt_path}")

    # Load scaler
    default_scaler = str(encoder_dir / "scaler.pkl")
    scaler_path = Path(enc_ckpt.get("scaler_path", default_scaler))

    # --- MLflow ---
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", str(args.output / "mlruns"))
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        mlflow.set_experiment("ewat_improvements")
        run = mlflow.start_run(run_name="precursor_train")
        mlflow.log_params({
            "k_values": str(args.k_values), "n_clusters": n_clusters,
            "reg_c": args.reg_c, "seed": args.seed,
            "classifier_type": getattr(args, "classifier_type", "lr"),
        })
    except Exception:
        run = None

    # --- Main loop over k values ---
    # auroc_table[k][cluster] = AUROC on test set
    auroc_table_val: dict[int, dict[int, float]] = {}
    auroc_table_test: dict[int, dict[int, float]] = {}
    # E3 (audit 2026-06): PR-AUC reporté à côté de l'AUROC — avec n_pos=3-8,
    # l'AUROC seule est optimiste (cf. M-6 : PR-AUC ∈ [0.166, 1.000] là où le
    # macro-AUROC affiche 0.920).
    pr_auc_table_val: dict[int, dict[int, float]] = {}
    pr_auc_table_test: dict[int, dict[int, float]] = {}
    classifiers: dict[int, PrecursorClassifier] = {}   # best classifier per k

    for k in args.k_values:
        print(f"\n--- k={k} steps ({k*30}s = {k*30//60}min {k*30%60}s) ---")

        ds_train = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="train")
        ds_val = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="val")
        ds_test = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="test")

        if scaler_path.exists():
            ds_train.load_scaler(scaler_path)
            ds_val.load_scaler(scaler_path)
            ds_test.load_scaler(scaler_path)

        print(f"  Embedding train ({len(ds_train)} ep) …")
        z_train, y_train = _embed_dataset(typer, ds_train, batch_size=32, device=device)

        print(f"  Embedding val ({len(ds_val)} ep) …")
        z_val, y_val = _embed_dataset(typer, ds_val, batch_size=32, device=device)

        print(f"  Embedding test ({len(ds_test)} ep) …")
        z_test, y_test = _embed_dataset(typer, ds_test, batch_size=32, device=device)

        clf = PrecursorClassifier(
            n_clusters=n_clusters, reg_c=args.reg_c, max_iter=args.max_iter,
            classifier_type=getattr(args, "classifier_type", "lr"),
        )
        clf.fit(z_train, y_train)
        classifiers[k] = clf

        auroc_val = clf.auroc_per_type(z_val, y_val)
        auroc_test = clf.auroc_per_type(z_test, y_test)
        auroc_table_val[k] = auroc_val
        auroc_table_test[k] = auroc_test
        pr_val = clf.pr_auc_per_type(z_val, y_val)
        pr_test = clf.pr_auc_per_type(z_test, y_test)
        pr_auc_table_val[k] = pr_val
        pr_auc_table_test[k] = pr_test

        mean_auc_val = float(np.nanmean(list(auroc_val.values())))
        mean_auc_test = float(np.nanmean(list(auroc_test.values())))
        mean_pr_val = float(np.nanmean(list(pr_val.values())))
        mean_pr_test = float(np.nanmean(list(pr_test.values())))
        print(f"  AUROC mean — val={mean_auc_val:.3f}  test={mean_auc_test:.3f}  "
              f"| PR-AUC mean — val={mean_pr_val:.3f}  test={mean_pr_test:.3f}")

        if run is not None:
            try:
                mlflow.log_metrics({
                    f"auroc_val_k{k}": mean_auc_val,
                    f"auroc_test_k{k}": mean_auc_test,
                    f"pr_auc_val_k{k}": mean_pr_val,
                    f"pr_auc_test_k{k}": mean_pr_test,
                }, step=k)
            except Exception:
                pass

    # --- Optimal k per cluster (selected from VAL to avoid test data snooping) ---
    k_optimal = find_optimal_k(auroc_table_val, n_clusters)
    baseline = baseline_auroc(n_clusters)

    # H3: at least one type has AUROC > 0.5 at its val-optimal k, measured on TEST
    h3_per_type = {
        c: auroc_table_test.get(k_optimal[c], {}).get(c, float("nan")) > baseline
        for c in range(n_clusters)
    }
    h3_pass = any(v for v in h3_per_type.values())
    print(f"\nH3 {'✓ PASS' if h3_pass else '✗ FAIL'}: "
          f"{sum(h3_per_type.values())} / {n_clusters} types AUROC > {baseline}")

    # --- Bootstrap CIs for AUROC at val-optimal k (on TEST set) ---
    auroc_ci: dict[int, dict] = {}
    if args.n_bootstrap > 0:
        print(f"\nBootstrap CIs (n={args.n_bootstrap}) on test AUROC at k*_val …")
        rng = np.random.default_rng(args.seed)
        for c in range(n_clusters):
            k_opt = k_optimal[c]
            clf = classifiers[k_opt]
            ds_test = PrecursorDataset(
                cluster_manifest, args.features_root, k=k_opt, split="test"
            )
            if scaler_path.exists():
                ds_test.load_scaler(scaler_path)
            z_test_c, y_test_c = _embed_dataset(typer, ds_test, batch_size=32, device=device)
            raw = clf.scores_per_type(z_test_c, y_test_c)
            if c in raw:
                y_true, y_score = raw[c]
                ci = bootstrap_auroc_ci(
                    y_true, y_score, n=args.n_bootstrap, rng=rng,
                    method=args.ci_method,   # Step 8 fix 8.4
                )
                auroc_ci[c] = ci.as_dict()
                print(f"  C{c}: AUROC = {ci}")
            else:
                auroc_ci[c] = {"estimate": float("nan"), "ci_lo": float("nan"),
                               "ci_hi": float("nan"), "alpha": 0.05,
                               "n_bootstrap": args.n_bootstrap}
                print(f"  C{c}: NaN (insufficient test samples)")
    else:
        auroc_ci = {}

    # --- Save classifiers at optimal k ---
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for c in range(n_clusters):
        k_opt = k_optimal[c]
        clf = classifiers[k_opt]
        clf.save(ckpt_dir / f"classifier_type{c}_k{k_opt}.pkl")

    # --- Save results ---
    # E8 (audit 2026-06): support test par cluster — permet d'appliquer le
    # filtre « reportable » (n_pos ≥ 5, cf. power analysis C-5) uniformément
    # dans toutes les agrégations en aval.
    n_pos_test = {str(c): int((y_test == c).sum()) for c in range(n_clusters)}
    summary = {
        "n_clusters": n_clusters,
        "k_values": args.k_values,
        "k_optimal": k_optimal,
        "n_pos_test": n_pos_test,
        "auroc_val": {str(k): v for k, v in auroc_table_val.items()},
        "auroc_test": {str(k): v for k, v in auroc_table_test.items()},
        "pr_auc_val": {str(k): v for k, v in pr_auc_table_val.items()},
        "pr_auc_test": {str(k): v for k, v in pr_auc_table_test.items()},
        "auroc_ci_test": {str(c): v for c, v in auroc_ci.items()},
        "h3_pass": h3_pass,
        "h3_per_type": {str(c): bool(v) for c, v in h3_per_type.items()},
        "baseline_auroc": baseline,
        "n_bootstrap": args.n_bootstrap,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    # Human-readable report
    has_ci = bool(auroc_ci)
    ci_header = "  95% CI              " if has_ci else ""
    lines = [
        "# Précurseurs typés — Résultats (Étape 3)\n",
        f"Clusters : {n_clusters}  |  k testé : {args.k_values}  (1 step = 30s)\n",
        f"H3 : {'✓ PASS' if h3_pass else '✗ FAIL'} "
        f"({sum(h3_per_type.values())}/{n_clusters} types AUROC > {baseline})\n",
        "## AUROC par type et par k (test set)",
        f"{'Type':<8}" + "".join(f"  k={k:>2}" for k in args.k_values)
        + f"  k*  AUROC(k*){ci_header}",
        "-" * (8 + 8 * len(args.k_values) + 14 + (22 if has_ci else 0)),
    ]
    for c in range(n_clusters):
        k_opt = k_optimal[c]
        best_auc = auroc_table_test.get(k_opt, {}).get(c, float("nan"))
        row = f"C{c:<7}"
        for k in args.k_values:
            auc = auroc_table_test.get(k, {}).get(c, float("nan"))
            row += f"  {auc:>5.3f}" if not np.isnan(auc) else "    NaN"
        row += f"  {k_opt:>3}  {best_auc:.3f}"
        if has_ci and c in auroc_ci:
            ci = auroc_ci[c]
            lo, hi = ci.get("ci_lo", float("nan")), ci.get("ci_hi", float("nan"))
            if not np.isnan(lo):
                row += f"  [{lo:.3f}, {hi:.3f}]"
            else:
                row += "  [NaN, NaN]       "
        lines.append(row)

    # E3 (audit 2026-06): tableau PR-AUC à côté de l'AUROC (n_pos faibles)
    lines += [
        "",
        "## PR-AUC par type et par k (test set) — E3 audit 2026-06",
        f"{'Type':<8}" + "".join(f"  k={k:>2}" for k in args.k_values)
        + "  PR-AUC(k*)  n_pos-aware: NaN si <2 positifs",
        "-" * (8 + 8 * len(args.k_values) + 12),
    ]
    for c in range(n_clusters):
        k_opt = k_optimal[c]
        best_pr = pr_auc_table_test.get(k_opt, {}).get(c, float("nan"))
        row = f"C{c:<7}"
        for k in args.k_values:
            pr = pr_auc_table_test.get(k, {}).get(c, float("nan"))
            row += f"  {pr:>5.3f}" if not np.isnan(pr) else "    NaN"
        row += f"  {best_pr:.3f}" if not np.isnan(best_pr) else "  NaN"
        lines.append(row)

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"Report: {args.output / 'results.md'}")

    if run is not None:
        try:
            mlflow.log_metrics({"h3_pass": float(h3_pass)})
            mlflow.end_run()
        except Exception:
            pass


def main() -> None:
    """Argparse entry point (legacy)."""
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
