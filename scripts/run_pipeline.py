"""Orchestrated EWAT pipeline runner.

Enchaîne les 4 étapes du pipeline (encodeur → typage → précurseurs → alertes)
pour une configuration donnée, avec logging MLflow de bout en bout.

Un MLflow run parent est créé pour la configuration complète ; chaque étape
logge ses métriques dans ce run.

Usage
-----
    python scripts/run_pipeline.py \\
        --dataset data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --output experiments/runs/my_run \\
        --seed 42 \\
        [--encoder-arch stgcn|gat] \\
        [--d-embed 64] \\
        [--d-proj 32|64|128] \\
        [--margin 0.5|1.0|1.5|2.0] \\
        [--n-neg-per-anchor 5|10|20] \\
        [--clustering-linkage ward|average|complete] \\
        [--clustering-metric euclidean|cosine] \\
        [--precursor-classifier lr|lr_tuned|rf|svc] \\
        [--k-values 1 2 3 4 5 6 8 10 12 15 20] \\
        [--use-layer-norm] \\
        [--epochs-encoder 100] \\
        [--epochs-typing 50] \\
        [--skip-alerts]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import mlflow

MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "file:///home/wassimbadraoui/repos/ewat/mlruns",
)
EXPERIMENT_NAME = "ewat_improvements"


def _run_step(cmd: list[str], step_name: str) -> int:
    """Run a sub-command and stream its output. Return exit code."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"CMD:  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Step {step_name!r} failed with code {result.returncode}", file=sys.stderr)
    return result.returncode


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EWAT pipeline orchestrator")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Path to dataset dir (contains split.json)")
    p.add_argument("--features-root", type=Path, required=True,
                   help="Path to feature store root (e.g. data/features/v3/)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory for this run (will create sub-dirs per step)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder-arch", default="stgcn", choices=["stgcn", "stgat"],
                   help="Encoder architecture")
    p.add_argument("--d-embed", type=int, default=64, help="Encoder embedding dimension")
    p.add_argument("--d-proj", type=int, default=32, help="Siamese projection dimension")
    p.add_argument("--margin", type=float, default=1.0, help="Contrastive loss margin")
    p.add_argument("--n-neg-per-anchor", type=int, default=5,
                   help="Negative pairs per anchor in siamese training")
    p.add_argument("--clustering-linkage", default="average",
                   choices=["ward", "average", "complete", "single"])
    p.add_argument("--clustering-metric", default="cosine",
                   choices=["euclidean", "cosine", "manhattan"])
    p.add_argument("--precursor-classifier", default="lr",
                   choices=["lr", "lr_tuned", "rf", "svc"],
                   help="Binary classifier per cluster type for precursors")
    p.add_argument("--k-values", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
                   help="Precursor horizon values in timesteps")
    p.add_argument("--use-layer-norm", action="store_true",
                   help="Enable TCN LayerNorm in encoder (disabled by default for v3 compat)")
    p.add_argument("--epochs-encoder", type=int, default=100)
    p.add_argument("--epochs-typing", type=int, default=50)
    p.add_argument("--skip-alerts", action="store_true",
                   help="Skip alert evaluation step (faster during sweeps)")
    p.add_argument("--run-name", type=str, default=None,
                   help="MLflow run name (auto-generated if not set)")
    return p


def run(args: argparse.Namespace) -> dict:
    """Run the full pipeline. Returns a dict of final H1/H3 metrics."""
    args.output.mkdir(parents=True, exist_ok=True)
    enc_dir = args.output / "encoder"
    typ_dir = args.output / "typing"
    prec_dir = args.output / "precursor"
    alert_dir = args.output / "alerts"

    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_name = args.run_name or (
            f"s{args.seed}_dp{args.d_proj}_m{args.margin}"
            f"_{args.clustering_linkage}_{args.clustering_metric}"
            f"_{args.precursor_classifier}"
        )
        parent_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "seed": args.seed,
            "encoder_arch": args.encoder_arch,
            "d_embed": args.d_embed,
            "d_proj": args.d_proj,
            "margin": args.margin,
            "n_neg_per_anchor": args.n_neg_per_anchor,
            "clustering_linkage": args.clustering_linkage,
            "clustering_metric": args.clustering_metric,
            "precursor_classifier": args.precursor_classifier,
            "k_values": str(args.k_values),
            "use_layer_norm": args.use_layer_norm,
        })
    except Exception:
        parent_run = None

    # ------------------------------------------------------------------ #
    # Step 1 — Encoder pre-training
    # ------------------------------------------------------------------ #
    enc_cmd = [
        sys.executable, "-m", "experiments.encoder.train",
        "--dataset", str(args.dataset),
        "--features-root", str(args.features_root),
        "--output", str(enc_dir),
        "--epochs", str(args.epochs_encoder),
        "--d-embed", str(args.d_embed),
        "--encoder-arch", args.encoder_arch,
        "--seed", str(args.seed),
    ]
    rc = _run_step(enc_cmd, "encoder_pretrain")
    if rc != 0:
        _finish(parent_run, {})
        return {}

    enc_summary = _load_json(enc_dir / "train_summary.json")
    enc_ckpt = enc_dir / "checkpoints" / "best_encoder.pt"

    # ------------------------------------------------------------------ #
    # Step 2 — Siamese typing + clustering
    # ------------------------------------------------------------------ #
    typing_cmd = [
        sys.executable, "-m", "experiments.typing.train",
        "--dataset", str(args.dataset),
        "--features-root", str(args.features_root),
        "--encoder-checkpoint", str(enc_ckpt),
        "--output", str(typ_dir),
        "--epochs", str(args.epochs_typing),
        "--d-proj", str(args.d_proj),
        "--margin", str(args.margin),
        "--n-neg-per-anchor", str(args.n_neg_per_anchor),
        "--clustering-linkage", args.clustering_linkage,
        "--clustering-metric", args.clustering_metric,
        "--seed", str(args.seed),
    ]
    rc = _run_step(typing_cmd, "siamese_typing")
    if rc != 0:
        _finish(parent_run, {})
        return {}

    typ_results = _load_json(typ_dir / "results.json")
    sil_test = typ_results.get("silhouette_test", float("nan"))
    h1_pass = typ_results.get("h1_pass", False)

    # ------------------------------------------------------------------ #
    # Step 3 — Precursor classifiers
    # ------------------------------------------------------------------ #
    prec_cmd = [
        sys.executable, "-m", "experiments.precursor.train",
        "--typing-dir", str(typ_dir),
        "--encoder-dir", str(enc_dir),
        "--features-root", str(args.features_root),
        "--output", str(prec_dir),
        "--classifier-type", args.precursor_classifier,
        "--k-values", *[str(k) for k in args.k_values],
        "--seed", str(args.seed),
    ]
    rc = _run_step(prec_cmd, "precursor_train")
    if rc != 0:
        _finish(parent_run, {})
        return {}

    prec_results = _load_json(prec_dir / "results.json")
    h3_pass = prec_results.get("h3_pass", False)

    # Compute mean AUROC at k* (test)
    auroc_test_at_k_star: list[float] = []
    k_optimal = prec_results.get("k_optimal", {})
    auroc_test_by_k = prec_results.get("auroc_test", {})
    for c_str, k_opt in k_optimal.items():
        auc = auroc_test_by_k.get(str(k_opt), {}).get(c_str, float("nan"))
        if not (auc != auc):  # not NaN
            auroc_test_at_k_star.append(float(auc))

    import numpy as np
    mean_auroc = float(np.nanmean(auroc_test_at_k_star)) if auroc_test_at_k_star else float("nan")

    # ------------------------------------------------------------------ #
    # Step 4 — Alert evaluation (optional)
    # ------------------------------------------------------------------ #
    if not args.skip_alerts:
        alert_cmd = [
            sys.executable, "-m", "experiments.alerts.eval",
            "--typing-dir", str(typ_dir),
            "--encoder-dir", str(enc_dir),
            "--precursor-dir", str(prec_dir),
            "--features-root", str(args.features_root),
            "--output", str(alert_dir),
        ]
        _run_step(alert_cmd, "alert_eval")

    # ------------------------------------------------------------------ #
    # Log final metrics and finish
    # ------------------------------------------------------------------ #
    final_metrics = {
        "h1_silhouette_test": sil_test,
        "h1_pass": float(h1_pass),
        "h3_mean_auroc": mean_auroc,
        "h3_pass": float(h3_pass),
        "encoder_best_val_loss": enc_summary.get("best_val_loss", float("nan")),
    }
    _finish(parent_run, final_metrics)

    (args.output / "pipeline_summary.json").write_text(json.dumps({
        **final_metrics,
        "config": vars(args),
    }, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — output: {args.output}")
    print(f"  H1 sil_test = {sil_test:.3f}  ({'PASS' if h1_pass else 'FAIL'})")
    print(f"  H3 AUROC    = {mean_auroc:.3f}  ({'PASS' if h3_pass else 'FAIL'})")
    print("=" * 60)
    return final_metrics


def _finish(mlflow_run: object, metrics: dict) -> None:
    if mlflow_run is None:
        return
    try:
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.end_run()
    except Exception:
        pass


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
