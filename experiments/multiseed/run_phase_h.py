"""Phase H — Multi-seed retrain orchestrator (10 graines, audit 2026-05-26).

Reproduces the Phase G retrain (single seed 42) across 10 seeds to measure
the statistical variance of:

- H1 silhouette test
- H3 AUROC peak (val + test)
- A1 Δ(far − near) distant-window
- best_epoch siamois (over-training diagnostic, cf. C-4)
- K_optimal stability

Each seed runs the full pipeline:

  1. Encoder STGCN (use_layer_norm=True, NaN-aware scaler, 80 epochs)
  2. Siamese typing (margin=2.0, d_proj=64, mining=semi-hard, avg+cosine)
  3. Precursor (class_weight='balanced' default, BCa CI)
  4. A1 distant-window on the trained pipeline (EWAT labels)

Outputs (per seed):
  experiments/multiseed/phase_h/seed_<S>/{encoder,typing,precursor,a1}/

Final aggregate:
  experiments/multiseed/phase_h/aggregate.json
  experiments/multiseed/phase_h/results.md

Usage
-----
    python -m experiments.multiseed.run_phase_h \\
        --dataset data/datasets/ewat_v4_strat \\
        --features-root data/features/v4 \\
        [--seeds 42 123 ... 99] [--skip-existing]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SEEDS = [42, 123, 456, 789, 1337, 0, 7, 17, 31, 99]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase H — Multi-seed retrain orchestrator")
    p.add_argument("--dataset", type=Path,
                   default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path,
                   default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/multiseed/phase_h"))
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--encoder-epochs", type=int, default=80)
    p.add_argument("--typing-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip seeds whose train_summary.json already exists")
    p.add_argument("--skip-a1", action="store_true",
                   help="Skip A1 distant-window step (faster, only train metrics)")
    # Phase H-bis (audit 2026-06, T3/M8): K fixé pour la comparabilité
    # multi-graines (Phase K : K∈[9,15] instable). 0 = sélection auto legacy.
    p.add_argument("--fixed-k", type=int, default=10,
                   help="K fixe passé à typing/train.py (0 = sélection auto)")
    return p


def _run(cmd: list[str], name: str, log_path: Path) -> int:
    """Run a subprocess, stream output to log, return exit code."""
    print(f"  ▶ {name} …")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"  ✓ {name} done in {elapsed:.1f}s (rc={proc.returncode})")
    return proc.returncode


def run_one_seed(seed: int, args: argparse.Namespace) -> dict:
    """Run encoder → typing → precursor → A1 for one seed.

    Returns a dict with the seed's headline metrics.
    """
    seed_dir = args.output / f"seed_{seed}"
    enc_dir = seed_dir / "encoder"
    typ_dir = seed_dir / "typing"
    prec_dir = seed_dir / "precursor"
    a1_dir = seed_dir / "a1"
    log_dir = seed_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_path = seed_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"== seed {seed} (skip — summary exists) ==")
        return json.loads(summary_path.read_text())

    print(f"\n== seed {seed} ==")
    t0 = time.time()

    # 1. Encoder
    enc_cmd = [
        sys.executable, "-m", "experiments.encoder.train",
        "--dataset", str(args.dataset),
        "--features-root", str(args.features_root),
        "--output", str(enc_dir),
        "--epochs", str(args.encoder_epochs),
        "--batch-size", str(args.batch_size),
        "--seed", str(seed),
        # Step 5 fix 5.3: use_layer_norm=True is the default in train.py
    ]
    rc = _run(enc_cmd, "encoder", log_dir / "encoder.log")
    if rc != 0:
        print(f"  ✗ encoder failed (rc={rc})")
        return {"seed": seed, "failed": "encoder"}
    enc_ckpt = enc_dir / "checkpoints" / "best_encoder.pt"

    # 2. Siamois + clustering
    typ_cmd = [
        sys.executable, "-m", "experiments.typing.train",
        "--dataset", str(args.dataset),
        "--features-root", str(args.features_root),
        "--encoder-checkpoint", str(enc_ckpt),
        "--output", str(typ_dir),
        "--epochs", str(args.typing_epochs),
        "--batch-size", str(args.batch_size),
        "--seed", str(seed),
        # Step 6: margin=2.0, d_proj=64, mining=semi-hard, avg+cosine are defaults
        # Audit 2026-06: checkpoint-criterion=silhouette est le défaut (M6)
    ]
    if args.fixed_k:
        typ_cmd += ["--fixed-k", str(args.fixed_k)]
    rc = _run(typ_cmd, "siamois", log_dir / "siamois.log")
    if rc != 0:
        print(f"  ✗ siamois failed (rc={rc})")
        return {"seed": seed, "failed": "siamois"}
    typ_res = json.loads((typ_dir / "results.json").read_text())

    # 3. Precursor
    prec_cmd = [
        sys.executable, "-m", "experiments.precursor.train",
        "--typing-dir", str(typ_dir),
        "--encoder-dir", str(enc_dir),
        "--features-root", str(args.features_root),
        "--output", str(prec_dir),
        "--k-values", "1", "2", "3", "4", "5", "6", "8", "10", "12", "15", "20",
        "--classifier-type", "lr",   # Step 8: class_weight='balanced' is default
        "--ci-method", "bca",
        "--n-bootstrap", "500",
        "--seed", str(seed),
    ]
    rc = _run(prec_cmd, "precursor", log_dir / "precursor.log")
    if rc != 0:
        print(f"  ✗ precursor failed (rc={rc})")
        return {"seed": seed, "failed": "precursor"}
    prec_res = json.loads((prec_dir / "results.json").read_text())

    # 4. A1 distant-window
    a1_res = {}
    if not args.skip_a1:
        a1_cmd = [
            sys.executable, "-m", "experiments.h3_robustness.distant_window",
            "--typing-dir", str(typ_dir),
            "--encoder-dir", str(enc_dir),
            "--features-root", str(args.features_root),
            "--output", str(a1_dir),
            "--k", "6",
            "--n-bootstrap", "500",
            "--seed", str(seed),
        ]
        rc = _run(a1_cmd, "A1", log_dir / "a1.log")
        if rc == 0:
            a1_res = json.loads((a1_dir / "results.json").read_text())

    # 5. Headline summary
    elapsed = time.time() - t0

    n_pos_test = {int(c): int(n) for c, n in
                  (prec_res.get("n_pos_test") or {}).items()}

    def _peak_mean(table: dict, reportable_only: bool = False) -> float:
        """Pic du macro-AUROC/PR-AUC sur k ; E8 : option filtre n_pos≥5."""
        vals = []
        for _k, table_k in table.items():
            valid = [
                v for c, v in table_k.items()
                if v is not None and v == v
                and (not reportable_only or n_pos_test.get(int(c), 0) >= 5)
            ]
            if valid:
                vals.append(sum(valid) / len(valid))
        return max(vals) if vals else float("nan")

    auroc_table = prec_res.get("auroc_test", {})
    pr_table = prec_res.get("pr_auc_test", {})
    peak_auroc_mean = _peak_mean(auroc_table)
    peak_auroc_reportable = _peak_mean(auroc_table, reportable_only=True)
    peak_pr_mean = _peak_mean(pr_table)
    peak_pr_reportable = _peak_mean(pr_table, reportable_only=True)

    summary = {
        "seed": seed,
        "elapsed_s": elapsed,
        "H1": {
            "silhouette_train": typ_res.get("silhouette_train"),
            "silhouette_val": typ_res.get("silhouette_val"),
            "silhouette_test": typ_res.get("silhouette_test"),
            # typing/results.json stores K under "k_optimal" (lowercase)
            "K_optimal": typ_res.get("k_optimal") or typ_res.get("K_optimal"),
            "h1_pass": typ_res.get("h1_pass"),
            "best_epoch_siamese": typ_res.get("best_epoch"),
        },
        "H3": {
            "auroc_peak_test_mean": peak_auroc_mean,
            # E3+E8 (audit 2026-06): PR-AUC + filtre reportable n_pos≥5
            "auroc_peak_test_mean_reportable": peak_auroc_reportable,
            "pr_auc_peak_test_mean": peak_pr_mean,
            "pr_auc_peak_test_mean_reportable": peak_pr_reportable,
            "n_pos_test": n_pos_test,
            "k_optimal": prec_res.get("k_optimal"),
            "h3_pass": prec_res.get("h3_pass"),
            "h3_per_type": prec_res.get("h3_per_type"),
            "auroc_ci_test": prec_res.get("auroc_ci_test"),
        },
        "A1": {
            "delta_far_near_macro": a1_res.get("delta_far_near_macro"),
            "verdict": a1_res.get("verdict"),
            "results_by_position": {
                pos: a1_res.get("results", {}).get(pos, {}).get("macro_auroc")
                for pos in ("last", "middle", "first")
            } if a1_res else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  → summary: sil_test={summary['H1']['silhouette_test']:.3f} "
          f"K={summary['H1']['K_optimal']} "
          f"AUROC_peak={summary['H3']['auroc_peak_test_mean']:.3f} "
          f"Δ(far-near)={summary['A1'].get('delta_far_near_macro')}")
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("Phase H — multi-seed retrain")
    print(f"  Dataset: {args.dataset}")
    print(f"  Features: {args.features_root}")
    print(f"  Output:  {args.output}")
    print(f"  Seeds:   {args.seeds}")

    summaries: list[dict] = []
    for seed in args.seeds:
        s = run_one_seed(seed, args)
        summaries.append(s)
        # Save intermediate after each seed (in case orchestrator is killed)
        (args.output / "all_summaries.json").write_text(
            json.dumps(summaries, indent=2)
        )

    print("\n=== Done ===")
    print(f"All summaries: {args.output / 'all_summaries.json'}")
    print(f"Now run: python -m experiments.multiseed.aggregate_phase_h --output {args.output}")


if __name__ == "__main__":
    main()
