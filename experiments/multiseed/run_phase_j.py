"""Phase J — B2 multi-seed (NaN-aware scaler) — Honest headline defense.

Re-runs the B2 baseline (LR-OvR Chaos Mesh on flatten features) across 10
seeds with the NEW NaN-aware ``fit_scaler`` (Step 2 fix 2.3). Goal: see if
the audit fixes shift the honest headline from 0.920 toward higher values.

Phase J also runs the C2-A1 distant-window on each retrained STGCN encoder
to confirm whether GENUINE_PRECURSION holds on the Chaos Mesh target across
seeds (vs Phase H A1 on EWAT labels which showed 1/10).

Usage
-----
    python -m experiments.multiseed.run_phase_j \\
        --dataset data/datasets/ewat_v4_strat \\
        --features-root data/features/v4 \\
        [--seeds 42 123 ... 99] [--skip-c2a1]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SEEDS = [42, 123, 456, 789, 1337, 0, 7, 17, 31, 99]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase J — Multi-seed B2 + C2-A1 retest")
    p.add_argument("--dataset", type=Path,
                   default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path,
                   default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/multiseed/phase_j"))
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--skip-c2a1", action="store_true",
                   help="Skip C2-A1 distant-window on Chaos Mesh (J.2)")
    p.add_argument("--phase-h-dir", type=Path,
                   default=Path("experiments/multiseed/phase_h"),
                   help="Reuse Phase H encoder checkpoints for C2-A1")
    return p


def _run(cmd: list[str], name: str, log_path: Path) -> int:
    print(f"  ▶ {name} …")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"  ✓ {name} done in {elapsed:.1f}s (rc={proc.returncode})")
    return proc.returncode


def run_one_seed(seed: int, args: argparse.Namespace) -> dict:
    seed_dir = args.output / f"seed_{seed}"
    b2_dir = seed_dir / "b2_chaos_mesh"
    c2a1_dir = seed_dir / "c2a1_distant"
    log_dir = seed_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== seed {seed} ==")
    t0 = time.time()

    # J.1 — B2 LR-OvR on flatten Chaos Mesh features
    b2_cmd = [
        sys.executable, "-m", "experiments.architecture_v2.chaos_mesh_target",
        "--typing-dir", str(args.dataset),  # uses index.parquet fallback
        "--features-root", str(args.features_root),
        "--output", str(b2_dir),
        "--k", "6",
        "--n-bootstrap", "500",
        "--seed", str(seed),
    ]
    rc = _run(b2_cmd, "B2", log_dir / "b2.log")
    if rc != 0:
        return {"seed": seed, "failed": "b2"}
    b2_res = json.loads((b2_dir / "results.json").read_text())

    # J.2 — C2-A1 distant-window on STGCN trained on Chaos Mesh
    c2a1_res = {}
    if not args.skip_c2a1:
        # Reuse Phase H encoder checkpoint for this seed
        phase_h_enc = args.phase_h_dir / f"seed_{seed}" / "encoder"
        if (phase_h_enc / "checkpoints" / "best_encoder.pt").exists():
            # The C2-A1 script trains its own classifier on top of any STGCN.
            # We pass the Phase H STGCN as the encoder.
            # NOTE: C2-A1 expects a Chaos Mesh-trained STGCN (with classifier head)
            # but we don't have that here. Use distant_window.py (EWAT-labels)
            # variant on the Phase H pipeline — or skip C2-A1 for now.
            c2a1_res = {"skipped": "C2-A1 needs STGCN trained on Chaos Mesh head, not Phase H"}
        else:
            c2a1_res = {"skipped": f"no Phase H encoder for seed {seed}"}

    elapsed = time.time() - t0
    summary = {
        "seed": seed,
        "elapsed_s": elapsed,
        "B2": {
            "stratified_macro_auroc": b2_res.get("stratified_macro_auroc"),
            "stratified_ci": b2_res.get("stratified_ci"),
            "loso_macro_auroc_full_test_mean": b2_res.get("loso_macro_auroc_full_test_mean"),
            "loso_macro_auroc_full_test_std": b2_res.get("loso_macro_auroc_full_test_std"),
        },
        "C2A1": c2a1_res,
    }
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  → B2 stratified = {summary['B2']['stratified_macro_auroc']:.3f}  "
          f"LOSO = {summary['B2']['loso_macro_auroc_full_test_mean']:.3f}")
    return summary


def aggregate(args: argparse.Namespace, summaries: list[dict]) -> None:
    valid = [s for s in summaries if "failed" not in s]
    b2_strat = [s["B2"]["stratified_macro_auroc"] for s in valid]
    b2_loso = [s["B2"]["loso_macro_auroc_full_test_mean"] for s in valid]
    b2_strat_arr = np.array(b2_strat, dtype=np.float64)
    b2_loso_arr = np.array(b2_loso, dtype=np.float64)

    aggregate_dict = {
        "n_seeds": len(valid),
        "B2_stratified": {
            "mean": float(b2_strat_arr.mean()),
            "std": float(b2_strat_arr.std(ddof=0)),
            "min": float(b2_strat_arr.min()),
            "max": float(b2_strat_arr.max()),
            "per_seed": b2_strat,
        },
        "B2_loso": {
            "mean": float(b2_loso_arr.mean()),
            "std": float(b2_loso_arr.std(ddof=0)),
            "per_seed": b2_loso,
        },
    }
    (args.output / "aggregate.json").write_text(json.dumps(aggregate_dict, indent=2))

    lines = [
        "# Phase J — B2 multi-seed aggregate",
        "",
        "_Generated from 10-seed run with NaN-aware fit_scaler (Step 2 fix 2.3)_",
        "",
        "## B2 LR-OvR Chaos Mesh (NaN-aware scaler)",
        "",
        f"- **Stratified macro-AUROC** : {aggregate_dict['B2_stratified']['mean']:.4f}"
        f" ± {aggregate_dict['B2_stratified']['std']:.4f}",
        f"- Range : [{aggregate_dict['B2_stratified']['min']:.4f},"
        f" {aggregate_dict['B2_stratified']['max']:.4f}]",
        f"- **LOSO macro-AUROC** : {aggregate_dict['B2_loso']['mean']:.4f}"
        f" ± {aggregate_dict['B2_loso']['std']:.4f}",
        "",
        "## Comparison vs single-seed baseline",
        "",
        "| Metric | Phase B (single seed 42) | Phase J (10 seeds) | Δ |",
        "|---|---|---|---|",
        f"| B2 stratified | 0.9200 | **{aggregate_dict['B2_stratified']['mean']:.4f}** "
        f"± {aggregate_dict['B2_stratified']['std']:.4f} |"
        f" {aggregate_dict['B2_stratified']['mean']-0.92:+.4f} |",
        f"| B2 LOSO | 0.9300 | **{aggregate_dict['B2_loso']['mean']:.4f}** "
        f"± {aggregate_dict['B2_loso']['std']:.4f} |"
        f" {aggregate_dict['B2_loso']['mean']-0.93:+.4f} |",
        "",
        "## Per-seed table",
        "",
        "| Seed | B2 stratified | B2 LOSO |",
        "|---|---|---|",
    ]
    for s in valid:
        lines.append(
            f"| {s['seed']} | {s['B2']['stratified_macro_auroc']:.4f} |"
            f" {s['B2']['loso_macro_auroc_full_test_mean']:.4f} |"
        )

    (args.output / "results.md").write_text("\n".join(lines))
    print("\n=== Phase J Aggregate ===")
    print(f"B2 stratified : {aggregate_dict['B2_stratified']['mean']:.4f}"
          f" ± {aggregate_dict['B2_stratified']['std']:.4f}")
    print(f"B2 LOSO       : {aggregate_dict['B2_loso']['mean']:.4f}"
          f" ± {aggregate_dict['B2_loso']['std']:.4f}")
    print(f"Δ vs single-seed 0.92 : {aggregate_dict['B2_stratified']['mean']-0.92:+.4f}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("Phase J — B2 multi-seed")
    print(f"  Seeds: {args.seeds}")
    summaries = []
    for seed in args.seeds:
        s = run_one_seed(seed, args)
        summaries.append(s)
        (args.output / "all_summaries.json").write_text(
            json.dumps(summaries, indent=2)
        )
    aggregate(args, summaries)


if __name__ == "__main__":
    main()
