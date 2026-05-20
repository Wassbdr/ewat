"""Sweep runner — parallélise des grilles de configurations via run_pipeline.py.

Sweeps disponibles
------------------
- ``clustering``   : 3 configs × N seeds (linkage/metric)
- ``siamese``      : d_proj × margin × N seeds
- ``precursor``    : classifier_type × N seeds
- ``multiseed``    : best config × 10 seeds (run after identifying best)
- ``full``         : clustering + siamese + precursor combinés (long)

Usage
-----
    # Sweep clustering (3 configs × 3 seeds = 9 runs, ~3h)
    python scripts/run_sweep.py --sweep clustering --n-jobs 4

    # Sweep siamese hyperparams (12 configs × 3 seeds = 36 runs, ~6h)
    python scripts/run_sweep.py --sweep siamese --n-jobs 4

    # Sweep classifiers précurseurs (3 × 3 seeds = 9 runs, ~1h)
    python scripts/run_sweep.py --sweep precursor --n-jobs 4

    # Validation finale 10 seeds avec best config (interactif)
    python scripts/run_sweep.py --sweep multiseed --n-seeds 10 \\
        --best-clustering-linkage average --best-clustering-metric cosine \\
        --best-d-proj 64 --best-margin 1.0 --best-classifier lr_tuned
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_pipeline.py"

DATASET = REPO_ROOT / "data" / "datasets" / "ewat_v3"
FEATURES_ROOT = REPO_ROOT / "data" / "features" / "v3"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"

BASE_SEEDS = [42, 123, 456]
MULTI_SEEDS = [42, 123, 456, 789, 1337, 0, 7, 17, 31, 99]


@dataclass
class SweepConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------

CLUSTERING_CONFIGS = [
    SweepConfig("ward_euclidean",  {"clustering_linkage": "ward",    "clustering_metric": "euclidean"}),
    SweepConfig("average_cosine",  {"clustering_linkage": "average", "clustering_metric": "cosine"}),
    SweepConfig("complete_cosine", {"clustering_linkage": "complete","clustering_metric": "cosine"}),
]

SIAMESE_CONFIGS = [
    SweepConfig(f"dp{dp}_m{m}", {"d_proj": dp, "margin": m})
    for dp in [32, 64, 128]
    for m in [0.5, 1.0, 1.5, 2.0]
]

PRECURSOR_CONFIGS = [
    SweepConfig("lr",       {"precursor_classifier": "lr"}),
    SweepConfig("lr_tuned", {"precursor_classifier": "lr_tuned"}),
    SweepConfig("rf",       {"precursor_classifier": "rf"}),
]


def _build_cmd(
    seed: int,
    output_dir: Path,
    extra_params: dict[str, Any],
    skip_alerts: bool = True,
) -> list[str]:
    """Build the run_pipeline.py command for a given config."""
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--dataset", str(DATASET),
        "--features-root", str(FEATURES_ROOT),
        "--output", str(output_dir),
        "--seed", str(seed),
        "--skip-alerts" if skip_alerts else "",
    ]
    cmd = [c for c in cmd if c]  # remove empty strings

    for key, val in extra_params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, list):
            cmd += [flag, *[str(v) for v in val]]
        elif isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd += [flag, str(val)]
    return cmd


def _run_single(args: tuple[list[str], Path]) -> dict:
    """Worker function: run a single pipeline command."""
    cmd, output_dir = args
    summary_path = output_dir / "pipeline_summary.json"
    if summary_path.exists():
        print(f"[SWEEP] Skipping (already done): {output_dir.name}", flush=True)
        summary = json.loads(summary_path.read_text())
        summary["output_dir"] = str(output_dir)
        return summary
    print(f"[SWEEP] Starting: {output_dir.name}", flush=True)
    result = subprocess.run(cmd, capture_output=False, check=False)
    summary_path = output_dir / "pipeline_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"error": f"exit_code={result.returncode}"}
    summary["output_dir"] = str(output_dir)
    print(
        f"[SWEEP] Done: {output_dir.name}  "
        f"H1={summary.get('h1_silhouette_test', 'N/A'):.3f}  "
        f"H3={summary.get('h3_mean_auroc', 'N/A'):.3f}",
        flush=True,
    )
    return summary


def _run_sweep(
    configs: list[SweepConfig],
    seeds: list[int],
    sweep_name: str,
    n_jobs: int,
    extra_fixed: dict[str, Any] | None = None,
    skip_alerts: bool = True,
) -> list[dict]:
    """Run a grid of configs × seeds in parallel."""
    extra_fixed = extra_fixed or {}
    jobs: list[tuple[list[str], Path]] = []

    for cfg in configs:
        for seed in seeds:
            params = {**cfg.params, **extra_fixed}
            run_dir = RUNS_DIR / sweep_name / f"{cfg.name}_s{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = _build_cmd(seed, run_dir, params, skip_alerts=skip_alerts)
            jobs.append((cmd, run_dir))

    print(f"\n[SWEEP] {sweep_name}: {len(jobs)} runs, {n_jobs} parallel workers")

    if n_jobs == 1:
        results = [_run_single(j) for j in jobs]
    else:
        actual_jobs = min(n_jobs, len(jobs))
        with multiprocessing.Pool(processes=actual_jobs) as pool:
            results = pool.map(_run_single, jobs)

    return results


def _print_results_table(results: list[dict], sweep_name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"SWEEP RESULTS — {sweep_name}")
    print(f"{'=' * 70}")
    print(f"{'Run':<40}  {'H1 sil':<10}  {'H3 AUROC':<10}  {'H1':<6}  {'H3':<6}")
    print("-" * 70)

    sorted_results = sorted(
        results,
        key=lambda r: r.get("h3_mean_auroc", 0),
        reverse=True,
    )
    for r in sorted_results:
        name = Path(r.get("output_dir", "?")).name
        sil = r.get("h1_silhouette_test", float("nan"))
        auc = r.get("h3_mean_auroc", float("nan"))
        h1 = "PASS" if r.get("h1_pass", 0) > 0.5 else "FAIL"
        h3 = "PASS" if r.get("h3_pass", 0) > 0.5 else "FAIL"
        sil_s = f"{sil:.3f}" if sil == sil else "NaN"
        auc_s = f"{auc:.3f}" if auc == auc else "NaN"
        print(f"{name:<40}  {sil_s:<10}  {auc_s:<10}  {h1:<6}  {h3:<6}")

    print(f"{'=' * 70}")
    best = sorted_results[0] if sorted_results else {}
    print(f"Best: {Path(best.get('output_dir', '?')).name}")
    print(f"  H1 sil_test = {best.get('h1_silhouette_test', float('nan')):.3f}")
    print(f"  H3 AUROC    = {best.get('h3_mean_auroc', float('nan')):.3f}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EWAT sweep runner")
    p.add_argument("--sweep", required=True,
                   choices=["clustering", "siamese", "precursor", "multiseed", "full"],
                   help="Which sweep to run")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Number of parallel workers (default: 4)")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Number of seeds for validation sweeps (default: 3)")
    p.add_argument("--skip-alerts", action="store_true", default=True,
                   help="Skip alert eval (default: True, faster)")
    p.add_argument("--with-alerts", action="store_true",
                   help="Include alert evaluation in each run")

    # Best-config options for --sweep multiseed
    p.add_argument("--best-clustering-linkage", default="average")
    p.add_argument("--best-clustering-metric", default="cosine")
    p.add_argument("--best-d-proj", type=int, default=64)
    p.add_argument("--best-margin", type=float, default=1.0)
    p.add_argument("--best-n-neg", type=int, default=5)
    p.add_argument("--best-classifier", default="lr",
                   choices=["lr", "lr_tuned", "rf", "svc"])
    p.add_argument("--best-use-layer-norm", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    skip_alerts = args.skip_alerts and not args.with_alerts
    seeds = MULTI_SEEDS[:args.n_seeds] if args.sweep != "multiseed" else MULTI_SEEDS

    if args.sweep == "clustering":
        results = _run_sweep(
            CLUSTERING_CONFIGS, seeds[:args.n_seeds],
            sweep_name="sweep_clustering", n_jobs=args.n_jobs,
            skip_alerts=skip_alerts,
        )
        _print_results_table(results, "clustering sweep")

    elif args.sweep == "siamese":
        results = _run_sweep(
            SIAMESE_CONFIGS, seeds[:args.n_seeds],
            sweep_name="sweep_siamese", n_jobs=args.n_jobs,
            extra_fixed={
                "clustering_linkage": args.best_clustering_linkage,
                "clustering_metric": args.best_clustering_metric,
            },
            skip_alerts=skip_alerts,
        )
        _print_results_table(results, "siamese sweep (d_proj × margin)")

    elif args.sweep == "precursor":
        results = _run_sweep(
            PRECURSOR_CONFIGS, seeds[:args.n_seeds],
            sweep_name="sweep_precursor", n_jobs=args.n_jobs,
            extra_fixed={
                "clustering_linkage": args.best_clustering_linkage,
                "clustering_metric": args.best_clustering_metric,
                "d_proj": args.best_d_proj,
                "margin": args.best_margin,
            },
            skip_alerts=skip_alerts,
        )
        _print_results_table(results, "precursor classifier sweep")

    elif args.sweep == "multiseed":
        best_params = {
            "clustering_linkage": args.best_clustering_linkage,
            "clustering_metric": args.best_clustering_metric,
            "d_proj": args.best_d_proj,
            "margin": args.best_margin,
            "n_neg_per_anchor": args.best_n_neg,
            "precursor_classifier": args.best_classifier,
        }
        if args.best_use_layer_norm:
            best_params["use_layer_norm"] = True

        best_cfg = SweepConfig("best", best_params)
        results = _run_sweep(
            [best_cfg], MULTI_SEEDS,
            sweep_name="sweep_multiseed", n_jobs=args.n_jobs,
            skip_alerts=skip_alerts,
        )
        _print_results_table(results, "multi-seed validation (10 seeds)")

        # Aggregate stats
        import numpy as np
        sils = [r.get("h1_silhouette_test", float("nan")) for r in results]
        aucs = [r.get("h3_mean_auroc", float("nan")) for r in results]
        print(f"\nAggregate (n={len(results)} seeds):")
        print(f"  sil_test  = {np.nanmean(sils):.3f} ± {np.nanstd(sils):.3f}")
        print(f"  AUROC     = {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}")
        agg = {
            "n_seeds": len(results),
            "sil_test_mean": float(np.nanmean(sils)),
            "sil_test_std": float(np.nanstd(sils)),
            "auroc_mean": float(np.nanmean(aucs)),
            "auroc_std": float(np.nanstd(aucs)),
            "best_params": best_params,
        }
        (RUNS_DIR / "sweep_multiseed" / "aggregate.json").write_text(
            json.dumps(agg, indent=2)
        )

    elif args.sweep == "full":
        # Sequential: clustering → siamese → precursor (using best of each)
        print("\n[FULL SWEEP] Phase 1/3: clustering")
        c_results = _run_sweep(
            CLUSTERING_CONFIGS, seeds[:args.n_seeds],
            sweep_name="sweep_full_clustering", n_jobs=args.n_jobs,
            skip_alerts=True,
        )
        _print_results_table(c_results, "clustering")
        best_c = max(c_results, key=lambda r: r.get("h3_mean_auroc", 0))
        best_cfg_data = json.loads(
            (Path(best_c["output_dir"]) / "pipeline_summary.json").read_text()
        ).get("config", {})
        best_linkage = best_cfg_data.get("clustering_linkage", "average")
        best_metric  = best_cfg_data.get("clustering_metric", "cosine")

        print(f"\n[FULL SWEEP] Best clustering: {best_linkage}/{best_metric}")
        print("\n[FULL SWEEP] Phase 2/3: siamese")
        s_results = _run_sweep(
            SIAMESE_CONFIGS, seeds[:args.n_seeds],
            sweep_name="sweep_full_siamese", n_jobs=args.n_jobs,
            extra_fixed={"clustering_linkage": best_linkage, "clustering_metric": best_metric},
            skip_alerts=True,
        )
        _print_results_table(s_results, "siamese")
        best_s = max(s_results, key=lambda r: r.get("h3_mean_auroc", 0))
        best_s_cfg = json.loads(
            (Path(best_s["output_dir"]) / "pipeline_summary.json").read_text()
        ).get("config", {})

        print("\n[FULL SWEEP] Phase 3/3: precursor")
        p_results = _run_sweep(
            PRECURSOR_CONFIGS, seeds[:args.n_seeds],
            sweep_name="sweep_full_precursor", n_jobs=args.n_jobs,
            extra_fixed={
                "clustering_linkage": best_linkage,
                "clustering_metric": best_metric,
                "d_proj": best_s_cfg.get("d_proj", 64),
                "margin": best_s_cfg.get("margin", 1.0),
            },
            skip_alerts=skip_alerts,
        )
        _print_results_table(p_results, "precursor")


if __name__ == "__main__":
    main()
