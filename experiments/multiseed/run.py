"""Évaluation multi-graines — robustesse de H1 et H3.

Lance le pipeline complet (encodeur → typage → précurseurs) sur N graines
différentes et rapporte mean ± std sur les métriques clés.

NB : chaque graine nécessite un réentraînement complet (~45 min CPU).
Prévoir une nuit pour N=5 graines.

Usage
-----
    python -m experiments.multiseed.run \\
        --dataset data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --encoder-dir experiments/encoder \\
        --output experiments/multiseed \\
        [--seeds 42 123 456 789 1337] \\
        [--epochs-encoder 100] [--epochs-typer 50]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ewat.utils.bootstrap import bootstrap_mean_ci


def _run(cmd: list[str]) -> int:
    """Run a subprocess command, streaming output. Returns return code."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _load_result(path: Path, key: str, default: float = float("nan")) -> float:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
        return float(data.get(key, default))
    except Exception:
        return default


def _load_best_auroc(precursor_results: Path, n_clusters: int) -> float:
    """Mean AUROC at k* (test set, ignoring NaN) from precursor results.json."""
    if not precursor_results.exists():
        return float("nan")
    data = json.loads(precursor_results.read_text())
    auroc_test = data.get("auroc_test", {})
    k_optimal = data.get("k_optimal", {})
    values = []
    for c in range(n_clusters):
        k = str(k_optimal.get(c, k_optimal.get(str(c), None)))
        if k is None:
            continue
        auc = auroc_test.get(k, {}).get(c, auroc_test.get(k, {}).get(str(c), float("nan")))
        if not np.isnan(float(auc)):
            values.append(float(auc))
    return float(np.nanmean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed evaluation for H1 and H3")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v3"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"),
                        help="Dir with scaler.pkl and checkpoints/ from the reference encoder")
    parser.add_argument("--output", type=Path, default=Path("experiments/multiseed"))
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 123, 456, 789, 1337, 2024, 31415, 27182, 161803, 6022],
        help="Default raised to 10 seeds for tighter aggregate CIs.",
    )
    parser.add_argument("--epochs-encoder", type=int, default=100)
    parser.add_argument("--epochs-typer", type=int, default=50)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument(
        "--n-bootstrap-sub", type=int, default=200,
        help="Bootstrap iterations forwarded to typing/precursor sub-runs "
             "(0 disables CIs in sub-runs).",
    )
    parser.add_argument(
        "--n-bootstrap-agg", type=int, default=2000,
        help="Bootstrap iterations for the cross-seed aggregate CIs.",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=20260506,
        help="RNG seed for the cross-seed bootstrap (reproducibility).",
    )
    parser.add_argument("--skip-encoder", action="store_true",
                        help="Skip encoder retraining (reuse existing checkpoints per seed)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict] = []

    for seed in args.seeds:
        seed_dir = args.output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        enc_out = seed_dir / "encoder"
        typing_out = seed_dir / "typing"
        prec_out = seed_dir / "precursor"

        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # --- Encoder ---
        enc_ckpt = enc_out / "checkpoints" / "best_encoder.pt"
        if not args.skip_encoder or not enc_ckpt.exists():
            rc = _run([
                sys.executable, "-m", "experiments.encoder.train",
                "--dataset", str(args.dataset),
                "--features-root", str(args.features_root),
                "--output", str(enc_out),
                "--epochs", str(args.epochs_encoder),
                "--seed", str(seed),
            ])
            if rc != 0:
                print(f"Encoder training FAILED for seed {seed}; skipping.")
                continue
        else:
            print("Encoder checkpoint exists, skipping retraining.")

        # --- Typer ---
        typing_ckpt = typing_out / "checkpoints" / "best_siamese.pt"
        if not typing_ckpt.exists():
            rc = _run([
                sys.executable, "-m", "experiments.typing.train",
                "--dataset", str(args.dataset),
                "--features-root", str(args.features_root),
                "--encoder-checkpoint", str(enc_ckpt),
                "--output", str(typing_out),
                "--epochs", str(args.epochs_typer),
                "--seed", str(seed),
                "--n-bootstrap", str(args.n_bootstrap_sub),
            ])
            if rc != 0:
                print(f"Typer training FAILED for seed {seed}; skipping.")
                continue
        else:
            print("Typer checkpoint exists, skipping retraining.")

        # --- Precursors ---
        prec_json = prec_out / "results.json"
        if not prec_json.exists():
            rc = _run([
                sys.executable, "-m", "experiments.precursor.train",
                "--typing-dir", str(typing_out),
                "--features-root", str(args.features_root),
                "--output", str(prec_out),
                "--k-values", *[str(k) for k in args.k_values],
                "--seed", str(seed),
                "--n-bootstrap", str(args.n_bootstrap_sub),
            ])
            if rc != 0:
                print(f"Precursor training FAILED for seed {seed}; skipping.")
                continue
        else:
            print("Precursor results exist, skipping retraining.")

        # --- Collect metrics ---
        typing_json = typing_out / "results.json"
        sil_test = _load_result(typing_json, "silhouette_test")
        sil_val = _load_result(typing_json, "silhouette_val")
        k_opt = _load_result(typing_json, "k_optimal")

        prec_data = json.loads(prec_json.read_text()) if prec_json.exists() else {}
        n_clusters = int(prec_data.get("n_clusters", 10))
        auroc_mean = _load_best_auroc(prec_json, n_clusters)
        h3_pass = bool(prec_data.get("h3_pass", False))
        n_pass = sum(1 for v in prec_data.get("h3_per_type", {}).values() if v)

        result = {
            "seed": seed,
            "silhouette_val": sil_val,
            "silhouette_test": sil_test,
            "k_optimal": k_opt,
            "auroc_mean_test": auroc_mean,
            "h3_pass": h3_pass,
            "n_clusters_pass": n_pass,
        }
        seed_results.append(result)
        print(f"\nSeed {seed} → sil_test={sil_test:.3f}  AUROC_mean={auroc_mean:.3f}  "
              f"H3={'PASS' if h3_pass else 'FAIL'} ({n_pass}/{n_clusters})")

    if not seed_results:
        print("No successful runs. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    sil_tests = [r["silhouette_test"] for r in seed_results if not np.isnan(r["silhouette_test"])]
    sil_vals = [r["silhouette_val"] for r in seed_results if not np.isnan(r["silhouette_val"])]
    aurocs = [r["auroc_mean_test"] for r in seed_results if not np.isnan(r["auroc_mean_test"])]

    rng = np.random.default_rng(args.bootstrap_seed)

    def _agg_block(values: list[float]) -> dict:
        if not values:
            return {
                "mean": float("nan"), "std": float("nan"),
                "se": float("nan"), "min": float("nan"), "max": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_bootstrap": args.n_bootstrap_agg, "values": [],
            }
        arr = np.asarray(values, dtype=float)
        ci = bootstrap_mean_ci(
            arr, n=args.n_bootstrap_agg, rng=rng, method="bca",
        )
        # Standard error of the mean across seeds (n − 1 normalisation).
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1) if len(arr) > 1 else 0.0), 4),
            "se": round(se, 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "ci_lo": round(ci.lo, 4),
            "ci_hi": round(ci.hi, 4),
            "n_bootstrap": args.n_bootstrap_agg,
            "values": [round(float(v), 4) for v in values],
        }

    summary = {
        "seeds": [r["seed"] for r in seed_results],
        "n_runs": len(seed_results),
        "n_bootstrap_sub": args.n_bootstrap_sub,
        "n_bootstrap_agg": args.n_bootstrap_agg,
        "bootstrap_seed": args.bootstrap_seed,
        "silhouette_test": _agg_block(sil_tests),
        "silhouette_val": _agg_block(sil_vals),
        "auroc_mean_test": _agg_block(aurocs),
        "h1_pass_all": all(s >= 0.3 for s in sil_tests),
        "h3_pass_all": all(r["h3_pass"] for r in seed_results),
        "per_seed": seed_results,
    }

    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    # Report
    lines = [
        "# Évaluation multi-graines\n",
        f"Graines : {summary['seeds']}  |  N runs réussis : {summary['n_runs']}\n",
        "## Résultats agrégés\n",
        "Bootstrap BCa sur la moyenne inter-graines "
        f"(n={summary['n_bootstrap_agg']}, seed={summary['bootstrap_seed']}).\n",
        "| Métrique | Moyenne | Std | SE | Min | Max | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in [("silhouette_test", "Silhouette test (H1)"),
                        ("silhouette_val", "Silhouette val"),
                        ("auroc_mean_test", "AUROC moyen test (H3)")]:
        d = summary[key]
        lines.append(
            f"| {label} | {d['mean']:.4f} | {d['std']:.4f} | {d['se']:.4f} "
            f"| {d.get('min', float('nan')):.4f} | {d.get('max', float('nan')):.4f} "
            f"| [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] |"
        )
    lines += [
        "",
        f"H1 PASS toutes graines : {'✓' if summary['h1_pass_all'] else '✗'}",
        f"H3 PASS toutes graines : {'✓' if summary['h3_pass_all'] else '✗'}",
        "",
        "## Détail par graine",
        f"{'Graine':<8}  {'sil_val':>8}  {'sil_test':>9}  {'AUROC':>7}  {'H3'}",
        "-" * 48,
    ]
    for r in seed_results:
        lines.append(
            f"{r['seed']:<8}  {r['silhouette_val']:>8.4f}  "
            f"{r['silhouette_test']:>9.4f}  {r['auroc_mean_test']:>7.4f}  "
            f"{'PASS' if r['h3_pass'] else 'FAIL'}"
        )

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")

    sil_block = summary["silhouette_test"]
    auc_block = summary["auroc_mean_test"]
    print("\n=== RÉSUMÉ ===")
    print(
        f"Silhouette test : {sil_block['mean']:.4f} ± {sil_block['std']:.4f}  "
        f"(SE={sil_block['se']:.4f}, 95% CI=[{sil_block['ci_lo']:.4f}, {sil_block['ci_hi']:.4f}])"
    )
    print(
        f"AUROC moyen test: {auc_block['mean']:.4f} ± {auc_block['std']:.4f}  "
        f"(SE={auc_block['se']:.4f}, 95% CI=[{auc_block['ci_lo']:.4f}, {auc_block['ci_hi']:.4f}])"
    )


if __name__ == "__main__":
    main()
