"""A4 — H3 AUROC filtered by minimum test positives (n_pos ≥ 5).

Question
--------
Some H3 clusters report AUROC = 1.0 with only n_pos = 1, 2, or 3 positives in
the test set. These are statistically meaningless: a single positive misclassified
flips AUROC from 1.0 to 0.5, and with n_pos < 4 the 95% bootstrap CI is wider
than the metric is useful.

Method
------
Reload the per-cluster AUROC table and bootstrap CIs from
``experiments/precursor/results.json`` (or any directory passed via
``--precursor-dir``). Filter to clusters with n_pos_test ≥ N_MIN. Recompute the
H3 headline and the mean AUROC over the *reportable* clusters.

Usage
-----
    python -m experiments.h3_robustness.filter_npos \\
        --precursor-dir experiments/precursor \\
        [--n-min 5] \\
        [--output experiments/h3_robustness/filter_npos]

The "n_pos_test" per cluster is inferred from the cluster_manifest splits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A4 — H3 filtered by n_pos≥N")
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/h3_robustness/filter_npos"))
    parser.add_argument("--n-min", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)

    precursor_results = json.loads((args.precursor_dir / "results.json").read_text())
    manifest = json.loads(
        (args.typing_dir / "cluster_artifacts" / "cluster_manifest.json").read_text()
    )

    test_clusters = [int(v["cluster"]) for v in manifest.values() if v["split"] == "test"]
    n_pos_per_cluster = Counter(test_clusters)

    n_clusters = int(precursor_results["n_clusters"])
    k_optimal = precursor_results["k_optimal"]
    auroc_test = precursor_results["auroc_test"]
    auroc_ci = precursor_results.get("auroc_ci_test", {})

    rows = []
    survivors = []
    for c in range(n_clusters):
        npos = int(n_pos_per_cluster.get(c, 0))
        k_opt = k_optimal.get(str(c), k_optimal.get(c))
        if k_opt is None:
            best_auc = float("nan")
        else:
            best_auc = float(auroc_test.get(str(k_opt), {}).get(str(c), float("nan")))
        ci = auroc_ci.get(str(c), {})
        ci_lo = ci.get("ci_lo", float("nan"))
        ci_hi = ci.get("ci_hi", float("nan"))
        rows.append({
            "cluster": c,
            "n_pos_test": npos,
            "k_opt": k_opt,
            "auroc_test": best_auc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "reportable": npos >= args.n_min and not np.isnan(best_auc),
        })
        if rows[-1]["reportable"]:
            survivors.append(best_auc)

    summary = {
        "n_min": args.n_min,
        "n_clusters_total": n_clusters,
        "n_clusters_reportable": len(survivors),
        "mean_auroc_reportable": float(np.mean(survivors)) if survivors else float("nan"),
        "std_auroc_reportable": float(np.std(survivors)) if survivors else float("nan"),
        "per_cluster": rows,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    # Markdown
    lines = [
        f"# A4 — H3 filtered by n_pos_test ≥ {args.n_min}",
        "",
        f"Source: `{args.precursor_dir}/results.json`",
        "",
        f"- Clusters total: {n_clusters}",
        f"- Clusters reportable (n_pos ≥ {args.n_min}): "
        f"{summary['n_clusters_reportable']} / {n_clusters}",
        f"- **Mean AUROC over reportable clusters = "
        f"{summary['mean_auroc_reportable']:.3f} ± "
        f"{summary['std_auroc_reportable']:.3f}**",
        "",
        f"Clusters with n_pos < {args.n_min} are statistically meaningless: a "
        "single misclassification flips AUROC dramatically, and the bootstrap CI "
        "is degenerate.",
        "",
        "## Per cluster",
        "",
        "| cluster | n_pos test | k* | AUROC test | 95% CI | reportable? |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rep = "✓" if r["reportable"] else "✗ (n_pos<5)"
        ci_str = (f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                  if not np.isnan(r["ci_lo"]) else "—")
        auc_str = f"{r['auroc_test']:.3f}" if not np.isnan(r["auroc_test"]) else "NaN"
        lines.append(
            f"| C{r['cluster']} | {r['n_pos_test']} | {r['k_opt']} | "
            f"{auc_str} | {ci_str} | {rep} |"
        )
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")
    print(f"Reportable clusters (n_pos ≥ {args.n_min}): "
          f"{summary['n_clusters_reportable']} / {n_clusters}")
    print(f"Mean AUROC over reportable = "
          f"{summary['mean_auroc_reportable']:.3f} ± "
          f"{summary['std_auroc_reportable']:.3f}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
