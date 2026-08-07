"""C-5 — Post-hoc statistical power analysis for EWAT stress tests.

Question (Plan unifié — C-5)
----------------------------
Avec n_test = 45 (ewat_v3) ou 45-57 (ewat_v4_strat), quelle est la puissance
statistique réelle des tests rapportés ? Certains clusters ont n_pos = 1 ou 2
→ AUROCs statistiquement bruités. Cette analyse quantifie :

  1. n_pos par cluster (combien de positifs dans le test set ?)
  2. Précision attendue de l'AUROC (largeur de l'IC bootstrap en fonction de
     n_pos) — formule asymptotique Hanley-McNeil ou bootstrap empirique.
  3. Power du test "AUROC > 0.5" via simulation Monte-Carlo.

Method
------
- Hanley-McNeil 1982 standard error of AUROC :
    SE(AUROC) ≈ sqrt[ AUROC(1-AUROC) + (n_pos-1)(Q1 - AUROC²) +
                     (n_neg-1)(Q2 - AUROC²) ] / sqrt(n_pos × n_neg)
  où Q1 = AUROC/(2-AUROC), Q2 = 2·AUROC²/(1+AUROC).
- Power (1 - β) : probabilité de rejeter H0 (AUROC=0.5) à un niveau α.

Sortie
------
- ``experiments/bench/power_analysis.json``
- ``experiments/bench/power_analysis.md``

Usage
-----
    python -m experiments.bench.power_analysis \\
        --dataset data/datasets/ewat_v3 \\
        --typing-dir experiments/typing \\
        [--n-min 5] [--alpha 0.05]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Hanley-McNeil standard error of AUROC
# ---------------------------------------------------------------------------

def hanley_mcneil_se(auroc: float, n_pos: int, n_neg: int) -> float:
    """Closed-form SE of AUROC under positivity-negativity asymmetry."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    A = auroc
    Q1 = A / (2.0 - A)
    Q2 = 2.0 * A * A / (1.0 + A)
    num = A * (1 - A) + (n_pos - 1) * (Q1 - A * A) + (n_neg - 1) * (Q2 - A * A)
    if num < 0:
        num = 0.0
    return float(np.sqrt(num / (n_pos * n_neg)))


def auroc_ci(auroc: float, n_pos: int, n_neg: int,
             alpha: float = 0.05) -> tuple[float, float]:
    """Normal-approximation CI for AUROC."""
    se = hanley_mcneil_se(auroc, n_pos, n_neg)
    if np.isnan(se):
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return float(max(0.0, auroc - z * se)), float(min(1.0, auroc + z * se))


def power_above_chance(auroc: float, n_pos: int, n_neg: int,
                       alpha: float = 0.05) -> float:
    """Power (1 - β) of detecting AUROC > 0.5 vs H0: AUROC = 0.5."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    se = hanley_mcneil_se(auroc, n_pos, n_neg)
    if np.isnan(se) or se < 1e-9:
        return float("nan")
    z_alpha = stats.norm.ppf(1 - alpha)
    z = (auroc - 0.5) / se - z_alpha
    return float(stats.norm.cdf(z))


def n_pos_required(auroc_target: float, n_neg: int, power_target: float = 0.8,
                   alpha: float = 0.05, max_n: int = 1000) -> int:
    """Minimum n_pos needed to detect AUROC=auroc_target with power=power_target."""
    for n_pos in range(2, max_n + 1):
        p = power_above_chance(auroc_target, n_pos, n_neg, alpha)
        if not np.isnan(p) and p >= power_target:
            return n_pos
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C-5 — Power analysis")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v3"))
    p.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"),
                   help="To extract cluster labels (defaults to ewat_v3)")
    p.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    p.add_argument("--output", type=Path, default=Path("experiments/bench"))
    p.add_argument("--n-min", type=int, default=5,
                   help="Threshold for 'reportable' clusters")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power-target", type=float, default=0.8)
    return p


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)

    # Try to load cluster_manifest first (ewat_v3 style)
    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        test_clusters = [int(v["cluster"]) for v in manifest.values()
                         if v["split"] == "test"]
        source = "cluster_manifest"
    else:
        # Fallback: read index.parquet and pretend scenario = cluster
        df = pd.read_parquet(args.dataset / "index.parquet")
        scenarios = sorted(df["scenario"].unique())
        sc_to_int = {s: i for i, s in enumerate(scenarios)}
        test_clusters = [sc_to_int[s] for s in df[df["split"] == "test"]["scenario"]]
        source = "scenario_as_cluster"

    n_pos_per = Counter(test_clusters)
    n_test = len(test_clusters)
    print(f"n_test = {n_test} | source = {source}")
    print(f"n_pos per cluster: {dict(sorted(n_pos_per.items()))}")

    # Load AUROC if available
    auroc_by_cluster: dict[int, float] = {}
    pr_results = args.precursor_dir / "results.json"
    if pr_results.exists():
        data = json.loads(pr_results.read_text())
        k_opt = data.get("k_optimal", {})
        auroc_test = data.get("auroc_test", {})
        for c_str, k in k_opt.items():
            c = int(c_str)
            v = auroc_test.get(str(k), {}).get(c_str)
            if v is not None:
                auroc_by_cluster[c] = float(v)
        print(f"Loaded AUROC for {len(auroc_by_cluster)} clusters from {pr_results}")

    # Per-cluster power analysis
    rows = []
    for c in sorted(n_pos_per):
        n_pos = n_pos_per[c]
        n_neg = n_test - n_pos
        auc = auroc_by_cluster.get(c, float("nan"))
        if not np.isnan(auc) and n_pos >= 1 and n_neg >= 1:
            se = hanley_mcneil_se(auc, n_pos, n_neg)
            lo, hi = auroc_ci(auc, n_pos, n_neg, args.alpha)
            power = power_above_chance(auc, n_pos, n_neg, args.alpha)
        else:
            se = lo = hi = power = float("nan")
        rows.append({
            "cluster": c,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": auc,
            "se_hm": se,
            "ci_lo": lo,
            "ci_hi": hi,
            "power_vs_0.5": power,
            "reportable": n_pos >= args.n_min,
        })

    # Sample-size sensitivity : how many n_pos for typical AUROC = {0.7, 0.8, 0.9}
    sensitivity = {}
    for target_auc in [0.7, 0.8, 0.9, 0.95]:
        n_req = n_pos_required(target_auc, n_neg=40, power_target=args.power_target,
                               alpha=args.alpha)
        sensitivity[f"auroc_{target_auc}"] = {
            "n_pos_required": n_req,
            "for_power": args.power_target,
            "for_n_neg": 40,
        }

    reportable = [r for r in rows if r["reportable"]]
    summary = {
        "dataset": str(args.dataset),
        "n_test": n_test,
        "n_clusters": len(rows),
        "n_clusters_reportable": len(reportable),
        "n_min_threshold": args.n_min,
        "alpha": args.alpha,
        "power_target": args.power_target,
        "per_cluster": rows,
        "sample_size_sensitivity": sensitivity,
        "mean_power_reportable": (
            float(np.mean([r["power_vs_0.5"] for r in reportable
                           if not np.isnan(r["power_vs_0.5"])]))
            if reportable else float("nan")
        ),
    }
    (args.output / "power_analysis.json").write_text(json.dumps(summary, indent=2))

    # Markdown
    lines = [
        "# C-5 — Statistical power analysis (post-hoc)",
        "",
        f"Dataset : `{args.dataset}` | n_test = {n_test} | "
        f"alpha = {args.alpha} | power target = {args.power_target} | "
        f"source = {source}",
        "",
        "## Per-cluster",
        "",
        "| cluster | n_pos | n_neg | AUROC | SE (HM) | 95% CI | Power vs 0.5 | reportable (n≥5) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rep = "✓" if r["reportable"] else "✗"
        auc_s = f"{r['auroc']:.3f}" if not np.isnan(r["auroc"]) else "NaN"
        se_s = f"{r['se_hm']:.3f}" if not np.isnan(r["se_hm"]) else "NaN"
        ci_s = (f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                if not np.isnan(r["ci_lo"]) else "—")
        pow_s = (f"{r['power_vs_0.5']:.2f}"
                 if not np.isnan(r["power_vs_0.5"]) else "NaN")
        lines.append(
            f"| C{r['cluster']} | {r['n_pos']} | {r['n_neg']} | {auc_s} | "
            f"{se_s} | {ci_s} | {pow_s} | {rep} |"
        )

    lines += [
        "",
        f"**Clusters reportables (n_pos ≥ {args.n_min}) :** "
        f"{len(reportable)} / {len(rows)}",
        f"**Power moyenne sur clusters reportables :** "
        f"{summary['mean_power_reportable']:.3f}",
        "",
        "## Sample-size sensitivity (n_pos requis pour power ≥ "
        f"{args.power_target}, n_neg = 40)",
        "",
        "| AUROC cible | n_pos requis |",
        "|---|---|",
    ]
    for k, v in sensitivity.items():
        target = k.replace("auroc_", "")
        lines.append(f"| {target} | {v['n_pos_required']} |")
    lines += [
        "",
        "## Lecture",
        "",
        f"- Avec n_test = {n_test}, {len(reportable)}/{len(rows)} clusters ont "
        f"n_pos ≥ {args.n_min} (seuil au-dessous duquel l'AUROC est statistiquement "
        "bruité — un seul mauvais classement fait basculer 1.0 → 0.5).",
        "- La SE Hanley-McNeil donne la précision attendue de l'AUROC ponctuelle ; "
        "les CI affichés ici sont *normaux* (approximation), à comparer aux CI "
        "bootstrap empiriques rapportés en `experiments/precursor/results.json`.",
        "- La sensibilité sample-size montre combien de positifs seraient "
        "nécessaires pour détecter avec puissance >80% un AUROC réel donné — "
        "utile pour dimensionner ewat_v5+.",
    ]
    (args.output / "power_analysis.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'power_analysis.md'}")
    print(f"\nReportable clusters (n_pos ≥ {args.n_min}): "
          f"{len(reportable)} / {len(rows)}")
    if reportable:
        print(f"Mean power vs chance: {summary['mean_power_reportable']:.3f}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
