"""H2b strict — Analyse de sensibilité du critère overlap et timing.

Post-traitement pur de per_episode.csv (produit par eval.py).
Ne relance PAS le streaming des épisodes.

Analyses
--------
1. Sensibilité du seuil d'overlap (0.3, 0.5, 0.7, 0.9)
   → Combien de clusters passent à chaque niveau ?

2. Test de Fisher exact C8 vs clusters drift pur (C5, C6, C9)
   → C8 (θ_{drift∩anomaly}) a-t-il un overlap significativement supérieur ?

3. Timing analysis par cluster
   → Distributions de (first_alert_t − injection_t) et
      (first_drift_flag_t − injection_t)
   → Drift flag précède-t-il l'alerte précurseur ?

Usage
-----
    python -m experiments.h2_overlap.eval_strict \\
        --input experiments/h2_overlap/per_episode.csv \\
        --output experiments/h2_overlap \\
        [--n-bootstrap 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from ewat.utils.bootstrap import bootstrap_proportion_ci

# Scenarios classified as drift bénin (no anomaly injection)
DRIFT_SCENARIOS: frozenset[str] = frozenset({
    "rolling_deploy", "drift_scale_up", "drift_config_change",
})
# θ_{drift∩anomaly} : faulty deploy overlap scenario
OVERLAP_SCENARIO = "faulty_deploy_overlap"
# θ_{drift∩anomaly} cluster id — derived dynamically in main()
OVERLAP_TARGET_CLUSTER = 8  # fallback for report labels only


def _derive_drift_pure_clusters(df: pd.DataFrame) -> list[int]:
    """Clusters whose dominant scenario is a known drift bénin — seed-agnostic."""
    dominant = df.groupby("cluster_gt")["scenario"].agg(lambda x: x.mode().iloc[0])
    return sorted(int(c) for c, s in dominant.items() if s in DRIFT_SCENARIOS)


def _derive_overlap_cluster(df: pd.DataFrame) -> int | None:
    """Cluster whose dominant scenario is the faulty_deploy_overlap."""
    dominant = df.groupby("cluster_gt")["scenario"].agg(lambda x: x.mode().iloc[0])
    matches = [int(c) for c, s in dominant.items() if s == OVERLAP_SCENARIO]
    return matches[0] if matches else None


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("first_drift_flag_t", "first_alert_t", "injection_t"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _cluster_stats(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> list[dict]:
    stats = []
    for c in sorted(df["cluster_gt"].unique()):
        rows = df[df["cluster_gt"] == c]
        n = len(rows)
        n_overlap = int(rows["is_overlap"].sum())
        dominant_scenario = rows["scenario"].mode().iloc[0]
        pct_drift = float(rows["has_drift_flag"].mean())
        pct_alert = float(rows["has_alert"].mean())
        ci = bootstrap_proportion_ci(n_overlap, n, n=n_bootstrap, rng=rng)
        stats.append({
            "cluster": int(c),
            "n": n,
            "dominant_scenario": dominant_scenario,
            "pct_drift_flag": round(pct_drift, 3),
            "pct_alert": round(pct_alert, 3),
            "n_overlap": n_overlap,
            "pct_overlap": round(n_overlap / n, 3),
            "ci_lo": round(ci.lo, 3),
            "ci_hi": round(ci.hi, 3),
        })
    return stats


def _sensitivity_table(stats: list[dict], thresholds: list[float]) -> list[dict]:
    rows = []
    for thr in thresholds:
        passing = [s for s in stats if s["pct_overlap"] > thr]
        rows.append({
            "threshold": thr,
            "n_clusters_passing": len(passing),
            "clusters_passing": [s["cluster"] for s in passing],
        })
    return rows


def _fisher_c8_vs_drift(
    df: pd.DataFrame,
    overlap_cluster: int,
    drift_pure_clusters: list[int],
) -> dict:
    """Fisher exact test: overlap cluster vs pooled drift-pur clusters."""
    c8 = df[df["cluster_gt"] == overlap_cluster]
    drift_pur = df[df["cluster_gt"].isin(drift_pure_clusters)]

    n_c8, n_c8_overlap = len(c8), int(c8["is_overlap"].sum())
    n_dp, n_dp_overlap = len(drift_pur), int(drift_pur["is_overlap"].sum())

    if n_c8 == 0 or n_dp == 0:
        return {"error": "insufficient data"}

    # 2×2 contingency: [overlap, no-overlap] × [C8, drift-pur]
    table = [
        [n_c8_overlap, n_c8 - n_c8_overlap],
        [n_dp_overlap, n_dp - n_dp_overlap],
    ]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    return {
        "c8_n": n_c8,
        "c8_n_overlap": n_c8_overlap,
        "c8_pct_overlap": round(n_c8_overlap / n_c8, 3),
        "drift_pur_clusters": drift_pure_clusters,
        "drift_pur_n": n_dp,
        "drift_pur_n_overlap": n_dp_overlap,
        "drift_pur_pct_overlap": round(n_dp_overlap / n_dp, 3),
        "odds_ratio": round(float(odds_ratio), 3),
        "p_value_one_sided": round(float(p_value), 4),
        "significant_alpha05": bool(p_value < 0.05),
        "contingency_table": table,
    }


def _timing_stats(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Per-cluster timing: alert vs injection and drift-flag vs injection."""
    result: dict[str, list[dict]] = {"per_cluster": []}
    for c in sorted(df["cluster_gt"].unique()):
        rows = df[df["cluster_gt"] == c].copy()
        # lead_alert: injection_t - first_alert_t  (positive = alerte avant injection)
        rows["lead_alert"] = rows["injection_t"] - rows["first_alert_t"]
        # lead_drift: injection_t - first_drift_flag_t
        rows["lead_drift"] = rows["injection_t"] - rows["first_drift_flag_t"]
        # timing_gap: first_alert_t - first_drift_flag_t (positive = alert après drift)
        rows["timing_gap"] = rows["first_alert_t"] - rows["first_drift_flag_t"]

        valid_lead = rows["lead_alert"].dropna()
        valid_gap = rows["timing_gap"].dropna()

        result["per_cluster"].append({
            "cluster": int(c),
            "dominant_scenario": rows["scenario"].mode().iloc[0],
            "n": len(rows),
            "lead_alert_mean": round(float(valid_lead.mean()), 2) if len(valid_lead) > 0 else None,
            "lead_alert_median": round(float(valid_lead.median()), 2) if len(valid_lead) > 0 else None,
            "lead_alert_pct_positive": round(float((valid_lead > 0).mean()), 3) if len(valid_lead) > 0 else None,
            "timing_gap_mean": round(float(valid_gap.mean()), 2) if len(valid_gap) > 0 else None,
            "timing_gap_median": round(float(valid_gap.median()), 2) if len(valid_gap) > 0 else None,
            "alert_before_drift_pct": round(float((valid_gap < 0).mean()), 3) if len(valid_gap) > 0 else None,
        })
    return result


def _write_report(
    out: Path,
    stats: list[dict],
    sensitivity: list[dict],
    fisher: dict,
    timing: dict,
) -> None:
    lines = [
        "# H2b strict — Analyse de sensibilité et timing\n",
        "Post-traitement de `per_episode.csv`. Aucun streaming d'épisodes.",
        "",
        "---",
        "",
        "## 1. Sensibilité du seuil d'overlap",
        "",
        f"{'Seuil':>7}  {'N pass':>6}  Clusters passing",
        "-" * 50,
    ]
    for row in sensitivity:
        lines.append(
            f"{row['threshold']:>7.1f}  {row['n_clusters_passing']:>6}  "
            f"{row['clusters_passing']}"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Fisher exact : C8 vs clusters drift pur (C5, C6, C9)",
        "",
    ]
    if "error" in fisher:
        lines.append(f"Erreur : {fisher['error']}")
    else:
        sig = "✓ SIGNIFICATIF" if fisher["significant_alpha05"] else "✗ NON SIGNIFICATIF"
        lines += [
            f"- C8 (faulty_deploy_overlap) : {fisher['c8_n_overlap']}/{fisher['c8_n']} "
            f"= {fisher['c8_pct_overlap']:.1%} overlap",
            f"- Drift pur (C5+C6+C9) poolés : {fisher['drift_pur_n_overlap']}/{fisher['drift_pur_n']} "
            f"= {fisher['drift_pur_pct_overlap']:.1%} overlap",
            f"- Odds ratio : {fisher['odds_ratio']:.3f}",
            f"- p-value (unilatéral, C8 > drift pur) : **{fisher['p_value_one_sided']:.4f}**",
            f"- **{sig}** (α = 0.05)",
            "",
            "**Interprétation** : " + (
                "C8 a un overlap significativement supérieur aux drifts purs — "
                "le régime θ_{drift∩anomaly} est statistiquement distinct."
                if fisher["significant_alpha05"] else
                "C8 n'a pas un overlap significativement supérieur aux drifts purs — "
                "H2b PASS reste trivial (le DriftDetector déclenche sur presque tout)."
            ),
        ]

    lines += [
        "",
        "---",
        "",
        "## 3. Timing analysis par cluster",
        "",
        "lead_alert = injection_t − first_alert_t (positif = alerte AVANT injection)",
        "timing_gap = first_alert_t − first_drift_flag_t (positif = alerte après drift flag)",
        "",
        f"{'C':>2}  {'Scénario':<26}  {'lead_alert':>10}  {'%avant_inj':>10}  "
        f"{'gap_med':>7}  {'%alert<drift':>12}",
        "-" * 75,
    ]
    for t in timing["per_cluster"]:
        lead = f"{t['lead_alert_median']:.1f}" if t["lead_alert_median"] is not None else "N/A"
        pct_pos = f"{t['lead_alert_pct_positive']:.0%}" if t["lead_alert_pct_positive"] is not None else "N/A"
        gap = f"{t['timing_gap_median']:.1f}" if t["timing_gap_median"] is not None else "N/A"
        pct_before = f"{t['alert_before_drift_pct']:.0%}" if t["alert_before_drift_pct"] is not None else "N/A"
        lines.append(
            f"C{t['cluster']:<1}  {t['dominant_scenario'][:26]:<26}  "
            f"{lead:>10}  {pct_pos:>10}  {gap:>7}  {pct_before:>12}"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Tableau récapitulatif des clusters (sensibilité 0.5)",
        "",
        f"{'C':>2}  {'N':>4}  {'Scénario dominant':<24}  "
        f"{'%drift':>7}  {'%alert':>7}  {'%overlap':>9}  {'CI 95%':>20}",
        "-" * 80,
    ]
    for s in stats:
        pass50 = "✓" if s["pct_overlap"] > 0.5 else " "
        lines.append(
            f"C{s['cluster']:<1}  {s['n']:>4}  {s['dominant_scenario'][:24]:<24}  "
            f"{s['pct_drift_flag']:>7.3f}  {s['pct_alert']:>7.3f}  "
            f"{s['pct_overlap']:>9.3f}  [{s['ci_lo']:.3f},{s['ci_hi']:.3f}]  {pass50}"
        )

    (out / "results_strict.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="H2b strict — sensibilité + Fisher + timing")
    parser.add_argument(
        "--input", type=Path,
        default=Path("experiments/h2_overlap/per_episode.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/h2_overlap"))
    parser.add_argument(
        "--overlap-thresholds", type=float, nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = _load_csv(args.input)
    print(f"Loaded {len(df)} episodes, {df['cluster_gt'].nunique()} clusters")

    drift_pure_clusters = _derive_drift_pure_clusters(df)
    overlap_cluster = _derive_overlap_cluster(df) or OVERLAP_TARGET_CLUSTER
    print(f"Drift-pure clusters (derived): {drift_pure_clusters}")
    print(f"Overlap target cluster (derived): C{overlap_cluster}")

    stats = _cluster_stats(df, args.n_bootstrap, rng)
    sensitivity = _sensitivity_table(stats, args.overlap_thresholds)
    fisher = _fisher_c8_vs_drift(df, overlap_cluster, drift_pure_clusters)
    timing = _timing_stats(df)

    # Save JSON
    results = {
        "sensitivity": sensitivity,
        "fisher_c8_vs_drift_pur": fisher,
        "timing": timing,
        "cluster_stats": stats,
    }
    (args.output / "results_strict.json").write_text(json.dumps(results, indent=2))

    # Save sensitivity CSV
    pd.DataFrame(sensitivity).to_csv(
        args.output / "sensitivity_table.csv", index=False
    )

    _write_report(args.output, stats, sensitivity, fisher, timing)

    # Print summary
    print("\n=== Sensibilité du seuil ===")
    for row in sensitivity:
        print(f"  > {row['threshold']:.1f} : {row['n_clusters_passing']}/10 clusters pass "
              f"  {row['clusters_passing']}")

    print("\n=== Fisher exact C8 vs drift pur ===")
    if "error" not in fisher:
        sig = "✓ SIGNIFICATIF" if fisher["significant_alpha05"] else "✗ NON SIGNIFICATIF"
        print(f"  C8: {fisher['c8_pct_overlap']:.1%}  vs  "
              f"drift pur: {fisher['drift_pur_pct_overlap']:.1%}  "
              f"OR={fisher['odds_ratio']:.3f}  p={fisher['p_value_one_sided']:.4f}  {sig}")

    print(f"\nOutputs: {args.output}/results_strict.md + sensitivity_table.csv + results_strict.json")


if __name__ == "__main__":
    main()
