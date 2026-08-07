"""Ontologie service-to-service — TE-KSG inter-services au sein des épisodes.

Contrairement à build.py (TE cluster→cluster sur trajectoires moyennes, biais
écologique), ce script calcule la TE entre les séries temporelles de chaque
paire de services *au sein de chaque épisode* du cluster, puis agrège par
moyenne hiérarchique (TE moyennée sur les épisodes, pas trajectoire moyenne
avant TE).

Question scientifique : lors d'une panne de type C_i, quel service propage
sa défaillance vers quel autre service ?

Outputs
-------
  experiments/ontology/service_causal.json  — résultats complets par cluster
  experiments/ontology/service_causal.md    — rapport lisible

Usage
-----
    python -m experiments.ontology.build_service \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        [--output experiments/ontology] \\
        [--regime injection] \\
        [--n-permutations 100] \\
        [--te-method univariate_sum] \\
        [--min-support 5] \\
        [--min-series-length 10] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ewat.ontology.causal import compute_service_causal_relations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Service-level causal ontology via TE-KSG (hierarchical estimator)"
    )
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/ontology"))
    parser.add_argument(
        "--regime", type=str, default=None,
        choices=["normal", "injection", "recovery", None],
        help="Restrict analysis to steps of this regime (default: full episode)",
    )
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument(
        "--te-method", choices=["univariate_sum", "multivariate"],
        default="univariate_sum",
        help="univariate_sum: fast, slight bias. multivariate: KSG-1 in ℝ^17, slower.",
    )
    parser.add_argument("--min-support", type=int, default=5,
                        help="Min episodes per cluster to attempt TE")
    parser.add_argument("--min-series-length", type=int, default=10,
                        help="Min timesteps after regime filtering for KSG")
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--k-knn", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--correction", choices=["bh", "holm", "none"], default="bh",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load cluster manifest
    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cluster manifest not found: {manifest_path}. "
            "Run experiments/typing/train.py first."
        )
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1

    print(f"Episodes: {len(cluster_manifest)}  |  Clusters: {n_clusters}")
    print(f"Regime: {args.regime or 'full episode'}  |  TE method: {args.te_method}")
    print(f"Permutations: {args.n_permutations}  |  p < {args.p_threshold} (BH-FDR)")

    results = compute_service_causal_relations(
        cluster_manifest=cluster_manifest,
        features_root=args.features_root,
        n_clusters=n_clusters,
        regime=args.regime,
        lag=args.lag,
        k_knn=args.k_knn,
        n_permutations=args.n_permutations,
        p_threshold=args.p_threshold,
        min_support=args.min_support,
        min_series_length=args.min_series_length,
        te_method=args.te_method,
        seed=args.seed,
        correction=args.correction,
    )

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    total_relations = sum(len(v) for v in results.values())
    output_data = {
        "n_clusters": n_clusters,
        "n_episodes": len(cluster_manifest),
        "regime": args.regime,
        "te_method": args.te_method,
        "n_permutations": args.n_permutations,
        "p_threshold": args.p_threshold,
        "total_significant_relations": total_relations,
        "clusters": {
            str(c): [asdict(r) for r in rels]
            for c, rels in results.items()
        },
    }
    json_path = args.output / "service_causal.json"
    json_path.write_text(json.dumps(output_data, indent=2))

    # -----------------------------------------------------------------------
    # Human-readable report
    # -----------------------------------------------------------------------
    lines = [
        "# Ontologie causale service→service (TE-KSG hiérarchique)\n",
        f"Épisodes : {len(cluster_manifest)}  |  Clusters : {n_clusters}",
        f"Régime analysé : **{args.regime or 'épisode complet'}**",
        f"Méthode TE : {args.te_method}  |  Permutations : {args.n_permutations}",
        f"Relations significatives (p < {args.p_threshold}, BH-FDR) : **{total_relations}**\n",
    ]

    if not results:
        lines.append("Aucune relation significative trouvée.")
    else:
        for c in sorted(results):
            scenario_counts: dict[str, int] = {}
            for ep_id, info in cluster_manifest.items():
                if int(info["cluster"]) == c:
                    sc = info.get("scenario", "unknown")
                    scenario_counts[sc] = scenario_counts.get(sc, 0) + 1
            dominant = max(scenario_counts, key=scenario_counts.get)

            lines.append(f"## Cluster C{c}  ({dominant}, {sum(scenario_counts.values())} épisodes)\n")
            rels = sorted(results[c], key=lambda r: -r.te_value)
            lines.append(f"{'Source':<25}  {'→  Target':<25}  {'TE':>8}  {'p_adj':>8}  {'n':>4}")
            lines.append("-" * 75)
            for r in rels:
                lines.append(
                    f"{r.source_service:<25}  →  {r.target_service:<22}  "
                    f"{r.te_value:>8.5f}  {r.p_value:>8.4f}  {r.support:>4}"
                )
            lines.append("")

    md_path = args.output / "service_causal.md"
    md_path.write_text("\n".join(lines))

    print(f"\n{total_relations} relations significatives trouvées.")
    print(f"JSON  : {json_path}")
    print(f"Report: {md_path}")

    for c in sorted(results):
        print(f"  C{c}: {len(results[c])} relations")


if __name__ == "__main__":
    main()
