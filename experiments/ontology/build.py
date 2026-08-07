"""Ontology build pipeline — Étape 2b.

Reads the cluster artifacts produced by experiments/typing/train.py and builds
the ontology O = (C, R) with three types of relations:
  1. Temporal  — C_i →^{Δt,σ} C_j via consecutive-episode analysis
  2. Causal    — C_i → C_j via Transfer Entropy (KSG estimator)
  3. Co-occurrence — C_i ↔ C_j via χ² test on scenario membership

Outputs
-------
  experiments/ontology/ontology.json  — full graph (OntologyGraph.save)
  experiments/ontology/results.md     — human-readable summary

Usage
-----
    python -m experiments.ontology.build \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        [--min-support 3] [--p-threshold 0.05] [--n-permutations 100]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_TRACKING_SILENT", "true")

import mlflow
from ewat.ontology.causal import compute_causal_relations, compute_service_causal_relations

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
from ewat.ontology.cooccurrence import compute_cooccurrence_relations
from ewat.ontology.graph import OntologyGraph
from ewat.ontology.temporal import compute_temporal_relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EWAT ontology (Étape 2b)")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"),
                        help="Output dir from experiments/typing/train.py")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"),
                        help="Feature store root (contains episode subdirs)")
    parser.add_argument("--output", type=Path, default=Path("experiments/ontology"),
                        help="Output directory")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--max-delta-seconds", type=float, default=7200.0)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--k-knn", type=int, default=5)
    parser.add_argument("--min-series-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-service-te",
        action="store_true",
        help="Also run hierarchical service→service TE (build_service.py logic)",
    )
    parser.add_argument(
        "--service-regime",
        type=str,
        default=None,
        choices=["normal", "injection", "recovery"],
        help="Regime filter for service-level TE (default: full episode)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load cluster artifacts
    artifacts_dir = args.typing_dir / "cluster_artifacts"
    manifest_path = artifacts_dir / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cluster manifest not found at {manifest_path}. "
            "Run experiments/typing/train.py first."
        )

    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    # Infer n_clusters from manifest
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1
    print(f"Loaded manifest: {len(cluster_manifest)} episodes, {n_clusters} clusters")

    graph = OntologyGraph(n_clusters=n_clusters)

    # -----------------------------------------------------------------------
    # 1. Temporal relations
    # -----------------------------------------------------------------------
    print(f"\n[1/3] Temporal relations (min_support={args.min_support}) …")
    temporal_rels = compute_temporal_relations(
        cluster_manifest=cluster_manifest,
        features_root=args.features_root,
        min_support=args.min_support,
        max_delta_seconds=args.max_delta_seconds,
    )
    for rel in temporal_rels:
        graph.add(rel)
    print(f"  → {len(temporal_rels)} temporal relations")

    # -----------------------------------------------------------------------
    # 2. Causal relations (TE-KSG)
    # -----------------------------------------------------------------------
    print(f"\n[2/3] Causal relations (TE-KSG, n_permutations={args.n_permutations}, "
          f"p<{args.p_threshold}) …")
    causal_rels = compute_causal_relations(
        cluster_manifest=cluster_manifest,
        features_root=args.features_root,
        n_clusters=n_clusters,
        lag=args.lag,
        k_knn=args.k_knn,
        n_permutations=args.n_permutations,
        p_threshold=args.p_threshold,
        min_support=args.min_support,
        min_series_length=args.min_series_length,
        seed=args.seed,
    )
    for rel in causal_rels:
        graph.add(rel)
    print(f"  → {len(causal_rels)} causal relations")

    # -----------------------------------------------------------------------
    # 3. Co-occurrence relations
    # -----------------------------------------------------------------------
    print(f"\n[3/3] Co-occurrence relations (p<{args.p_threshold}) …")
    cooc_rels = compute_cooccurrence_relations(
        cluster_manifest=cluster_manifest,
        n_clusters=n_clusters,
        p_threshold=args.p_threshold,
        min_cooccurrences=args.min_support,
    )
    for rel in cooc_rels:
        graph.add(rel)
    print(f"  → {len(cooc_rels)} co-occurrence relations")

    n_service_causal = 0
    if args.include_service_te:
        print(
            f"\n[4/4] Service-level TE (hierarchical, regime={args.service_regime or 'full'}) …"
        )
        service_results = compute_service_causal_relations(
            cluster_manifest=cluster_manifest,
            features_root=args.features_root,
            n_clusters=n_clusters,
            regime=args.service_regime,
            lag=args.lag,
            k_knn=args.k_knn,
            n_permutations=args.n_permutations,
            p_threshold=args.p_threshold,
            min_support=args.min_support,
            min_series_length=10,
            seed=args.seed,
        )
        n_service_causal = sum(len(v) for v in service_results.values())
        from dataclasses import asdict

        service_path = args.output / "service_causal.json"
        service_path.write_text(
            json.dumps(
                {
                    "n_clusters": n_clusters,
                    "regime": args.service_regime,
                    "total_significant_relations": n_service_causal,
                    "clusters": {
                        str(c): [asdict(r) for r in rels]
                        for c, rels in service_results.items()
                    },
                },
                indent=2,
            )
        )
        print(f"  → {n_service_causal} service→service relations → {service_path}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    ontology_path = args.output / "ontology.json"
    graph.save(ontology_path)
    print(f"\nOntology saved to {ontology_path}")
    print(graph.summary())

    # Human-readable report
    report_lines = [
        "# Ontologie EWAT — Résultats\n",
        f"Clusters : {n_clusters}",
        f"Épisodes : {len(cluster_manifest)}\n",
        "## Relations temporelles",
    ]
    temporal = graph.filter_by_type("temporal")
    if temporal:
        for r in sorted(temporal, key=lambda x: -x.support):
            report_lines.append(
                f"  C{r.source} → C{r.target}  "
                f"Δt={r.delta_t_mean:.0f}±{r.delta_t_std:.0f}s  "
                f"support={r.support}"
            )
    else:
        report_lines.append("  (aucune)")

    report_lines.extend(["\n## Relations causales (TE-KSG)"])
    causal = graph.filter_by_type("causal")
    if causal:
        for r in sorted(causal, key=lambda x: -x.strength):
            report_lines.append(
                f"  C{r.source} → C{r.target}  "
                f"TE={r.strength:.4f}  p={r.p_value:.3f}  support={r.support}"
            )
    else:
        report_lines.append("  (aucune)")

    report_lines.extend(["\n## Relations de co-occurrence (χ²)"])
    cooc = graph.filter_by_type("cooccurrence")
    if cooc:
        for r in sorted(cooc, key=lambda x: -x.strength):
            report_lines.append(
                f"  C{r.source} ↔ C{r.target}  "
                f"χ²={r.strength:.2f}  p={r.p_value:.3f}  support={r.support}"
            )
    else:
        report_lines.append("  (aucune)")

    report_lines.extend(
        [
            "\n## Relations causales service→service (TE hiérarchique)",
            f"  Cluster-level TE (section ci-dessus) : {len(causal)} relations",
            f"  Service-level TE (--include-service-te) : {n_service_causal} relations",
            "  Voir `service_causal.md` pour le détail par cluster.",
        ]
    )

    results_path = args.output / "results.md"
    results_path.write_text("\n".join(report_lines))
    print(f"Report: {results_path}")

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("ewat_ontology_build")
        with mlflow.start_run(run_name="ontology_build"):
            mlflow.log_params({
                "n_clusters": n_clusters,
                "n_episodes": len(cluster_manifest),
                "min_support": args.min_support,
                "p_threshold": args.p_threshold,
                "n_permutations": args.n_permutations,
                "lag": args.lag,
                "k_knn": args.k_knn,
                "seed": args.seed,
            })
            mlflow.log_metrics({
                "n_temporal_relations": len(temporal_rels),
                "n_causal_relations": len(causal_rels),
                "n_cooccurrence_relations": len(cooc_rels),
                "n_total_relations": len(temporal_rels) + len(causal_rels) + len(cooc_rels),
            })
    except Exception:
        pass


if __name__ == "__main__":
    main()
