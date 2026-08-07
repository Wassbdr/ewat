"""End-to-end orchestrator for the EWAT OWL/RDF anomaly ontology pipeline.

Phases (cf. plan ``oublie-la-phase-jury-tidy-reef.md``):

  P1  Build TBox (classes, properties, axioms, literature annotations).
  P2  Populate ABox from cluster manifest + fiches + scenarios registry.
  P3  Enrich with service-level propagation (filtered ``service_causal.json``).
  P3' Inject existing temporal cross-cluster transitions as ``precedes``.
  P4  (assumes synthetic episodes already generated via
      ``scripts/synthesize_composite_episodes.py``)
  P5  Extract causal/co-occurrence relations from composite synthetics
      (multivariate KSG-1 TE + by-construction co-occurrence) and inject
      into the ABox; run HermiT.
  Save the full ontology in RDF/XML + Turtle.

Outputs
-------
- ``data/ontology/full_ontology.owl`` (RDF/XML, canonical)
- ``data/ontology/full_ontology.ttl`` (Turtle, human-readable)
- ``experiments/ontology_v2/build_summary.json`` (counts per phase)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ewat.ontology.composite_causal import (
    extract_pairwise_causal,
    extract_pairwise_cooccurrence,
)
from ewat.ontology.graph import OntologyRelation
from ewat.ontology.owl_export import EmpiricalSources, build_abox
from ewat.ontology.reasoning import (
    add_causal_relations_to_abox,
    add_cooccurrence_relations_to_abox,
    add_temporal_relations_to_abox,
    run_reasoner,
)
from ewat.ontology.service_propagation import enrich_with_service_propagation

log = logging.getLogger("build_owl")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-owl", type=Path,
                        default=Path("data/ontology/full_ontology.owl"))
    parser.add_argument("--output-ttl", type=Path,
                        default=Path("data/ontology/full_ontology.ttl"))
    parser.add_argument("--synthetic-root", type=Path,
                        default=Path("data/features/v3_synthetic"),
                        help="Directory of synth_cascade_*/synth_overlay_* episodes")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--ubiquity-threshold", type=float, default=0.5)
    parser.add_argument("--summary", type=Path,
                        default=Path("experiments/ontology_v2/build_summary.json"))
    args = parser.parse_args()

    summary: dict = {"phases": {}}

    # ── Phase 1+2 ─────────────────────────────────────────────────────────
    log.info("[P1+P2] Building TBox and ABox")
    sources = EmpiricalSources.default(args.repo_root)
    abox = build_abox(sources)
    summary["phases"]["P1_P2"] = {
        "n_classes": len(abox.tbox.classes),
        "n_object_properties": len(abox.tbox.object_properties),
        "n_data_properties": len(abox.tbox.data_properties),
        "n_clusters": abox.n_clusters,
        "n_signatures": abox.n_signatures,
        "n_feature_weights": abox.n_feature_weights,
        "n_services": abox.n_services,
        "n_individuals_total": len(abox.individuals),
    }
    log.info("  TBox: %d classes, %d obj props, %d data props",
             len(abox.tbox.classes),
             len(abox.tbox.object_properties),
             len(abox.tbox.data_properties))
    log.info("  ABox: %d individuals (%d clusters, %d feature weights)",
             len(abox.individuals), abox.n_clusters, abox.n_feature_weights)

    # ── Phase 3 ────────────────────────────────────────────────────────────
    log.info("[P3] Enriching with service-level propagation")
    prop_report = enrich_with_service_propagation(
        abox,
        args.repo_root / "experiments/ontology/service_causal.json",
        ubiquity_threshold=args.ubiquity_threshold,
    )
    summary["phases"]["P3"] = {
        "n_input_edges": prop_report.n_input_edges,
        "n_after_filter": prop_report.n_after_specificity_filter,
        "n_clusters_enriched": prop_report.n_clusters_enriched,
        "dropped_ubiquitous_pairs": prop_report.dropped_ubiquitous_pairs,
    }

    # ── Phase 3' (temporal cross-cluster transitions) ─────────────────────
    log.info("[P3'] Injecting temporal precedes relations from ontology.json")
    onto_path = args.repo_root / "experiments/ontology/ontology.json"
    if onto_path.exists():
        existing = json.loads(onto_path.read_text())
        temporal_rels = [OntologyRelation(**r) for r in existing["relations"]]
        n_temp = add_temporal_relations_to_abox(
            abox.individuals, temporal_rels, abox.tbox.ontology,
        )
        summary["phases"]["P3_prime"] = {"n_precedes_added": n_temp}
    else:
        summary["phases"]["P3_prime"] = {"warning": "ontology.json missing"}

    # ── Phase 5: composite causal + cooccurrence ─────────────────────────
    log.info("[P5] Extracting causal/cooccurrence from synthetic composites")
    if args.synthetic_root.exists():
        manifest = args.repo_root / \
            "experiments/typing/cluster_artifacts/cluster_manifest.json"
        causal_rels = extract_pairwise_causal(
            args.synthetic_root, manifest,
            n_permutations=args.n_permutations,
            p_threshold=args.p_threshold,
        )
        coocc_rels = extract_pairwise_cooccurrence(
            args.synthetic_root, manifest,
        )
        n_c = add_causal_relations_to_abox(
            abox.individuals, causal_rels, abox.tbox.ontology,
        )
        n_co = add_cooccurrence_relations_to_abox(
            abox.individuals, coocc_rels, abox.tbox.ontology,
        )
        summary["phases"]["P5"] = {
            "n_synthetic_episodes": len(list(args.synthetic_root.iterdir())),
            "n_causal_significant": len(causal_rels),
            "n_causal_added": n_c,
            "n_cooccurrence_significant": len(coocc_rels),
            "n_cooccurrence_added": n_co,
            "causal_relations": [
                {"source": r.source, "target": r.target,
                 "strength": r.strength, "p_value": r.p_value}
                for r in causal_rels
            ],
        }
    else:
        log.warning(
            "Synthetic root %s missing — skipping P5 causal extraction "
            "(run scripts/synthesize_composite_episodes.py first)",
            args.synthetic_root,
        )
        summary["phases"]["P5"] = {"skipped": True}

    # ── Reasoning + save ──────────────────────────────────────────────────
    log.info("[Reasoning] Running HermiT")
    rep = run_reasoner(abox.tbox.ontology, reasoner="hermit")
    summary["phases"]["reasoning"] = {
        "elapsed_s": rep.elapsed_s,
        "consistent": rep.consistent,
        "inconsistent_classes": rep.inconsistent_classes,
        "n_class_triples_before": rep.n_class_triples_before,
        "n_class_triples_after": rep.n_class_triples_after,
        "materialised_class_triples": rep.materialised_class_triples,
    }

    log.info("[Save] Writing %s and %s", args.output_owl, args.output_ttl)
    args.output_owl.parent.mkdir(parents=True, exist_ok=True)
    abox.save(args.output_owl, fmt="rdfxml")
    abox.save_turtle(args.output_ttl)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, default=str))
    log.info("[Done] Summary written to %s", args.summary)


if __name__ == "__main__":
    main()
