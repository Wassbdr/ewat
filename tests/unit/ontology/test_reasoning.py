"""Tests for the reasoning + SPARQL query layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ewat.ontology.graph import OntologyRelation
from ewat.ontology.owl_export import EmpiricalSources, build_abox
from ewat.ontology.queries import CANONICAL_QUERIES, run_query
from ewat.ontology.reasoning import (
    add_causal_relations_to_abox,
    add_cooccurrence_relations_to_abox,
    add_temporal_relations_to_abox,
    extract_entailment_diff,
    run_reasoner,
)
from ewat.ontology.service_propagation import enrich_with_service_propagation


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def enriched_abox():
    src = EmpiricalSources.default(REPO_ROOT)
    abox = build_abox(src)
    enrich_with_service_propagation(
        abox, REPO_ROOT / "experiments/ontology/service_causal.json",
    )
    existing = json.loads(
        (REPO_ROOT / "experiments/ontology/ontology.json").read_text()
    )
    rels = [OntologyRelation(**r) for r in existing["relations"]]
    add_temporal_relations_to_abox(abox.individuals, rels, abox.tbox.ontology)
    return abox


# ---------------------------------------------------------------------------
# Relation injection helpers
# ---------------------------------------------------------------------------


def test_add_temporal_relations_skips_self_loops():
    src = EmpiricalSources.default(REPO_ROOT)
    abox = build_abox(src)
    fake_rels = [
        OntologyRelation(
            source=0, target=0, relation_type="temporal", strength=10.0,
        ),
        OntologyRelation(
            source=0, target=3, relation_type="temporal", strength=4.0,
        ),
    ]
    n = add_temporal_relations_to_abox(
        abox.individuals, fake_rels, abox.tbox.ontology,
    )
    assert n == 1  # only the cross-cluster transition kept


def test_add_causal_relations_only_keeps_causal_type():
    src = EmpiricalSources.default(REPO_ROOT)
    abox = build_abox(src)
    fake_rels = [
        OntologyRelation(
            source=0, target=2, relation_type="causal", strength=0.5,
            p_value=0.01,
        ),
        OntologyRelation(
            source=1, target=3, relation_type="temporal", strength=2.0,
        ),
        OntologyRelation(
            source=4, target=5, relation_type="cooccurrence", strength=0.8,
            p_value=0.02,
        ),
    ]
    n = add_causal_relations_to_abox(
        abox.individuals, fake_rels, abox.tbox.ontology,
    )
    assert n == 1


def test_add_cooccurrence_relations_only_keeps_cooccurrence_type():
    src = EmpiricalSources.default(REPO_ROOT)
    abox = build_abox(src)
    fake_rels = [
        OntologyRelation(
            source=0, target=2, relation_type="cooccurrence", strength=0.7,
            p_value=0.01,
        ),
    ]
    n = add_cooccurrence_relations_to_abox(
        abox.individuals, fake_rels, abox.tbox.ontology,
    )
    assert n == 1


# ---------------------------------------------------------------------------
# HermiT reasoning
# ---------------------------------------------------------------------------


def test_run_reasoner_reports_consistency(enriched_abox):
    report = run_reasoner(enriched_abox.tbox.ontology, reasoner="hermit")
    assert report.consistent
    assert report.elapsed_s < 30.0  # plan: < 30s on full ABox
    assert report.inconsistent_classes == []


def test_extract_entailment_diff_returns_propagation_and_composite(
    enriched_abox,
):
    run_reasoner(enriched_abox.tbox.ontology, reasoner="hermit")
    diff = extract_entailment_diff(enriched_abox.tbox.ontology.world)
    assert diff.n_propagation_triples > 0
    # Asserted Drift_With_Anomaly ⊑ Composite_Anomaly → cluster_8 surfaces
    assert any("cluster_8" in name for name in diff.composite_anomaly_instances)


# ---------------------------------------------------------------------------
# Canonical queries
# ---------------------------------------------------------------------------


def test_all_composites_query_returns_cluster_8(enriched_abox):
    run_reasoner(enriched_abox.tbox.ontology, reasoner="hermit")
    res = run_query(
        enriched_abox.tbox.ontology.world,
        CANONICAL_QUERIES["all_composites"],
        select_vars=["anomaly"],
    )
    names = [r["anomaly"].name for r in res]
    assert any("cluster_8" in n for n in names)


def test_signatures_sharing_heavy_features_returns_results(enriched_abox):
    res = run_query(
        enriched_abox.tbox.ontology.world,
        CANONICAL_QUERIES["signatures_sharing_heavy_features"],
    )
    # At least a few features per cluster cross the 0.2 weight threshold
    assert len(res) >= 5


def test_fast_precursors_query_runs(enriched_abox):
    # Pre-condition: temporal relations have been injected as precedes
    res = run_query(
        enriched_abox.tbox.ontology.world,
        CANONICAL_QUERIES["fast_precursors_of_composite"],
    )
    # Lead time of cluster_8 is 5*30=150s ≤ 300s; if any cluster precedes
    # cluster_8 and has leadtime <= 300, it appears.
    assert isinstance(res, list)


def test_all_five_canonical_queries_are_valid_sparql(enriched_abox):
    """Smoke test that every query parses and returns a list (possibly empty)."""
    for name, query in CANONICAL_QUERIES.items():
        res = run_query(enriched_abox.tbox.ontology.world, query)
        assert isinstance(res, list), f"query {name} did not return a list"
