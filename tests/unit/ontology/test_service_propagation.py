"""Tests for service-level propagation enrichment."""

from __future__ import annotations

from pathlib import Path

import pytest

from ewat.ontology.owl_export import EmpiricalSources, build_abox
from ewat.ontology.service_propagation import (
    DEFAULT_UBIQUITY_THRESHOLD,
    ServiceEdge,
    _count_pair_ubiquity,
    _filter_specific,
    _normalize_service,
    enrich_with_service_propagation,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_CAUSAL = REPO_ROOT / "experiments/ontology/service_causal.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_normalize_service_replaces_hyphens():
    assert _normalize_service("load-generator") == "load_generator"
    assert _normalize_service("frontend") == "frontend"
    assert _normalize_service("product-catalog") == "product_catalog"


def _make_edge(src: str, tgt: str, te: float = 0.05) -> ServiceEdge:
    return ServiceEdge(src, tgt, te, p_value=0.01, support=20)


def test_count_pair_ubiquity():
    per_cluster = {
        0: [_make_edge("a", "b"), _make_edge("c", "d")],
        1: [_make_edge("a", "b")],
        2: [_make_edge("c", "d")],
    }
    counts = _count_pair_ubiquity(per_cluster)
    assert counts[("a", "b")] == 2
    assert counts[("c", "d")] == 2


def test_filter_specific_drops_ubiquitous_pair():
    per_cluster = {
        i: [_make_edge("a", "b"), _make_edge(f"src_{i}", f"tgt_{i}")]
        for i in range(4)
    }
    kept, dropped = _filter_specific(per_cluster, ubiquity_threshold=0.5)
    # ("a", "b") appears in 4/4 clusters > 50% → dropped
    assert ("a", "b") in dropped
    for edges in kept.values():
        assert all((e.source_service, e.target_service) != ("a", "b") for e in edges)


def test_filter_specific_keeps_rare_pair():
    per_cluster = {
        0: [_make_edge("rare_src", "rare_tgt")],
        1: [_make_edge("common_src", "common_tgt")],
        2: [_make_edge("common_src", "common_tgt")],
    }
    kept, dropped = _filter_specific(per_cluster, ubiquity_threshold=0.5)
    assert ("rare_src", "rare_tgt") not in dropped
    assert ("common_src", "common_tgt") in dropped


# ---------------------------------------------------------------------------
# End-to-end enrichment on the real artefacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def abox_with_propagation():
    src = EmpiricalSources.default(REPO_ROOT)
    abox = build_abox(src)
    report = enrich_with_service_propagation(
        abox, SERVICE_CAUSAL, ubiquity_threshold=DEFAULT_UBIQUITY_THRESHOLD,
    )
    return abox, report


def test_input_edges_match_124(abox_with_propagation):
    _, report = abox_with_propagation
    assert report.n_input_edges == 124


def test_filter_drops_known_ubiquitous_pairs(abox_with_propagation):
    _, report = abox_with_propagation
    dropped = set(report.dropped_ubiquitous_pairs)
    # load-generator → frontend is the canonical traffic-graph edge
    # appearing in 8/8 active clusters
    assert ("load-generator", "frontend") in dropped


def test_at_least_eight_clusters_enriched(abox_with_propagation):
    _, report = abox_with_propagation
    # 8 active clusters (C5 + C6 had 0 TE-significant edges originally)
    assert report.n_clusters_enriched >= 7


def test_drift_cluster_5_has_no_propagation(abox_with_propagation):
    """C5 (drift_rolling_deploy) has zero significant service-level TE
    relations in service_causal.json — a benign drift does not propagate.
    Scientifically validating result; preserved after enrichment."""
    abox, _ = abox_with_propagation
    a5 = abox.individuals["anomaly_cluster_5"]
    assert len(a5.propagatesThrough) == 0


def test_drift_cluster_6_has_no_propagation(abox_with_propagation):
    """C6 (drift_config_change) — same rationale as C5."""
    abox, _ = abox_with_propagation
    a6 = abox.individuals["anomaly_cluster_6"]
    assert len(a6.propagatesThrough) == 0


def test_anomaly_clusters_have_propagation(abox_with_propagation):
    """C0 (memory_pressure) is the largest cluster (59 episodes) and has
    the most service-level TE relations — should propagate widely."""
    abox, _ = abox_with_propagation
    a0 = abox.individuals["anomaly_cluster_0"]
    assert len(a0.propagatesThrough) >= 1


def test_propagation_targets_are_canonical_services(abox_with_propagation):
    abox, _ = abox_with_propagation
    expected_iris = {
        "service_frontend", "service_cart", "service_ad",
        "service_recommendation", "service_product_catalog",
        "service_load_generator",
    }
    for cid in range(10):
        a = abox.individuals[f"anomaly_cluster_{cid}"]
        for tgt in a.propagatesThrough:
            assert tgt.name in expected_iris


def test_filter_reduces_edges(abox_with_propagation):
    _, report = abox_with_propagation
    assert report.n_after_specificity_filter < report.n_input_edges


def test_consistency_preserved_after_enrichment(abox_with_propagation):
    """ABox + propagation must remain HermiT-consistent."""
    import owlready2 as owl

    abox, _ = abox_with_propagation
    onto = abox.tbox.ontology
    with onto:
        owl.sync_reasoner_hermit(infer_property_values=False, debug=0)
    assert list(onto.inconsistent_classes()) == []
