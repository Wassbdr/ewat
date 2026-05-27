"""Tests for the empirical ABox builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from ewat.ontology.owl_export import (
    GRID_STEP_SECONDS,
    EmpiricalSources,
    build_abox,
    export_ontology,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def sources() -> EmpiricalSources:
    return EmpiricalSources.default(REPO_ROOT)


@pytest.fixture(scope="module")
def abox(sources):
    return build_abox(sources)


# ---------------------------------------------------------------------------
# Source availability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr",
    [
        "cluster_manifest", "fiches_dir", "cluster_semantics",
        "ontology_temporal", "precursor_results", "scenarios_registry",
        "ontology_config",
    ],
)
def test_source_path_exists(sources, attr):
    assert getattr(sources, attr).exists()


# ---------------------------------------------------------------------------
# Cardinality of the ABox
# ---------------------------------------------------------------------------


def test_abox_has_ten_clusters(abox):
    assert abox.n_clusters == 10


def test_abox_has_ten_signatures(abox):
    assert abox.n_signatures == 10


def test_abox_has_six_services(abox):
    assert abox.n_services == 6


def test_abox_feature_weights_positive(abox):
    # Each cluster has between 4 and 17 non-zero feature weights;
    # ewat_v3 totals ~107 across all clusters.
    assert 50 <= abox.n_feature_weights <= 170


def test_abox_total_individuals_consistent(abox):
    # 10 clusters + 10 anomalies + 10 signatures + n_feat_weights + 6 services
    expected = 10 + 10 + 10 + abox.n_feature_weights + abox.n_services
    assert len(abox.individuals) == expected


# ---------------------------------------------------------------------------
# Individuals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", range(10))
def test_each_cluster_has_empirical_individual(abox, cid):
    assert f"cluster_{cid}" in abox.individuals


@pytest.mark.parametrize("cid", range(10))
def test_each_cluster_has_anomaly_individual(abox, cid):
    assert f"anomaly_cluster_{cid}" in abox.individuals


@pytest.mark.parametrize("cid", range(10))
def test_each_cluster_has_signature(abox, cid):
    sig = abox.individuals[f"signature_cluster_{cid}"]
    assert len(sig.hasFeatureWeight) >= 1


def test_services_match_canonical_set(abox):
    services = {
        name for name in abox.individuals if name.startswith("service_")
    }
    expected = {
        "service_frontend", "service_cart", "service_ad",
        "service_recommendation", "service_product_catalog",
        "service_load_generator",
    }
    assert services == expected


# ---------------------------------------------------------------------------
# Class assignment from scenario mapping
# ---------------------------------------------------------------------------


def test_cluster_8_is_drift_with_anomaly(abox):
    """faulty_deploy_overlap dominant → Drift_With_Anomaly (Composite_Anomaly)."""
    a8 = abox.individuals["anomaly_cluster_8"]
    class_names = {c.name for c in a8.is_a}
    # Drift_With_Anomaly is a subclass of Composite_Anomaly
    assert "Drift_With_Anomaly" in class_names


def test_cluster_2_is_hard_exhaustion(abox):
    """resource_leak dominant → HardExhaustion."""
    a2 = abox.individuals["anomaly_cluster_2"]
    class_names = {c.name for c in a2.is_a}
    assert "HardExhaustion" in class_names


def test_cluster_7_is_cpu_saturation(abox):
    """cpu_starvation dominant → CPU_Saturation."""
    a7 = abox.individuals["anomaly_cluster_7"]
    class_names = {c.name for c in a7.is_a}
    assert "CPU_Saturation" in class_names


# ---------------------------------------------------------------------------
# Relations populated
# ---------------------------------------------------------------------------


def test_anomalies_link_to_their_cluster(abox):
    a3 = abox.individuals["anomaly_cluster_3"]
    c3 = abox.individuals["cluster_3"]
    assert c3 in a3.observedIn


def test_anomalies_have_signature(abox):
    a0 = abox.individuals["anomaly_cluster_0"]
    sig0 = abox.individuals["signature_cluster_0"]
    assert sig0 in a0.hasSignature


def test_anomalies_affect_at_least_one_service(abox):
    """At least 8/10 clusters should have non-empty target service union
    (a few clusters with marginal scenario distributions may have empty
    intersection with canonical services)."""
    n_with_targets = sum(
        1 for cid in range(10)
        if len(abox.individuals[f"anomaly_cluster_{cid}"].affects) > 0
    )
    assert n_with_targets >= 8


def test_feature_weight_pair_is_complete(abox):
    fw = abox.individuals["signature_cluster_0"].hasFeatureWeight[0]
    assert fw.featureName is not None
    assert fw.weightValue is not None
    assert fw.weightValue > 0.0


# ---------------------------------------------------------------------------
# Data properties (temporal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", range(10))
def test_temporal_duration_is_positive(abox, cid):
    a = abox.individuals[f"anomaly_cluster_{cid}"]
    assert a.temporalDuration is not None
    assert a.temporalDuration > 0.0


@pytest.mark.parametrize("cid", range(10))
def test_temporal_lead_time_is_positive(abox, cid):
    a = abox.individuals[f"anomaly_cluster_{cid}"]
    assert a.temporalLeadTime is not None
    assert a.temporalLeadTime > 0.0


def test_lead_time_is_grid_aligned(abox):
    """Lead times should be integer multiples of GRID_STEP_SECONDS."""
    for cid in range(10):
        lead = abox.individuals[f"anomaly_cluster_{cid}"].temporalLeadTime
        assert (lead % GRID_STEP_SECONDS) == 0


# ---------------------------------------------------------------------------
# HermiT consistency
# ---------------------------------------------------------------------------


def test_abox_is_logically_consistent(abox):
    """HermiT must accept the merged TBox + ABox without inconsistency."""
    import owlready2 as owl

    onto = abox.tbox.ontology
    with onto:
        owl.sync_reasoner_hermit(infer_property_values=False, debug=0)
    inconsistent = list(onto.inconsistent_classes())
    assert inconsistent == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_writes_both_formats(tmp_path: Path, sources):
    paths = export_ontology(sources, tmp_path)
    assert paths["rdfxml"].exists()
    assert paths["turtle"].exists()
    assert paths["rdfxml"].stat().st_size > 10_000
    assert paths["turtle"].stat().st_size > 10_000


def test_exported_turtle_is_parseable(tmp_path: Path, sources):
    import rdflib

    paths = export_ontology(sources, tmp_path)
    g = rdflib.Graph()
    g.parse(str(paths["turtle"]), format="turtle")
    # ABox should add hundreds of triples on top of the TBox.
    assert len(g) > 500
