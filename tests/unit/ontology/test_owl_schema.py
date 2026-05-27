"""Structural tests for the OWL TBox.

Reasoning behaviour (HermiT classification of individuals) is covered by
``tests/unit/ontology/test_reasoning.py`` (Phase 5) — that requires a Java
runtime and AllDifferent axioms which are emitted by the ABox builder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ewat.ontology.literature_taxonomy import (
    CLASS_LITERATURE,
    PROPERTY_LITERATURE,
)
from ewat.ontology.owl_schema import DEFAULT_IRI, build_tbox, export_tbox


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tbox():
    return build_tbox()


def test_tbox_default_iri(tbox):
    assert tbox.iri == DEFAULT_IRI


def test_tbox_expected_class_count(tbox):
    assert len(tbox.classes) == 29


def test_tbox_expected_property_counts(tbox):
    assert len(tbox.object_properties) == 11
    assert len(tbox.data_properties) == 6


@pytest.mark.parametrize(
    "name",
    [
        "Anomaly", "Signature", "Service", "EmpiricalCluster", "FeatureWeight",
        "RecoveryPattern", "Mitigation",
        "Resource_Anomaly", "Saturation", "CPU_Saturation", "Memory_Saturation",
        "Network_Saturation", "Disk_Saturation", "HardExhaustion",
        "Liveness_Anomaly", "Functional_Anomaly", "Latency_Anomaly",
        "Network_Anomaly", "Configuration_Anomaly", "Deployment_Anomaly",
        "Composite_Anomaly", "Drift_With_Anomaly", "CascadingFailure",
        "Drift", "Benign_Drift", "Scaling_Drift", "Deployment_Drift",
        "Configuration_Drift", "Traffic_Drift",
    ],
)
def test_class_present(tbox, name):
    assert name in tbox.classes


@pytest.mark.parametrize(
    "name",
    [
        "hasSignature", "affects", "observedIn", "causes", "isCausedBy",
        "precedes", "coOccursWith", "propagatesThrough", "hasComponent",
        "hasFeatureWeight", "mitigatedBy",
    ],
)
def test_object_property_present(tbox, name):
    assert name in tbox.object_properties


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_cpu_saturation_subclasses_saturation(tbox):
    sat = tbox.classes["Saturation"]
    cpu = tbox.classes["CPU_Saturation"]
    assert sat in cpu.ancestors()


def test_saturation_subclasses_resource_anomaly(tbox):
    res = tbox.classes["Resource_Anomaly"]
    sat = tbox.classes["Saturation"]
    assert res in sat.ancestors()


def test_hard_exhaustion_subclasses_saturation(tbox):
    sat = tbox.classes["Saturation"]
    he = tbox.classes["HardExhaustion"]
    assert sat in he.ancestors()


def test_cascading_failure_subclasses_composite(tbox):
    comp = tbox.classes["Composite_Anomaly"]
    cf = tbox.classes["CascadingFailure"]
    assert comp in cf.ancestors()


def test_drift_disjoint_from_anomaly_hierarchy(tbox):
    """Drift and Anomaly are top-level disjoint subtrees of owl:Thing."""
    drift = tbox.classes["Drift"]
    anomaly = tbox.classes["Anomaly"]
    # Neither in the other's ancestor chain
    assert anomaly not in drift.ancestors()
    assert drift not in anomaly.ancestors()


# ---------------------------------------------------------------------------
# Property characteristics
# ---------------------------------------------------------------------------


def test_causes_is_transitive(tbox):
    import owlready2 as owl

    causes = tbox.object_properties["causes"]
    assert owl.TransitiveProperty in causes.is_a


def test_causes_is_asymmetric(tbox):
    import owlready2 as owl

    causes = tbox.object_properties["causes"]
    assert owl.AsymmetricProperty in causes.is_a


def test_causes_is_irreflexive(tbox):
    import owlready2 as owl

    causes = tbox.object_properties["causes"]
    assert owl.IrreflexiveProperty in causes.is_a


def test_co_occurs_with_is_symmetric(tbox):
    import owlready2 as owl

    co = tbox.object_properties["coOccursWith"]
    assert owl.SymmetricProperty in co.is_a


def test_precedes_is_transitive(tbox):
    import owlready2 as owl

    precedes = tbox.object_properties["precedes"]
    assert owl.TransitiveProperty in precedes.is_a


def test_has_component_is_transitive(tbox):
    import owlready2 as owl

    hc = tbox.object_properties["hasComponent"]
    assert owl.TransitiveProperty in hc.is_a


def test_propagates_through_implies_affects(tbox):
    propagates = tbox.object_properties["propagatesThrough"]
    affects = tbox.object_properties["affects"]
    assert affects in propagates.is_a


def test_is_caused_by_is_inverse_of_causes(tbox):
    causes = tbox.object_properties["causes"]
    is_caused_by = tbox.object_properties["isCausedBy"]
    assert is_caused_by.inverse_property == causes


# ---------------------------------------------------------------------------
# Equivalence axioms
# ---------------------------------------------------------------------------


def test_composite_anomaly_has_equivalence_axiom(tbox):
    comp = tbox.classes["Composite_Anomaly"]
    assert len(comp.equivalent_to) >= 1


def test_cascading_failure_has_equivalence_axiom(tbox):
    cf = tbox.classes["CascadingFailure"]
    assert len(cf.equivalent_to) >= 1


# ---------------------------------------------------------------------------
# Literature annotations
# ---------------------------------------------------------------------------


def test_every_class_has_literature_annotation(tbox):
    for name in tbox.classes:
        assert name in CLASS_LITERATURE, f"{name} missing literature"


def test_every_object_property_has_literature_annotation(tbox):
    for name in tbox.object_properties:
        assert name in PROPERTY_LITERATURE, f"{name} missing literature"


def test_class_comment_contains_reference_year(tbox):
    cpu = tbox.classes["CPU_Saturation"]
    annotations = list(cpu.comment)
    assert any("2013" in a for a in annotations)  # Gregg 2013


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_writes_both_formats(tmp_path: Path):
    paths = export_tbox(tmp_path)
    assert paths["rdfxml"].exists()
    assert paths["turtle"].exists()
    assert paths["rdfxml"].stat().st_size > 1000
    assert paths["turtle"].stat().st_size > 1000


def test_exported_turtle_is_parseable(tmp_path: Path):
    import rdflib

    paths = export_tbox(tmp_path)
    g = rdflib.Graph()
    g.parse(str(paths["turtle"]), format="turtle")
    assert len(g) > 100  # at least a hundred triples
