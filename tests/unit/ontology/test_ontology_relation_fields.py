"""Tests for OntologyRelation new fields (Step 7 audit fixes).

Covers fixes 7.3 (is_from_synthetic) and 7.4 (p_raw vs p_value separation).
"""

import json

import pytest

from ewat.ontology.graph import OntologyGraph, OntologyRelation
from ewat.ontology.causal import ServiceCausalRelation


def test_relation_defaults_p_raw_and_is_synthetic():
    """New fields have safe defaults so existing callers don't break."""
    r = OntologyRelation(source=0, target=1, relation_type="causal", strength=0.1)
    assert r.p_raw is None
    assert r.is_from_synthetic is False


def test_relation_with_p_raw_and_p_adjusted():
    """Both raw and adjusted p-values can be stored side by side."""
    r = OntologyRelation(
        source=0, target=1, relation_type="causal", strength=0.15,
        p_value=0.03, p_raw=0.002, support=10,
    )
    assert r.p_value == 0.03
    assert r.p_raw == 0.002
    assert r.p_raw < r.p_value, "raw < adjusted under BH-FDR"


def test_relation_is_from_synthetic_flag():
    r = OntologyRelation(
        source=2, target=5, relation_type="causal", strength=0.08,
        p_value=0.04, p_raw=0.01, is_from_synthetic=True,
    )
    assert r.is_from_synthetic is True


def test_graph_to_dict_includes_new_fields():
    """Round-trip OntologyGraph → dict → JSON preserves p_raw and is_synthetic."""
    g = OntologyGraph(n_clusters=3)
    g.add(OntologyRelation(
        source=0, target=1, relation_type="causal", strength=0.2,
        p_value=0.04, p_raw=0.005, is_from_synthetic=True,
    ))
    serialised = g.to_dict()
    assert "relations" in serialised
    rel = serialised["relations"][0]
    assert rel["p_raw"] == 0.005
    assert rel["is_from_synthetic"] is True
    # JSON round-trip preserves fields
    payload = json.dumps(serialised)
    parsed = json.loads(payload)
    assert parsed["relations"][0]["p_raw"] == 0.005


def test_service_causal_relation_p_raw_default_none():
    """ServiceCausalRelation now exposes p_raw, default None."""
    r = ServiceCausalRelation(
        cluster=4,
        source_service="frontend",
        target_service="cart",
        te_value=0.05,
        p_value=0.03,
        support=12,
    )
    assert r.p_raw is None


def test_service_causal_relation_with_p_raw():
    r = ServiceCausalRelation(
        cluster=4,
        source_service="frontend",
        target_service="cart",
        te_value=0.05,
        p_value=0.04,
        support=12,
        p_raw=0.003,
    )
    assert r.p_raw == 0.003
