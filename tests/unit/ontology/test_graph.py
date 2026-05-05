"""Tests for OntologyGraph and OntologyRelation."""

import json
import tempfile
from pathlib import Path

import pytest

from ewat.ontology.graph import OntologyGraph, OntologyRelation


# ---------------------------------------------------------------------------
# OntologyRelation
# ---------------------------------------------------------------------------

def test_relation_default_fields():
    rel = OntologyRelation(source=0, target=1, relation_type="temporal", strength=3.0)
    assert rel.p_value is None
    assert rel.delta_t_mean is None
    assert rel.delta_t_std is None
    assert rel.support == 0


def test_relation_all_fields():
    rel = OntologyRelation(
        source=2, target=5, relation_type="causal", strength=0.42,
        p_value=0.03, support=10,
    )
    assert rel.source == 2
    assert rel.target == 5
    assert rel.strength == pytest.approx(0.42)
    assert rel.p_value == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# OntologyGraph construction
# ---------------------------------------------------------------------------

def test_graph_add():
    g = OntologyGraph(n_clusters=5)
    g.add(OntologyRelation(0, 1, "temporal", 4.0, support=4))
    assert len(g.relations) == 1


def test_graph_filter_by_type():
    g = OntologyGraph(n_clusters=5)
    g.add(OntologyRelation(0, 1, "temporal", 3.0, support=3))
    g.add(OntologyRelation(1, 2, "causal", 0.5, p_value=0.02))
    g.add(OntologyRelation(0, 2, "cooccurrence", 4.2, p_value=0.01))
    g.add(OntologyRelation(2, 3, "temporal", 5.0, support=5))

    temporal = g.filter_by_type("temporal")
    causal = g.filter_by_type("causal")
    cooc = g.filter_by_type("cooccurrence")

    assert len(temporal) == 2
    assert len(causal) == 1
    assert len(cooc) == 1


def test_graph_filter_unknown_type_returns_empty():
    g = OntologyGraph(n_clusters=3)
    g.add(OntologyRelation(0, 1, "temporal", 1.0))
    assert g.filter_by_type("nonexistent") == []


# ---------------------------------------------------------------------------
# Serialization / deserialization
# ---------------------------------------------------------------------------

def test_graph_to_dict_structure():
    g = OntologyGraph(n_clusters=4)
    g.add(OntologyRelation(0, 1, "causal", 0.3, p_value=0.04, support=6))
    d = g.to_dict()
    assert d["n_clusters"] == 4
    assert len(d["relations"]) == 1
    assert d["relations"][0]["source"] == 0
    assert d["relations"][0]["relation_type"] == "causal"


def test_graph_save_and_load_roundtrip():
    g = OntologyGraph(n_clusters=3)
    g.add(OntologyRelation(0, 2, "temporal", 5.0, delta_t_mean=60.0, delta_t_std=10.0, support=5))
    g.add(OntologyRelation(1, 2, "cooccurrence", 6.5, p_value=0.01, support=3))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ontology.json"
        g.save(path)

        g2 = OntologyGraph.load(path)

    assert g2.n_clusters == 3
    assert len(g2.relations) == 2
    temporal = g2.filter_by_type("temporal")
    assert len(temporal) == 1
    assert temporal[0].delta_t_mean == pytest.approx(60.0)
    assert temporal[0].delta_t_std == pytest.approx(10.0)


def test_graph_save_is_valid_json():
    g = OntologyGraph(n_clusters=2)
    g.add(OntologyRelation(0, 1, "causal", 0.1, p_value=0.05, support=5))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "o.json"
        g.save(path)
        data = json.loads(path.read_text())
    assert "relations" in data
    assert "n_clusters" in data


def test_graph_summary_contains_counts():
    g = OntologyGraph(n_clusters=10)
    g.add(OntologyRelation(0, 1, "temporal", 3.0))
    g.add(OntologyRelation(2, 3, "causal", 0.5, p_value=0.02))
    summary = g.summary()
    assert "temporal=1" in summary
    assert "causal=1" in summary
    assert "n_clusters=10" in summary


def test_graph_load_none_fields_preserved():
    g = OntologyGraph(n_clusters=2)
    g.add(OntologyRelation(0, 1, "temporal", 3.0, delta_t_mean=None, delta_t_std=None))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "o.json"
        g.save(path)
        g2 = OntologyGraph.load(path)
    assert g2.relations[0].delta_t_mean is None
