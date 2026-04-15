"""Tests for graph.validation."""

from __future__ import annotations

from graph.types import ServiceGraph, WeightedEdge
from graph.validation import validate_graph, validate_graph_sequence


def _valid_graph() -> ServiceGraph:
    return ServiceGraph(
        services=["svc-a", "svc-b", "svc-c"],
        edges=[
            WeightedEdge(
                source="svc-a",
                target="svc-b",
                volume=4,
                latency_median_s=0.1,
                error_rate=0.0,
            )
        ],
        timestamp=1000.0,
    )


def test_validate_graph_detects_isolated_node() -> None:
    report = validate_graph(_valid_graph(), min_edges=1)

    assert report.is_valid
    assert report.isolated_nodes == ["svc-c"]


def test_validate_graph_detects_phantom_edge() -> None:
    graph = ServiceGraph(
        services=["svc-a", "svc-b"],
        edges=[
            WeightedEdge(
                source="svc-a",
                target="svc-missing",
                volume=1,
                latency_median_s=0.2,
                error_rate=0.0,
            )
        ],
        timestamp=1000.0,
    )

    report = validate_graph(graph)

    assert not report.is_valid
    assert report.phantom_edges == [("svc-a", "svc-missing")]
    assert any("phantom" in err.lower() for err in report.errors)


def test_validate_graph_expected_services_warning() -> None:
    report = validate_graph(_valid_graph(), expected_services=["svc-a", "svc-b", "svc-z"])

    assert report.is_valid
    assert report.missing_services == ["svc-z"]
    assert report.warnings


def test_validate_graph_sequence_service_inconsistency() -> None:
    g1 = _valid_graph()
    g2 = ServiceGraph(
        services=["svc-a", "svc-b", "svc-d"],
        edges=g1.edges,
        timestamp=1015.0,
    )

    reports = validate_graph_sequence([g1, g2])

    assert len(reports) == 2
    assert reports[0].is_valid
    assert not reports[1].is_valid
    assert any("service list differs" in err.lower() for err in reports[1].errors)
