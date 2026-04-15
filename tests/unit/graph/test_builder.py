"""Tests for graph.builder — ServiceGraphBuilder.

Uses synthetic spans to validate edge detection, aggregation, and filtering.
"""

from __future__ import annotations

import pytest

from graph.builder import ServiceGraphBuilder
from graph.types import ServiceGraph
from telemetry.collectors.trace_collector import Span

# ---------------------------------------------------------------------------
# Fixtures: synthetic span topologies
# ---------------------------------------------------------------------------


def _span(
    trace_id: str,
    span_id: str,
    parent_span_id: str,
    service_name: str,
    duration_s: float = 0.1,
    start_time_s: float | None = None,
    status_code: str = "OK",
) -> Span:
    """Convenience span factory."""
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        service_name=service_name,
        duration_s=duration_s,
        start_time_s=start_time_s,
        status_code=status_code,
    )


def _linear_trace() -> list[Span]:
    """A → B → C linear call chain (1 trace, 3 spans).

    Edges: A→B, B→C
    """
    return [
        _span("t1", "s1", "", "svc-a", duration_s=0.5),        # root
        _span("t1", "s2", "s1", "svc-b", duration_s=0.3),      # A calls B
        _span("t1", "s3", "s2", "svc-c", duration_s=0.1),      # B calls C
    ]


def _fan_out_trace() -> list[Span]:
    """A → B, A → C fan-out (1 trace, 3 spans).

    Edges: A→B, A→C
    """
    return [
        _span("t2", "s10", "", "svc-a", duration_s=0.5),
        _span("t2", "s11", "s10", "svc-b", duration_s=0.2),
        _span("t2", "s12", "s10", "svc-c", duration_s=0.15),
    ]


def _intra_service_trace() -> list[Span]:
    """A → A (internal call, same service).

    Should NOT produce an edge.
    """
    return [
        _span("t3", "s20", "", "svc-a", duration_s=0.5),
        _span("t3", "s21", "s20", "svc-a", duration_s=0.1),
    ]


def _multi_call_trace() -> list[Span]:
    """A calls B three times (3 child spans), with one error.

    Edge A→B: volume=3, error_rate=1/3
    """
    return [
        _span("t4", "s30", "", "svc-a", duration_s=1.0),
        _span("t4", "s31", "s30", "svc-b", duration_s=0.1, status_code="OK"),
        _span("t4", "s32", "s30", "svc-b", duration_s=0.3, status_code="OK"),
        _span("t4", "s33", "s30", "svc-b", duration_s=0.5, status_code="ERROR"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServiceGraphBuilder:
    """Test suite for ServiceGraphBuilder.build()."""

    def test_linear_trace_produces_two_edges(self) -> None:
        builder = ServiceGraphBuilder(edge_presence_threshold=0)
        graph = builder.build(_linear_trace(), timestamp=1000.0)

        assert isinstance(graph, ServiceGraph)
        assert graph.n_services == 3
        assert graph.n_edges == 2

        edge_keys = {(e.source, e.target) for e in graph.edges}
        assert ("svc-a", "svc-b") in edge_keys
        assert ("svc-b", "svc-c") in edge_keys

    def test_fan_out_trace_produces_two_edges(self) -> None:
        builder = ServiceGraphBuilder()
        graph = builder.build(_fan_out_trace(), timestamp=1000.0)

        assert graph.n_edges == 2
        edge_keys = {(e.source, e.target) for e in graph.edges}
        assert ("svc-a", "svc-b") in edge_keys
        assert ("svc-a", "svc-c") in edge_keys

    def test_intra_service_calls_ignored(self) -> None:
        builder = ServiceGraphBuilder()
        graph = builder.build(_intra_service_trace(), timestamp=1000.0)

        assert graph.n_edges == 0
        assert graph.n_services == 1  # only svc-a

    def test_multi_call_aggregation(self) -> None:
        builder = ServiceGraphBuilder()
        graph = builder.build(_multi_call_trace(), timestamp=1000.0)

        assert graph.n_edges == 1
        edge = graph.edges[0]
        assert edge.source == "svc-a"
        assert edge.target == "svc-b"
        assert edge.volume == 3
        assert edge.latency_median_s == pytest.approx(0.3, abs=1e-6)
        assert edge.error_rate == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_edge_presence_threshold_filters(self) -> None:
        """Threshold=3 should exclude edges with volume < 3."""
        builder = ServiceGraphBuilder(edge_presence_threshold=3)
        spans = _linear_trace()  # each edge has volume=1
        graph = builder.build(spans, timestamp=1000.0)

        # volume=1 ≤ threshold=3 → all edges filtered
        assert graph.n_edges == 0

    def test_explicit_services_list(self) -> None:
        builder = ServiceGraphBuilder()
        graph = builder.build(
            _linear_trace(),
            services=["svc-a", "svc-b", "svc-c", "svc-d"],
            timestamp=1000.0,
        )

        # svc-d is in the service list but has no spans → still a node
        assert graph.n_services == 4
        assert "svc-d" in graph.services
        # Only edges that exist in spans
        assert graph.n_edges == 2

    def test_unknown_service_in_spans_excluded(self) -> None:
        """If services list is given, spans from unknown services are ignored."""
        builder = ServiceGraphBuilder()
        graph = builder.build(
            _linear_trace(),
            services=["svc-a", "svc-b"],  # svc-c not in list
            timestamp=1000.0,
        )

        # B→C edge should be excluded because svc-c is not in services
        assert graph.n_edges == 1
        assert graph.edges[0].target == "svc-b"

    def test_empty_spans_produces_empty_graph(self) -> None:
        builder = ServiceGraphBuilder()
        graph = builder.build([], services=["svc-a"], timestamp=1000.0)

        assert graph.n_services == 1
        assert graph.n_edges == 0

    def test_from_config(self) -> None:
        """Test the from_config factory."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"graph": {"edge_presence_threshold": 5}})
        builder = ServiceGraphBuilder.from_config(cfg)
        assert builder._threshold == 5

    def test_services_are_sorted(self) -> None:
        builder = ServiceGraphBuilder()
        spans = _linear_trace()  # svc-a, svc-b, svc-c
        graph = builder.build(spans, timestamp=1000.0)

        assert graph.services == sorted(graph.services)

    def test_combined_traces(self) -> None:
        """Multiple traces combined should merge edges correctly."""
        builder = ServiceGraphBuilder()
        # Two traces both with A→B
        spans = _linear_trace() + _fan_out_trace()
        graph = builder.build(spans, timestamp=1000.0)

        # A→B appears in both traces → volume should be 2
        ab_edges = [e for e in graph.edges if e.source == "svc-a" and e.target == "svc-b"]
        assert len(ab_edges) == 1
        assert ab_edges[0].volume == 2

    def test_build_windowed_uses_span_timestamps(self) -> None:
        builder = ServiceGraphBuilder()
        spans = [
            _span("t1", "s1", "", "svc-a", start_time_s=10.0),
            _span("t1", "s2", "s1", "svc-b", start_time_s=11.0),
            _span("t2", "s3", "", "svc-a", start_time_s=40.0),
            _span("t2", "s4", "s3", "svc-c", start_time_s=41.0),
        ]

        graphs = builder.build_windowed(spans, window_s=20.0, stride_s=15.0)

        assert len(graphs) >= 2
        # Early window should contain A->B, late window should contain A->C
        edge_sets = [{(e.source, e.target) for e in graph.edges} for graph in graphs]
        merged = set().union(*edge_sets)
        assert ("svc-a", "svc-b") in merged
        assert ("svc-a", "svc-c") in merged

    def test_build_windowed_requires_positive_params(self) -> None:
        builder = ServiceGraphBuilder()
        with pytest.raises(ValueError):
            builder.build_windowed(_linear_trace(), window_s=0.0, stride_s=10.0)
        with pytest.raises(ValueError):
            builder.build_windowed(_linear_trace(), window_s=10.0, stride_s=0.0)

    def test_build_from_collector_uses_backend(self) -> None:
        class _Backend:
            def fetch_spans(self, _start: float, _end: float) -> list[Span]:
                return _linear_trace()

        class _Collector:
            def __init__(self) -> None:
                self._backend = _Backend()

        builder = ServiceGraphBuilder()
        graph = builder.build_from_collector(_Collector(), timestamp=1000.0, window_s=60.0)

        assert graph.n_edges == 2
