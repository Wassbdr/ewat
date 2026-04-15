"""Tests for graph.serialization."""

from __future__ import annotations

import json

import numpy as np

from graph.serialization import (
    load_graph,
    load_graph_sequence,
    save_adjacency_tensor_bulk,
    save_graph,
    save_graph_sequence,
)
from graph.types import ServiceGraph, WeightedEdge


def _graph(ts: float) -> ServiceGraph:
    return ServiceGraph(
        services=["svc-a", "svc-b", "svc-c"],
        edges=[
            WeightedEdge(
                source="svc-a",
                target="svc-b",
                volume=3,
                latency_median_s=0.12,
                error_rate=0.2,
            ),
            WeightedEdge(
                source="svc-b",
                target="svc-c",
                volume=2,
                latency_median_s=0.08,
                error_rate=0.0,
            ),
        ],
        timestamp=ts,
    )


def test_save_load_graph_roundtrip(tmp_path) -> None:
    graph = _graph(ts=1000.0)
    base = tmp_path / "g0"

    save_graph(graph, base)
    loaded = load_graph(base)

    assert loaded.services == graph.services
    assert loaded.timestamp == graph.timestamp
    assert loaded.edges == graph.edges
    np.testing.assert_allclose(loaded.adjacency_tensor(), graph.adjacency_tensor())


def test_save_load_graph_sequence_roundtrip(tmp_path) -> None:
    graphs = [_graph(1000.0), _graph(1015.0)]

    save_graph_sequence(graphs, tmp_path, prefix="graph")
    loaded = load_graph_sequence(tmp_path, prefix="graph")

    assert len(loaded) == 2
    assert loaded[0].timestamp == 1000.0
    assert loaded[1].timestamp == 1015.0


def test_save_adjacency_tensor_bulk(tmp_path) -> None:
    graphs = [_graph(1000.0), _graph(1015.0), _graph(1030.0)]
    base = tmp_path / "adjacency"

    save_adjacency_tensor_bulk(graphs, base)

    with np.load(f"{base}.npz") as payload:
        adjacency = payload["adjacency"]
        timestamps = payload["timestamps"]

    assert adjacency.shape == (3, 3, 3, 3)
    assert timestamps.shape == (3,)
    np.testing.assert_allclose(timestamps, [1000.0, 1015.0, 1030.0])

    with open(f"{base}_services.json") as f:
        services = json.load(f)
    assert services == ["svc-a", "svc-b", "svc-c"]
