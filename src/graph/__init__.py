"""graph — EWAT service graph construction package.

Builds G(t) = (V, E(t), w_E(t)) from OTel trace spans and produces
A(t) ∈ ℝ^{N×N×3} for the STGCN encoder.

    from graph import ServiceGraphBuilder, ServiceGraph

    builder = ServiceGraphBuilder(edge_presence_threshold=0)
    graph = builder.build(spans)
    A_t = graph.adjacency_tensor()  # shape (N, N, 3)
"""

from graph.adjacency import (
    binary_adjacency,
    channel_adjacency,
    normalised_adjacency,
    weighted_adjacency,
)
from graph.builder import ServiceGraphBuilder
from graph.diagnostics import GraphStats, compute_stats, stats_to_dict
from graph.serialization import (
    load_graph,
    load_graph_sequence,
    save_adjacency_tensor_bulk,
    save_graph,
    save_graph_sequence,
)
from graph.types import ServiceGraph, WeightedEdge
from graph.validation import (
    ValidationReport,
    validate_graph,
    validate_graph_sequence,
)

__all__ = [
    # Builder
    "ServiceGraphBuilder",
    # Types
    "ServiceGraph",
    "WeightedEdge",
    # Adjacency
    "weighted_adjacency",
    "binary_adjacency",
    "normalised_adjacency",
    "channel_adjacency",
    # Diagnostics
    "GraphStats",
    "compute_stats",
    "stats_to_dict",
    # Serialization
    "save_graph",
    "load_graph",
    "save_graph_sequence",
    "load_graph_sequence",
    "save_adjacency_tensor_bulk",
    # Validation
    "ValidationReport",
    "validate_graph",
    "validate_graph_sequence",
]
