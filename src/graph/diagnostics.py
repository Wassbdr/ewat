"""Graph diagnostics — structural statistics for G(t).

Computes per-snapshot metrics useful for monitoring data quality
and understanding the baseline graph topology.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from graph.types import ServiceGraph


@dataclass
class GraphStats:
    """Structural statistics for a single graph snapshot.

    Attributes
    ----------
    n_nodes:
        Number of services |V|.
    n_edges:
        Number of directed edges |E(t)|.
    density:
        Edge density = |E| / (|V| × (|V|-1)). 0 for degenerate cases.
    avg_degree:
        Mean out-degree.
    max_degree:
        Maximum out-degree.
    n_connected_components:
        Number of weakly connected components (treating directed as undirected).
    diameter:
        Graph diameter on the largest weakly connected component (undirected).
    largest_component_size:
        Number of nodes in the largest weakly connected component.
    total_volume:
        Sum of call volumes across all edges.
    mean_latency:
        Mean median latency across edges (seconds).
    mean_error_rate:
        Mean error rate across edges.
    timestamp:
        Snapshot timestamp.
    """

    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    max_degree: int
    n_connected_components: int
    diameter: int
    largest_component_size: int
    total_volume: int
    mean_latency: float
    mean_error_rate: float
    timestamp: float


def compute_stats(graph: ServiceGraph) -> GraphStats:
    """Compute structural statistics for a graph snapshot.

    Parameters
    ----------
    graph:
        A :class:`ServiceGraph` instance.

    Returns
    -------
    GraphStats
    """
    n = graph.n_services
    m = graph.n_edges

    # Density
    max_edges = n * (n - 1) if n > 1 else 1
    density = m / max_edges if max_edges > 0 else 0.0

    # Degree distribution (out-degree)
    out_degree: dict[str, int] = {s: 0 for s in graph.services}
    for edge in graph.edges:
        if edge.source in out_degree:
            out_degree[edge.source] += 1

    degrees = list(out_degree.values())
    avg_deg = float(np.mean(degrees)) if degrees else 0.0
    max_deg = max(degrees) if degrees else 0

    # Connected components (weakly connected = treat as undirected)
    neighbors = _build_undirected_neighbors(graph)
    components = _weak_components(graph.services, neighbors)
    n_components = len(components)
    largest_component_size = max((len(component) for component in components), default=0)
    diameter = _graph_diameter(components, neighbors)

    # Edge weight stats
    total_vol = sum(e.volume for e in graph.edges)
    mean_lat = (
        float(np.mean([e.latency_median_s for e in graph.edges]))
        if graph.edges
        else 0.0
    )
    mean_err = (
        float(np.mean([e.error_rate for e in graph.edges]))
        if graph.edges
        else 0.0
    )

    return GraphStats(
        n_nodes=n,
        n_edges=m,
        density=density,
        avg_degree=avg_deg,
        max_degree=max_deg,
        n_connected_components=n_components,
        diameter=diameter,
        largest_component_size=largest_component_size,
        total_volume=total_vol,
        mean_latency=mean_lat,
        mean_error_rate=mean_err,
        timestamp=graph.timestamp,
    )


def _build_undirected_neighbors(graph: ServiceGraph) -> dict[str, set[str]]:
    """Build undirected neighborhood map from directed edges."""
    neighbors: dict[str, set[str]] = {service: set() for service in graph.services}
    for edge in graph.edges:
        if edge.source in neighbors and edge.target in neighbors:
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
    return neighbors


def _weak_components(
    services: list[str],
    neighbors: dict[str, set[str]],
) -> list[set[str]]:
    """Return weakly connected components from an undirected adjacency map."""
    visited: set[str] = set()
    components: list[set[str]] = []

    for service in services:
        if service in visited:
            continue

        component: set[str] = set()
        queue: deque[str] = deque([service])
        visited.add(service)

        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def _component_diameter(component: set[str], neighbors: dict[str, set[str]]) -> int:
    """Compute diameter of one connected component via BFS from each node."""
    if len(component) <= 1:
        return 0

    diameter = 0
    for source in component:
        distances: dict[str, int] = {source: 0}
        queue: deque[str] = deque([source])

        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if neighbor in component and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        diameter = max(diameter, max(distances.values(), default=0))

    return diameter


def _graph_diameter(
    components: list[set[str]],
    neighbors: dict[str, set[str]],
) -> int:
    """Compute diameter on the largest weakly connected component."""
    if not components:
        return 0

    largest = max(components, key=len)
    return _component_diameter(largest, neighbors)


def stats_to_dict(stats: GraphStats) -> dict[str, float]:
    """Convert GraphStats to a flat dict (for CSV/DataFrame export).

    Returns
    -------
    dict
        Keys match GraphStats field names.
    """
    return {
        "timestamp": stats.timestamp,
        "n_nodes": float(stats.n_nodes),
        "n_edges": float(stats.n_edges),
        "density": stats.density,
        "avg_degree": stats.avg_degree,
        "max_degree": float(stats.max_degree),
        "n_connected_components": float(stats.n_connected_components),
        "diameter": float(stats.diameter),
        "largest_component_size": float(stats.largest_component_size),
        "total_volume": float(stats.total_volume),
        "mean_latency": stats.mean_latency,
        "mean_error_rate": stats.mean_error_rate,
    }
