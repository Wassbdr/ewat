"""Graph validation — detect structural anomalies in G(t).

Checks for issues that would compromise dataset quality:
- Isolated nodes (services with zero edges)
- Missing services (expected but not in graph)
- Phantom edges (edges to/from unknown services)
- Degenerate graphs (too sparse or too dense)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from graph.types import ServiceGraph

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Results from validating a ServiceGraph.

    Attributes
    ----------
    is_valid:
        True if no critical issues found.
    isolated_nodes:
        Services with zero in-degree and zero out-degree.
    low_volume_edges:
        Edges with suspiciously low call volume (< threshold).
    phantom_edges:
        Edges that reference unknown source/target services.
    missing_services:
        Expected services not present in the graph.
    warnings:
        Non-critical issues (informational).
    errors:
        Critical issues that should block dataset inclusion.
    """

    is_valid: bool = True
    isolated_nodes: list[str] = field(default_factory=list)
    low_volume_edges: list[tuple[str, str, int]] = field(default_factory=list)
    phantom_edges: list[tuple[str, str]] = field(default_factory=list)
    missing_services: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [f"Valid: {self.is_valid}"]
        if self.isolated_nodes:
            lines.append(f"Isolated nodes ({len(self.isolated_nodes)}): {self.isolated_nodes}")
        if self.low_volume_edges:
            lines.append(f"Low volume edges: {len(self.low_volume_edges)}")
        if self.phantom_edges:
            lines.append(f"Phantom edges ({len(self.phantom_edges)}): {self.phantom_edges}")
        if self.missing_services:
            lines.append(f"Missing services: {self.missing_services}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


def validate_graph(
    graph: ServiceGraph,
    expected_services: list[str] | None = None,
    min_edges: int = 1,
    min_volume: int = 1,
) -> ValidationReport:
    """Validate a ServiceGraph for dataset quality.

    Parameters
    ----------
    graph:
        The graph to validate.
    expected_services:
        If provided, check that all these services are present.
    min_edges:
        Minimum number of edges for a valid graph.
    min_volume:
        Minimum call volume for an edge to not be flagged.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    # Check: at least some services
    if graph.n_services == 0:
        report.is_valid = False
        report.errors.append("Graph has 0 services")
        return report

    # Check: duplicate services
    if len(set(graph.services)) != graph.n_services:
        report.is_valid = False
        report.errors.append("Graph contains duplicate service names")

    svc_set = set(graph.services)

    # Check: phantom edges (references to unknown nodes)
    for edge in graph.edges:
        if edge.source not in svc_set or edge.target not in svc_set:
            report.phantom_edges.append((edge.source, edge.target))
    if report.phantom_edges:
        report.is_valid = False
        report.errors.append(
            f"Graph has {len(report.phantom_edges)} phantom edges"
        )

    # Check: minimum edges
    if graph.n_edges < min_edges:
        report.is_valid = False
        report.errors.append(
            f"Graph has {graph.n_edges} edges, minimum is {min_edges}"
        )

    # Check: isolated nodes (no incoming or outgoing edges)
    sources = {e.source for e in graph.edges if e.source in svc_set}
    targets = {e.target for e in graph.edges if e.target in svc_set}
    connected = sources | targets
    for svc in graph.services:
        if svc not in connected:
            report.isolated_nodes.append(svc)

    if len(report.isolated_nodes) == graph.n_services and graph.n_services > 1:
        report.is_valid = False
        report.errors.append("All nodes are isolated (no edges at all)")

    # Check: expected services present
    if expected_services is not None:
        for expected in expected_services:
            if expected not in svc_set:
                report.missing_services.append(expected)
        if report.missing_services:
            report.warnings.append(
                f"{len(report.missing_services)} expected services missing"
            )

    # Check: low volume edges
    for edge in graph.edges:
        if edge.volume < min_volume:
            report.low_volume_edges.append(
                (edge.source, edge.target, edge.volume)
            )

    if report.low_volume_edges:
        report.warnings.append(
            f"{len(report.low_volume_edges)} edges with volume < {min_volume}"
        )

    return report


def validate_graph_sequence(
    graphs: list[ServiceGraph],
    expected_services: list[str] | None = None,
    max_nan_ratio: float = 0.5,
) -> list[ValidationReport]:
    """Validate a temporal sequence of graphs.

    Parameters
    ----------
    graphs:
        List of graphs G(t_1), ..., G(t_T).
    expected_services:
        Expected service list (should be consistent across time).
    max_nan_ratio:
        Maximum fraction of empty graphs allowed.

    Returns
    -------
    list[ValidationReport]
        One report per graph.
    """
    reports = []
    n_empty = 0
    reference_services: list[str] | None = None

    for graph in graphs:
        report = validate_graph(graph, expected_services=expected_services)
        if reference_services is None:
            reference_services = graph.services
        elif graph.services != reference_services:
            report.is_valid = False
            report.errors.append("Service list differs from first graph in sequence")
        if graph.n_edges == 0:
            n_empty += 1
        reports.append(report)

    if graphs and n_empty / len(graphs) > max_nan_ratio:
        logger.warning(
            "Graph sequence: %d/%d graphs are empty (%.0f%% > %.0f%% threshold)",
            n_empty,
            len(graphs),
            100.0 * n_empty / len(graphs),
            100.0 * max_nan_ratio,
        )

    return reports
