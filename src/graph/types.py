"""Service graph data types for G(t) = (V, E(t), w_E(t)).

V = Kubernetes services (not pods). |V| = N constant.
Edges are weighted: w_E(t) : E(t) → ℝ³, e_ij(t) ↦ (volume, latence_med, taux_erreur).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class WeightedEdge:
    """A directed, weighted edge between two services.

    Parameters
    ----------
    source:
        Source service name (caller).
    target:
        Target service name (callee).
    volume:
        Number of calls from source to target in the observation window.
    latency_median_s:
        Median latency of calls in seconds.
    error_rate:
        Fraction of calls that resulted in an error (0..1).
    """

    source: str
    target: str
    volume: int
    latency_median_s: float
    error_rate: float

    @property
    def weight_vector(self) -> npt.NDArray[np.float32]:
        """Return ℝ³ weight vector (volume, latency_med, error_rate)."""
        return np.array(
            [float(self.volume), self.latency_median_s, self.error_rate],
            dtype=np.float32,
        )


@dataclass
class ServiceGraph:
    """G(t) — a snapshot of the service dependency graph.

    Attributes
    ----------
    services:
        Sorted list of N service names (graph nodes = V).
    edges:
        List of weighted directed edges observed in the time window.
    timestamp:
        Unix timestamp (seconds) of the snapshot.
    """

    services: list[str]
    edges: list[WeightedEdge] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def n_services(self) -> int:
        """Number of nodes |V|."""
        return len(self.services)

    @property
    def n_edges(self) -> int:
        """Number of observed edges |E(t)|."""
        return len(self.edges)

    @property
    def service_index(self) -> dict[str, int]:
        """Mapping service_name → row/column index."""
        return {s: i for i, s in enumerate(self.services)}

    def adjacency_tensor(self) -> npt.NDArray[np.float32]:
        """Build A(t) ∈ ℝ^{N×N×3}.

        Channels:
            0 — volume (call count)
            1 — median latency (seconds)
            2 — error rate (fraction)

        Returns
        -------
        np.ndarray
            Shape (N, N, 3), float32. Zero where no edge exists.
        """
        n = self.n_services
        adj = np.zeros((n, n, 3), dtype=np.float32)
        idx = self.service_index
        for edge in self.edges:
            i = idx.get(edge.source)
            j = idx.get(edge.target)
            if i is not None and j is not None:
                adj[i, j] = edge.weight_vector
        return adj

    def adjacency_binary(self) -> npt.NDArray[np.float32]:
        """Build binary adjacency matrix (1 where edge exists, 0 otherwise).

        Returns
        -------
        np.ndarray
            Shape (N, N), float32.
        """
        n = self.n_services
        adj = np.zeros((n, n), dtype=np.float32)
        idx = self.service_index
        for edge in self.edges:
            i = idx.get(edge.source)
            j = idx.get(edge.target)
            if i is not None and j is not None:
                adj[i, j] = 1.0
        return adj
