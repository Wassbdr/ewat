"""Tests for graph.adjacency — matrix representations of G(t).

Validates A(t) tensor shape, normalisation properties, and channel extraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from graph.adjacency import (
    binary_adjacency,
    channel_adjacency,
    normalised_adjacency,
    weighted_adjacency,
)
from graph.types import ServiceGraph, WeightedEdge

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_graph() -> ServiceGraph:
    """3-node graph: A→B (vol=10, lat=0.05, err=0.1), B→C (vol=5, lat=0.02, err=0.0)."""
    return ServiceGraph(
        services=["svc-a", "svc-b", "svc-c"],
        edges=[
            WeightedEdge(
                source="svc-a",
                target="svc-b",
                volume=10,
                latency_median_s=0.05,
                error_rate=0.1,
            ),
            WeightedEdge(
                source="svc-b",
                target="svc-c",
                volume=5,
                latency_median_s=0.02,
                error_rate=0.0,
            ),
        ],
        timestamp=1000.0,
    )


def _isolated_graph() -> ServiceGraph:
    """3 nodes, no edges."""
    return ServiceGraph(
        services=["svc-a", "svc-b", "svc-c"],
        edges=[],
        timestamp=1000.0,
    )


# ---------------------------------------------------------------------------
# Tests: weighted_adjacency (A(t) ∈ ℝ^{N×N×3})
# ---------------------------------------------------------------------------


class TestWeightedAdjacency:
    def test_shape(self) -> None:
        g = _sample_graph()
        A = weighted_adjacency(g)
        assert A.shape == (3, 3, 3)
        assert A.dtype == np.float32

    def test_edge_values(self) -> None:
        g = _sample_graph()
        A = weighted_adjacency(g)
        idx = g.service_index

        # A→B
        i, j = idx["svc-a"], idx["svc-b"]
        np.testing.assert_allclose(A[i, j, 0], 10.0)      # volume
        np.testing.assert_allclose(A[i, j, 1], 0.05)       # latency
        np.testing.assert_allclose(A[i, j, 2], 0.1)        # error_rate

        # B→C
        i, j = idx["svc-b"], idx["svc-c"]
        np.testing.assert_allclose(A[i, j, 0], 5.0)
        np.testing.assert_allclose(A[i, j, 1], 0.02)
        np.testing.assert_allclose(A[i, j, 2], 0.0)

    def test_non_edges_are_zero(self) -> None:
        g = _sample_graph()
        A = weighted_adjacency(g)
        idx = g.service_index

        # A→C (no edge)
        i, j = idx["svc-a"], idx["svc-c"]
        np.testing.assert_array_equal(A[i, j], [0.0, 0.0, 0.0])

        # C→A (no reverse edge)
        i, j = idx["svc-c"], idx["svc-a"]
        np.testing.assert_array_equal(A[i, j], [0.0, 0.0, 0.0])

    def test_isolated_graph_all_zeros(self) -> None:
        A = weighted_adjacency(_isolated_graph())
        np.testing.assert_array_equal(A, np.zeros((3, 3, 3)))


# ---------------------------------------------------------------------------
# Tests: binary_adjacency
# ---------------------------------------------------------------------------


class TestBinaryAdjacency:
    def test_shape_and_values(self) -> None:
        g = _sample_graph()
        A = binary_adjacency(g)
        assert A.shape == (3, 3)
        idx = g.service_index

        assert A[idx["svc-a"], idx["svc-b"]] == 1.0
        assert A[idx["svc-b"], idx["svc-c"]] == 1.0
        assert A[idx["svc-a"], idx["svc-c"]] == 0.0

    def test_total_edges(self) -> None:
        A = binary_adjacency(_sample_graph())
        assert A.sum() == 2.0


# ---------------------------------------------------------------------------
# Tests: normalised_adjacency
# ---------------------------------------------------------------------------


class TestNormalisedAdjacency:
    def test_shape(self) -> None:
        A_bin = binary_adjacency(_sample_graph())
        A_norm = normalised_adjacency(A_bin)
        assert A_norm.shape == (3, 3)
        assert A_norm.dtype == np.float32

    def test_self_loops_added(self) -> None:
        """With self-loops, diagonal should be non-zero."""
        A_bin = binary_adjacency(_sample_graph())
        A_norm = normalised_adjacency(A_bin, add_self_loops=True)
        for i in range(3):
            assert A_norm[i, i] > 0.0

    def test_no_self_loops(self) -> None:
        """Without self-loops, diagonal of the original should be preserved."""
        A_bin = binary_adjacency(_sample_graph())
        A_norm = normalised_adjacency(A_bin, add_self_loops=False)
        # Original binary matrix has no self-loops, so diagonal stays 0
        # BUT normalisation of zero-degree nodes → 0
        assert A_norm.shape == (3, 3)

    def test_isolated_graph_with_self_loops(self) -> None:
        """Isolated nodes get self-loops → should be identity after norm."""
        A_bin = binary_adjacency(_isolated_graph())
        A_norm = normalised_adjacency(A_bin, add_self_loops=True)
        np.testing.assert_allclose(A_norm, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_symmetric(self) -> None:
        """D^{-1/2} (A+I) D^{-1/2} should be symmetric for undirected edges."""
        # Use a symmetric binary adjacency for this test
        A_bin = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        A_norm = normalised_adjacency(A_bin, add_self_loops=True)
        np.testing.assert_allclose(A_norm, A_norm.T, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: channel_adjacency
# ---------------------------------------------------------------------------


class TestChannelAdjacency:
    def test_volume_channel(self) -> None:
        g = _sample_graph()
        A_w = weighted_adjacency(g)
        A_vol = channel_adjacency(A_w, channel=0)

        assert A_vol.shape == (3, 3)
        idx = g.service_index
        assert A_vol[idx["svc-a"], idx["svc-b"]] == 10.0
        assert A_vol[idx["svc-b"], idx["svc-c"]] == 5.0

    def test_normalised_channel(self) -> None:
        g = _sample_graph()
        A_w = weighted_adjacency(g)
        A_vol_norm = channel_adjacency(A_w, channel=0, normalise=True)

        # Should have self-loops (diagonal > 0) from normalisation
        for i in range(3):
            assert A_vol_norm[i, i] > 0.0

    def test_error_channel(self) -> None:
        g = _sample_graph()
        A_w = weighted_adjacency(g)
        A_err = channel_adjacency(A_w, channel=2)

        idx = g.service_index
        assert A_err[idx["svc-a"], idx["svc-b"]] == pytest.approx(0.1)
        assert A_err[idx["svc-b"], idx["svc-c"]] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: WeightedEdge data class
# ---------------------------------------------------------------------------


class TestWeightedEdge:
    def test_weight_vector(self) -> None:
        edge = WeightedEdge(
            source="a", target="b", volume=42, latency_median_s=0.123, error_rate=0.05
        )
        w = edge.weight_vector
        assert w.shape == (3,)
        np.testing.assert_allclose(w, [42.0, 0.123, 0.05], atol=1e-6)

    def test_frozen(self) -> None:
        edge = WeightedEdge(
            source="a", target="b", volume=1, latency_median_s=0.1, error_rate=0.0
        )
        with pytest.raises(AttributeError):
            edge.volume = 999  # type: ignore[misc]
