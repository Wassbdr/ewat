"""Adjacency matrix utilities for the service graph.

Provides helper functions to convert G(t) into various matrix
representations consumed by the STGCN encoder (Step 1).

Key representations:
- A(t) ∈ ℝ^{N×N×3}   — weighted adjacency tensor (volume, latency, error_rate)
- Â ∈ ℝ^{N×N}         — normalised adjacency with self-loops (GCN convention)
- D^{-1/2} A D^{-1/2} — symmetric normalisation for spectral graph convolution
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from graph.types import ServiceGraph


def weighted_adjacency(graph: ServiceGraph) -> npt.NDArray[np.float32]:
    """Return A(t) ∈ ℝ^{N×N×3}.

    Delegates to ``graph.adjacency_tensor()`` — this function exists as a
    convenience entry point for the module.

    Parameters
    ----------
    graph:
        A :class:`ServiceGraph` snapshot.

    Returns
    -------
    np.ndarray
        Shape (N, N, 3), float32.
    """
    return graph.adjacency_tensor()


def binary_adjacency(graph: ServiceGraph) -> npt.NDArray[np.float32]:
    """Return binary adjacency matrix.

    Parameters
    ----------
    graph:
        A :class:`ServiceGraph` snapshot.

    Returns
    -------
    np.ndarray
        Shape (N, N), float32. 1.0 where edge exists, 0.0 otherwise.
    """
    return graph.adjacency_binary()


def normalised_adjacency(
    a_bin: npt.NDArray[np.float32],
    add_self_loops: bool = True,
) -> npt.NDArray[np.float32]:
    """Symmetric normalisation: D^{-1/2} Â D^{-1/2}.

    Standard GCN preprocessing (Kipf & Welling, 2017). Used by the STGCN
    encoder for spectral graph convolution.

    Parameters
    ----------
    a_bin:
        Binary adjacency matrix of shape (N, N).
    add_self_loops:
        If True, add the identity matrix (Â = A + I) before normalising.

    Returns
    -------
    np.ndarray
        Shape (N, N), float32. Symmetrically normalised adjacency.
    """
    n = a_bin.shape[0]
    adj = a_bin.copy()

    if add_self_loops:
        adj = adj + np.eye(n, dtype=np.float32)

    # Degree matrix
    deg = np.diag(adj.sum(axis=1))

    # D^{-1/2} — handle zero-degree nodes
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = np.diag(deg) > 0
    deg_inv_sqrt[nonzero, nonzero] = 1.0 / np.sqrt(np.diag(deg)[nonzero])

    # D^{-1/2} A D^{-1/2}
    return (deg_inv_sqrt @ adj @ deg_inv_sqrt).astype(np.float32)


def channel_adjacency(
    a_weighted: npt.NDArray[np.float32],
    channel: int,
    normalise: bool = False,
) -> npt.NDArray[np.float32]:
    """Extract a single channel from A(t) ∈ ℝ^{N×N×3}.

    Parameters
    ----------
    a_weighted:
        Weighted adjacency tensor of shape (N, N, 3).
    channel:
        Channel index: 0 = volume, 1 = latency_median, 2 = error_rate.
    normalise:
        If True, apply symmetric normalisation to the extracted channel.

    Returns
    -------
    np.ndarray
        Shape (N, N), float32.
    """
    a_ch = a_weighted[:, :, channel].copy()
    if normalise:
        # Binarise first (any non-zero → 1), then normalise the structure
        a_bin = (a_ch > 0).astype(np.float32)
        return normalised_adjacency(a_bin)
    return a_ch
