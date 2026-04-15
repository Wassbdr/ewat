"""Graph serialization — save/load ServiceGraph to/from disk.

Uses numpy .npz for the adjacency tensor and JSON for metadata.
Designed for bulk dataset storage in data/raw/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from graph.types import ServiceGraph, WeightedEdge

logger = logging.getLogger(__name__)


def save_graph(graph: ServiceGraph, path: str | Path) -> None:
    """Save a ServiceGraph to disk.

    Creates two files:
        - ``{path}.npz`` — adjacency tensor A(t) ∈ ℝ^{N×N×3}
        - ``{path}.json`` — services list, edges metadata, timestamp

    Parameters
    ----------
    graph:
        The graph to save.
    path:
        Base path (without extension). E.g. ``data/raw/run/graph_0001``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save adjacency tensor
    adj = graph.adjacency_tensor()
    np.savez_compressed(f"{path}.npz", adjacency=adj)

    # Save metadata
    meta = {
        "services": graph.services,
        "timestamp": graph.timestamp,
        "n_edges": graph.n_edges,
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "volume": e.volume,
                "latency_median_s": e.latency_median_s,
                "error_rate": e.error_rate,
            }
            for e in graph.edges
        ],
    }
    with open(f"{path}.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_graph(path: str | Path) -> ServiceGraph:
    """Load a ServiceGraph from disk.

    Expects:
        - ``{path}.npz`` — adjacency tensor
        - ``{path}.json`` — metadata

    Parameters
    ----------
    path:
        Base path (without extension).

    Returns
    -------
    ServiceGraph
    """
    path = Path(path)

    with open(f"{path}.json") as f:
        meta = json.load(f)

    edges = [
        WeightedEdge(
            source=e["source"],
            target=e["target"],
            volume=e["volume"],
            latency_median_s=e["latency_median_s"],
            error_rate=e["error_rate"],
        )
        for e in meta["edges"]
    ]

    return ServiceGraph(
        services=meta["services"],
        edges=edges,
        timestamp=meta["timestamp"],
    )


def save_graph_sequence(
    graphs: list[ServiceGraph],
    directory: str | Path,
    prefix: str = "graph",
) -> None:
    """Save a temporal sequence of graphs.

    Creates files: ``{directory}/{prefix}_0000.{npz,json}``, etc.

    Parameters
    ----------
    graphs:
        List of ServiceGraph snapshots.
    directory:
        Output directory.
    prefix:
        Filename prefix.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for i, graph in enumerate(graphs):
        save_graph(graph, directory / f"{prefix}_{i:04d}")

    logger.info("Saved %d graphs to %s", len(graphs), directory)


def load_graph_sequence(
    directory: str | Path,
    prefix: str = "graph",
) -> list[ServiceGraph]:
    """Load a temporal sequence of graphs.

    Parameters
    ----------
    directory:
        Directory containing graph files.
    prefix:
        Filename prefix used during save.

    Returns
    -------
    list[ServiceGraph]
        Sorted by filename index.
    """
    directory = Path(directory)
    json_files = sorted(directory.glob(f"{prefix}_*.json"))

    graphs = []
    for json_file in json_files:
        base = str(json_file).removesuffix(".json")
        graphs.append(load_graph(base))

    logger.info("Loaded %d graphs from %s", len(graphs), directory)
    return graphs


def save_adjacency_tensor_bulk(
    graphs: list[ServiceGraph],
    path: str | Path,
) -> None:
    """Save all graphs as a single bulk adjacency tensor.

    Saves A ∈ ℝ^{T×N×N×3} as a single .npz file, plus services.json.
    More efficient for training data loading than per-graph files.

    Parameters
    ----------
    graphs:
        List of T ServiceGraph snapshots. All must share the same service list.
    path:
        Output path (without extension).
    """
    if not graphs:
        raise ValueError("Cannot save empty graph sequence")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    services = graphs[0].services
    n = len(services)
    t = len(graphs)

    bulk = np.zeros((t, n, n, 3), dtype=np.float32)
    timestamps = np.zeros(t, dtype=np.float64)

    for i, graph in enumerate(graphs):
        if graph.services != services:
            raise ValueError(
                f"Graph {i} has different services: {graph.services} vs {services}"
            )
        bulk[i] = graph.adjacency_tensor()
        timestamps[i] = graph.timestamp

    np.savez_compressed(
        f"{path}.npz",
        adjacency=bulk,
        timestamps=timestamps,
    )

    with open(f"{path}_services.json", "w") as f:
        json.dump(services, f)

    logger.info(
        "Saved bulk adjacency tensor shape %s to %s.npz",
        bulk.shape,
        path,
    )
