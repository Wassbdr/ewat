"""Temporal relations between cluster types.

Discovers C_i →^{Δt,σ} C_j relations: anomaly type C_i tends to be followed
by C_j after Δt ± σ seconds in the experimental timeline.

Algorithm
---------
1. Load episode start timestamps from labels.parquet (min timestamp per episode).
2. Sort all episodes by start time.
3. Iterate consecutive episode pairs within max_delta_seconds.
4. For each transition (cluster_i → cluster_j), accumulate Δt values.
5. Emit OntologyRelation if support ≥ min_support.

This captures temporal succession in the experimental sequence — useful for
understanding which anomaly types tend to precede others in production.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ewat.ontology.graph import OntologyRelation


def _episode_start_time(features_root: Path, episode_id: str) -> float:
    """Return the earliest timestamp in labels.parquet for this episode."""
    parquet_path = features_root / episode_id / "labels.parquet"
    df = pd.read_parquet(parquet_path, columns=["timestamp"])
    return float(df["timestamp"].min())


def compute_temporal_relations(
    cluster_manifest: dict[str, dict],
    features_root: Path,
    min_support: int = 3,
    max_delta_seconds: float = 7200.0,
) -> list[OntologyRelation]:
    """Discover temporal succession relations between cluster types.

    Parameters
    ----------
    cluster_manifest:  {episode_id → {"cluster": int, "split": str, "scenario": str}}
    features_root:     Root of feature store (contains episode subdirs).
    min_support:       Minimum number of observed transitions to emit a relation.
    max_delta_seconds: Max time gap (s) for two episodes to be "consecutive".

    Returns
    -------
    List of OntologyRelation with relation_type="temporal".
    """
    features_root = Path(features_root)

    # Build (episode_id, cluster, start_time) list
    records: list[tuple[float, str, int]] = []
    for ep_id, info in cluster_manifest.items():
        try:
            t0 = _episode_start_time(features_root, ep_id)
            records.append((t0, ep_id, int(info["cluster"])))
        except Exception:
            continue

    # Sort by start time
    records.sort(key=lambda x: x[0])

    # Accumulate transition Δt values
    # transition_deltas[i][j] = list of Δt values (seconds)
    transition_deltas: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for k in range(len(records) - 1):
        t0, _, c0 = records[k]
        t1, _, c1 = records[k + 1]
        delta = t1 - t0
        if delta <= max_delta_seconds:
            transition_deltas[c0][c1].append(delta)

    # Emit relations
    relations: list[OntologyRelation] = []
    for src, targets in transition_deltas.items():
        for tgt, deltas in targets.items():
            if len(deltas) < min_support:
                continue
            arr = np.array(deltas)
            relations.append(OntologyRelation(
                source=src,
                target=tgt,
                relation_type="temporal",
                strength=float(len(deltas)),    # support count as strength
                delta_t_mean=float(arr.mean()),
                delta_t_std=float(arr.std()),
                support=len(deltas),
            ))

    return relations
