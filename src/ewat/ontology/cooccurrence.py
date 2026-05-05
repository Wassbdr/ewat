"""Co-occurrence relations via χ² test.

For each pair of cluster types (i, j), tests whether they tend to appear in the
same scenario more often than expected by chance.

Co-occurrence is defined at the scenario level: a scenario S "contains" cluster
type C_k if at least one of its episodes was assigned to cluster k.

χ² test (1 degree of freedom) with Yates continuity correction compares the
observed co-occurrence count to the expected count under independence.

Reference: Agresti (2002) — Categorical Data Analysis.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import chi2

from ewat.ontology.graph import OntologyRelation


def compute_cooccurrence_relations(
    cluster_manifest: dict[str, dict],
    n_clusters: int,
    p_threshold: float = 0.05,
    min_cooccurrences: int = 2,
) -> list[OntologyRelation]:
    """Discover co-occurrence relations between cluster type pairs.

    Parameters
    ----------
    cluster_manifest:   {episode_id → {"cluster": int, "scenario": str, ...}}
    n_clusters:         Total number of cluster types.
    p_threshold:        Maximum χ² p-value to emit a relation.
    min_cooccurrences:  Minimum observed co-occurrence count.

    Returns
    -------
    List of OntologyRelation with relation_type="cooccurrence".
    """
    # Build scenario → set of cluster types
    scenario_clusters: dict[str, set[int]] = {}
    for info in cluster_manifest.values():
        sc = info["scenario"]
        c = int(info["cluster"])
        scenario_clusters.setdefault(sc, set()).add(c)

    scenarios = list(scenario_clusters.values())
    n_scenarios = len(scenarios)

    if n_scenarios < 2:
        return []

    # Count per-cluster scenario membership
    cluster_count = np.zeros(n_clusters, dtype=int)
    for cs in scenarios:
        for c in cs:
            if c < n_clusters:
                cluster_count[c] += 1

    # Co-occurrence matrix
    co_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for cs in scenarios:
        members = [c for c in cs if c < n_clusters]
        for ci, cj in combinations(members, 2):
            co_matrix[ci, cj] += 1
            co_matrix[cj, ci] += 1

    relations: list[OntologyRelation] = []

    for i, j in combinations(range(n_clusters), 2):
        observed = co_matrix[i, j]
        if observed < min_cooccurrences:
            continue

        n_i = cluster_count[i]
        n_j = cluster_count[j]
        expected = n_i * n_j / n_scenarios

        if expected <= 0:
            continue

        # Yates-corrected χ²
        chi2_stat = (max(0.0, abs(observed - expected) - 0.5) ** 2) / expected
        p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))

        if p_val < p_threshold:
            relations.append(OntologyRelation(
                source=i, target=j,
                relation_type="cooccurrence",
                strength=float(chi2_stat),
                p_value=p_val,
                support=observed,
            ))

    return relations
