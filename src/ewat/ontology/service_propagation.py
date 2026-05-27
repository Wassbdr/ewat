"""Service-level propagation enrichment for the EWAT ABox.

Reads the existing service-level Transfer Entropy graph (124 relations on
ewat_v3, ``experiments/ontology/service_causal.json``) and adds
``propagatesThrough`` triplets to the per-cluster anomaly individuals.

Specificity filtering
---------------------
Some service edges are ubiquitous (``load-generator → frontend`` appears in
8 / 8 active clusters — it is the structural traffic graph of Online
Boutique, not a propagation pattern specific to any anomaly type). To avoid
polluting the ontology with these tautologies, an edge ``(src → tgt)`` is
kept for cluster ``C`` only if it is present in **at most**
``ubiquity_threshold * n_active_clusters`` clusters (default 0.5 → keep
edges appearing in less than half of active clusters).

The non-filtered edges are exposed as ``propagatesThrough`` targets on the
anomaly individual. Because the TBox declares
``propagatesThrough some Service ⊑ affects some Service`` the
``affects`` triplets are implied transitively (we do not duplicate them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ewat.ontology.owl_export import ABoxArtefact


DEFAULT_UBIQUITY_THRESHOLD = 0.5  # keep edges present in < 50% of active clusters


@dataclass(frozen=True)
class ServiceEdge:
    """One source → target propagation edge with its TE value."""

    source_service: str
    target_service: str
    te_value: float
    p_value: float
    support: int


@dataclass
class PropagationReport:
    """Summary of the enrichment pass."""

    n_input_edges: int
    n_after_specificity_filter: int
    n_clusters_enriched: int
    dropped_ubiquitous_pairs: list[tuple[str, str]]


def _normalize_service(name: str) -> str:
    """Map ``service_causal.json`` service names to canonical OWL individual
    keys (replaces hyphens with underscores)."""
    return name.replace("-", "_")


def _load_service_causal(path: Path) -> dict[int, list[ServiceEdge]]:
    """Read the JSON and group significant edges per cluster id."""
    raw = json.loads(Path(path).read_text())
    out: dict[int, list[ServiceEdge]] = {}
    for cid_str, edges in raw.get("clusters", {}).items():
        cid = int(cid_str)
        out[cid] = [
            ServiceEdge(
                source_service=e["source_service"],
                target_service=e["target_service"],
                te_value=float(e["te_value"]),
                p_value=float(e["p_value"]),
                support=int(e["support"]),
            )
            for e in edges
        ]
    return out


def _count_pair_ubiquity(
    per_cluster: dict[int, list[ServiceEdge]],
) -> dict[tuple[str, str], int]:
    """For each (src, tgt) pair, count in how many distinct clusters it
    appears at least once."""
    counts: dict[tuple[str, str], int] = {}
    for cid, edges in per_cluster.items():
        seen_pairs = {(e.source_service, e.target_service) for e in edges}
        for pair in seen_pairs:
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _filter_specific(
    per_cluster: dict[int, list[ServiceEdge]],
    ubiquity_threshold: float,
) -> tuple[dict[int, list[ServiceEdge]], list[tuple[str, str]]]:
    """Apply the specificity filter and return (kept_edges, dropped_pairs)."""
    pair_counts = _count_pair_ubiquity(per_cluster)
    n_active = sum(1 for edges in per_cluster.values() if edges)
    if n_active == 0:
        return per_cluster, []
    # Strict fractional comparison: a pair appearing in strictly more than
    # ``ubiquity_threshold`` of the active clusters is considered
    # ubiquitous and dropped. Avoids round-to-even edge cases.
    dropped: set[tuple[str, str]] = {
        pair for pair, count in pair_counts.items()
        if (count / n_active) > ubiquity_threshold
    }
    kept: dict[int, list[ServiceEdge]] = {
        cid: [e for e in edges
              if (e.source_service, e.target_service) not in dropped]
        for cid, edges in per_cluster.items()
    }
    return kept, sorted(dropped)


def enrich_with_service_propagation(
    abox: ABoxArtefact,
    service_causal_path: Path,
    ubiquity_threshold: float = DEFAULT_UBIQUITY_THRESHOLD,
) -> PropagationReport:
    """Mutate the ABox by adding ``propagatesThrough`` triplets.

    Parameters
    ----------
    abox:
        Output of :func:`ewat.ontology.owl_export.build_abox`.
    service_causal_path:
        Path to ``experiments/ontology/service_causal.json``.
    ubiquity_threshold:
        Edges present in more than this fraction of active clusters are
        dropped as non-specific (default 0.5).

    Returns
    -------
    PropagationReport
        Summary of how many edges survived the filter and which pairs were
        dropped.
    """
    per_cluster = _load_service_causal(service_causal_path)
    n_input = sum(len(edges) for edges in per_cluster.values())

    kept_per_cluster, dropped_pairs = _filter_specific(
        per_cluster, ubiquity_threshold,
    )
    n_after_filter = sum(len(edges) for edges in kept_per_cluster.values())

    onto = abox.tbox.ontology
    n_clusters_enriched = 0
    with onto:
        for cid, edges in kept_per_cluster.items():
            if not edges:
                continue
            an_name = f"anomaly_cluster_{cid}"
            anomaly = abox.individuals.get(an_name)
            if anomaly is None:
                continue
            unique_targets: list[Any] = []
            seen: set[str] = set()
            for e in edges:
                key = _normalize_service(e.target_service)
                svc_indiv = abox.individuals.get(f"service_{key}")
                if svc_indiv is None:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                unique_targets.append(svc_indiv)
            if unique_targets:
                for target in unique_targets:
                    if target not in anomaly.propagatesThrough:
                        anomaly.propagatesThrough.append(target)
                n_clusters_enriched += 1

    return PropagationReport(
        n_input_edges=n_input,
        n_after_specificity_filter=n_after_filter,
        n_clusters_enriched=n_clusters_enriched,
        dropped_ubiquitous_pairs=dropped_pairs,
    )
