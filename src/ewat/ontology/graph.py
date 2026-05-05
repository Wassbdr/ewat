"""Ontology graph data structures.

O = (C, R) where C = set of cluster types and R = set of typed relations.

Three relation types:
  temporal     — C_i →^{Δt,σ} C_j (temporal succession with mean lag and std)
  causal       — C_i → C_j via Transfer Entropy (TE-KSG)
  cooccurrence — C_i ↔ C_j via χ² test on scenario co-occurrence
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OntologyRelation:
    source: int           # source cluster id
    target: int           # target cluster id
    relation_type: str    # "temporal" | "causal" | "cooccurrence"
    strength: float       # TE value / χ² statistic / transition count
    p_value: float | None = None
    delta_t_mean: float | None = None  # seconds (temporal only)
    delta_t_std: float | None = None   # seconds (temporal only)
    support: int = 0                   # number of observations


@dataclass
class OntologyGraph:
    n_clusters: int
    relations: list[OntologyRelation] = field(default_factory=list)

    def add(self, rel: OntologyRelation) -> None:
        self.relations.append(rel)

    def filter_by_type(self, relation_type: str) -> list[OntologyRelation]:
        return [r for r in self.relations if r.relation_type == relation_type]

    def to_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "relations": [asdict(r) for r in self.relations],
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> OntologyGraph:
        data = json.loads(Path(path).read_text())
        rels = [OntologyRelation(**r) for r in data["relations"]]
        return cls(n_clusters=data["n_clusters"], relations=rels)

    def summary(self) -> str:
        temporal = self.filter_by_type("temporal")
        causal = self.filter_by_type("causal")
        cooccurrence = self.filter_by_type("cooccurrence")
        return (
            f"OntologyGraph(n_clusters={self.n_clusters}, "
            f"temporal={len(temporal)}, causal={len(causal)}, "
            f"cooccurrence={len(cooccurrence)})"
        )
