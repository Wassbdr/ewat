"""HermiT-based reasoning utilities for the EWAT anomaly ontology.

Wraps owlready2's reasoner integration with two practical helpers:

- :func:`run_reasoner` — calls HermiT, returns the wall-clock duration and
  the list of inconsistent classes.
- :func:`extract_entailment_diff` — compares the world's class assertions
  on individuals before and after reasoning to report which triples were
  *materialised* by the reasoner (the "ROI" of the equivalence axioms).

Caveats
-------
owlready2's HermiT integration materialises class taxonomy entailments but
does not always propagate them into individuals' ``.is_a`` attribute,
especially when classification depends on qualified cardinality
restrictions. The :func:`extract_entailment_diff` helper therefore queries
via SPARQL on the post-reasoning world, which is reliable for both class
hierarchy and property entailments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import owlready2 as owl

from ewat.ontology.queries import PREFIXES


@dataclass
class ReasoningReport:
    """Outcome of running HermiT on a populated ontology."""

    elapsed_s: float
    inconsistent_classes: list[str]
    n_individuals: int
    n_class_triples_before: int
    n_class_triples_after: int

    @property
    def consistent(self) -> bool:
        return not self.inconsistent_classes

    @property
    def materialised_class_triples(self) -> int:
        return self.n_class_triples_after - self.n_class_triples_before


def _count_class_triples(onto: owl.Ontology) -> int:
    """Count distinct ``individual a Class`` triples on named individuals."""
    n = 0
    for ind in onto.individuals():
        n += len(list(ind.is_a))
    return n


def run_reasoner(
    onto: owl.Ontology,
    *,
    reasoner: str = "hermit",
    infer_property_values: bool = True,
    debug: int = 0,
) -> ReasoningReport:
    """Run the configured reasoner on the ontology.

    Parameters
    ----------
    onto:
        owlready2 Ontology (TBox + ABox loaded).
    reasoner:
        ``"hermit"`` (default) or ``"pellet"``.
    infer_property_values:
        Forwarded to owlready2; enables data/object property entailments.
    debug:
        owlready2 debug level (0 = silent).
    """
    before = _count_class_triples(onto)
    n_individuals = len(list(onto.individuals()))
    t0 = time.time()
    if reasoner == "hermit":
        with onto:
            owl.sync_reasoner_hermit(
                infer_property_values=infer_property_values, debug=debug,
            )
    elif reasoner == "pellet":
        with onto:
            owl.sync_reasoner_pellet(
                infer_property_values=infer_property_values,
                infer_data_property_values=infer_property_values,
                debug=debug,
            )
    else:
        raise ValueError(f"unknown reasoner: {reasoner!r}")
    elapsed = time.time() - t0
    after = _count_class_triples(onto)
    inconsistent = [c.name for c in onto.inconsistent_classes()]
    return ReasoningReport(
        elapsed_s=elapsed,
        inconsistent_classes=inconsistent,
        n_individuals=n_individuals,
        n_class_triples_before=before,
        n_class_triples_after=after,
    )


@dataclass
class EntailmentDiff:
    """SPARQL-derived snapshot of the entailment closure.

    ``n_typing_triples`` counts ``individual rdf:type Class`` triples on
    named individuals after reasoning (closed-world projection of HermiT's
    class assertions, including transitive subclass closure).
    ``n_causal_pairs`` counts ``(s, ewat:causes, t)`` triples.
    """

    n_typing_triples: int
    n_causal_pairs: int
    n_propagation_triples: int
    composite_anomaly_instances: list[str] = field(default_factory=list)


def extract_entailment_diff(
    world: owl.World,
    ontology_iri_hash: str = "http://ewat.devoteam.com/ontology#",
) -> EntailmentDiff:
    """Snapshot key entailments via SPARQL after :func:`run_reasoner`.

    Parameters
    ----------
    world:
        owlready2 World whose ontology has just been reasoned on.
    ontology_iri_hash:
        Base IRI with the trailing ``#`` (needed for typed SPARQL queries).
    """
    # Class assertions (rdf:type) on named individuals
    typing_rows = list(world.sparql(PREFIXES + """
        SELECT (COUNT(*) AS ?n) WHERE {
            ?ind rdf:type ?cls .
            ?cls rdfs:subClassOf* ewat:Anomaly .
        }
    """))
    n_typing = int(typing_rows[0][0]) if typing_rows else 0

    causal_rows = list(world.sparql(PREFIXES + """
        SELECT (COUNT(*) AS ?n) WHERE {
            ?s ewat:causes ?t .
        }
    """))
    n_causal = int(causal_rows[0][0]) if causal_rows else 0

    prop_rows = list(world.sparql(PREFIXES + """
        SELECT (COUNT(*) AS ?n) WHERE {
            ?s ewat:propagatesThrough ?svc .
        }
    """))
    n_prop = int(prop_rows[0][0]) if prop_rows else 0

    comp_rows = list(world.sparql(PREFIXES + """
        SELECT DISTINCT ?anomaly WHERE {
            ?anomaly rdf:type ?subClass .
            ?subClass rdfs:subClassOf* ewat:Composite_Anomaly .
        }
    """))
    composites = [
        r[0].name if hasattr(r[0], "name") else str(r[0])
        for r in comp_rows
    ]

    return EntailmentDiff(
        n_typing_triples=n_typing,
        n_causal_pairs=n_causal,
        n_propagation_triples=n_prop,
        composite_anomaly_instances=composites,
    )


# ---------------------------------------------------------------------------
# Causes/co-occurrence loaders (Phase 4-5 bridge)
# ---------------------------------------------------------------------------


def add_causal_relations_to_abox(
    abox_individuals: dict[str, Any],
    relations: list[Any],
    onto: owl.Ontology,
) -> int:
    """Attach ``OntologyRelation(relation_type='causal')`` to anomaly
    individuals in the ABox.

    Returns the number of triples emitted.
    """
    n = 0
    with onto:
        for rel in relations:
            if rel.relation_type != "causal":
                continue
            src = abox_individuals.get(f"anomaly_cluster_{rel.source}")
            tgt = abox_individuals.get(f"anomaly_cluster_{rel.target}")
            if src is None or tgt is None:
                continue
            if tgt not in src.causes:
                src.causes.append(tgt)
                n += 1
    return n


def add_cooccurrence_relations_to_abox(
    abox_individuals: dict[str, Any],
    relations: list[Any],
    onto: owl.Ontology,
) -> int:
    """Same as :func:`add_causal_relations_to_abox` for co-occurrence."""
    n = 0
    with onto:
        for rel in relations:
            if rel.relation_type != "cooccurrence":
                continue
            a = abox_individuals.get(f"anomaly_cluster_{rel.source}")
            b = abox_individuals.get(f"anomaly_cluster_{rel.target}")
            if a is None or b is None:
                continue
            if b not in a.coOccursWith:
                a.coOccursWith.append(b)
                n += 1
    return n


def add_temporal_relations_to_abox(
    abox_individuals: dict[str, Any],
    relations: list[Any],
    onto: owl.Ontology,
) -> int:
    """Attach ``OntologyRelation(relation_type='temporal')`` cross-cluster
    transitions as ``precedes`` triples (auto-transitions C_i → C_i are
    excluded as they encode injection duration, not precedence)."""
    n = 0
    with onto:
        for rel in relations:
            if rel.relation_type != "temporal":
                continue
            if rel.source == rel.target:
                continue  # skip self-loops (trivial)
            src = abox_individuals.get(f"anomaly_cluster_{rel.source}")
            tgt = abox_individuals.get(f"anomaly_cluster_{rel.target}")
            if src is None or tgt is None:
                continue
            if tgt not in src.precedes:
                src.precedes.append(tgt)
                n += 1
    return n
