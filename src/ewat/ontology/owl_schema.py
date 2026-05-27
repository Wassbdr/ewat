"""OWL TBox declaration for the EWAT anomaly ontology.

The TBox (terminological box) defines the classes, object properties, data
properties, and equivalence axioms. Instances (ABox) are populated by
``owl_export.py`` from empirical artefacts (cluster manifest, signature
fiches, service-level causal graph).

Design notes
------------
1. ``causes`` is declared transitive + asymmetric + irreflexive so that
   HermiT can compute the transitive closure of the causal graph
   automatically. Important for queries like "all anomalies causally
   downstream of Memory_Saturation".
2. ``Composite_Anomaly`` is defined by an *equivalence* axiom on cardinality
   of ``hasComponent`` (≥ 2). HermiT will therefore classify any individual
   with ≥ 2 ``hasComponent`` assertions as a Composite_Anomaly, even if not
   explicitly typed.
3. ``CascadingFailure`` is a Composite_Anomaly whose components form at
   least one ``precedes`` chain — again classified by HermiT.
4. The ``propagatesThrough some Service ⊑ affects some Service`` subsumption
   means that asserting propagation automatically asserts impact, halving
   the work of the ABox builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import owlready2 as owl

from ewat.ontology.literature_taxonomy import (
    CLASS_LITERATURE,
    PROPERTY_LITERATURE,
    validate_property_mapping,
    validate_taxonomy_mapping,
)


DEFAULT_IRI = "http://ewat.devoteam.com/ontology"


@dataclass
class TBoxArtefact:
    """Container exposing the loaded ontology + a few useful handles."""

    ontology: owl.Ontology
    iri: str
    classes: dict[str, Any]
    object_properties: dict[str, Any]
    data_properties: dict[str, Any]

    def save(self, path: Path, fmt: str = "rdfxml") -> None:
        """Serialize the TBox to disk.

        Parameters
        ----------
        path:
            Output file path.
        fmt:
            One of ``"rdfxml"``, ``"ntriples"``. Turtle output is produced
            via :func:`save_turtle` (rdflib round-trip).
        """
        self.ontology.save(file=str(path), format=fmt)

    def save_turtle(self, path: Path) -> None:
        """Save as Turtle by routing through rdflib (owlready2 cannot emit
        Turtle directly)."""
        import rdflib  # local import to keep owlready2-only paths light

        tmp = path.with_suffix(".rdfxml.tmp")
        self.ontology.save(file=str(tmp), format="rdfxml")
        graph = rdflib.Graph()
        graph.parse(str(tmp), format="xml")
        graph.serialize(destination=str(path), format="turtle")
        tmp.unlink()


def _annotate_with_literature(cls: Any, key: str) -> None:
    """Attach literature references to a class or property as rdfs:comment."""
    refs = CLASS_LITERATURE.get(key) or [PROPERTY_LITERATURE.get(key)]
    refs = [r for r in refs if r is not None]
    if not refs:
        return
    cls.comment = [ref.to_annotation() for ref in refs]


def build_tbox(iri: str = DEFAULT_IRI) -> TBoxArtefact:
    """Build the EWAT anomaly ontology TBox in a fresh owlready2 world.

    Returns
    -------
    TBoxArtefact
        Handle exposing the ontology and dictionaries of named entities so
        callers can populate the ABox without re-querying the world.
    """
    world = owl.World()
    onto = world.get_ontology(iri)

    classes: dict[str, Any] = {}
    obj_props: dict[str, Any] = {}
    data_props: dict[str, Any] = {}

    with onto:
        # ── Top-level classes ────────────────────────────────────────────
        class Anomaly(owl.Thing):
            pass

        class Signature(owl.Thing):
            pass

        class Service(owl.Thing):
            pass

        class EmpiricalCluster(owl.Thing):
            pass

        class FeatureWeight(owl.Thing):
            pass

        class RecoveryPattern(owl.Thing):
            pass

        class Mitigation(owl.Thing):
            pass

        # ── Anomaly taxonomy (is-a) ──────────────────────────────────────
        class Resource_Anomaly(Anomaly):
            pass

        class Saturation(Resource_Anomaly):
            pass

        class CPU_Saturation(Saturation):
            pass

        class Memory_Saturation(Saturation):
            pass

        class Network_Saturation(Saturation):
            pass

        class Disk_Saturation(Saturation):
            pass

        class HardExhaustion(Saturation):
            pass

        class Liveness_Anomaly(Anomaly):
            pass

        class Functional_Anomaly(Anomaly):
            pass

        class Latency_Anomaly(Anomaly):
            pass

        class Network_Anomaly(Anomaly):
            pass

        class Configuration_Anomaly(Anomaly):
            pass

        class Deployment_Anomaly(Anomaly):
            pass

        class Composite_Anomaly(Anomaly):
            pass

        class Drift_With_Anomaly(Composite_Anomaly):
            pass

        class CascadingFailure(Composite_Anomaly):
            pass

        # ── Drift taxonomy (orthogonal to Anomaly) ───────────────────────
        class Drift(owl.Thing):
            pass

        class Benign_Drift(Drift):
            pass

        class Scaling_Drift(Benign_Drift):
            pass

        class Deployment_Drift(Benign_Drift):
            pass

        class Configuration_Drift(Benign_Drift):
            pass

        class Traffic_Drift(Benign_Drift):
            pass

        # ── Object properties ────────────────────────────────────────────
        class hasSignature(owl.ObjectProperty):
            domain = [Anomaly]
            range = [Signature]

        class affects(owl.ObjectProperty):
            domain = [Anomaly]
            range = [Service]

        class observedIn(owl.ObjectProperty):
            domain = [Anomaly]
            range = [EmpiricalCluster]

        class causes(owl.ObjectProperty, owl.TransitiveProperty,
                     owl.IrreflexiveProperty, owl.AsymmetricProperty):
            domain = [Anomaly]
            range = [Anomaly]

        class isCausedBy(owl.ObjectProperty):
            inverse_property = causes

        class precedes(owl.ObjectProperty, owl.TransitiveProperty,
                       owl.AsymmetricProperty):
            domain = [Anomaly]
            range = [Anomaly]

        class coOccursWith(owl.ObjectProperty, owl.SymmetricProperty,
                           owl.ReflexiveProperty):
            domain = [Anomaly]
            range = [Anomaly]

        class propagatesThrough(owl.ObjectProperty):
            domain = [Anomaly]
            range = [Service]

        class hasComponent(owl.ObjectProperty, owl.TransitiveProperty):
            domain = [Composite_Anomaly]
            range = [Anomaly]

        class hasFeatureWeight(owl.ObjectProperty):
            domain = [Signature]
            range = [FeatureWeight]

        class mitigatedBy(owl.ObjectProperty):
            domain = [Anomaly]
            range = [Mitigation]

        # Subsumption axiom: propagation implies impact
        propagatesThrough.is_a.append(affects)

        # ── Data properties ──────────────────────────────────────────────
        class featureName(owl.DataProperty, owl.FunctionalProperty):
            domain = [FeatureWeight]
            range = [str]

        class weightValue(owl.DataProperty, owl.FunctionalProperty):
            domain = [FeatureWeight]
            range = [float]

        class temporalDuration(owl.DataProperty, owl.FunctionalProperty):
            domain = [Anomaly]
            range = [float]

        class temporalLeadTime(owl.DataProperty, owl.FunctionalProperty):
            domain = [Anomaly]
            range = [float]

        class severity(owl.DataProperty, owl.FunctionalProperty):
            domain = [Anomaly]
            range = [str]

        class confidence(owl.DataProperty):
            range = [float]

        # ── Equivalence axioms (drive HermiT classification) ─────────────
        Composite_Anomaly.equivalent_to.append(
            Anomaly & (hasComponent.min(2, Anomaly))
        )
        CascadingFailure.equivalent_to.append(
            Composite_Anomaly
            & (hasComponent.some(Anomaly & precedes.some(Anomaly)))
        )

    # Collect named entities for ABox builders.
    for name in [
        "Anomaly", "Signature", "Service", "EmpiricalCluster", "FeatureWeight",
        "RecoveryPattern", "Mitigation",
        "Resource_Anomaly", "Saturation", "CPU_Saturation", "Memory_Saturation",
        "Network_Saturation", "Disk_Saturation", "HardExhaustion",
        "Liveness_Anomaly", "Functional_Anomaly", "Latency_Anomaly",
        "Network_Anomaly", "Configuration_Anomaly", "Deployment_Anomaly",
        "Composite_Anomaly", "Drift_With_Anomaly", "CascadingFailure",
        "Drift", "Benign_Drift", "Scaling_Drift", "Deployment_Drift",
        "Configuration_Drift", "Traffic_Drift",
    ]:
        entity = onto[name]
        if entity is None:
            raise RuntimeError(f"class {name!r} not present in ontology")
        classes[name] = entity
        _annotate_with_literature(entity, name)

    for name in [
        "hasSignature", "affects", "observedIn", "causes", "isCausedBy",
        "precedes", "coOccursWith", "propagatesThrough", "hasComponent",
        "hasFeatureWeight", "mitigatedBy",
    ]:
        entity = onto[name]
        if entity is None:
            raise RuntimeError(f"object property {name!r} not present")
        obj_props[name] = entity
        if name in PROPERTY_LITERATURE:
            _annotate_with_literature(entity, name)

    for name in [
        "featureName", "weightValue", "temporalDuration", "temporalLeadTime",
        "severity", "confidence",
    ]:
        entity = onto[name]
        if entity is None:
            raise RuntimeError(f"data property {name!r} not present")
        data_props[name] = entity

    # Cross-check that every literature-tracked class is actually in the TBox.
    missing_class = validate_taxonomy_mapping(list(classes))
    missing_prop = validate_property_mapping(list(obj_props))
    if missing_class:
        raise RuntimeError(
            f"classes lack literature annotation: {missing_class}"
        )
    if missing_prop:
        raise RuntimeError(
            f"properties lack literature annotation: {missing_prop}"
        )

    return TBoxArtefact(
        ontology=onto,
        iri=iri,
        classes=classes,
        object_properties=obj_props,
        data_properties=data_props,
    )


def export_tbox(output_dir: Path, iri: str = DEFAULT_IRI) -> dict[str, Path]:
    """Build and save the TBox in both RDF/XML and Turtle.

    Returns a dict ``{format: path}`` of the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artefact = build_tbox(iri=iri)
    rdfxml_path = output_dir / "taxonomy.owl"
    ttl_path = output_dir / "taxonomy.ttl"
    artefact.save(rdfxml_path, fmt="rdfxml")
    artefact.save_turtle(ttl_path)
    return {"rdfxml": rdfxml_path, "turtle": ttl_path}
