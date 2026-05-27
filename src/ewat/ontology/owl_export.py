"""ABox population for the EWAT anomaly ontology.

Reads empirical artefacts (cluster manifest, signature fiches, cluster
semantics, temporal/precursor metadata) and produces RDF instances that
satisfy the TBox declared in :mod:`ewat.ontology.owl_schema`.

The exported ontology is the union of the TBox + ABox + an ``AllDifferent``
axiom over every named individual. The latter is required for HermiT to
discharge qualified cardinality restrictions (Open World Assumption +
absence of Unique Name Assumption otherwise blocks ``Composite_Anomaly``
classification).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import owlready2 as owl
import yaml

from ewat.ontology.owl_schema import DEFAULT_IRI, TBoxArtefact, build_tbox


GRID_STEP_SECONDS = 30.0  # ewat_v3 sampling cadence


@dataclass
class EmpiricalSources:
    """Bundle of file paths feeding the ABox builder."""

    cluster_manifest: Path
    fiches_dir: Path
    cluster_semantics: Path
    ontology_temporal: Path
    precursor_results: Path
    scenarios_registry: Path
    ontology_config: Path

    @classmethod
    def default(cls, root: Path) -> EmpiricalSources:
        return cls(
            cluster_manifest=root / "experiments/typing/cluster_artifacts/cluster_manifest.json",
            fiches_dir=root / "experiments/typing/fiches",
            cluster_semantics=root / "experiments/typing/cluster_semantics.json",
            ontology_temporal=root / "experiments/ontology/ontology.json",
            precursor_results=root / "experiments/precursor/results.json",
            scenarios_registry=root / "k8s/chaos-mesh/registry.yaml",
            ontology_config=root / "configs/ontology.yaml",
        )


@dataclass
class ABoxArtefact:
    """Container exposing the populated ontology + summary counts."""

    tbox: TBoxArtefact
    individuals: dict[str, Any] = field(default_factory=dict)
    n_clusters: int = 0
    n_signatures: int = 0
    n_feature_weights: int = 0
    n_services: int = 0

    def save(self, path: Path, fmt: str = "rdfxml") -> None:
        self.tbox.ontology.save(file=str(path), format=fmt)

    def save_turtle(self, path: Path) -> None:
        import rdflib

        tmp = path.with_suffix(".rdfxml.tmp")
        self.tbox.ontology.save(file=str(tmp), format="rdfxml")
        graph = rdflib.Graph()
        graph.parse(str(tmp), format="xml")
        graph.serialize(destination=str(path), format="turtle")
        tmp.unlink()


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def _scenario_targets(registry: dict) -> dict[str, list[str]]:
    """Map scenario name -> list of target services from the registry."""
    out: dict[str, list[str]] = {}
    for s in registry.get("scenarios", []):
        out[s["name"]] = list(s.get("targets", []))
    return out


def _aggregate_targets_per_cluster(
    manifest: dict[str, dict],
    scenario_targets: dict[str, list[str]],
    canonical_services: set[str],
) -> dict[int, set[str]]:
    """For each cluster, union the target services of its constituent scenarios,
    restricted to the canonical service set."""
    per_cluster: dict[int, set[str]] = {}
    for info in manifest.values():
        cid = int(info["cluster"])
        scenario = info["scenario"]
        targets = scenario_targets.get(scenario, [])
        bucket = per_cluster.setdefault(cid, set())
        for t in targets:
            if t in canonical_services:
                bucket.add(t)
    return per_cluster


def _scenario_distribution_per_cluster(
    manifest: dict[str, dict],
) -> dict[int, Counter]:
    """Empirical scenario counts per cluster id."""
    per_cluster: dict[int, Counter] = {}
    for info in manifest.values():
        cid = int(info["cluster"])
        per_cluster.setdefault(cid, Counter())[info["scenario"]] += 1
    return per_cluster


def _self_loop_durations(temporal: dict) -> dict[int, float]:
    """For each cluster, extract delta_t_mean from its self-loop temporal
    relation (Ci → Ci). Returns seconds.
    """
    out: dict[int, float] = {}
    for rel in temporal.get("relations", []):
        if rel.get("relation_type") != "temporal":
            continue
        if rel["source"] == rel["target"]:
            out[int(rel["source"])] = float(rel.get("delta_t_mean") or 0.0)
    return out


def _lead_times_seconds(precursor: dict) -> dict[int, float]:
    """k_optimal (in steps) → seconds via GRID_STEP_SECONDS."""
    k_opt = precursor.get("k_optimal", {})
    return {int(cid): float(k) * GRID_STEP_SECONDS for cid, k in k_opt.items()}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_abox(
    sources: EmpiricalSources,
    tbox: TBoxArtefact | None = None,
    iri: str = DEFAULT_IRI,
) -> ABoxArtefact:
    """Populate the ABox from empirical artefacts.

    Returns an :class:`ABoxArtefact` that owns the merged TBox+ABox ontology.
    """
    if tbox is None:
        tbox = build_tbox(iri=iri)

    onto = tbox.ontology
    C = tbox.classes
    P = tbox.object_properties
    D = tbox.data_properties

    # --- Load all artefacts -------------------------------------------------
    manifest = _load_json(sources.cluster_manifest)
    semantics = _load_json(sources.cluster_semantics)
    temporal = _load_json(sources.ontology_temporal)
    precursor = _load_json(sources.precursor_results)
    ontology_cfg = _load_yaml(sources.ontology_config)
    registry = _load_yaml(sources.scenarios_registry)

    canonical_services: list[str] = list(ontology_cfg["services"])
    scenario_to_class: dict[str, str] = ontology_cfg["scenario_to_class"]
    scenario_targets = _scenario_targets(registry)

    cluster_targets = _aggregate_targets_per_cluster(
        manifest, scenario_targets, set(canonical_services),
    )
    scenario_dist = _scenario_distribution_per_cluster(manifest)
    self_loop_durations = _self_loop_durations(temporal)
    lead_times = _lead_times_seconds(precursor)

    individuals: dict[str, Any] = {}

    with onto:
        # --- Service individuals -------------------------------------------
        for svc in canonical_services:
            iri_name = f"service_{svc.replace('-', '_')}"
            individuals[iri_name] = C["Service"](iri_name)

        # --- One EmpiricalCluster + one Anomaly + one Signature per cluster
        n_signatures = 0
        n_feature_weights = 0
        n_clusters = int(semantics["n_clusters"])

        for cid_str, sem in semantics["clusters"].items():
            cid = int(cid_str)
            dominant_scenario: str = sem["dominant_scenario"]
            class_name = scenario_to_class.get(dominant_scenario)
            if class_name is None:
                raise ValueError(
                    f"scenario {dominant_scenario!r} (cluster {cid}) is missing "
                    f"from configs/ontology.yaml scenario_to_class"
                )
            anomaly_class = C[class_name]

            # EmpiricalCluster individual
            ec_name = f"cluster_{cid}"
            ec = C["EmpiricalCluster"](ec_name)
            individuals[ec_name] = ec

            # Anomaly individual (typed by literature class)
            an_name = f"anomaly_cluster_{cid}"
            anomaly = anomaly_class(an_name)
            individuals[an_name] = anomaly
            anomaly.observedIn = [ec]

            # Signature + FeatureWeight reified instances
            sig_name = f"signature_cluster_{cid}"
            sig = C["Signature"](sig_name)
            individuals[sig_name] = sig
            anomaly.hasSignature = [sig]
            n_signatures += 1

            fiche_path = sources.fiches_dir / f"cluster_{cid}.json"
            if not fiche_path.exists():
                raise FileNotFoundError(f"missing fiche: {fiche_path}")
            fiche = _load_json(fiche_path)
            feature_importance: dict[str, float] = fiche["feature_importance"]
            for feat_name, weight in feature_importance.items():
                if weight <= 0.0:
                    continue  # skip zero-weight features (queue_depth, retry_rate, ...)
                fw_name = f"fw_cluster_{cid}_{feat_name}"
                fw = C["FeatureWeight"](fw_name)
                fw.featureName = feat_name
                fw.weightValue = float(weight)
                sig.hasFeatureWeight.append(fw)
                individuals[fw_name] = fw
                n_feature_weights += 1

            # affects = canonical target services aggregated from scenarios
            for svc in cluster_targets.get(cid, set()):
                svc_indiv = individuals[f"service_{svc.replace('-', '_')}"]
                anomaly.affects.append(svc_indiv)

            # Data properties
            duration = self_loop_durations.get(cid, 0.0)
            if duration > 0.0:
                anomaly.temporalDuration = duration
            lead = lead_times.get(cid, 0.0)
            if lead > 0.0:
                anomaly.temporalLeadTime = lead

        # --- AllDifferent over every named individual ---------------------
        # Required so HermiT can count distinct hasComponent targets when
        # discharging Composite_Anomaly ≡ Anomaly ⊓ hasComponent.min(2, Anomaly).
        owl.AllDifferent(list(individuals.values()))

    return ABoxArtefact(
        tbox=tbox,
        individuals=individuals,
        n_clusters=n_clusters,
        n_signatures=n_signatures,
        n_feature_weights=n_feature_weights,
        n_services=len(canonical_services),
    )


def export_ontology(
    sources: EmpiricalSources,
    output_dir: Path,
    iri: str = DEFAULT_IRI,
) -> dict[str, Path]:
    """Build the full TBox + ABox and save in RDF/XML and Turtle.

    Returns ``{format: path}`` of the written artefacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artefact = build_abox(sources, iri=iri)
    rdfxml_path = output_dir / "ewat_instances.owl"
    ttl_path = output_dir / "ewat_instances.ttl"
    artefact.save(rdfxml_path, fmt="rdfxml")
    artefact.save_turtle(ttl_path)
    return {"rdfxml": rdfxml_path, "turtle": ttl_path}
