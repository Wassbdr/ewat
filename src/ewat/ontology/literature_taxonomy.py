"""Literature-grounded taxonomy mapping for the EWAT anomaly ontology.

Each OWL class in the EWAT ontology is anchored to one or more peer-reviewed
references so that the taxonomy is defensible and traceable. The mapping is a
plain Python dict consumed at TBox build time (``owl_schema.build_tbox``) and
exported as ``rdfs:isDefinedBy`` annotations into the OWL artifact.

Primary sources
---------------
- Soldani & Brogi (2022) — "Anomalies, failures, and recovery plans for
  microservices: A survey", ACM Computing Surveys, 55(7):1–35.
  Used for: anomaly families (resource, network, deployment, configuration,
  composite) and recovery patterns.
- Fu et al. (2025) — "RCA in microservices: a survey", IEEE Transactions on
  Services Computing, in press.
  Used for: causal propagation taxonomy, fault propagation graphs.
- Gregg (2013) — "Systems Performance: Enterprise and the Cloud", Prentice
  Hall. USE method (Utilization, Saturation, Errors).
  Used for: ``Saturation`` and its sub-classes (CPU/Memory/Network/Disk).
- Aniello, Bonomi, Lombardi, Zelli & Baldoni (2014) — "An architecture for
  automatic scaling of replicated services", NETYS.
  Used for: ``CascadingFailure`` propagation patterns.
- Kubernetes documentation — liveness/readiness probes, OOMKill semantics.
  Used for: ``Liveness_Anomaly``, ``HardExhaustion``.

The dict below is the single source of truth. If a class is added to the OWL
schema without an entry here, ``validate_taxonomy_mapping`` will raise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiteratureRef:
    """A single bibliographic reference attached to an OWL class."""

    citation: str
    year: int
    section: str = ""
    note: str = ""

    def to_annotation(self) -> str:
        """Render as a one-line annotation suitable for ``rdfs:comment``."""
        base = f"{self.citation} ({self.year})"
        if self.section:
            base += f", {self.section}"
        if self.note:
            base += f" — {self.note}"
        return base


# Mapping: OWL class name -> list of LiteratureRef
# Keep in sync with owl_schema.py — validated at TBox load time.
CLASS_LITERATURE: dict[str, list[LiteratureRef]] = {
    # ── Top-level ─────────────────────────────────────────────────────────
    "Anomaly": [
        LiteratureRef(
            "Soldani & Brogi", 2022, "§2",
            "umbrella concept for any observable deviation from nominal regime",
        ),
    ],
    "Signature": [
        LiteratureRef(
            "Fu et al.", 2025, "§3.2",
            "observable feature pattern that characterises an anomaly type",
        ),
    ],
    "Service": [
        LiteratureRef(
            "Soldani & Brogi", 2022, "§1",
            "Kubernetes Deployment/Service abstraction",
        ),
    ],
    "EmpiricalCluster": [
        LiteratureRef(
            "EWAT internal", 2026, "Étape 2",
            "siamois + agglomerative clustering output (K=10 on ewat_v3)",
        ),
    ],
    "FeatureWeight": [
        LiteratureRef(
            "Fu et al.", 2025, "§4.1",
            "reified pair (feature_name, weight) — permutation importance",
        ),
    ],
    "RecoveryPattern": [
        LiteratureRef(
            "Soldani & Brogi", 2022, "§5",
            "recovery plan archetypes (restart, rollback, scale, isolate)",
        ),
    ],
    "Mitigation": [
        LiteratureRef(
            "Soldani & Brogi", 2022, "§5.2",
            "concrete corrective action attached to a recovery pattern",
        ),
    ],
    # ── Resource anomalies (USE method) ───────────────────────────────────
    "Resource_Anomaly": [
        LiteratureRef("Gregg", 2013, "ch. 2", "USE method top-level family"),
    ],
    "Saturation": [
        LiteratureRef(
            "Gregg", 2013, "ch. 2",
            "queueing or rejection because demand exceeds capacity",
        ),
    ],
    "CPU_Saturation": [
        LiteratureRef("Gregg", 2013, "ch. 6", "run-queue length, CPU contention"),
        LiteratureRef("Soldani & Brogi", 2022, "§3.1", "CPU starvation pattern"),
    ],
    "Memory_Saturation": [
        LiteratureRef("Gregg", 2013, "ch. 7", "page scanning, swap-in/out"),
        LiteratureRef("Soldani & Brogi", 2022, "§3.1", "memory pressure pattern"),
    ],
    "Network_Saturation": [
        LiteratureRef("Gregg", 2013, "ch. 10", "interface utilisation > 70%"),
        LiteratureRef(
            "Soldani & Brogi", 2022, "§3.1",
            "noisy-neighbour effect on shared network plane",
        ),
    ],
    "Disk_Saturation": [
        LiteratureRef("Gregg", 2013, "ch. 9", "I/O queue length, await metric"),
    ],
    "HardExhaustion": [
        LiteratureRef(
            "Kubernetes docs", 2024, "OOMKill",
            "container terminated by kernel when cgroup memory limit hit",
        ),
        LiteratureRef(
            "Soldani & Brogi", 2022, "§3.1",
            "resource leak archetype leading to exhaustion",
        ),
    ],
    # ── Liveness / functional / latency ───────────────────────────────────
    "Liveness_Anomaly": [
        LiteratureRef(
            "Kubernetes docs", 2024, "Liveness probes",
            "probe-based liveness failure (pod restart loop)",
        ),
        LiteratureRef("Soldani & Brogi", 2022, "§3.2", "crash failure family"),
    ],
    "Functional_Anomaly": [
        LiteratureRef(
            "Soldani & Brogi", 2022, "§3.3",
            "application-level logical error (incorrect output)",
        ),
    ],
    "Latency_Anomaly": [
        LiteratureRef(
            "Fu et al.", 2025, "§3.1",
            "slow response without explicit error (gray failure)",
        ),
    ],
    "Network_Anomaly": [
        LiteratureRef("Soldani & Brogi", 2022, "§3.1", "network partition / loss"),
    ],
    "Configuration_Anomaly": [
        LiteratureRef(
            "Soldani & Brogi", 2022, "§3.4",
            "misconfiguration (probes, resource limits, routing)",
        ),
    ],
    "Deployment_Anomaly": [
        LiteratureRef(
            "Soldani & Brogi", 2022, "§3.4",
            "failed rollout, image pull error, faulty deploy",
        ),
    ],
    # ── Composite ─────────────────────────────────────────────────────────
    "Composite_Anomaly": [
        LiteratureRef(
            "Aniello et al.", 2014, "§3",
            "cascading or co-occurring failures across services",
        ),
        LiteratureRef("Fu et al.", 2025, "§5", "fault propagation graphs"),
    ],
    "Drift_With_Anomaly": [
        LiteratureRef(
            "EWAT formalisation", 2026, "θ_{drift∩anomaly}",
            "regime where a benign drift co-occurs with a true anomaly",
        ),
    ],
    "CascadingFailure": [
        LiteratureRef("Aniello et al.", 2014, "§3", "temporal cascade A → B → C"),
    ],
    # ── Drift taxonomy (orthogonal) ───────────────────────────────────────
    "Drift": [
        LiteratureRef(
            "EWAT formalisation", 2026, "§Régimes",
            "distribution shift orthogonal to anomaly (benign by default)",
        ),
    ],
    "Benign_Drift": [
        LiteratureRef(
            "Hinder et al.", 2024, "§2",
            "concept drift in unsupervised data streams (no failure attached)",
        ),
    ],
    "Scaling_Drift": [
        LiteratureRef(
            "EWAT formalisation", 2026, "drift_scale_up",
            "horizontal pod autoscaler scaling event",
        ),
    ],
    "Deployment_Drift": [
        LiteratureRef(
            "EWAT formalisation", 2026, "drift_rolling_deploy",
            "rolling deployment (successful)",
        ),
    ],
    "Configuration_Drift": [
        LiteratureRef(
            "EWAT formalisation", 2026, "drift_config_change",
            "ConfigMap or Deployment patch (benign)",
        ),
    ],
    "Traffic_Drift": [
        LiteratureRef(
            "EWAT formalisation", 2026, "drift_traffic_ramp",
            "load-generator ramp-up (workload distribution shift)",
        ),
    ],
}


# Mapping: object property name -> single LiteratureRef
PROPERTY_LITERATURE: dict[str, LiteratureRef] = {
    "hasSignature": LiteratureRef(
        "Fu et al.", 2025, "§3.2", "anomaly → observable feature pattern",
    ),
    "affects": LiteratureRef(
        "Soldani & Brogi", 2022, "§4", "anomaly → impacted service",
    ),
    "observedIn": LiteratureRef(
        "EWAT internal", 2026, "Étape 2b",
        "anomaly concept → empirical cluster (traceability)",
    ),
    "causes": LiteratureRef(
        "Fu et al.", 2025, "§5",
        "directed causal link (Transfer Entropy KSG, BH-FDR p<0.05)",
    ),
    "isCausedBy": LiteratureRef(
        "Fu et al.", 2025, "§5", "inverse of causes (auto-materialised by HermiT)",
    ),
    "precedes": LiteratureRef(
        "Aniello et al.", 2014, "§3", "temporal precedence within an episode",
    ),
    "coOccursWith": LiteratureRef(
        "Soldani & Brogi", 2022, "§4.2",
        "symmetric co-occurrence within an episode (χ² significant)",
    ),
    "propagatesThrough": LiteratureRef(
        "Fu et al.", 2025, "§5.1",
        "anomaly propagates along a service-to-service edge",
    ),
    "hasComponent": LiteratureRef(
        "Aniello et al.", 2014, "§3",
        "composite anomaly is made of two or more sub-anomalies",
    ),
    "hasFeatureWeight": LiteratureRef(
        "EWAT internal", 2026, "Étape 2b",
        "signature → reified feature weight (permutation importance)",
    ),
    "mitigatedBy": LiteratureRef(
        "Soldani & Brogi", 2022, "§5.2",
        "anomaly → applicable mitigation action",
    ),
}


def validate_taxonomy_mapping(declared_classes: list[str]) -> list[str]:
    """Return the list of OWL class names declared in the schema but missing
    from :data:`CLASS_LITERATURE`. An empty list means the mapping is complete.
    """
    missing = [c for c in declared_classes if c not in CLASS_LITERATURE]
    return missing


def validate_property_mapping(declared_properties: list[str]) -> list[str]:
    """Same as :func:`validate_taxonomy_mapping` for object properties."""
    missing = [p for p in declared_properties if p not in PROPERTY_LITERATURE]
    return missing
