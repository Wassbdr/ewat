"""Garde-fou CI pour les tests d'ontologie.

Plusieurs tests de ce dossier (ABox empirique, raisonnement, propagation de
services, causalité, temporel) chargent des artefacts expérimentaux sous
``experiments/`` (cluster_manifest, fiches, service_causal, ontology.json…).
Ces artefacts sont **gitignorés** (non versionnés) → absents d'un checkout propre
en CI, ce qui faisait échouer ces tests sur ``FileNotFoundError``.

On skip proprement tout le dossier quand le manifest de clusters — pivot dont
dépendent ces artefacts — n'est pas présent. En local (artefacts présents) les
tests s'exécutent normalement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLUSTER_MANIFEST = (
    _REPO_ROOT / "experiments" / "typing" / "cluster_artifacts" / "cluster_manifest.json"
)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if _CLUSTER_MANIFEST.exists():
        return
    skip = pytest.mark.skip(
        reason="artefacts experiments/ absents (gitignorés, non versionnés) — skip en CI"
    )
    here = Path(__file__).parent.resolve()
    needs_artifacts = {
        "test_owl_export.py",
        "test_reasoning.py",
        "test_service_propagation.py",
        "test_causal.py",
        "test_temporal.py",
    }
    for item in items:
        p = Path(str(item.fspath)).resolve()
        if here in p.parents and p.name in needs_artifacts:
            item.add_marker(skip)
