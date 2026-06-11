"""Empreintes de cohérence train/inférence (M15, audit 2026-06).

L'AlertAssembler charge le scaler depuis un pickle séparé des checkpoints :
rien ne garantissait que le scaler appliqué à l'inférence soit celui de
l'entraînement (re-fit, fichier remplacé, mauvais répertoire). Les scripts
d'entraînement enregistrent ``scaler_fingerprint(scaler)`` dans le checkpoint
(clé ``scaler_sha256``) ; l'assembleur recompare au chargement.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.preprocessing import StandardScaler


def scaler_fingerprint(scaler: StandardScaler) -> str:
    """SHA-256 déterministe des paramètres appris d'un StandardScaler.

    Couvre ``mean_`` et ``scale_`` (les deux seuls paramètres appliqués par
    ``transform``). Deux scalers fittés sur les mêmes données donnent la même
    empreinte ; tout re-fit sur des données différentes la change.
    """
    h = hashlib.sha256()
    for attr in ("mean_", "scale_"):
        value = getattr(scaler, attr, None)
        if value is None:
            h.update(b"<unfitted>")
            continue
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        h.update(arr.tobytes())
    return h.hexdigest()
