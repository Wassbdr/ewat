"""Alert dataclass — sortie finale du pipeline EWAT.

Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Alert:
    """Un avertissement émis par le pipeline EWAT pour un type d'anomalie donné.

    Attributes
    ----------
    cluster_id:
        Indice du type d'anomalie C_i dans l'ontologie.
    probability:
        p̂_i(t) ∈ [0, 1] — probabilité que l'anomalie de type C_i survienne
        dans les prochains k*_i pas de temps.
    horizon_steps:
        k*_i — horizon optimal en nombre de pas (1 pas = 30 s).
    horizon_seconds:
        k*_i × 30 — horizon en secondes.
    fiche:
        Fiche descriptive du cluster C_i (chargée depuis JSON) ou dict vide
        si non disponible.
    timestamp:
        Horodatage de l'observation courante (secondes depuis epoch).
    episode_id:
        Identifiant de l'épisode courant (optionnel, utile pour le debug).
    """

    cluster_id: int
    probability: float
    horizon_steps: int
    horizon_seconds: float
    fiche: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    episode_id: str = ""
    drift_flag: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")
        if self.horizon_steps < 1:
            raise ValueError(f"horizon_steps must be >= 1, got {self.horizon_steps}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "probability": self.probability,
            "horizon_steps": self.horizon_steps,
            "horizon_seconds": self.horizon_seconds,
            "fiche": self.fiche,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "drift_flag": self.drift_flag,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Alert:
        return cls(
            cluster_id=int(d["cluster_id"]),
            probability=float(d["probability"]),
            horizon_steps=int(d["horizon_steps"]),
            horizon_seconds=float(d["horizon_seconds"]),
            fiche=dict(d.get("fiche", {})),
            timestamp=float(d.get("timestamp", 0.0)),
            episode_id=str(d.get("episode_id", "")),
            drift_flag=bool(d.get("drift_flag", False)),
        )
