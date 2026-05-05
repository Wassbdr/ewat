"""AlertAssembler — assemblage de la sortie finale du pipeline EWAT.

Prend en entrée :
- SiameseTyper (déjà chargé)
- PrecursorClassifiers par type (dict {cluster_id → PrecursorClassifier})
- k_optimal par type (dict {cluster_id → k*})
- fiches par cluster (dict {cluster_id → dict})

Et émet la liste des Alert(t) à partir du signal courant.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ewat.alerts.alert import Alert
from ewat.precursor.model import PrecursorClassifier
from ewat.typing.siamese import SiameseTyper

STEP_SECONDS = 30.0


class AlertAssembler:
    """Assemble les alertes à partir du signal courant.

    Parameters
    ----------
    typer:
        SiameseTyper chargé depuis le checkpoint de typage.
    classifiers:
        Dictionnaire {cluster_id → PrecursorClassifier} chargé depuis les
        checkpoints de précurseurs.
    k_optimal:
        Dictionnaire {cluster_id → k*} — horizon optimal par type.
    fiches:
        Dictionnaire {cluster_id → dict} — fiche descriptive par type.
    threshold:
        Seuil de probabilité minimum pour émettre une alerte (défaut 0.5).
    scaler:
        StandardScaler fitté sur les données train. Si fourni, appliqué au
        signal avant l'encodage (même normalisation que lors de l'entraînement).
    device:
        Device PyTorch pour l'inférence de l'encodeur.
    """

    def __init__(
        self,
        typer: SiameseTyper,
        classifiers: dict[int, PrecursorClassifier],
        k_optimal: dict[int, int],
        fiches: dict[int, dict[str, Any]],
        threshold: float = 0.5,
        scaler: StandardScaler | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.typer = typer
        self.classifiers = classifiers
        self.k_optimal = k_optimal
        self.fiches = fiches
        self.threshold = threshold
        self.scaler = scaler
        self.device = device or torch.device("cpu")
        self.typer = self.typer.to(self.device).eval()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_experiment_dirs(
        cls,
        typing_dir: Path,
        encoder_dir: Path,
        precursor_dir: Path,
        threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> AlertAssembler:
        """Charge l'assembleur depuis les répertoires d'expériences standard."""
        from ewat.encoder.stgcn import STGCNEncoder

        enc_ckpt = torch.load(
            encoder_dir / "checkpoints" / "best_encoder.pt",
            map_location="cpu",
            weights_only=False,
        )
        encoder = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
        encoder.load_state_dict(enc_ckpt["encoder_state"])

        typer_ckpt = torch.load(
            typing_dir / "checkpoints" / "best_siamese.pt",
            map_location="cpu",
            weights_only=False,
        )
        typer = SiameseTyper(encoder, d_proj=32)
        typer.load_state_dict(typer_ckpt["typer_state"])

        results = json.loads((precursor_dir / "results.json").read_text())
        k_optimal: dict[int, int] = {int(k): int(v) for k, v in results["k_optimal"].items()}
        n_clusters: int = results["n_clusters"]

        classifiers: dict[int, PrecursorClassifier] = {}
        for c in range(n_clusters):
            k_opt = k_optimal[c]
            ckpt_path = precursor_dir / "checkpoints" / f"classifier_type{c}_k{k_opt}.pkl"
            if ckpt_path.exists():
                classifiers[c] = PrecursorClassifier.load(ckpt_path)

        fiches: dict[int, dict[str, Any]] = {}
        fiches_dir = typing_dir / "fiches"
        if fiches_dir.exists():
            for fiche_path in fiches_dir.glob("cluster_*.json"):
                try:
                    cid = int(fiche_path.stem.split("_")[1])
                    fiches[cid] = json.loads(fiche_path.read_text())
                except (ValueError, IndexError):
                    pass

        # Load StandardScaler (same normalization as training)
        scaler: StandardScaler | None = None
        default_scaler = str(enc_ckpt.get("scaler_path", str(encoder_dir / "scaler.pkl")))
        scaler_path = Path(default_scaler)
        if scaler_path.exists():
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)

        return cls(
            typer=typer,
            classifiers=classifiers,
            k_optimal=k_optimal,
            fiches=fiches,
            threshold=threshold,
            scaler=scaler,
            device=device or torch.device("cpu"),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        signal: np.ndarray,
        adjacency: np.ndarray,
        timestamp: float = 0.0,
        episode_id: str = "",
    ) -> list[Alert]:
        """Émet les alertes depuis le signal courant.

        Parameters
        ----------
        signal:
            Signal courant S(t) de forme (T, N, 17) — T timesteps disponibles.
        adjacency:
            Matrice d'adjacence A(t) de forme (T, N, N, 3).
        timestamp:
            Horodatage du pas courant (secondes depuis epoch).
        episode_id:
            Identifiant de l'épisode (optionnel).

        Returns
        -------
        list[Alert]
            Alertes émises, triées par probabilité décroissante, filtrées par
            le seuil.
        """
        # Apply same normalisation as EpisodeDataset
        signal = signal.astype(np.float32)
        if self.scaler is not None:
            t_len, n_nodes, d = signal.shape
            flat = signal.reshape(-1, d)
            flat = np.where(np.isnan(flat), 0.0, flat)
            flat = self.scaler.transform(flat).astype(np.float32)
            signal = flat.reshape(t_len, n_nodes, d)
        else:
            signal = np.nan_to_num(signal, nan=0.0)
        adjacency = np.nan_to_num(adjacency.astype(np.float32), nan=0.0)

        t_total = signal.shape[0]
        alerts: list[Alert] = []

        for cluster_id, clf in self.classifiers.items():
            k = self.k_optimal.get(cluster_id, 2)
            actual_k = min(k, t_total)

            sig_window = signal[-actual_k:]  # (actual_k, N, 17)
            adj_window = adjacency[-actual_k:]  # (actual_k, N, N, 3)

            if actual_k < k:
                pad_t = k - actual_k
                n_nodes = signal.shape[1]
                sig_window = np.concatenate(
                    [np.zeros((pad_t, n_nodes, 17), dtype=np.float32), sig_window], axis=0
                )
                adj_window = np.concatenate(
                    [np.zeros((pad_t, n_nodes, n_nodes, 3), dtype=np.float32), adj_window],
                    axis=0,
                )

            sig_t = torch.from_numpy(sig_window).float().unsqueeze(0).to(self.device)
            adj_t = torch.from_numpy(adj_window).float().unsqueeze(0).to(self.device)
            z = self.typer.embed(sig_t, adj_t).cpu().numpy()  # (1, d_proj)

            proba = clf.predict_proba(z)  # (1, n_clusters)
            p_i = float(proba[0, cluster_id])

            if p_i >= self.threshold:
                alerts.append(
                    Alert(
                        cluster_id=cluster_id,
                        probability=p_i,
                        horizon_steps=k,
                        horizon_seconds=k * STEP_SECONDS,
                        fiche=self.fiches.get(cluster_id, {}),
                        timestamp=timestamp,
                        episode_id=episode_id,
                    )
                )

        alerts.sort(key=lambda a: a.probability, reverse=True)
        return alerts
