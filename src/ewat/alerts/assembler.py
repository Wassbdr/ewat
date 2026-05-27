"""AlertAssembler — assemblage de la sortie finale du pipeline EWAT.

Prend en entrée :
- ``SiameseTyper`` (déjà chargé)
- Précurseurs par type (``dict {cluster_id → PrecursorClassifier}``)
- ``k_optimal`` par type (``dict {cluster_id → k*}``)
- Fiches par cluster (``dict {cluster_id → dict}``)

et émet la liste des ``Alert(t)`` à partir du signal courant.

Optimisations
=============

- Encodage groupé par ``k*`` : tous les classifiers partageant la même
  fenêtre sont passés en une seule forward STGCN, ce qui réduit le coût
  d'inférence de O(C) à O(|distinct k*|).
- Topologie et hyperparamètres du détecteur de drift chargés depuis les
  artefacts d'entraînement (``arch`` dans le checkpoint encodeur,
  ``drift_calibration.json``) au lieu d'être hardcodés.

Réinitialisation du DriftDetector
=================================

Le détecteur est ré-initialisé entre épisodes pour éviter qu'un état
résiduel d'un épisode précédent ne fausse les détections du suivant. La
règle est :

- Si l'appelant passe ``episode_id=None`` → reset systématique avant
  chaque appel (mode défensif, recommandé pour les simulations
  rejouables).
- Sinon → reset uniquement quand ``episode_id`` change.

L'ancien comportement (``episode_id: str = ""``) avait le défaut de ne
*jamais* re-initialiser le détecteur tant que l'appelant ne passait pas
explicitement un identifiant. Ce mode est toujours disponible si l'on
passe ``episode_id=""`` aux deux premiers appels.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ewat.alerts.alert import Alert
from ewat.drift.detector import DriftDetector
from ewat.drift.mmd import RFFKernel
from ewat.precursor.model import PrecursorClassifier
from ewat.typing.siamese import SiameseTyper

DEFAULT_STEP_SECONDS = 30.0
DEFAULT_DRIFT_EPSILON = 0.5226
DEFAULT_DRIFT_RFF_DIM = 256
DEFAULT_DRIFT_WINDOW_REF = 5
DEFAULT_DRIFT_WINDOW_CUR = 5
DEFAULT_DRIFT_POST = 3

# Sentinel: differs from any plausible string episode_id
_RESET_ALWAYS = object()


class AlertAssembler:
    """Assemble les alertes à partir du signal courant.

    Parameters
    ----------
    typer:
        ``SiameseTyper`` chargé depuis le checkpoint de typage.
    classifiers:
        ``{cluster_id → PrecursorClassifier}``.
    k_optimal:
        ``{cluster_id → k*}`` — horizon optimal par type.
    fiches:
        ``{cluster_id → dict}`` — fiche descriptive par type.
    threshold:
        Seuil de probabilité minimum pour émettre une alerte (défaut 0.5).
    scaler:
        ``StandardScaler`` fitté sur les données d'entraînement. Si fourni,
        appliqué au signal avant l'encodage (même normalisation qu'à
        l'entraînement).
    drift_detector:
        ``DriftDetector`` optionnel. Si fourni, les alertes précurseurs sont
        supprimées lorsque le détecteur indique un état DRIFT (flag=True).
    step_seconds:
        Durée d'un pas du signal en secondes (défaut 30 s). Sert à convertir
        ``k*`` en horizon temporel.
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
        drift_detector: DriftDetector | None = None,
        step_seconds: float = DEFAULT_STEP_SECONDS,
        device: torch.device | None = None,
    ) -> None:
        self.typer = typer
        self.classifiers = classifiers
        self.k_optimal = k_optimal
        self.fiches = fiches
        self.threshold = threshold
        self.scaler = scaler
        self.drift_detector = drift_detector
        self.step_seconds = float(step_seconds)
        self.device = device or torch.device("cpu")
        self.typer = self.typer.to(self.device).eval()
        self._last_episode_id: object = _RESET_ALWAYS

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
        drift_calibration_path: Path | None = None,
    ) -> AlertAssembler:
        """Charge l'assembleur depuis les répertoires d'expériences standard.

        Architecture et calibration sont lues depuis les artefacts :

        - ``encoder_dir/checkpoints/best_encoder.pt`` peut contenir une clé
          ``arch`` ; sinon les valeurs par défaut historiques (17, 6, 64, 64)
          sont utilisées avec un avertissement.
        - ``drift_calibration_path`` (ou ``encoder_dir / "drift_calibration.json"``
          par défaut) peut contenir ``epsilon_drift``, ``rff_dim``, etc.

        L'instance retournée groupe automatiquement les classifiers par ``k*``
        au moment de :meth:`predict`.
        """
        from ewat.encoder.stgcn import STGCNEncoder

        enc_ckpt = torch.load(
            encoder_dir / "checkpoints" / "best_encoder.pt",
            map_location="cpu",
            weights_only=False,
        )

        arch = enc_ckpt.get("arch") or {}
        d_feat = int(arch.get("d_feat", 17))
        n_nodes = int(arch.get("n_nodes", 6))
        d_hidden = int(arch.get("d_hidden", 64))
        d_embed = int(arch.get("d_embed", 64))
        n_gcn_layers = int(arch.get("n_gcn_layers", 2))
        tcn_kernel = int(arch.get("tcn_kernel", 3))
        tcn_layers = int(arch.get("tcn_layers", 2))
        n_adj_ch = int(arch.get("n_adj_ch", 3))

        encoder = STGCNEncoder(
            d_feat=d_feat,
            n_nodes=n_nodes,
            d_hidden=d_hidden,
            d_embed=d_embed,
            n_gcn_layers=n_gcn_layers,
            tcn_kernel=tcn_kernel,
            tcn_layers=tcn_layers,
            n_adj_ch=n_adj_ch,
        )
        encoder.load_state_dict(enc_ckpt["encoder_state"])

        typer_ckpt = torch.load(
            typing_dir / "checkpoints" / "best_siamese.pt",
            map_location="cpu",
            weights_only=False,
        )
        d_proj = int(typer_ckpt.get("d_proj", 32))
        typer = SiameseTyper(encoder, d_proj=d_proj)
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

        scaler: StandardScaler | None = None
        default_scaler = str(enc_ckpt.get("scaler_path", str(encoder_dir / "scaler.pkl")))
        scaler_path = Path(default_scaler)
        if scaler_path.exists():
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)

        # Drift calibration (loaded from JSON if available, else defaults).
        drift_cfg: dict[str, Any] = {}
        cal_path = drift_calibration_path or encoder_dir / "drift_calibration.json"
        if cal_path.exists():
            drift_cfg = json.loads(Path(cal_path).read_text())

        kernel = RFFKernel(
            rff_dim=int(drift_cfg.get("rff_dim", DEFAULT_DRIFT_RFF_DIM)),
            seed=int(drift_cfg.get("seed", 42)),
        )
        drift_detector = DriftDetector(
            kernel=kernel,
            epsilon_drift=float(drift_cfg.get("epsilon_drift", DEFAULT_DRIFT_EPSILON)),
            window_ref_size=int(drift_cfg.get("window_ref_size", DEFAULT_DRIFT_WINDOW_REF)),
            window_cur_size=int(drift_cfg.get("window_cur_size", DEFAULT_DRIFT_WINDOW_CUR)),
            post_drift_window_s=int(drift_cfg.get("post_drift_window_s", DEFAULT_DRIFT_POST)),
        )

        step_seconds = float(arch.get("step_seconds", DEFAULT_STEP_SECONDS))

        return cls(
            typer=typer,
            classifiers=classifiers,
            k_optimal=k_optimal,
            fiches=fiches,
            threshold=threshold,
            scaler=scaler,
            drift_detector=drift_detector,
            step_seconds=step_seconds,
            device=device or torch.device("cpu"),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _maybe_reset_drift(self, episode_id: str | None) -> None:
        """Reset the drift detector when the episode changes.

        - ``episode_id is None`` → reset on every call (defensive default).
        - ``episode_id`` differs from the previous one → reset.
        - Same ``episode_id`` as previous call → keep state.
        """
        if self.drift_detector is None:
            return
        if episode_id is None:
            self.drift_detector.reset()
            self._last_episode_id = _RESET_ALWAYS
            return
        if self._last_episode_id is _RESET_ALWAYS or episode_id != self._last_episode_id:
            self.drift_detector.reset()
            self._last_episode_id = episode_id

    @torch.no_grad()
    def predict(
        self,
        signal: np.ndarray,
        adjacency: np.ndarray,
        timestamp: float = 0.0,
        episode_id: str | None = None,
        regime_mask: np.ndarray | None = None,
    ) -> list[Alert]:
        """Émet les alertes depuis le signal courant.

        Parameters
        ----------
        signal:
            Signal courant ``S(t)`` de forme ``(T, N, 17)``.
        adjacency:
            Matrice d'adjacence ``A(t)`` de forme ``(T, N, N, 3)``.
        timestamp:
            Horodatage du pas courant (secondes depuis epoch).
        episode_id:
            Identifiant d'épisode. ``None`` → reset systématique du
            DriftDetector ; ``""`` ou autre chaîne → reset uniquement
            si différent du précédent appel.
        regime_mask:
            ``(T,)`` boolean array (``True`` = pas en régime normal). Si
            fourni, la fenêtre de précurseur est extraite des ``k`` derniers
            pas marqués ``True`` — alignement strict avec
            :class:`PrecursorDataset` qui filtre ``regime == "normal"``.
            Sinon, la fenêtre est ``signal[-k:]`` (comportement par défaut,
            sans filtre).
        """
        self._maybe_reset_drift(episode_id)

        signal = signal.astype(np.float32)
        if self.scaler is not None:
            t_len, n_nodes, d = signal.shape
            flat = signal.reshape(-1, d)
            nan_mask = np.isnan(flat)
            flat = np.where(nan_mask, self.scaler.mean_, flat)  # impute → 0 in scaled space
            flat = self.scaler.transform(flat).astype(np.float32)
            signal = flat.reshape(t_len, n_nodes, d)
        else:
            signal = np.nan_to_num(signal, nan=0.0)
        adjacency = np.nan_to_num(adjacency.astype(np.float32), nan=0.0)

        # Look-through: run precursors even during drift (θ_{drift∩anomaly} must be detected).
        # Alerts emitted during drift carry drift_flag=True so the caller can filter if needed.
        drift_flag = False
        if self.drift_detector is not None:
            drift_result = self.drift_detector.update(signal[-1].astype(np.float64))
            drift_flag = drift_result.flag

        if not self.classifiers:
            # Step 9 fix 9.5 (audit 2026-05-26): log instead of silent no-op
            # so misconfigured pipelines are immediately visible.
            import logging
            logging.getLogger(__name__).warning(
                "AlertAssembler.predict called with no classifiers loaded — "
                "returning empty alerts. Ensure precursor checkpoints exist."
            )
            return []

        t_total = signal.shape[0]
        n_nodes = signal.shape[1]
        feature_dim = signal.shape[2]
        adj_channels = adjacency.shape[-1]

        # Group classifiers by their effective window length so that one
        # encoder forward pass can serve every classifier in the group.
        groups: dict[int, list[int]] = defaultdict(list)
        for cluster_id in self.classifiers:
            k = int(self.k_optimal.get(cluster_id, 2))
            groups[k].append(cluster_id)

        ep_id_str = episode_id or ""

        if regime_mask is not None:
            regime_mask = np.asarray(regime_mask, dtype=bool).ravel()
            if regime_mask.shape[0] != t_total:
                raise ValueError(
                    f"regime_mask length {regime_mask.shape[0]} != signal T {t_total}"
                )
            normal_indices = np.where(regime_mask)[0]
        else:
            normal_indices = None

        alerts: list[Alert] = []
        for k, cluster_ids in groups.items():
            if normal_indices is not None and len(normal_indices) > 0:
                idx = normal_indices[-k:]
                sig_window = signal[idx]
                adj_window = adjacency[idx]
                actual_k = len(idx)
            else:
                actual_k = min(k, t_total)
                sig_window = signal[-actual_k:]
                adj_window = adjacency[-actual_k:]

            if actual_k < k:
                pad_t = k - actual_k
                sig_window = np.concatenate(
                    [
                        np.zeros((pad_t, n_nodes, feature_dim), dtype=np.float32),
                        sig_window,
                    ],
                    axis=0,
                )
                adj_window = np.concatenate(
                    [
                        np.zeros(
                            (pad_t, n_nodes, n_nodes, adj_channels), dtype=np.float32
                        ),
                        adj_window,
                    ],
                    axis=0,
                )

            sig_t = torch.from_numpy(sig_window).float().unsqueeze(0).to(self.device)
            adj_t = torch.from_numpy(adj_window).float().unsqueeze(0).to(self.device)
            z = self.typer.embed(sig_t, adj_t).cpu().numpy()  # (1, d_proj)

            for cluster_id in cluster_ids:
                clf = self.classifiers[cluster_id]
                proba = clf.predict_proba(z)
                p_i = float(proba[0, cluster_id])

                if p_i >= self.threshold:
                    alerts.append(
                        Alert(
                            cluster_id=cluster_id,
                            probability=p_i,
                            horizon_steps=k,
                            horizon_seconds=k * self.step_seconds,
                            fiche=self.fiches.get(cluster_id, {}),
                            timestamp=timestamp,
                            episode_id=ep_id_str,
                            drift_flag=drift_flag,
                        )
                    )

        alerts.sort(key=lambda a: a.probability, reverse=True)
        return alerts
