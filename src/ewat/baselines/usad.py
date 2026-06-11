"""USAD — UnSupervised Anomaly Detection (Audibert et al., KDD 2020).

Première baseline publiée du projet (E1, audit 2026-06) : sans point de
comparaison externe, le headline B2 = 0.920 n'est pas positionnable.

Architecture (fidèle au papier, adaptée aux fenêtres épisodiques EWAT) :

- un encodeur E partagé et deux décodeurs D1, D2 (AE1 = D1∘E, AE2 = D2∘E) ;
- entraînement adversarial en deux phases par batch :
  AE1 minimise ‖W − AE1(W)‖² + (1/n)‖W − AE2(AE1(W))‖²,
  AE2 minimise ‖W − AE2(W)‖² − (1/n)‖W − AE2(AE1(W))‖²
  (pondération 1/n décroissante avec l'époque n, comme dans le papier) ;
- score d'anomalie : α‖W − AE1(W)‖² + β‖W − AE2(AE1(W))‖², α+β=1.

Entrée : fenêtres aplaties (k × N × d) — la MÊME featurisation que B2
(instance norm en amont, hors de ce module) pour une comparaison à
protocole constant.

Référence : J. Audibert, P. Michiardi, F. Guyard, S. Marti, M. A. Zuluaga,
« USAD: UnSupervised Anomaly Detection on Multivariate Time Series », KDD 2020.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int) -> None:
        super().__init__()
        h1, h2 = max(d_in // 2, d_latent), max(d_in // 4, d_latent)
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, d_latent), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, d_latent: int, d_out: int) -> None:
        super().__init__()
        h1, h2 = max(d_out // 4, d_latent), max(d_out // 2, d_latent)
        self.net = nn.Sequential(
            nn.Linear(d_latent, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class USAD(nn.Module):
    """Modèle USAD : encodeur partagé + deux décodeurs adversariaux.

    Parameters
    ----------
    d_in:
        Dimension de la fenêtre aplatie (k × N × d_feat).
    d_latent:
        Dimension du goulot (papier : ~d_in/20 ; défaut 40 pour d_in ≈ 612).
    """

    def __init__(self, d_in: int, d_latent: int = 40) -> None:
        super().__init__()
        self.d_in = d_in
        self.encoder = _Encoder(d_in, d_latent)
        self.decoder1 = _Decoder(d_latent, d_in)
        self.decoder2 = _Decoder(d_latent, d_in)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne (w1, w2, w2_of_w1) = AE1(W), AE2(W), AE2(AE1(W))."""
        z = self.encoder(w)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w2_of_w1 = self.decoder2(self.encoder(w1))
        return w1, w2, w2_of_w1


class USADDetector:
    """Wrapper fit/score sklearn-like autour de :class:`USAD`.

    fit() sur les fenêtres NORMALES uniquement (entraînement non supervisé) ;
    anomaly_score() pour la détection, latent() pour brancher un classifieur
    de typage en aval (volet (b) de l'évaluation E1).
    """

    def __init__(
        self,
        d_latent: int = 40,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        alpha: float = 0.5,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.d_latent = d_latent
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha          # poids de ‖W−AE1(W)‖² ; beta = 1−alpha
        self.seed = seed
        self.device = torch.device(device)
        self.model: USAD | None = None
        self.history: list[dict] = []

    def fit(self, X: np.ndarray) -> USADDetector:
        """Entraîne sur (n, d_in) fenêtres normales aplaties."""
        torch.manual_seed(self.seed)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.model = USAD(d_in=X_t.shape[1], d_latent=self.d_latent).to(self.device)
        opt1 = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.decoder1.parameters()), lr=self.lr,
        )
        opt2 = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.decoder2.parameters()), lr=self.lr,
        )
        gen = torch.Generator().manual_seed(self.seed)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            inv_n = 1.0 / epoch
            perm = torch.randperm(len(X_t), generator=gen)
            tot1 = tot2 = 0.0
            n_batches = 0
            for start in range(0, len(X_t), self.batch_size):
                w = X_t[perm[start:start + self.batch_size]].to(self.device)

                # Phase 1 : AE1 minimise rec1 + (1/n)·rec(W, AE2(AE1(W)))
                w1, _, w2_of_w1 = self.model(w)
                loss1 = ((1 - inv_n) * (w - w1).pow(2).mean()
                         + inv_n * (w - w2_of_w1).pow(2).mean())
                opt1.zero_grad()
                loss1.backward()
                opt1.step()

                # Phase 2 : AE2 minimise rec2 − (1/n)·rec(W, AE2(AE1(W)))
                _, w2, w2_of_w1 = self.model(w)
                loss2 = ((1 - inv_n) * (w - w2).pow(2).mean()
                         - inv_n * (w - w2_of_w1).pow(2).mean())
                opt2.zero_grad()
                loss2.backward()
                opt2.step()

                tot1 += float(loss1)
                tot2 += float(loss2)
                n_batches += 1
            self.history.append({"epoch": epoch,
                                 "loss1": tot1 / max(n_batches, 1),
                                 "loss2": tot2 / max(n_batches, 1)})
        self.model.eval()
        return self

    @torch.no_grad()
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Score USAD : α‖W−AE1(W)‖² + (1−α)‖W−AE2(AE1(W))‖² par fenêtre."""
        if self.model is None:
            raise RuntimeError("fit() must be called before anomaly_score()")
        w = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        w1, _, w2_of_w1 = self.model(w)
        s = (self.alpha * (w - w1).pow(2).mean(dim=1)
             + (1.0 - self.alpha) * (w - w2_of_w1).pow(2).mean(dim=1))
        return s.cpu().numpy()

    @torch.no_grad()
    def latent(self, X: np.ndarray) -> np.ndarray:
        """Représentation latente E(W) — pour le volet typage (LR aval)."""
        if self.model is None:
            raise RuntimeError("fit() must be called before latent()")
        w = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        return self.model.encoder(w).cpu().numpy()
