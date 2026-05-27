"""Mahalanobis-based out-of-distribution detection.

Reference
---------
Lee, K., Lee, K., Lee, H., & Shin, J. (2018).
*A Simple Unified Framework for Detecting Out-of-Distribution Samples and
Adversarial Attacks.* NeurIPS.

Idea
----
Fit class-conditional Gaussians on the training feature space, sharing a tied
covariance Σ across all classes (more robust on small per-class samples than
per-class Σ). At inference, the Mahalanobis distance to the *nearest* class
mean acts as an in-distribution score; values above a threshold flag the
input as OOD.

Compared to OpenMax (Bendale & Boult 2016):

- **Pro** : no Weibull tail fit (more robust on n_class ≈ 25).
- **Pro** : single tied covariance regularises with shared structure.
- **Pro** : easier to calibrate threshold via a validation OOD pool.
- **Con** : assumes Gaussian class clouds (OK for L2-normalised embeddings).

API mirrors :class:`OpenMax` for drop-in comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class MahalanobisOOD:
    """Mahalanobis OOD detector with class-conditional means + tied covariance.

    Step 9 fix 9.2 (audit 2026-05-26): alternative to OpenMax for cases where
    the per-class Weibull tail fit is unreliable (Unknown AUROC = 0.55 on
    EWAT v4_strat with OpenMax).

    Parameters
    ----------
    n_classes:
        Number of *known* classes K. Output has K+1 columns (col ``K`` = unknown).
    shrinkage:
        Ledoit-Wolf-style shrinkage applied to the tied covariance toward
        diagonal: ``Σ_reg = (1 − shrinkage) Σ + shrinkage · diag(diag(Σ))``.
        Default 0.05 to stabilise inversion on small samples (n < d).
    threshold_mode:
        ``"none"`` (default): return raw normalised scores. Unknown prob =
        ``1 − max(softmax(-dist²/2))``.
        ``"calibrated"``: caller supplies a threshold via :meth:`set_threshold`
        learned on a validation set (more accurate but requires holdout).
    """

    n_classes: int
    shrinkage: float = 0.05
    threshold_mode: Literal["none", "calibrated"] = "none"

    class_means_: np.ndarray = field(default=None, init=False)
    precision_: np.ndarray = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)
    _threshold: float | None = field(default=None, init=False)
    _n_features: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.shrinkage <= 1.0):
            raise ValueError(f"shrinkage must be in [0, 1], got {self.shrinkage}")
        if self.threshold_mode not in ("none", "calibrated"):
            raise ValueError(
                f"threshold_mode must be 'none' or 'calibrated', got {self.threshold_mode!r}"
            )

    def fit(self, activations: np.ndarray, labels: np.ndarray) -> "MahalanobisOOD":
        """Fit class means + tied covariance.

        Parameters
        ----------
        activations: ``(n, d)`` features.
        labels:      ``(n,)`` integer labels in ``[0, K)``.
        """
        if activations.ndim != 2:
            raise ValueError(f"activations must be 2D, got {activations.shape}")
        labels = np.asarray(labels)
        n, d = activations.shape
        self._n_features = d
        self.class_means_ = np.zeros((self.n_classes, d), dtype=np.float64)

        # Per-class mean
        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() >= 1:
                self.class_means_[c] = activations[mask].mean(axis=0)

        # Tied covariance: average of per-class deviations from class means
        residuals_blocks = []
        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() >= 2:
                residuals_blocks.append(activations[mask] - self.class_means_[c])
        if not residuals_blocks:
            cov = np.eye(d, dtype=np.float64)
        else:
            residuals = np.vstack(residuals_blocks).astype(np.float64)
            cov = residuals.T @ residuals / max(len(residuals) - 1, 1)
        # Shrinkage toward diagonal
        diag_cov = np.diag(np.diag(cov))
        cov_reg = (1 - self.shrinkage) * cov + self.shrinkage * diag_cov
        # Add jitter so inversion is numerically stable on small samples
        jitter = 1e-6 * np.trace(cov_reg) / max(d, 1)
        cov_reg += jitter * np.eye(d)
        self.precision_ = np.linalg.pinv(cov_reg)
        self.fitted_ = True
        return self

    def _mahalanobis_dists_sq(self, x: np.ndarray) -> np.ndarray:
        """Return ``(n, K)`` squared Mahalanobis distances."""
        # (x - μ_c)^T Σ⁻¹ (x - μ_c) for every class c
        n = x.shape[0]
        K = self.n_classes
        dists = np.empty((n, K), dtype=np.float64)
        for c in range(K):
            diff = x - self.class_means_[c]
            dists[:, c] = np.einsum("ij,jk,ik->i", diff, self.precision_, diff)
        return dists

    def set_threshold(self, threshold: float) -> None:
        """Set a calibrated threshold (only used when ``threshold_mode='calibrated'``)."""
        self._threshold = float(threshold)
        self.threshold_mode = "calibrated"

    def unknown_score(self, activations: np.ndarray) -> np.ndarray:
        """Return ``(n,)`` ``p(unknown | x)`` ∈ [0, 1].

        High values indicate OOD inputs. The score uses the absolute minimum
        Mahalanobis distance (nearest class) compared against the training
        distribution of that distance:

            min_d_sq = min_c (x - μ_c)^T Σ⁻¹ (x - μ_c)
            unknown  = 1 - exp(-min_d_sq / (2 · d))   ∈ [0, 1)

        The ``d`` normalisation puts the expected ``min_d_sq`` of an
        in-distribution sample near the feature dimension (mean of χ²_d).
        OOD samples produce large ``min_d_sq`` → unknown → 1.
        """
        if not self.fitted_:
            raise RuntimeError("MahalanobisOOD must be fit before unknown_score.")
        dists_sq = self._mahalanobis_dists_sq(activations)
        min_d_sq = dists_sq.min(axis=1)   # (n,)
        return 1.0 - np.exp(-min_d_sq / (2.0 * max(self._n_features, 1)))

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        """Return ``(n, K+1)`` probability matrix (col ``K`` = unknown)."""
        if not self.fitted_:
            raise RuntimeError("MahalanobisOOD must be fit before predict_proba.")
        if activations.shape[1] != self._n_features:
            raise ValueError(
                f"activations.shape[1]={activations.shape[1]} != fit dim {self._n_features}"
            )
        dists_sq = self._mahalanobis_dists_sq(activations)
        unknown = self.unknown_score(activations)
        # Known class distribution via softmax over -d²/2
        scores = -0.5 * dists_sq
        scores = scores - scores.max(axis=1, keepdims=True)
        soft = np.exp(scores)
        soft = soft / soft.sum(axis=1, keepdims=True)
        # Renormalise so that ``known[i] · (1 - unknown[i])`` integrates to 1
        known = soft * (1.0 - unknown[:, None])
        return np.concatenate([known, unknown[:, None]], axis=1)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Return argmax class index in ``[0, K]``. ``K`` = unknown."""
        return self.predict_proba(activations).argmax(axis=1)
