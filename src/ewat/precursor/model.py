"""PrecursorClassifier — one-vs-rest binary classifiers for anomaly type prediction.

For each cluster type C_i we train a LogisticRegression classifier:
  f_i(z_pre) → p̂_i ∈ [0,1]

where z_pre ∈ ℝ^d is the embedding of the pre-injection window (output of
SiameseTyper.embed() or STGCNEncoder).

AUROC evaluation
----------------
For each type C_i:
  AUROC_i = roc_auc_score(y_i, p̂_i)   where y_i = (labels == i)

H3 validation (formalisation.md)
---------------------------------
H3 is confirmed if AUROC_i > 0.5 (baseline) for at least some types.
H3 is falsified if AUROC_i < baseline ∀i, ∀k.

Optimal horizon
---------------
k*_i = argmax_k AUROC_i(k)   over k_values (in timesteps)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

VALID_CLASSIFIER_TYPES = ("lr", "lr_tuned", "rf", "svc")


class PrecursorClassifier:
    """One-vs-rest binary classifiers per cluster type.

    Parameters
    ----------
    n_clusters:       Number of cluster types.
    reg_c:            Inverse regularisation for LogisticRegression (``"lr"`` only).
    max_iter:         Max solver iterations (``"lr"`` and ``"lr_tuned"`` only).
    classifier_type:  One of ``"lr"`` (default, backward-compatible),
                      ``"lr_tuned"`` (LogisticRegressionCV over C grid),
                      ``"rf"`` (RandomForest, balanced), ``"svc"`` (CalibratedSVC, balanced).
    """

    def __init__(
        self,
        n_clusters: int,
        reg_c: float = 1.0,
        max_iter: int = 500,
        classifier_type: str = "lr",
    ) -> None:
        if classifier_type not in VALID_CLASSIFIER_TYPES:
            raise ValueError(
                f"classifier_type must be one of {VALID_CLASSIFIER_TYPES}, got {classifier_type!r}"
            )
        self.n_clusters = n_clusters
        self.reg_c = reg_c
        self.max_iter = max_iter
        self.classifier_type = classifier_type
        self._classifiers: dict[int, Any] = {}

    def _build_binary_clf(self) -> Any:
        if self.classifier_type == "lr_tuned":
            from sklearn.linear_model import LogisticRegressionCV
            return LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
                max_iter=self.max_iter,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
                cv=5,
                scoring="roc_auc",
            )
        if self.classifier_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
        if self.classifier_type == "svc":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.svm import SVC
            return CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced", probability=False),
                cv=5,
                method="sigmoid",
            )
        # "lr" — default, backward-compatible (no class_weight change)
        return LogisticRegression(C=self.reg_c, max_iter=self.max_iter, solver="lbfgs")

    def fit(self, z: np.ndarray, labels: np.ndarray) -> None:
        """Fit one binary classifier per cluster type.

        Parameters
        ----------
        z:      (N_ep, d_embed) embeddings.
        labels: (N_ep,) cluster labels in [0, n_clusters).
        """
        for c in range(self.n_clusters):
            y = (labels == c).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                # Degenerate — skip; predict_proba will return 0 or 1 for all
                self._classifiers[c] = None
                continue
            clf = self._build_binary_clf()
            clf.fit(z, y)
            self._classifiers[c] = clf

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        """Return (N_ep, n_clusters) probability matrix.

        Column i = P(type C_i) according to the one-vs-rest classifier for C_i.
        """
        out = np.zeros((len(z), self.n_clusters), dtype=np.float32)
        for c in range(self.n_clusters):
            clf = self._classifiers.get(c)
            if clf is None:
                out[:, c] = 0.5
            else:
                out[:, c] = clf.predict_proba(z)[:, 1]
        return out

    def scores_per_type(
        self, z: np.ndarray, labels: np.ndarray
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return raw (y_true, y_score) pairs per cluster type.

        Useful for bootstrap CI computation. Types with fewer than 2 positives
        or 2 negatives are omitted from the result.

        Returns
        -------
        {cluster_id → (y_true, y_score)} — binary labels and predicted proba.
        """
        proba = self.predict_proba(z)
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c in range(self.n_clusters):
            y = (labels == c).astype(int)
            if y.sum() < 2 or (len(y) - y.sum()) < 2:
                continue
            result[c] = (y, proba[:, c])
        return result

    def auroc_per_type(
        self, z: np.ndarray, labels: np.ndarray
    ) -> dict[int, float]:
        """Compute AUROC for each cluster type on the given embeddings.

        Returns {cluster_id → AUROC}, where AUROC is NaN if fewer than
        2 positive examples are present (can't compute AUROC).
        """
        proba = self.predict_proba(z)
        results: dict[int, float] = {}
        for c in range(self.n_clusters):
            y = (labels == c).astype(int)
            if y.sum() < 2 or (len(y) - y.sum()) < 2:
                results[c] = float("nan")
                continue
            results[c] = float(roc_auc_score(y, proba[:, c]))
        return results

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> PrecursorClassifier:
        with open(path, "rb") as f:
            return pickle.load(f)


def find_optimal_k(
    auroc_table: dict[int, dict[int, float]],
    n_clusters: int,
) -> dict[int, int]:
    """Find optimal horizon k per cluster type.

    Parameters
    ----------
    auroc_table: {k_steps → {cluster_id → AUROC}}
    n_clusters:  Total number of cluster types.

    Returns
    -------
    {cluster_id → k_optimal} — k with highest AUROC per type.
    """
    k_values = sorted(auroc_table.keys())
    result: dict[int, int] = {}
    for c in range(n_clusters):
        best_k, best_auroc = k_values[0], -1.0
        for k in k_values:
            auc = auroc_table[k].get(c, float("nan"))
            if not np.isnan(auc) and auc > best_auroc:
                best_auroc = auc
                best_k = k
        result[c] = best_k
    return result


def baseline_auroc(n_clusters: int) -> float:
    """AUROC of a random classifier (0.5 — H3 threshold)."""
    return 0.5
