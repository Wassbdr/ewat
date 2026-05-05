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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class PrecursorClassifier:
    """One-vs-rest logistic regression classifiers per cluster type.

    Parameters
    ----------
    n_clusters:  Number of cluster types.
    C:           Inverse regularisation strength for LogisticRegression.
    max_iter:    Max iterations for LogisticRegression solver.
    """

    def __init__(self, n_clusters: int, reg_c: float = 1.0, max_iter: int = 500) -> None:
        self.n_clusters = n_clusters
        self.reg_c = reg_c
        self.max_iter = max_iter
        self._classifiers: dict[int, LogisticRegression] = {}

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
                self._classifiers[c] = None  # type: ignore[assignment]
                continue
            clf = LogisticRegression(C=self.reg_c, max_iter=self.max_iter, solver="lbfgs")
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
