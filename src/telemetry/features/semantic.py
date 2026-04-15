"""Semantic anomaly scorer for log lines using SentenceBERT.

L_SEMANTIC_ANOMALY = mean cosine distance of log embeddings to the normal centroid μ_v.

The centroid μ_v is computed from a reference window of "normal" log lines
(e.g. first W_ref seconds after deployment stabilises). It is persisted so that
future windows can be scored without recomputing.

Model: all-MiniLM-L6-v2 (384 dims, ~23M params, <1s per batch on CPU).
This model is the SentenceBERT default for sentence similarity and is available
offline via `sentence-transformers`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticAnomalyScorer:
    """Embed log lines and score their anomaly against a reference centroid.

    Parameters
    ----------
    centroid:
        Pre-computed normal centroid μ_v of shape (384,). Pass ``None`` to
        defer until :meth:`fit` is called.
    model_name:
        SentenceBERT model identifier. Defaults to all-MiniLM-L6-v2.
    batch_size:
        Number of lines encoded per forward pass.
    """

    def __init__(
        self,
        centroid: npt.NDArray[np.float32] | None = None,
        model_name: str = _MODEL_NAME,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._centroid: npt.NDArray[np.float32] | None = centroid
        self._model: object | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference_lines: list[str]) -> "SemanticAnomalyScorer":
        """Compute μ_v from reference log lines.

        Parameters
        ----------
        reference_lines:
            List of log line strings from the normal reference window.

        Returns
        -------
        self
        """
        if not reference_lines:
            raise ValueError("reference_lines must be non-empty to compute a centroid")
        embeddings = self._embed(reference_lines)
        self._centroid = embeddings.mean(axis=0).astype(np.float32)
        return self

    def score(self, log_lines: list[str]) -> float:
        """Return the mean cosine distance to μ_v.

        Parameters
        ----------
        log_lines:
            Log lines from the current window.

        Returns
        -------
        float
            Score in [0, 1] (higher → more anomalous). Returns ``nan`` when
            ``log_lines`` is empty or ``fit`` has not been called.
        """
        if not log_lines:
            return float("nan")
        if self._centroid is None:
            logger.warning("SemanticAnomalyScorer.score called before fit(); returning nan")
            return float("nan")

        embeddings = self._embed(log_lines)
        return float(_mean_cosine_distance(embeddings, self._centroid))

    def save_centroid(self, path: str | Path) -> None:
        """Persist μ_v to a .npy file.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``data/centroid_ewat.npy``).
        """
        if self._centroid is None:
            raise RuntimeError("No centroid to save; call fit() first")
        np.save(path, self._centroid)

    def load_centroid(self, path: str | Path) -> "SemanticAnomalyScorer":
        """Load μ_v from a .npy file.

        Parameters
        ----------
        path:
            Path to a .npy file produced by :meth:`save_centroid`.

        Returns
        -------
        self
        """
        self._centroid = np.load(path).astype(np.float32)
        return self

    @property
    def centroid(self) -> npt.NDArray[np.float32] | None:
        """The reference centroid μ_v, or None if not yet fitted."""
        return self._centroid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            self._model = SentenceTransformer(self._model_name)
            logger.debug("SemanticAnomalyScorer: loaded model '%s'", self._model_name)

    def _embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return L2-normalised embeddings of shape (len(texts), 384)."""
        self._load_model()
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------

def _mean_cosine_distance(
    embeddings: npt.NDArray[np.float32],
    centroid: npt.NDArray[np.float32],
) -> float:
    """Mean cosine distance between each row of ``embeddings`` and ``centroid``.

    Since both embeddings and centroid are L2-normalised,
    cosine_distance = 1 - dot(e, c).

    Parameters
    ----------
    embeddings:
        (N, D) array of L2-normalised vectors.
    centroid:
        (D,) L2-normalised reference vector.

    Returns
    -------
    float
        Mean cosine distance in [0, 1].
    """
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
    # embeddings are already normalised by SentenceTransformer
    cosine_sims = embeddings @ centroid_norm
    return float(np.mean(1.0 - cosine_sims))
