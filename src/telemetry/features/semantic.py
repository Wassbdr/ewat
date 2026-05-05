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

# Module-level singleton: share one model across all scorer instances in a process.
# Avoids loading the 23M-param model once per service per episode.
_SHARED_MODEL: object | None = None


def _get_shared_model(model_name: str) -> object:
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _SHARED_MODEL = SentenceTransformer(model_name)
        logger.info("SemanticAnomalyScorer: loaded shared model '%s'", model_name)
    return _SHARED_MODEL


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
        # Per-instance cache: text → L2-normalised embedding vector.
        # Log lines often repeat across timesteps (health checks, load patterns);
        # caching avoids redundant forward passes.
        self._embed_cache: dict[str, npt.NDArray[np.float32]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference_lines: list[str]) -> SemanticAnomalyScorer:
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

    def load_centroid(self, path: str | Path) -> SemanticAnomalyScorer:
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

    def warmup(self) -> None:
        """Pre-load the SentenceBERT model into memory.

        Call this at process startup rather than letting the model load lazily
        mid-run.  A lazy load during collection triggers a ~1.5 GB PyTorch
        allocation at an unpredictable point, which can OOM-kill WSL when
        Prometheus/Jaeger port-forwards are already consuming memory.
        """
        _get_shared_model(self._model_name)
        logger.info("SemanticAnomalyScorer: model '%s' pre-loaded", self._model_name)

    def _load_model(self) -> None:
        pass  # model is now managed via the module-level singleton

    def _embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return L2-normalised embeddings of shape (len(texts), 384).

        Hits the per-instance cache first; only encodes texts not seen before.
        """
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model: SentenceTransformer = _get_shared_model(self._model_name)  # type: ignore[assignment]

        # Split into cached and uncached texts
        uncached = [t for t in texts if t not in self._embed_cache]
        if uncached:
            # Deduplicate before encoding to avoid redundant work
            unique_uncached = list(dict.fromkeys(uncached))
            vectors = model.encode(
                unique_uncached,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            for text, vec in zip(unique_uncached, vectors):
                self._embed_cache[text] = vec

        return np.stack([self._embed_cache[t] for t in texts])


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
