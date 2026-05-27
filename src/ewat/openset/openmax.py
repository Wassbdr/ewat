"""OpenMax — open-set recognition via Extreme Value Theory.

Reference
---------
Bendale, A., & Boult, T. (2016). *Towards Open Set Deep Networks.* CVPR.

Idea
----
A standard softmax classifier always allocates 100% of probability mass to one
of its K known classes — including for inputs that resemble none of them. OpenMax
fits a Weibull distribution on the *tails* of the per-class distance distribution
of correctly-classified training samples (distances of activation vectors to their
class mean). At inference, this Weibull is used to compute a per-class
"recognition probability"; the residual mass is reallocated to a synthetic
"unknown" class (index = K).

This implementation
-------------------
- Works on **logits** (pre-softmax activation vectors) of shape (n, K).
- Distance = Euclidean distance to class mean of correctly-classified train
  samples. Other distances (cosine, Mahalanobis) are supported via the ``metric``
  argument.
- Fits Weibull on the largest ``tail_size`` distances per class (default 20).
- Inference: revise per-class score, redistribute residual to unknown.

Differences vs. the original
----------------------------
- We skip the "activation vector" α-revision originally applied to the top-α
  classes; here we revise all K classes uniformly. For 15 well-separated chaos
  scenarios this is sufficient and simpler.
- For embeddings (not logits) the same API can be used by passing the encoder
  embedding (B, d) as ``activations``; see ``OpenMax(use_embeddings=True)``.

Usage
-----
>>> openmax = OpenMax(n_classes=15, tail_size=20, alpha_rank=15)
>>> openmax.fit(train_logits, train_labels)              # fit Weibull per class
>>> probas_open = openmax.predict_proba(test_logits)     # (n, K+1) with col K = unknown
>>> probas_open.argmax(axis=1)                           # K  = unknown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.stats as st


@dataclass
class OpenMax:
    """Fit-then-predict OpenMax open-set recogniser.

    Parameters
    ----------
    n_classes:
        Number of *known* classes K. The output will have K+1 columns,
        with column ``K`` representing the "unknown" class.
    tail_size:
        Number of largest distances used to fit the Weibull per class. Must be
        >= 3. Default: 20 (Bendale & Boult use ~20–40 in their paper).
    alpha_rank:
        Number of top classes (sorted by activation) for which the OpenMax
        revision is applied. Default: ``n_classes`` (revise all). Lower values
        increase robustness when many classes share similar activation.
    metric:
        Distance metric for tail fit and inference. ``"euclidean"`` (default) or
        ``"cosine"``.

    Attributes
    ----------
    class_means_:
        ``(K, d)`` array of class mean activations from training.
    weibulls_:
        List of K fitted ``scipy.stats.weibull_min`` instances. ``weibulls_[c]``
        models the distribution of distances of train samples of class c to
        ``class_means_[c]``.
    fitted_:
        ``True`` after :meth:`fit` has been called.
    """

    n_classes: int
    tail_size: int = 20
    # Step 9 fix 9.1 (audit 2026-05-26): tail_size_ratio adapts tail_size to
    # per-class sample count: ``tail_size = max(3, int(n_class * ratio))``.
    # When ``tail_size_ratio`` is set (non-None), it OVERRIDES the static
    # ``tail_size`` per class. Default None preserves backward compat (fixed
    # tail_size=20 globally).
    tail_size_ratio: float | None = None
    alpha_rank: int | None = None
    metric: Literal["euclidean", "cosine"] = "euclidean"

    class_means_: np.ndarray = field(default=None, init=False)
    weibulls_: list = field(default_factory=list, init=False)
    fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.tail_size < 3:
            raise ValueError(f"tail_size must be >= 3, got {self.tail_size}")
        if self.tail_size_ratio is not None:
            if not (0.0 < self.tail_size_ratio <= 1.0):
                raise ValueError(
                    f"tail_size_ratio must be in (0, 1], got {self.tail_size_ratio}"
                )
        if self.alpha_rank is None:
            self.alpha_rank = self.n_classes
        if self.metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric must be 'euclidean' or 'cosine', got {self.metric!r}")

    # -------------------------------------------------------------- distance
    def _distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute distance from each row of ``x`` (n, d) to each row of
        ``centroids`` (K, d), returning ``(n, K)``."""
        if self.metric == "euclidean":
            return np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=-1)
        # cosine
        x_n = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
        c_n = centroids / (np.linalg.norm(centroids, axis=-1, keepdims=True) + 1e-12)
        return 1.0 - x_n @ c_n.T

    # ------------------------------------------------------------------- fit
    def fit(self, activations: np.ndarray, labels: np.ndarray) -> "OpenMax":
        """Fit class means + per-class Weibull on training activations.

        Parameters
        ----------
        activations:
            ``(n, d)`` activation vectors — typically the penultimate layer
            embeddings used for distance computation.
        labels:
            ``(n,)`` integer class labels in ``[0, K)``.
        """
        if activations.ndim != 2:
            raise ValueError(f"activations must be 2D (n, d), got {activations.shape}")
        labels = np.asarray(labels)
        n, d = activations.shape
        self._n_features = d

        self.class_means_ = np.zeros((self.n_classes, d), dtype=np.float64)
        self.weibulls_ = [None] * self.n_classes
        # Step 9 fix 9.1 (audit 2026-05-26): expose effective tail size per class
        self._effective_tail_size: dict[int, int] = {}
        for c in range(self.n_classes):
            mask = (labels == c)
            n_c = int(mask.sum())
            # Compute effective tail size for this class
            if self.tail_size_ratio is not None:
                effective_tail = max(3, int(n_c * self.tail_size_ratio))
            else:
                effective_tail = self.tail_size
            self._effective_tail_size[c] = effective_tail
            if n_c < effective_tail:
                # Not enough samples for this class → fit a degenerate Weibull
                # with very low concentration (always low recognition prob)
                if n_c >= 2:
                    pts = activations[mask]
                    self.class_means_[c] = pts.mean(axis=0)
                    dists = self._distance(pts, self.class_means_[c:c+1]).flatten()
                    tail = np.sort(dists)[-min(len(dists), 3):]
                else:
                    # Truly degenerate — fallback to zero mean, dummy Weibull
                    self.class_means_[c] = np.zeros(d, dtype=np.float64)
                    tail = np.array([1.0, 1.0, 1.0])
            else:
                pts = activations[mask]
                self.class_means_[c] = pts.mean(axis=0)
                dists = self._distance(pts, self.class_means_[c:c+1]).flatten()
                # Top-k largest distances (tail of the distribution)
                tail = np.sort(dists)[-effective_tail:]
            # Fit Weibull on the tail
            try:
                shape, loc, scale = st.weibull_min.fit(tail, floc=0)
                self.weibulls_[c] = (shape, loc, scale)
            except Exception:
                # If fitting fails (rare), use a degenerate large-scale Weibull
                self.weibulls_[c] = (1.0, 0.0, max(tail.max(), 1.0))

        self.fitted_ = True
        return self

    # ----------------------------------------------------------- predict_proba
    def predict_proba(
        self, activations: np.ndarray, logits: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``(n, K+1)`` probability matrix with column ``K`` = unknown.

        Parameters
        ----------
        activations:
            ``(n, d)`` — same feature space used for ``fit`` (distance space).
        logits:
            Optional ``(n, K)`` class scores to be revised by the recognition.
            If ``None``, the closed-set classification scores are derived from
            negative distances to class means (``softmax(-dist)``) — useful
            when activations are already in the classifier output space.

        Algorithm
        ---------
        1. Compute distances to class means in activation space → d_c.
        2. w_c = 1 - F_Weibull_c(d_c) ∈ [0, 1] (recognition).
        3. Derive a closed-set "pseudo-logit" vector p_c per sample:
           p_c = -d_c if ``logits`` not given, else ``logits[:, c]``.
        4. For each c in top-α (sorted by p_c desc): revised_c = p_c · w_c,
           unknown_score += p_c · (1 - w_c) · linear_taper.
        5. Softmax over (revised_logits, unknown_score) → (n, K+1).
        """
        if not self.fitted_:
            raise RuntimeError("OpenMax must be fit before predict_proba.")
        if activations.ndim != 2:
            raise ValueError(f"activations must be 2D, got {activations.shape}")
        if activations.shape[1] != self._n_features:
            raise ValueError(
                f"activations.shape[1]={activations.shape[1]} doesn't match fit "
                f"feature dim {self._n_features}"
            )
        n = activations.shape[0]
        K = self.n_classes

        # Distances of each input to each class mean
        dists = self._distance(activations, self.class_means_)   # (n, K)

        # CDF probabilities per (input, class)
        cdf = np.zeros_like(dists)
        for c in range(K):
            shape, loc, scale = self.weibulls_[c]
            cdf[:, c] = st.weibull_min.cdf(dists[:, c], shape, loc=loc, scale=scale)
        recognition = 1.0 - cdf   # ∈ [0,1], 1 = clearly known, 0 = far in tail

        if logits is None:
            # Recognition-only mode (no external logits):
            #   p(unknown) = 1 - max_c(recognition_c)
            #   p(c)       = recognition_c / sum_c recognition_c  · (1 - p(unknown))
            unknown = 1.0 - recognition.max(axis=1)            # (n,)
            denom = recognition.sum(axis=1, keepdims=True) + 1e-12
            known = recognition / denom * (1.0 - unknown[:, None])
            return np.concatenate([known, unknown[:, None]], axis=1)

        # Logit-revision mode (canonical OpenMax)
        if logits.shape != (n, K):
            raise ValueError(f"logits.shape={logits.shape} expected ({n}, {K})")
        class_logits = logits.astype(np.float64)
        order = np.argsort(-class_logits, axis=1)
        revised = class_logits.copy()
        unknown_score = np.zeros(n, dtype=np.float64)
        for i in range(n):
            top = order[i, : self.alpha_rank]
            for rank, c in enumerate(top):
                weight = (self.alpha_rank - rank) / self.alpha_rank
                w_c = recognition[i, c]
                delta = class_logits[i, c] * (1.0 - w_c) * weight
                revised[i, c] = class_logits[i, c] - delta
                unknown_score[i] += delta

        full = np.concatenate([revised, unknown_score[:, None]], axis=1)
        full = full - full.max(axis=1, keepdims=True)
        exp = np.exp(full)
        return exp / exp.sum(axis=1, keepdims=True)

    # ----------------------------------------------------------------- predict
    def predict(
        self, activations: np.ndarray, logits: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return argmax class index in ``[0, K]``. ``K`` = unknown."""
        return self.predict_proba(activations, logits=logits).argmax(axis=1)

    # ---------------------------------------------------------- unknown score
    def unknown_score(
        self, activations: np.ndarray, logits: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convenience: return only the unknown-class probability ``p(unknown|x)``."""
        return self.predict_proba(activations, logits=logits)[:, -1]
