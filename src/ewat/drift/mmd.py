"""MMD² estimator via Random Fourier Features (RFF).

Implements the approximate Maximum Mean Discrepancy test statistic:

    MMD²(X_ref, X_cur) = ‖μ_φ(X_ref) − μ_φ(X_cur)‖²

where φ(x) = √(2/D) · cos(W x + b) are random Fourier features that
approximate the RBF kernel with bandwidth σ.

Complexity: O((n_ref + n_cur) · d · D) vs. O(n²·d) for exact MMD.

References
----------
Rahimi & Recht (2007) — Random features for large-scale kernel machines.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt


class RFFKernel:
    """Random Fourier Feature approximation of an RBF kernel.

    Parameters
    ----------
    sigma:
        RBF bandwidth. If ``None``, calibrated lazily from the first call to
        :meth:`fit_sigma` or by :meth:`mmd_squared`.
    rff_dim:
        Number of random features D (higher → more accurate, more memory).
    seed:
        PRNG seed for reproducibility.
    """

    def __init__(
        self,
        sigma: float | None = None,
        rff_dim: int = 256,
        seed: int = 42,
    ) -> None:
        self._sigma = sigma
        self._rff_dim = rff_dim
        self._rng = np.random.default_rng(seed)
        self._W: npt.NDArray[np.float64] | None = None
        self._b: npt.NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> float | None:
        return self._sigma

    def fit_sigma(self, X_ref: npt.NDArray[np.float64]) -> "RFFKernel":
        """Set σ to the median pairwise distance of X_ref (heuristic of Gretton et al.).

        Only a random subsample (≤ 500 rows) is used for efficiency.
        NaN values are imputed with column means before computing distances.
        """
        X_ref = _impute_column_mean(np.asarray(X_ref, dtype=np.float64))
        n = X_ref.shape[0]
        idx = self._rng.choice(n, size=min(n, 500), replace=False)
        sub = X_ref[idx]
        dists = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        upper = dists[np.triu_indices(len(sub), k=1)]
        with np.errstate(all="ignore"):
            median_dist = float(np.nanmedian(upper))
        # Use np.maximum to safely handle NaN (falls back to 1e-8)
        self._sigma = float(np.nanmax([median_dist, 1e-8]))
        # Invalidate feature projections so they are re-drawn with new σ
        self._W = None
        self._b = None
        return self

    def phi(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map X ∈ ℝ^{n×d} to random Fourier features ∈ ℝ^{n×D}.

        Uses the cached random projection W, b, initialising them on first
        call (requires sigma to be set).
        """
        if self._sigma is None:
            raise RuntimeError("sigma must be set before calling phi(); call fit_sigma() first")
        if self._W is None:
            d = X.shape[1]
            self._W = self._rng.standard_normal((d, self._rff_dim)) / self._sigma
            self._b = self._rng.uniform(0.0, 2.0 * np.pi, (self._rff_dim,))
        return np.sqrt(2.0 / self._rff_dim) * np.cos(X @ self._W + self._b)

    def mmd_squared(
        self,
        X_ref: npt.NDArray[np.float64],
        X_cur: npt.NDArray[np.float64],
    ) -> float:
        """Compute MMD²(X_ref, X_cur) via RFF mean embeddings.

        If sigma has not been set, calibrates it from X_ref on the fly.

        Parameters
        ----------
        X_ref:
            Reference window, shape (n_ref, d).
        X_cur:
            Current window, shape (n_cur, d).

        Returns
        -------
        float
            MMD² ≥ 0.  Returns 0.0 for degenerate inputs.
        """
        X_ref = np.asarray(X_ref, dtype=np.float64)
        X_cur = np.asarray(X_cur, dtype=np.float64)

        if X_ref.ndim == 1:
            X_ref = X_ref.reshape(1, -1)
        if X_cur.ndim == 1:
            X_cur = X_cur.reshape(1, -1)

        if X_ref.shape[0] == 0 or X_cur.shape[0] == 0:
            return 0.0

        # Ignore NaN dimensions consistently (same mask for both windows)
        valid_cols = ~(np.isnan(X_ref).all(axis=0) | np.isnan(X_cur).all(axis=0))
        if not valid_cols.any():
            return 0.0
        X_ref = X_ref[:, valid_cols]
        X_cur = X_cur[:, valid_cols]

        # Replace remaining NaNs with column means (per-window) to avoid NaN propagation
        X_ref = _impute_column_mean(X_ref)
        X_cur = _impute_column_mean(X_cur)

        if self._sigma is None:
            self.fit_sigma(X_ref)

        mu_ref = self.phi(X_ref).mean(axis=0)
        mu_cur = self.phi(X_cur).mean(axis=0)
        return float(np.dot(mu_ref - mu_cur, mu_ref - mu_cur))


def _impute_column_mean(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Replace NaNs in each column with that column's mean (or 0 if all-NaN)."""
    out = X.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nanmean(out, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return out
