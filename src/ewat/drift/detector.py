"""Sliding-window drift detector based on MMD-RFF.

Implements the look-through mechanism from the EWAT formalisation:

    MMD² < ε_drift → NORMAL  (signal passed as-is)
    MMD² ≥ ε_drift + post-drift test positive → DRIFT  (flag added)
    MMD² ≥ ε_drift + post-drift test negative → RECALIBRATE (W_ref ← W_cur)

The "post-drift test" is a second MMD² computation over a lookahead window.
If the distribution *stays* anomalous for post_drift_window_s seconds, the
regime is confirmed as DRIFT; otherwise it is treated as a benign change and
the reference window is recalibrated.

References
----------
EWAT formalisation §Pipeline étape 0 — Détection de drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from src.ewat.drift.mmd import RFFKernel


@dataclass
class DriftResult:
    """Output of one detector update step."""

    flag: bool
    mmd2: float
    regime: Literal["normal", "drift", "recalibrate"]


class DriftDetector:
    """Sliding-window MMD drift detector with look-through.

    Parameters
    ----------
    kernel:
        Configured :class:`RFFKernel`.  sigma may be None at construction time
        — it will be calibrated from the first reference window.
    epsilon_drift:
        Detection threshold.  If ``None``, the detector will not flag any
        drift (useful until calibration is done).
    window_ref_size:
        Number of timesteps in the reference window W_ref.
    window_cur_size:
        Number of timesteps in the current window W_cur.
    post_drift_window_s:
        Number of timesteps to observe after a putative drift before
        confirming DRIFT vs. RECALIBRATE.
    """

    def __init__(
        self,
        kernel: RFFKernel,
        epsilon_drift: float | None = None,
        window_ref_size: int = 300,
        window_cur_size: int = 60,
        post_drift_window_s: int = 120,
    ) -> None:
        self._kernel = kernel
        self._epsilon_drift = epsilon_drift
        self._ref_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=window_ref_size)
        self._cur_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=window_cur_size)
        self._post_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=post_drift_window_s)
        self._window_ref_size = window_ref_size
        self._window_cur_size = window_cur_size
        self._post_drift_window_s = post_drift_window_s
        # State machine
        self._pending_drift: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def epsilon_drift(self) -> float | None:
        return self._epsilon_drift

    @epsilon_drift.setter
    def epsilon_drift(self, value: float) -> None:
        self._epsilon_drift = value

    def update(self, S_t: npt.NDArray[np.float64]) -> DriftResult:
        """Consume one timestep of the flattened feature vector.

        Parameters
        ----------
        S_t:
            Feature snapshot at time t.  Shape (N·d,) or (N, d) — will be
            flattened to 1-D.

        Returns
        -------
        DriftResult
        """
        row = np.asarray(S_t, dtype=np.float64).ravel()

        # Always accumulate in current buffer
        self._cur_buf.append(row)

        # If epsilon is not set yet or reference not warm, no detection
        if self._epsilon_drift is None or len(self._ref_buf) < self._window_ref_size:
            # Warm up reference with the current snapshot
            self._ref_buf.append(row)
            return DriftResult(flag=False, mmd2=0.0, regime="normal")

        if len(self._cur_buf) < self._window_cur_size:
            return DriftResult(flag=False, mmd2=0.0, regime="normal")

        X_ref = np.stack(list(self._ref_buf))
        X_cur = np.stack(list(self._cur_buf))
        mmd2 = self._kernel.mmd_squared(X_ref, X_cur)

        if mmd2 < self._epsilon_drift:
            if self._pending_drift:
                # False alarm — confirm recalibrate
                self._pending_drift = False
                self._recalibrate()
                return DriftResult(flag=False, mmd2=mmd2, regime="recalibrate")
            return DriftResult(flag=False, mmd2=mmd2, regime="normal")

        # mmd2 ≥ ε_drift
        if not self._pending_drift:
            self._pending_drift = True
            self._post_buf.clear()

        self._post_buf.append(row)

        if len(self._post_buf) < self._post_drift_window_s:
            # Still observing post-drift window — conservatively flag DRIFT
            return DriftResult(flag=True, mmd2=mmd2, regime="drift")

        # Post-drift window full — re-test
        X_post = np.stack(list(self._post_buf))
        mmd2_post = self._kernel.mmd_squared(X_ref, X_post)

        if mmd2_post >= self._epsilon_drift:
            # Sustained drift confirmed
            return DriftResult(flag=True, mmd2=mmd2_post, regime="drift")

        # Distribution returned to normal — recalibrate
        self._pending_drift = False
        self._recalibrate()
        return DriftResult(flag=False, mmd2=mmd2_post, regime="recalibrate")

    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self._ref_buf.clear()
        self._cur_buf.clear()
        self._post_buf.clear()
        self._pending_drift = False

    def load_reference(self, X_ref: npt.NDArray[np.float64]) -> None:
        """Seed the reference buffer from an array of shape (n, d).

        Allows bypassing the warm-up period when a saved reference window is
        available.
        """
        self._ref_buf.clear()
        rows = np.asarray(X_ref, dtype=np.float64)
        for row in rows[-self._window_ref_size:]:
            self._ref_buf.append(row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recalibrate(self) -> None:
        """W_ref ← W_cur (slide the reference window forward)."""
        self._ref_buf.clear()
        for row in self._cur_buf:
            self._ref_buf.append(row)
        self._post_buf.clear()
        # Invalidate cached RFF projections so they are re-drawn for new σ
        self._kernel._W = None
        self._kernel._b = None
        self._kernel._sigma = None
