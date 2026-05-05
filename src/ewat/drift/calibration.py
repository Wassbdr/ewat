"""Calibrate the drift threshold ε_drift.

Strategy: compute MMD²(W_ref, W_cur) for all drift episodes in the *train*
split (where ground-truth drift is known), and set ε_drift to the
``percentile``-th percentile of that distribution.  This ensures ε_drift is
above the noise floor while still capturing real drifts.

A matching distribution is computed for normal episodes to verify that
ε_drift provides separability: the desired invariant is

    max(MMD²_normal) < ε_drift < min(MMD²_drift_confirmed)

Usage
=====

    from src.ewat.drift.calibration import calibrate_epsilon
    epsilon = calibrate_epsilon(
        train_drift_episodes,
        train_normal_episodes,
        kernel,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.ewat.drift.mmd import RFFKernel

logger = logging.getLogger(__name__)


def _episode_mmd2_sequence(
    signal: npt.NDArray[np.float64],
    kernel: RFFKernel,
    window_ref_size: int,
    window_cur_size: int,
) -> list[float]:
    """Slide reference and current windows over one episode and collect MMD² values.

    Parameters
    ----------
    signal:
        Episode signal, shape (T, N, d_feat).  Flattened to (T, N*d_feat).
    kernel:
        RFFKernel (sigma may be None — calibrated from the reference window).
    window_ref_size, window_cur_size:
        Window sizes in timesteps.

    Returns
    -------
    list of float
        One MMD² per step after both windows are warm.
    """
    T = signal.shape[0]
    flat = signal.reshape(T, -1).astype(np.float64)

    if T < window_ref_size + window_cur_size:
        return []

    ref_win = flat[:window_ref_size]
    if kernel.sigma is None:
        kernel.fit_sigma(ref_win)

    mmd2s: list[float] = []
    for t in range(window_ref_size, T - window_cur_size + 1):
        cur_win = flat[t:t + window_cur_size]
        mmd2s.append(kernel.mmd_squared(ref_win, cur_win))

    return mmd2s


def calibrate_epsilon(
    drift_signals: list[npt.NDArray[np.float64]],
    normal_signals: list[npt.NDArray[np.float64]],
    kernel: RFFKernel,
    window_ref_size: int = 300,
    window_cur_size: int = 60,
    percentile: float = 95.0,
) -> float:
    """Compute ε_drift as percentile_th of the MMD² distribution over drift episodes.

    Parameters
    ----------
    drift_signals:
        List of (T, N, d) signal arrays from labelled drift train episodes.
    normal_signals:
        List of (T, N, d) signal arrays from normal train episodes.
    kernel:
        RFFKernel (shared; sigma may be None).
    window_ref_size, window_cur_size:
        Sliding window sizes in timesteps.
    percentile:
        Percentile of the drift MMD² distribution used as the threshold.

    Returns
    -------
    float
        Calibrated ε_drift.
    """
    drift_mmd2: list[float] = []
    for sig in drift_signals:
        drift_mmd2.extend(_episode_mmd2_sequence(sig, kernel, window_ref_size, window_cur_size))

    normal_mmd2: list[float] = []
    for sig in normal_signals:
        normal_mmd2.extend(_episode_mmd2_sequence(sig, kernel, window_ref_size, window_cur_size))

    if not drift_mmd2:
        raise ValueError("no drift MMD² values computed — check window sizes vs. episode lengths")

    epsilon = float(np.percentile(drift_mmd2, percentile))
    logger.info(
        "calibrate_epsilon: epsilon_drift=%.6f (p%.0f of %d drift MMD² values)",
        epsilon, percentile, len(drift_mmd2),
    )
    if normal_mmd2:
        max_normal = float(np.max(normal_mmd2))
        logger.info(
            "  normal MMD² max=%.6f  separability gap=%.6f",
            max_normal, epsilon - max_normal,
        )
        if max_normal >= epsilon:
            logger.warning(
                "Separability gap is negative (%.6f): ε_drift does not cleanly separate "
                "drift from normal.  Consider increasing the percentile or collecting more data.",
                epsilon - max_normal,
            )
    return epsilon


def save_calibration(
    epsilon: float,
    output_path: str | Path,
    extra: dict | None = None,
) -> None:
    """Persist calibrated epsilon to a JSON file.

    Parameters
    ----------
    epsilon:
        Calibrated threshold.
    output_path:
        Destination path (e.g. ``experiments/drift_separation/epsilon_calibrated.json``).
    extra:
        Optional dict with additional metadata to embed (percentile, n_drift_mmd2, etc.).
    """
    payload = {"epsilon_drift": epsilon}
    if extra:
        payload.update(extra)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved calibration → %s", out)
