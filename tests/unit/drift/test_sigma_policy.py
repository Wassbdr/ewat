"""Tests sigma_policy à la recalibration (M2, audit 2026-06)."""

from __future__ import annotations

import numpy as np
import pytest

from ewat.drift.detector import DriftDetector
from ewat.drift.mmd import RFFKernel


def _make_detector(sigma_policy: str) -> DriftDetector:
    return DriftDetector(
        kernel=RFFKernel(sigma=1.0, rff_dim=64, seed=0),
        epsilon_drift=0.05,
        window_ref_size=10,
        window_cur_size=5,
        post_drift_window_s=3,
        sigma_policy=sigma_policy,
    )


def _drive_to_recalibration(det: DriftDetector) -> None:
    """Warm-up normal → pic transitoire → retour normal ⇒ RECALIBRATE.

    Séquence déterministe (zéros/pic constant) : après le pic, la fenêtre
    post se remplit de zéros → re-test négatif → branche RECALIBRATE.
    """
    regimes = []
    for _ in range(15):  # warm-up + normal
        regimes.append(det.update(np.zeros(4)).regime)
    for _ in range(3):  # pic (pending drift, post window se remplit)
        regimes.append(det.update(np.ones(4) * 30.0).regime)
    for _ in range(10):  # retour à la normale → re-test négatif → recalibrate
        regimes.append(det.update(np.zeros(4)).regime)
    assert "recalibrate" in regimes, f"recalibration jamais atteinte: {set(regimes)}"


def test_sigma_kept_by_default():
    det = _make_detector("keep")
    _drive_to_recalibration(det)
    # σ et les projections RFF survivent à la recalibration : l'échelle du
    # MMD² reste comparable au ε calibré.
    assert det._kernel._sigma == 1.0
    assert det._kernel._W is not None


def test_sigma_refit_legacy_changes_scale():
    det = _make_detector("refit")
    _drive_to_recalibration(det)
    # legacy : σ invalidé puis re-fitté sur la nouvelle fenêtre dès le MMD²
    # suivant — ici il s'effondre (1e-8, fenêtre plate), démontrant le saut
    # d'échelle du MMD² face à un ε resté fixe (la raison du défaut "keep").
    assert det._kernel._sigma != 1.0


def test_default_policy_is_keep():
    det = DriftDetector(kernel=RFFKernel(sigma=1.0, rff_dim=16, seed=0))
    assert det._sigma_policy == "keep"


def test_invalid_policy_raises():
    with pytest.raises(ValueError, match="sigma_policy"):
        DriftDetector(kernel=RFFKernel(sigma=1.0, rff_dim=16, seed=0),
                      sigma_policy="reset")


def test_detection_still_works_after_keep_recalibration():
    det = _make_detector("keep")
    _drive_to_recalibration(det)
    # la recalibration vide ref_buf (5/10 rows) → re-chauffe d'abord
    for _ in range(6):
        det.update(np.zeros(4))
    # un vrai drift soutenu après recalibration doit toujours être détecté
    flags = [det.update(np.ones(4) * 80.0).flag for _ in range(10)]
    assert any(flags)
