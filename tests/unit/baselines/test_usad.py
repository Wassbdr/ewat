"""Tests USAD (E1, audit 2026-06) — shapes, convergence jouet, sémantique."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ewat.baselines.usad import USAD, USADDetector


def _toy_data(n: int = 200, d: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # données normales : structure linéaire de rang faible + bruit
    basis = rng.normal(size=(4, d))
    coefs = rng.normal(size=(n, 4))
    return (coefs @ basis + 0.05 * rng.normal(size=(n, d))).astype(np.float32)


def test_model_forward_shapes():
    model = USAD(d_in=24, d_latent=4)
    w = torch.randn(8, 24)
    w1, w2, w2_of_w1 = model(w)
    assert w1.shape == w2.shape == w2_of_w1.shape == (8, 24)


def test_fit_reduces_reconstruction_loss():
    X = _toy_data()
    det = USADDetector(d_latent=4, epochs=30, seed=0)
    det.fit(X)
    assert det.history[-1]["loss1"] < det.history[0]["loss1"]


def test_anomalies_score_higher_than_normal():
    X = _toy_data()
    det = USADDetector(d_latent=4, epochs=60, seed=0)
    det.fit(X)
    normal_scores = det.anomaly_score(_toy_data(n=50, seed=1))
    rng = np.random.default_rng(2)
    anomalies = rng.normal(loc=4.0, scale=2.0, size=(50, 24)).astype(np.float32)
    anom_scores = det.anomaly_score(anomalies)
    assert np.median(anom_scores) > np.median(normal_scores) * 2, \
        "les anomalies hors-distribution doivent scorer nettement plus haut"


def test_deterministic_given_seed():
    X = _toy_data(n=60)
    s1 = USADDetector(d_latent=4, epochs=5, seed=7).fit(X).anomaly_score(X)
    s2 = USADDetector(d_latent=4, epochs=5, seed=7).fit(X).anomaly_score(X)
    np.testing.assert_allclose(s1, s2, rtol=1e-5)


def test_latent_shape():
    X = _toy_data(n=30)
    det = USADDetector(d_latent=6, epochs=3, seed=0).fit(X)
    z = det.latent(X)
    assert z.shape == (30, 6)
    assert np.isfinite(z).all()


def test_score_before_fit_raises():
    det = USADDetector()
    with pytest.raises(RuntimeError, match="fit"):
        det.anomaly_score(np.zeros((2, 10), dtype=np.float32))


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        USADDetector(alpha=1.5)
