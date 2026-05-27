"""Tests for Step 8 audit fixes on PrecursorClassifier + k_stability_check."""

import numpy as np
import pytest

from ewat.precursor.model import (
    PrecursorClassifier,
    k_stability_check,
)


def test_lr_default_uses_class_weight_balanced():
    """Step 8 fix 8.1: default 'lr' must pass class_weight='balanced'."""
    clf = PrecursorClassifier(n_clusters=3, classifier_type="lr")
    inner = clf._build_binary_clf()
    # sklearn stores it on the LogisticRegression instance
    assert getattr(inner, "class_weight", None) == "balanced"


def test_lr_handles_severe_imbalance():
    """With class_weight=balanced, LR should still produce > 0.5 AUROC on
    a separable but heavily imbalanced binary problem."""
    rng = np.random.default_rng(0)
    # 49 negatives, 1 positive — extreme imbalance like test set C9
    n_neg, n_pos = 49, 5   # use 5 positives to ensure stable AUROC
    d = 8
    z = np.vstack([
        rng.normal(0, 1, size=(n_neg, d)),
        rng.normal(3, 1, size=(n_pos, d)),   # well-separated cluster
    ])
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    # n_clusters=2 to fit binary OvR
    clf = PrecursorClassifier(n_clusters=2, classifier_type="lr")
    clf.fit(z, y)
    proba = clf.predict_proba(z)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, proba[:, 1])
    assert auc > 0.9, f"separable + balanced LR should achieve high AUROC, got {auc}"


def test_k_stability_check_returns_distribution_per_cluster():
    """k_stability_check returns the bootstrap distribution of k*."""
    rng = np.random.default_rng(42)
    # Synthetic: 3 cluster types, 30 episodes total, 2 k candidates
    n_ep, d, n_clusters = 30, 8, 3
    z_val_by_k = {
        2: rng.normal(0, 1, size=(n_ep, d)).astype(np.float32),
        6: rng.normal(0, 1, size=(n_ep, d)).astype(np.float32),
    }
    y_val = rng.integers(0, n_clusters, size=n_ep)
    result = k_stability_check(
        z_val_by_k, y_val, n_clusters,
        n_bootstrap=50, seed=0,
    )
    assert set(result.keys()) == {0, 1, 2}
    for c, r in result.items():
        assert "k_star_mode" in r
        assert "k_star_std" in r
        assert "distribution" in r
        assert "n_eligible" in r
        assert set(r["distribution"].keys()) == {2, 6}


def test_k_stability_check_handles_empty_cluster():
    """A cluster with no positives in val should produce n_eligible=0."""
    rng = np.random.default_rng(0)
    n_ep, d = 20, 4
    # Only labels 0 and 1 — cluster 2 has no positives
    y_val = np.repeat(np.array([0, 1]), n_ep // 2)
    z_val_by_k = {
        2: rng.normal(size=(n_ep, d)).astype(np.float32),
        4: rng.normal(size=(n_ep, d)).astype(np.float32),
    }
    result = k_stability_check(
        z_val_by_k, y_val, n_clusters=3, n_bootstrap=30, seed=1,
    )
    assert result[2]["n_eligible"] == 0
    assert result[2]["k_star_mode"] == -1


def test_k_stability_check_stable_when_one_k_dominates():
    """If only one k can fit (because the other has degenerate embeddings),
    k_star_mode should be that k consistently."""
    rng = np.random.default_rng(0)
    n_ep, d, n_clusters = 30, 8, 3
    # k=4 has clean separation; k=2 has identical noise (k=2 cannot separate)
    sep = np.concatenate([rng.normal(0, 1, size=(n_ep // 2, d)),
                          rng.normal(5, 1, size=(n_ep // 2, d))])
    z_val_by_k = {
        2: rng.normal(size=(n_ep, d)).astype(np.float32),   # noise only
        4: sep.astype(np.float32),                          # separated
    }
    y_val = np.concatenate([np.zeros(n_ep // 2, dtype=int),
                            np.ones(n_ep // 2, dtype=int)])
    # n_clusters=2 to focus on label 1
    result = k_stability_check(
        z_val_by_k, y_val, n_clusters=2, n_bootstrap=50, seed=0,
    )
    # k=4 should dominate for cluster 1
    assert result[1]["k_star_mode"] == 4
