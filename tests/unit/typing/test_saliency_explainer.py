"""Tests for ``ewat.typing.saliency_explainer`` (and the deprecated shim)."""

from __future__ import annotations

import warnings
from pathlib import Path

import json
import numpy as np
import pytest
import torch

from ewat.encoder.stgcn import STGCNEncoder
from ewat.typing.saliency_explainer import (
    compute_cluster_kernel_shap,
    compute_cluster_saliency,
    write_cluster_fiches,
)


class _FakeDataset:
    """Lightweight stand-in for ``EpisodeDataset`` (no disk IO)."""

    FEATURE_NAMES = [f"f{i:02d}" for i in range(17)]

    def __init__(self, n: int = 10, T: int = 8, N: int = 6, d: int = 17, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._items = []
        for i in range(n):
            sig = torch.from_numpy(rng.normal(0, 1, (T, N, d)).astype(np.float32))
            adj = torch.from_numpy(rng.uniform(0, 1, (T, N, N, 3)).astype(np.float32))
            self._items.append(
                {
                    "signal": sig,
                    "adjacency": adj,
                    "scenario": f"sc_{i % 3}",
                    "episode_id": f"ep_{i:03d}",
                    "T": T,
                }
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]

    def __iter__(self):
        return iter(self._items)


@pytest.fixture()
def encoder():
    return STGCNEncoder(
        d_feat=17, n_nodes=6, d_hidden=16, d_embed=8,
        n_gcn_layers=1, tcn_kernel=3, tcn_layers=1, n_adj_ch=3, dropout=0.0,
    )


@pytest.fixture()
def dataset():
    return _FakeDataset(n=12)


# ---------------------------------------------------------------------------
# compute_cluster_saliency
# ---------------------------------------------------------------------------


def test_saliency_returns_one_vector_per_cluster(encoder, dataset):
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    out = compute_cluster_saliency(encoder, dataset, labels, max_samples_per_cluster=2)
    assert set(out.keys()) == {0, 1, 2}
    for v in out.values():
        assert v.shape == (17,)
        assert np.all(v >= 0.0)
        assert pytest.approx(float(v.sum()), rel=1e-3) == 1.0


def test_saliency_label_mismatch_raises(encoder, dataset):
    labels = np.zeros(len(dataset) + 5, dtype=int)
    with pytest.raises(ValueError):
        compute_cluster_saliency(encoder, dataset, labels)


def test_saliency_seed_reproducible(encoder, dataset):
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    a = compute_cluster_saliency(encoder, dataset, labels, seed=7, max_samples_per_cluster=2)
    b = compute_cluster_saliency(encoder, dataset, labels, seed=7, max_samples_per_cluster=2)
    for cid in a:
        np.testing.assert_allclose(a[cid], b[cid])


# ---------------------------------------------------------------------------
# compute_cluster_kernel_shap
# ---------------------------------------------------------------------------


def test_kernel_shap_returns_normalised(encoder, dataset):
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    out = compute_cluster_kernel_shap(
        encoder, dataset, labels,
        clusters=[0],
        n_bg=4,
        n_samples_per_episode=8,
        max_episodes_per_cluster=2,
    )
    assert set(out.keys()) == {0}
    v = out[0]
    assert v.shape == (17,)
    assert np.all(v >= 0.0)
    assert pytest.approx(float(v.sum()), rel=1e-3) == 1.0


def test_kernel_shap_subset_of_clusters(encoder, dataset):
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    out = compute_cluster_kernel_shap(
        encoder, dataset, labels,
        clusters=[1, 2],
        n_bg=3,
        n_samples_per_episode=8,
        max_episodes_per_cluster=1,
    )
    assert set(out.keys()) == {1, 2}


# ---------------------------------------------------------------------------
# write_cluster_fiches
# ---------------------------------------------------------------------------


def test_fiches_written(tmp_path: Path, dataset):
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    importance = {0: np.ones(17) / 17, 1: np.ones(17) / 17, 2: np.ones(17) / 17}
    write_cluster_fiches(importance, labels, dataset, tmp_path, method="saliency")
    fiches = sorted((tmp_path / "fiches").glob("cluster_*.json"))
    assert len(fiches) == 3
    payload = json.loads(fiches[0].read_text())
    assert payload["method"] == "saliency"
    assert "feature_importance" in payload
    assert payload["n_episodes"] >= 1


def test_fiches_method_propagated(tmp_path: Path, dataset):
    labels = np.array([0] * len(dataset))
    importance = {0: np.ones(17) / 17}
    write_cluster_fiches(importance, labels, dataset, tmp_path, method="kernel_shap")
    payload = json.loads((tmp_path / "fiches" / "cluster_0.json").read_text())
    assert payload["method"] == "kernel_shap"


# ---------------------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------------------


def test_shap_explainer_module_emits_deprecation_warning():
    # Import inside the test to surface the warning each time.
    import importlib

    import ewat.typing.shap_explainer as legacy

    importlib.reload(legacy)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        importlib.reload(legacy)
    deps = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert deps, "expected DeprecationWarning on import"


def test_legacy_compute_cluster_shap_delegates(encoder, dataset):
    from ewat.typing.shap_explainer import compute_cluster_shap

    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    out = compute_cluster_shap(
        encoder, dataset, labels, max_samples_per_cluster=2, seed=42,
    )
    assert set(out.keys()) == {0, 1, 2}
    for v in out.values():
        assert v.shape == (17,)
