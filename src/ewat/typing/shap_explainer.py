"""Gradient-based per-cluster feature importance.

Computes feature attributions for each cluster to identify which of the 17
input features drive the encoder embedding.  Outputs one "fiche" per cluster:
a JSON file with feature importances and the scenario distribution.

Implementation
--------------
We use **gradient × input** attribution (a fast proxy for SHAP Gradient
Explainer).  For each episode, we propagate gradients through the encoder
from the sum of the embedding back to the mean signal, giving a (17,) feature
importance vector.  Per-cluster importance is the mean |grad × input| over
cluster members.

This approach requires only 1 backward pass per episode (vs. d_embed backward
passes for SHAP GradientExplainer) and is substantially faster on CPU.

Usage
-----
>>> cluster_shap = compute_cluster_shap(encoder, dataset, result.labels)
>>> write_cluster_fiches(cluster_shap, scenario_dist, output_dir)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ewat.encoder.dataset import EpisodeDataset
from ewat.encoder.stgcn import STGCNEncoder


def compute_cluster_shap(
    encoder: STGCNEncoder,
    dataset: EpisodeDataset,
    cluster_labels: np.ndarray,
    n_bg: int = 50,  # kept for API compatibility, not used in gradient method
    device: torch.device | None = None,
    max_samples_per_cluster: int = 50,
) -> dict[int, np.ndarray]:
    """Compute gradient×input feature importance per cluster.

    Parameters
    ----------
    encoder:                  Trained STGCNEncoder.
    dataset:                  EpisodeDataset with correct scaler.
    cluster_labels:           (N_ep,) cluster assignments.
    n_bg:                     Unused (kept for API compatibility).
    device:                   Torch device.  Defaults to CPU.
    max_samples_per_cluster:  Cap samples per cluster for speed.

    Returns
    -------
    {cluster_id: importance (17,)} — mean |grad × input| per feature, averaged
    over nodes and time, then normalised to sum to 1.
    """
    if device is None:
        device = torch.device("cpu")

    encoder = encoder.to(device).eval()
    n_ep = len(dataset)
    assert len(cluster_labels) == n_ep, (
        f"cluster_labels length {len(cluster_labels)} != dataset length {n_ep}"
    )

    cluster_ids = sorted(set(int(c) for c in cluster_labels))
    shap_per_cluster: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(42)

    for cid in cluster_ids:
        ep_idxs = np.where(cluster_labels == cid)[0]
        # Subsample for speed
        if len(ep_idxs) > max_samples_per_cluster:
            ep_idxs = rng.choice(ep_idxs, size=max_samples_per_cluster, replace=False)

        importances: list[np.ndarray] = []
        for idx in ep_idxs:
            item = dataset[int(idx)]
            sig = item["signal"].to(device).unsqueeze(0)   # (1, T, N, 17)
            adj = item["adjacency"].to(device).unsqueeze(0)  # (1, T, N, N, 3)
            sig = sig.requires_grad_(True)

            z = encoder(sig, adj)          # (1, d_embed)
            z.sum().backward()             # scalar backward → 1 pass

            # grad × input: (1, T, N, 17)
            saliency = (sig.grad * sig.detach()).abs()
            # Mean over batch, T, N → (17,)
            importance = saliency.squeeze(0).mean(dim=(0, 1)).detach().cpu().numpy()
            importances.append(importance)

        if importances:
            mean_imp = np.stack(importances).mean(axis=0)  # (17,)
            # Normalise so values sum to 1
            total = mean_imp.sum()
            if total > 0:
                mean_imp = mean_imp / total
            shap_per_cluster[cid] = mean_imp.astype(np.float32)
        else:
            shap_per_cluster[cid] = np.ones(17, dtype=np.float32) / 17

    return shap_per_cluster


def write_cluster_fiches(
    cluster_shap: dict[int, np.ndarray],
    cluster_labels: np.ndarray,
    dataset: EpisodeDataset,
    output_dir: Path,
    feature_names: list[str] | None = None,
) -> None:
    """Write one JSON fiche per cluster.

    Parameters
    ----------
    cluster_shap:  {cluster_id → (17,) importance array}
    cluster_labels: (N_ep,) cluster assignments
    dataset:       EpisodeDataset (to read scenario names)
    output_dir:    Directory to write fiches/ subdirectory
    feature_names: 17-element list of feature names (defaults to FEATURE_NAMES)
    """
    if feature_names is None:
        feature_names = EpisodeDataset.FEATURE_NAMES

    fiches_dir = Path(output_dir) / "fiches"
    fiches_dir.mkdir(parents=True, exist_ok=True)

    # Scenario distribution per cluster
    for cid, importance in sorted(cluster_shap.items()):
        ep_idxs = np.where(cluster_labels == cid)[0]
        scenarios: dict[str, int] = {}
        for i in ep_idxs:
            sc = dataset[int(i)]["scenario"]
            scenarios[sc] = scenarios.get(sc, 0) + 1

        # Top features by importance
        ranked = sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        fiche = {
            "cluster_id": cid,
            "n_episodes": int(len(ep_idxs)),
            "scenario_distribution": scenarios,
            "feature_importance": {name: float(val) for name, val in ranked},
            "top5_features": [name for name, _ in ranked[:5]],
        }

        fiche_path = fiches_dir / f"cluster_{cid}.json"
        fiche_path.write_text(json.dumps(fiche, indent=2))
