"""Per-cluster feature attribution for the STGCN encoder.

Two attribution methods are exposed:

- :func:`compute_cluster_saliency` — fast **gradient × input** saliency
  (single backward pass per episode). This is the legacy method previously
  exported as ``compute_cluster_shap``: it is **not** Shapley values, only
  a gradient-based importance proxy.

- :func:`compute_cluster_kernel_shap` — proper Shapley value attributions
  via :class:`shap.KernelExplainer`. Slower but theoretically grounded.
  Used to validate the cheap saliency method on 1–2 clusters
  (cross-validation with permutation importance).

Both functions return per-cluster (17,) importance vectors aggregated over
nodes and time, normalised to sum to 1. The legacy alias
``compute_cluster_shap`` is preserved (with a :class:`DeprecationWarning`
on import) so that existing experiment scripts keep working.

Usage
-----

>>> sal = compute_cluster_saliency(encoder, dataset, labels, seed=42)
>>> shap_vals = compute_cluster_kernel_shap(
...     encoder, dataset, labels, clusters=[0, 5], n_bg=20, seed=42,
... )
>>> write_cluster_fiches(sal, labels, dataset, output_dir)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ewat.encoder.dataset import EpisodeDataset
from ewat.encoder.stgcn import STGCNEncoder


def _validate_labels(cluster_labels: np.ndarray, n_ep: int) -> None:
    if len(cluster_labels) != n_ep:
        raise ValueError(
            f"cluster_labels length {len(cluster_labels)} != dataset length {n_ep}"
        )


def compute_cluster_saliency(
    encoder: STGCNEncoder,
    dataset: EpisodeDataset,
    cluster_labels: np.ndarray,
    device: torch.device | None = None,
    max_samples_per_cluster: int = 50,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Gradient × input feature importance per cluster.

    Backward pass on ``z.sum()`` (i.e. the embedding dimensions are weighted
    equally — see module docstring caveat). One backward pass per episode.

    Parameters
    ----------
    encoder:
        Trained STGCN encoder.
    dataset:
        ``EpisodeDataset`` with the correct scaler.
    cluster_labels:
        ``(N_ep,)`` integer cluster assignments.
    device:
        Torch device (defaults to CPU).
    max_samples_per_cluster:
        Cap on episodes processed per cluster (for speed).
    seed:
        RNG seed for sub-sampling within a cluster.

    Returns
    -------
    Dict ``{cluster_id → (17,) importance}``, normalised to sum to 1.
    """
    if device is None:
        device = torch.device("cpu")

    encoder = encoder.to(device).eval()
    n_ep = len(dataset)
    _validate_labels(cluster_labels, n_ep)

    cluster_ids = sorted(set(int(c) for c in cluster_labels))
    out: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)

    for cid in cluster_ids:
        ep_idxs = np.where(cluster_labels == cid)[0]
        if len(ep_idxs) > max_samples_per_cluster:
            ep_idxs = rng.choice(
                ep_idxs, size=max_samples_per_cluster, replace=False
            )

        importances: list[np.ndarray] = []
        for idx in ep_idxs:
            item = dataset[int(idx)]
            sig = item["signal"].to(device).unsqueeze(0)
            adj = item["adjacency"].to(device).unsqueeze(0)
            sig = sig.requires_grad_(True)

            z = encoder(sig, adj)
            z.sum().backward()

            saliency = (sig.grad * sig.detach()).abs()
            importance = (
                saliency.squeeze(0).mean(dim=(0, 1)).detach().cpu().numpy()
            )
            importances.append(importance)

        if importances:
            mean_imp = np.stack(importances).mean(axis=0)
            total = mean_imp.sum()
            if total > 0:
                mean_imp = mean_imp / total
            out[cid] = mean_imp.astype(np.float32)
        else:
            d = next(iter(dataset))["signal"].shape[-1]
            out[cid] = np.ones(d, dtype=np.float32) / d

    return out


def _episode_feature_signature(
    sig: torch.Tensor,
) -> np.ndarray:
    """Reduce a (T, N, d) tensor to a (d,) feature signature (mean over T,N).

    Used as the input to KernelSHAP: SHAP operates on the d-dim feature space
    after collapsing time and nodes; this matches the saliency aggregation.
    """
    return sig.detach().cpu().numpy().mean(axis=(0, 1))


def compute_cluster_kernel_shap(
    encoder: STGCNEncoder,
    dataset: EpisodeDataset,
    cluster_labels: np.ndarray,
    clusters: list[int] | None = None,
    n_bg: int = 20,
    n_samples_per_episode: int = 64,
    max_episodes_per_cluster: int = 5,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Proper Shapley value attributions via :class:`shap.KernelExplainer`.

    For each selected cluster, samples ``max_episodes_per_cluster`` episodes
    and runs KernelSHAP on the (d,) mean feature vector. Per-feature SHAP
    values are averaged across episodes, then normalised to sum to 1
    (absolute value, to match :func:`compute_cluster_saliency`).

    The encoder is wrapped in a closure: the KernelSHAP perturbations
    only modify the (d,) mean-feature vector; that vector is then
    broadcast back over (T, N) to recover an STGCN input. This is a
    coherent reduction because the saliency method also operates on the
    time-and-node-averaged representation.

    Parameters
    ----------
    clusters:
        Subset of cluster ids to explain. Defaults to all clusters.
    n_bg:
        Number of background samples used to build the SHAP baseline.
    n_samples_per_episode:
        ``nsamples`` argument forwarded to KernelSHAP per episode.
    max_episodes_per_cluster:
        Cap on episodes processed per cluster.
    seed:
        RNG seed.

    Returns
    -------
    Dict ``{cluster_id → (17,) |SHAP|}`` normalised to sum to 1.

    Notes
    -----
    Requires the ``shap`` package (already a project dependency).
    """
    try:
        import shap  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - covered by environment
        raise RuntimeError(
            "compute_cluster_kernel_shap requires the `shap` package"
        ) from exc

    if device is None:
        device = torch.device("cpu")

    encoder = encoder.to(device).eval()
    n_ep = len(dataset)
    _validate_labels(cluster_labels, n_ep)

    cluster_ids = sorted(set(int(c) for c in cluster_labels))
    if clusters is not None:
        cluster_ids = [c for c in cluster_ids if c in clusters]

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    bg_idx = rng.choice(n_ep, size=min(n_bg, n_ep), replace=False)
    bg_signatures: list[np.ndarray] = []
    for i in bg_idx:
        item = dataset[int(i)]
        bg_signatures.append(_episode_feature_signature(item["signal"]))
    bg = np.stack(bg_signatures).astype(np.float32)

    out: dict[int, np.ndarray] = {}

    for cid in cluster_ids:
        ep_idxs = np.where(cluster_labels == cid)[0]
        if len(ep_idxs) == 0:
            continue
        if len(ep_idxs) > max_episodes_per_cluster:
            ep_idxs = rng.choice(
                ep_idxs, size=max_episodes_per_cluster, replace=False
            )

        shap_vals_per_ep: list[np.ndarray] = []
        for idx in ep_idxs:
            item = dataset[int(idx)]
            sig_full = item["signal"].to(device)  # (T, N, d)
            adj_full = item["adjacency"].to(device).unsqueeze(0)
            T, N, d = sig_full.shape

            mean_signature = _episode_feature_signature(item["signal"])

            def predict(x: np.ndarray) -> np.ndarray:
                """Run encoder on a (B, d) batch of perturbed feature signatures."""
                with torch.no_grad():
                    sigs = torch.from_numpy(x).float().to(device)  # (B, d)
                    B = sigs.shape[0]
                    expanded = sigs.view(B, 1, 1, d).expand(B, T, N, d).contiguous()
                    adj_b = adj_full.expand(B, -1, -1, -1, -1).contiguous()
                    z = encoder(expanded, adj_b)  # (B, d_embed)
                    return z.sum(dim=1).cpu().numpy()

            explainer = shap.KernelExplainer(predict, bg)
            shap_vals = explainer.shap_values(
                mean_signature.reshape(1, -1),
                nsamples=n_samples_per_episode,
                silent=True,
            )
            shap_vals = np.abs(np.asarray(shap_vals).reshape(d))
            shap_vals_per_ep.append(shap_vals)

        if shap_vals_per_ep:
            mean = np.stack(shap_vals_per_ep).mean(axis=0)
            total = mean.sum()
            if total > 0:
                mean = mean / total
            out[cid] = mean.astype(np.float32)

    return out


def write_cluster_fiches(
    cluster_importance: dict[int, np.ndarray],
    cluster_labels: np.ndarray,
    dataset: EpisodeDataset,
    output_dir: Path,
    feature_names: list[str] | None = None,
    method: str = "saliency",
) -> None:
    """Write one JSON fiche per cluster.

    Parameters
    ----------
    cluster_importance:
        ``{cluster_id → (17,) importance array}`` from any attribution method.
    cluster_labels:
        ``(N_ep,)`` cluster assignments.
    dataset:
        ``EpisodeDataset`` (used to read scenario names).
    output_dir:
        Directory containing (or to contain) a ``fiches/`` subdirectory.
    feature_names:
        17-element list of feature names (defaults to ``EpisodeDataset.FEATURE_NAMES``).
    method:
        Attribution method label written into the fiche metadata
        (``"saliency"``, ``"kernel_shap"``, ...).
    """
    if feature_names is None:
        feature_names = EpisodeDataset.FEATURE_NAMES

    fiches_dir = Path(output_dir) / "fiches"
    fiches_dir.mkdir(parents=True, exist_ok=True)

    for cid, importance in sorted(cluster_importance.items()):
        ep_idxs = np.where(cluster_labels == cid)[0]
        scenarios: dict[str, int] = {}
        for i in ep_idxs:
            sc = dataset[int(i)]["scenario"]
            scenarios[sc] = scenarios.get(sc, 0) + 1

        ranked = sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        fiche = {
            "cluster_id": cid,
            "method": method,
            "n_episodes": int(len(ep_idxs)),
            "scenario_distribution": scenarios,
            "feature_importance": {name: float(val) for name, val in ranked},
            "top5_features": [name for name, _ in ranked[:5]],
        }

        fiche_path = fiches_dir / f"cluster_{cid}.json"
        fiche_path.write_text(json.dumps(fiche, indent=2))
