"""Backward-compatibility shim for ``ewat.typing.shap_explainer``.

The previous ``compute_cluster_shap`` function performed **gradient × input
saliency**, not Shapley value attribution. The implementation has been moved
to :mod:`ewat.typing.saliency_explainer`, where a *real* KernelSHAP
implementation is also available (``compute_cluster_kernel_shap``).

Importing from this module emits a :class:`DeprecationWarning`. Update your
imports to::

    from ewat.typing.saliency_explainer import (
        compute_cluster_saliency,
        compute_cluster_kernel_shap,
        write_cluster_fiches,
    )
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from ewat.encoder.dataset import EpisodeDataset
from ewat.encoder.stgcn import STGCNEncoder
from ewat.typing.saliency_explainer import (
    compute_cluster_kernel_shap,
    compute_cluster_saliency,
    write_cluster_fiches,
)

__all__ = [
    "compute_cluster_shap",
    "compute_cluster_kernel_shap",
    "compute_cluster_saliency",
    "write_cluster_fiches",
]

warnings.warn(
    "ewat.typing.shap_explainer is deprecated; the implementation is "
    "saliency (gradient × input), not Shapley values. Use "
    "ewat.typing.saliency_explainer.compute_cluster_saliency or "
    "compute_cluster_kernel_shap instead.",
    DeprecationWarning,
    stacklevel=2,
)


def compute_cluster_shap(
    encoder: STGCNEncoder,
    dataset: EpisodeDataset,
    cluster_labels: np.ndarray,
    n_bg: int = 50,  # noqa: ARG001  — kept for API compatibility
    device: torch.device | None = None,
    max_samples_per_cluster: int = 50,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Deprecated alias for :func:`compute_cluster_saliency`."""
    return compute_cluster_saliency(
        encoder=encoder,
        dataset=dataset,
        cluster_labels=cluster_labels,
        device=device,
        max_samples_per_cluster=max_samples_per_cluster,
        seed=seed,
    )
