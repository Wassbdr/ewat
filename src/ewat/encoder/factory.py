"""Encoder factory — pick STGCN or STGAT from a config dict.

Used by training scripts and ``alerts.AlertAssembler`` so the encoder
implementation can be swapped without touching downstream code.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ewat.encoder.stgat import STGATEncoder
from ewat.encoder.stgcn import STGCNEncoder

ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "stgcn": STGCNEncoder,
    "stgat": STGATEncoder,
}


def build_encoder(architecture: str, **kwargs: Any) -> nn.Module:
    """Instantiate an encoder by name.

    Parameters
    ----------
    architecture:
        ``"stgcn"`` (default) or ``"stgat"``.
    **kwargs:
        Forwarded to the encoder constructor. Constructor arguments differ
        slightly: STGCN takes ``n_gcn_layers``, STGAT takes ``n_gat_layers``
        / ``n_heads``. Unknown kwargs raise ``TypeError``.
    """
    arch = architecture.lower()
    if arch not in ENCODER_REGISTRY:
        raise ValueError(
            f"unknown encoder architecture {architecture!r}; "
            f"expected one of {sorted(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[arch](**kwargs)
