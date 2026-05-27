"""Encoder factory — pick STGCN or STGAT from a config dict.

Used by training scripts and ``alerts.AlertAssembler`` so the encoder
implementation can be swapped without touching downstream code.
"""

from __future__ import annotations

from typing import Any, Mapping

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


def detect_use_layer_norm(state_dict: Mapping[str, Any]) -> bool:
    """Detect whether an STGCN checkpoint was trained with ``use_layer_norm=True``.

    Step 5 fix 5.3 (audit 2026-05-26): legacy v3 checkpoints embed the TCN
    LayerNorm weights as ``tcn_blocks.<i>.norm.weight`` / ``.bias``. The
    constructor flag ``use_layer_norm`` defaults to ``False`` for backward
    compatibility, but downstream code (precursor training, alerts, distant-
    window analysis) repeatedly had to detect the flag from state dict keys.
    Centralising this here removes the duplicated detection logic across
    callers.

    Parameters
    ----------
    state_dict:
        The encoder state dict (typically ``checkpoint["encoder_state"]``).

    Returns
    -------
    bool
        ``True`` if any ``tcn_blocks.*.norm.weight`` key is present.

    Example
    -------
    >>> ckpt = torch.load("best_encoder.pt", map_location="cpu", weights_only=False)
    >>> use_ln = detect_use_layer_norm(ckpt["encoder_state"])
    >>> encoder = build_encoder("stgcn", use_layer_norm=use_ln, ...)
    >>> encoder.load_state_dict(ckpt["encoder_state"])
    """
    return any(
        ".norm.weight" in k for k in state_dict if "tcn_blocks" in k
    )


def build_encoder_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    state_dict_key: str = "encoder_state",
    arch_key: str = "arch",
    default_architecture: str = "stgcn",
) -> nn.Module:
    """Build an encoder pre-configured to load a given checkpoint.

    Step 5 fix 5.3 (audit 2026-05-26): convenience helper that:

    1. Reads ``arch`` metadata from the checkpoint if present.
    2. Auto-detects ``use_layer_norm`` from the state dict (see
       :func:`detect_use_layer_norm`).
    3. Falls back to historical defaults (d_feat=17, n_nodes=6, d_hidden=64,
       d_embed=64) if metadata is missing.

    The returned encoder is **not** yet loaded; the caller must explicitly
    invoke ``encoder.load_state_dict(checkpoint[state_dict_key])``.
    """
    state_dict = checkpoint[state_dict_key]
    arch_meta = checkpoint.get(arch_key) or {}
    use_ln = detect_use_layer_norm(state_dict)
    return build_encoder(
        arch_meta.get("architecture", default_architecture),
        d_feat=int(arch_meta.get("d_feat", 17)),
        n_nodes=int(arch_meta.get("n_nodes", 6)),
        d_hidden=int(arch_meta.get("d_hidden", 64)),
        d_embed=int(arch_meta.get("d_embed", 64)),
        use_layer_norm=use_ln,
    )
