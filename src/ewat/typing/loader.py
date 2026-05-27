"""Utility to load a SiameseTyper from experiment checkpoints.

Single source of truth for loading encoder + typer; avoids copy-paste
between experiments/verification/verify_h1_h3.py and experiments/rcaeval/eval_fewshot.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ewat.encoder.stgcn import STGCNEncoder
from ewat.typing.siamese import SiameseTyper


def load_typer(
    typing_dir: Path,
    encoder_dir: Path,
    device: torch.device | None = None,
) -> SiameseTyper:
    """Load a SiameseTyper from experiment checkpoint directories.

    Architecture hyperparameters are read from the checkpoint's ``arch`` dict
    when available; otherwise canonical ewat_v3 defaults are used.

    Parameters
    ----------
    typing_dir:
        Directory containing ``checkpoints/best_siamese.pt``.
    encoder_dir:
        Directory containing ``checkpoints/best_encoder.pt``.
    device:
        Target device. Defaults to CPU.

    Returns
    -------
    SiameseTyper in eval mode on ``device``.
    """
    device = device or torch.device("cpu")

    enc_ckpt = torch.load(
        Path(encoder_dir) / "checkpoints" / "best_encoder.pt",
        map_location="cpu",
        weights_only=False,
    )
    arch = enc_ckpt.get("arch") or {}
    encoder = STGCNEncoder(
        d_feat=int(arch.get("d_feat", 17)),
        n_nodes=int(arch.get("n_nodes", 6)),
        d_hidden=int(arch.get("d_hidden", 64)),
        d_embed=int(arch.get("d_embed", 64)),
        n_gcn_layers=int(arch.get("n_gcn_layers", 2)),
        tcn_kernel=int(arch.get("tcn_kernel", 3)),
        tcn_layers=int(arch.get("tcn_layers", 2)),
        n_adj_ch=int(arch.get("n_adj_ch", 3)),
    )
    encoder.load_state_dict(enc_ckpt["encoder_state"])

    typer_ckpt = torch.load(
        Path(typing_dir) / "checkpoints" / "best_siamese.pt",
        map_location="cpu",
        weights_only=False,
    )
    d_proj = int(typer_ckpt.get("d_proj", 32))
    typer = SiameseTyper(encoder, d_proj=d_proj)
    typer.load_state_dict(typer_ckpt["typer_state"])
    return typer.to(device).eval()
