"""EWAT Step 2 — Siamese contrastive typing.

Architecture
============

    z_e  ──► ProjectionHead ──► z_proj  ─┐
                                          ├─► cosine_distance ──► ContrastiveLoss
    z_e' ──► ProjectionHead ──► z_proj' ─┘

- **ProjectionHead**: MLP(d_embed → d_hidden → d_proj) + L2 normalisation.
  L2 normalisation makes cosine distance equivalent to Euclidean distance in
  the unit-sphere projection space.

- **SiameseTyper**: wraps an STGCNEncoder (optionally frozen) with a
  ProjectionHead.  Exposes `embed()` and `distance()`.

- **ContrastiveLoss**: hinge loss with margin.
    same pair    → d²
    diff pair    → max(0, margin − d)²
  With L2-normalised projections, d ∈ [0, 2] so margin=1.0 (half the range).

Input shapes
============
- signal:    (B, T, N, d_feat)   — normalised signal (from EpisodeDataset)
- adjacency: (B, T, N, N, C)     — adjacency tensor (3 channels)

Output
======
- embed():    (B, d_proj)  — unit-sphere projection
- distance(): (B,)         — cosine distance ∈ [0, 2]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ewat.encoder.stgcn import STGCNEncoder


class ProjectionHead(nn.Module):
    """MLP(d_in → d_in//2 → d_proj) followed by L2 normalisation.

    Parameters
    ----------
    d_in:   Input dimension (encoder embedding_dim, typically 64).
    d_proj: Output projection dimension (typically 32).
    """

    def __init__(self, d_in: int = 64, d_proj: int = 32) -> None:
        super().__init__()
        d_hidden = max(d_proj, d_in // 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_in) → (B, d_proj), L2 normalised (‖z‖₂ = 1)."""
        out = self.mlp(z)
        return F.normalize(out, p=2, dim=-1)


class SiameseTyper(nn.Module):
    """STGCNEncoder (optionally frozen) + ProjectionHead.

    Parameters
    ----------
    encoder:        Pre-trained STGCNEncoder.
    d_proj:         Projection head output dimension.
    freeze_encoder: If True, encoder weights are frozen (only head is trained).
    """

    def __init__(
        self,
        encoder: STGCNEncoder,
        d_proj: int = 32,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ProjectionHead(d_in=encoder.embedding_dim, d_proj=d_proj)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def embed(self, signal: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Encode and project a batch of episodes.

        Parameters
        ----------
        signal:    (B, T, N, d_feat)
        adjacency: (B, T, N, N, C)

        Returns
        -------
        (B, d_proj) — unit-sphere projection
        """
        z_e = self.encoder(signal, adjacency)  # (B, d_embed)
        return self.head(z_e)                  # (B, d_proj)

    def distance(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Cosine distance between two batches of projections.

        Parameters
        ----------
        z_i, z_j: (B, d_proj) — L2-normalised projections

        Returns
        -------
        (B,) — cosine distance = 1 − cosine_similarity ∈ [0, 2]
        """
        cos_sim = (z_i * z_j).sum(dim=-1).clamp(-1.0, 1.0)
        return 1.0 - cos_sim

    def forward(
        self,
        signal_i: torch.Tensor,
        adjacency_i: torch.Tensor,
        signal_j: torch.Tensor,
        adjacency_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed a pair and return (z_i, z_j, distance).

        Convenience forward for the training loop.
        """
        z_i = self.embed(signal_i, adjacency_i)
        z_j = self.embed(signal_j, adjacency_j)
        return z_i, z_j, self.distance(z_i, z_j)


class ContrastiveLoss(nn.Module):
    """Hinge contrastive loss with margin.

    For a pair (z_i, z_j) with binary label is_same ∈ {True, False}:

        same:  loss = d²
        diff:  loss = max(0, margin − d)²

    Parameters
    ----------
    margin: Minimum distance for negative pairs (default 1.0).
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, dist: torch.Tensor, is_same: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        dist:    (B,) — cosine distances ∈ [0, 2]
        is_same: (B,) bool — True if same scenario type

        Returns
        -------
        Scalar loss
        """
        same_mask = is_same.float()
        diff_mask = 1.0 - same_mask

        loss_same = same_mask * dist.pow(2)
        loss_diff = diff_mask * F.relu(self.margin - dist).pow(2)

        return (loss_same + loss_diff).mean()
