"""SimCLR-style contrastive pre-training for the STGCN encoder.

The classical EWAT pre-training (``experiments.encoder.train``) is a
reconstruction objective. SimCLR is an alternative self-supervised pretext
task that often produces better embedding geometries for downstream
contrastive fine-tuning, because the pretext loss already pushes
"different episodes" apart in representation space.

Components
----------

* :func:`augment_temporal_pair` — produces two stochastic views of an
  episode signal using temporal augmentations (random crop, time jitter,
  Gaussian feature noise, channel masking). Designed to preserve the
  semantic content (regime label) of an episode while randomising the
  temporal alignment / scale that the encoder should be invariant to.

* :class:`SimCLRHead` — small MLP projection head ``ℝ^{d_e} → ℝ^{d_proj}``
  followed by L2 normalisation, as in Chen et al. (SimCLR, 2020).

* :func:`nt_xent_loss` — NT-Xent (normalised temperature-scaled cross
  entropy) with cosine similarity. Numerically stable in fp32.

* :class:`SimCLRTrainer` — convenience wrapper that runs one training
  epoch given a ``DataLoader`` returning ``EpisodeDataset`` items.

References
----------
- Chen et al. (2020) — A Simple Framework for Contrastive Learning of Visual
  Representations (SimCLR). https://arxiv.org/abs/2002.05709
- Eldele et al. (2021) — TS-TCC: Time-Series Representation Learning via
  Temporal and Contextual Contrasting. (used here as inspiration for the
  temporal augmentation recipe)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ewat.encoder.stgcn import STGCNEncoder


# --------------------------------------------------------------------------- #
# Temporal augmentations
# --------------------------------------------------------------------------- #

@dataclass
class AugmentationConfig:
    """Hyperparameters for SimCLR-style temporal augmentations.

    All probabilities are independent: each augmentation is applied with
    its own probability, so a view may receive multiple augmentations.
    """

    crop_min: float = 0.6              # min fraction of T to keep on random crop
    crop_max: float = 1.0
    crop_prob: float = 1.0             # always crop (cheap, very effective)
    jitter_std: float = 0.05           # Gaussian noise stddev (post-scaling)
    jitter_prob: float = 0.7
    mask_prob: float = 0.4             # probability of applying channel masking
    mask_ratio: float = 0.15           # fraction of feature channels to zero
    timewarp_prob: float = 0.0         # disabled by default (expensive on CPU)
    seed: int = 0


def _resample_to_T(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linearly resample ``x = (T, …)`` along the time axis to ``target_T``."""
    if x.shape[0] == target_T:
        return x
    T = x.shape[0]
    idx_old = torch.linspace(0, T - 1, T, device=x.device)
    idx_new = torch.linspace(0, T - 1, target_T, device=x.device)
    weights_lo = (idx_old.unsqueeze(0) - idx_new.unsqueeze(1)).abs()  # (target_T, T)
    nearest = weights_lo.argmin(dim=1)
    return x[nearest]


def augment_temporal_pair(
    signal: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    cfg: AugmentationConfig,
    rng: torch.Generator,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Produce two stochastic views of one episode for SimCLR.

    Parameters
    ----------
    signal:    ``(T, N, d_feat)``  episode signal (already scaled).
    adjacency: ``(T, N, N, C)``    episode adjacency.
    cfg:       :class:`AugmentationConfig`.
    rng:       Torch :class:`Generator` for reproducibility.

    Returns
    -------
    Two ``(signal_view, adjacency_view)`` tuples whose temporal length
    may differ from the original (after random cropping). The caller is
    responsible for collating views into a batch.
    """
    s1, a1 = _augment_one(signal, adjacency, cfg=cfg, rng=rng)
    s2, a2 = _augment_one(signal, adjacency, cfg=cfg, rng=rng)
    return (s1, a1), (s2, a2)


def _augment_one(
    signal: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    cfg: AugmentationConfig,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = signal.shape[0]
    d_feat = signal.shape[-1]

    sig = signal
    adj = adjacency

    if torch.rand((), generator=rng) < cfg.crop_prob:
        frac = cfg.crop_min + (
            cfg.crop_max - cfg.crop_min
        ) * torch.rand((), generator=rng).item()
        new_T = max(2, int(round(T * frac)))
        if new_T < T:
            start_max = T - new_T
            start = int(torch.randint(0, start_max + 1, (1,), generator=rng).item())
            sig = sig[start: start + new_T]
            adj = adj[start: start + new_T]

    if torch.rand((), generator=rng) < cfg.jitter_prob:
        noise = torch.randn(sig.shape, generator=rng) * cfg.jitter_std
        sig = sig + noise

    if torch.rand((), generator=rng) < cfg.mask_prob:
        n_mask = max(1, int(round(d_feat * cfg.mask_ratio)))
        feat_idx = torch.randperm(d_feat, generator=rng)[:n_mask]
        mask = torch.ones(d_feat, dtype=sig.dtype, device=sig.device)
        mask[feat_idx] = 0.0
        sig = sig * mask[None, None, :]

    if cfg.timewarp_prob > 0 and torch.rand((), generator=rng) < cfg.timewarp_prob:
        warp_factor = 0.7 + 0.6 * torch.rand((), generator=rng).item()
        new_T = max(2, int(round(sig.shape[0] * warp_factor)))
        sig = _resample_to_T(sig, new_T)
        adj = _resample_to_T(adj, new_T)

    return sig, adj


def collate_simclr_views(
    views: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length views to the longest T and stack.

    Returns ``(signal, adjacency, lengths)``. Mirrors
    :func:`ewat.encoder.dataset.collate_episodes`.
    """
    max_T = max(s.shape[0] for s, _ in views)
    sigs, adjs, lens = [], [], []
    for s, a in views:
        T = s.shape[0]
        pad = max_T - T
        if pad > 0:
            s = torch.cat([s, torch.zeros(pad, *s.shape[1:], dtype=s.dtype)], dim=0)
            a = torch.cat([a, torch.zeros(pad, *a.shape[1:], dtype=a.dtype)], dim=0)
        sigs.append(s)
        adjs.append(a)
        lens.append(T)
    return (
        torch.stack(sigs),
        torch.stack(adjs),
        torch.tensor(lens, dtype=torch.long),
    )


# --------------------------------------------------------------------------- #
# Projection head + NT-Xent loss
# --------------------------------------------------------------------------- #

class SimCLRHead(nn.Module):
    """Two-layer MLP projection head with L2-normalised output."""

    def __init__(self, d_in: int, d_proj: int = 64) -> None:
        super().__init__()
        d_hidden = max(d_proj, d_in)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """NT-Xent loss (Chen et al. 2020).

    Parameters
    ----------
    z1, z2:      ``(B, d)`` projections of two augmented views. Assumed
                 already L2-normalised — :class:`SimCLRHead` does this.
    temperature: SimCLR temperature τ. Default 0.5 works well for
                 small-batch CPU training; lower (≈0.1) is typical with
                 large GPU batches.

    Returns
    -------
    Scalar loss averaged over the 2B positive pairs.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1/z2 shape mismatch: {z1.shape} vs {z2.shape}")
    B = z1.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    z = torch.cat([z1, z2], dim=0)               # (2B, d)
    sim = (z @ z.T) / float(temperature)         # (2B, 2B)
    eye = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)             # exclude self-similarity

    # The positive of row i is its augmented twin: i ↔ i+B (and vice versa).
    pos_idx = torch.arange(2 * B, device=sim.device)
    pos_idx = (pos_idx + B) % (2 * B)
    return F.cross_entropy(sim, pos_idx)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

@dataclass
class SimCLRState:
    """Lightweight bookkeeping returned by :meth:`SimCLRTrainer.run_epoch`."""

    epoch: int
    train_loss: float
    n_batches: int
    history: list[float] = field(default_factory=list)


class SimCLRTrainer:
    """Run SimCLR pre-training over an :class:`EpisodeDataset`.

    The trainer is intentionally minimal — no MLflow / checkpointing here —
    so it can be invoked from a script (``experiments/encoder/simclr_train.py``)
    or from a unit test.

    Parameters
    ----------
    encoder:        STGCN encoder shared across views.
    head:           :class:`SimCLRHead`.
    optimizer:      Optimiser over ``encoder.parameters() + head.parameters()``.
    aug_cfg:        Augmentation config.
    temperature:    NT-Xent temperature.
    device:         Device.
    seed:           RNG seed for the augmentation generator.
    """

    def __init__(
        self,
        encoder: STGCNEncoder,
        head: SimCLRHead,
        optimizer: torch.optim.Optimizer,
        *,
        aug_cfg: AugmentationConfig | None = None,
        temperature: float = 0.5,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.encoder = encoder
        self.head = head
        self.optimizer = optimizer
        self.cfg = aug_cfg or AugmentationConfig(seed=seed)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self.history: list[float] = []

    def _make_views(
        self, episodes: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        view1, view2 = [], []
        for ep in episodes:
            sig = ep["signal"]
            adj = ep["adjacency"]
            (s1, a1), (s2, a2) = augment_temporal_pair(
                sig, adj, cfg=self.cfg, rng=self._gen,
            )
            view1.append((s1, a1))
            view2.append((s2, a2))
        sig1, adj1, len1 = collate_simclr_views(view1)
        sig2, adj2, len2 = collate_simclr_views(view2)
        return sig1, adj1, len1, sig2, adj2, len2

    def run_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int = 1,
    ) -> SimCLRState:
        self.encoder.train()
        self.head.train()
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            episodes = batch if isinstance(batch, list) else _split_batch(batch)
            sig1, adj1, len1, sig2, adj2, len2 = self._make_views(episodes)
            sig1 = sig1.to(self.device)
            adj1 = adj1.to(self.device)
            len1 = len1.to(self.device)
            sig2 = sig2.to(self.device)
            adj2 = adj2.to(self.device)
            len2 = len2.to(self.device)

            z1 = self.head(self.encoder(sig1, adj1, lengths=len1))
            z2 = self.head(self.encoder(sig2, adj2, lengths=len2))
            loss = nt_xent_loss(z1, z2, temperature=self.temperature)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach())
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        self.history.append(avg)
        return SimCLRState(
            epoch=epoch, train_loss=avg, n_batches=n_batches, history=list(self.history),
        )


def _split_batch(batch: dict) -> list[dict]:
    """Split a collated batch (signal=(B,T,N,17), adjacency=(B,T,N,N,3))
    back into per-episode dicts. The trainer applies augmentations
    *per-episode* before re-batching, so we need this inverse op.
    """
    B = batch["signal"].shape[0]
    out = []
    lengths = batch.get("T")
    for i in range(B):
        T_i = int(lengths[i].item()) if lengths is not None else batch["signal"].shape[1]
        out.append({
            "signal": batch["signal"][i, :T_i],
            "adjacency": batch["adjacency"][i, :T_i],
        })
    return out
