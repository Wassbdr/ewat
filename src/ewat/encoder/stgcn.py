"""EWAT Step 1 — STGCN Encoder.

Maps an episode window to a fixed-size embedding:

    z_e = Enc_θ(S̃_{[t-W, t+δ]}, A(t)) ∈ ℝ^{d_e}

Architecture
============

For each timestep t:
  1. **Spatial GCN**: X_t ∈ ℝ^{N×d_in}  →  H_t ∈ ℝ^{N×d_hidden}
     using either the **dynamic** per-step adjacency ``A(t)`` (default) or
     the time-averaged adjacency ``A_bar = mean_t A(t)`` (toggle via
     ``dynamic_graph=False`` — kept as an ablation variant).
     Adjacency is normalised D^{-1/2} A D^{-1/2} (symmetric).
     Multi-channel: 3 adjacency channels combined via learned weights.

  2. **Temporal TCN**: H ∈ ℝ^{T×N×d_hidden}  →  z ∈ ℝ^{T'×N×d_hidden}
     1-D causal convolution over the time axis per node, followed by a
     channel-wise LayerNorm (previously dead code).

  3. **Readout**: masked mean-pool over ``T'`` (using episode lengths from
     the collate function) and ``N`` → ℝ^{d_hidden}. Falls back to the
     unmasked global mean when ``lengths`` is not provided.

  4. **MLP head**: ℝ^{d_hidden}  →  ℝ^{d_e}

Input shapes
============
- signal:    ``(B, T, N, d_feat)``   — normalised S(t)
- adjacency: ``(B, T, N, N, C)``     — raw adjacency tensor (3 channels)
- lengths (optional): ``(B,)`` long tensor with the *valid* length of each
  episode (the rest is zero-padded by ``collate_episodes``).

Output shape
============
- ``z_e``: ``(B, d_e)``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpatialGCNLayer(nn.Module):
    """Single-hop spectral GCN with multi-channel adjacency.

    Combines C adjacency channels via learned scalar weights, then applies
    ``D^{-1/2} A_combined D^{-1/2}`` normalisation before the linear
    transform.

    The forward accepts either a static (B, N, N, C) adjacency or a dynamic
    (B, T, N, N, C) adjacency: in the latter case the GCN is applied per
    timestep, producing one (B, T, N, out_features) tensor.
    """

    def __init__(self, in_features: int, out_features: int, n_adj_channels: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ch_weights = nn.Parameter(torch.ones(n_adj_channels) / n_adj_channels)

    def _normalised_adj(self, adj: torch.Tensor) -> torch.Tensor:
        ch_w = F.softmax(self.ch_weights, dim=0)  # (C,)
        a = (adj * ch_w).sum(dim=-1)              # (..., N, N)
        deg = a.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (..., N, 1)
        d_inv_sqrt = deg.pow(-0.5)
        return d_inv_sqrt * a * d_inv_sqrt.transpose(-1, -2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Either ``(B, N, in_features)`` (static) or
            ``(B, T, N, in_features)`` (dynamic).
        adj:
            Either ``(B, N, N, C)`` (static) or ``(B, T, N, N, C)``
            (dynamic). Shapes must be consistent with ``x``.
        """
        a_norm = self._normalised_adj(adj)
        if x.dim() == 4 and adj.dim() == 5:
            # Dynamic: matmul along the last two dims via ``matmul`` (bcast over T).
            out = torch.matmul(a_norm, x)
        else:
            out = torch.bmm(a_norm, x)
        return self.linear(out) + self.bias


class _TemporalBlock(nn.Module):
    """Causal 1-D convolution over the time axis, applied node-wise.

    Input/output shape: ``(B, N, T, d)`` with causal padding on the left so
    that each output timestep only sees past inputs. The per-channel
    LayerNorm (``self.norm``) exists as a parameter but is intentionally
    NOT applied in the forward pass — the v3 checkpoint was trained without
    it. Enable for newly trained models on v4 data.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, d, T)"""
        out = self.conv(x)
        out = out[..., : -self._pad] if self._pad else out
        out = F.gelu(out)
        return self.drop(out)


class STGCNEncoder(nn.Module):
    """Spatio-temporal encoder: ``(B, T, N, 17)`` + ``(B, T, N, N, 3)`` → ``(B, d_e)``.

    Parameters
    ----------
    d_feat:
        Number of input features per node (17 by default).
    n_nodes:
        Number of nodes N (6 by default).
    d_hidden:
        Hidden dimension of the GCN layers.
    d_embed:
        Output embedding dimension d_e (64 by default).
    n_gcn_layers:
        Number of stacked spatial GCN layers.
    tcn_kernel:
        Kernel size for the temporal convolution.
    tcn_layers:
        Number of stacked TCN blocks.
    n_adj_ch:
        Number of adjacency channels (3: volume, latency, error_rate).
    dropout:
        Dropout rate applied after each block.
    dynamic_graph:
        If ``True`` (default), the GCN uses the per-timestep adjacency
        ``A(t)``. If ``False``, the time-averaged adjacency
        ``A_bar = mean_t A(t)`` is used instead — this matches the
        historical behaviour and is kept as an ablation variant.
    """

    def __init__(
        self,
        d_feat: int = 17,
        n_nodes: int = 6,
        d_hidden: int = 64,
        d_embed: int = 64,
        n_gcn_layers: int = 2,
        tcn_kernel: int = 3,
        tcn_layers: int = 2,
        n_adj_ch: int = 3,
        dropout: float = 0.1,
        dynamic_graph: bool = True,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(d_feat, d_hidden)

        gcn_dims = [d_hidden] * (n_gcn_layers + 1)
        self.gcn_layers = nn.ModuleList([
            _SpatialGCNLayer(gcn_dims[i], gcn_dims[i + 1], n_adj_ch)
            for i in range(n_gcn_layers)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(gcn_dims[i + 1]) for i in range(n_gcn_layers)
        ])

        self.tcn_blocks = nn.ModuleList([
            _TemporalBlock(
                channels=d_hidden,
                kernel_size=tcn_kernel,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(tcn_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_embed),
        )

        self._d_embed = d_embed
        self._dynamic_graph = bool(dynamic_graph)

    @property
    def embedding_dim(self) -> int:
        return self._d_embed

    @property
    def dynamic_graph(self) -> bool:
        return self._dynamic_graph

    def forward(
        self,
        signal: torch.Tensor,
        adjacency: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        signal:
            ``(B, T, N, d_feat)``.
        adjacency:
            ``(B, T, N, N, C)`` — raw (non-normalised) adjacency.
        lengths:
            Optional ``(B,)`` long tensor of valid timesteps per episode.
            When provided, padded timesteps are excluded from the temporal
            mean pool.

        Returns
        -------
        z_e : ``(B, d_embed)``
        """
        B, T, N, _ = signal.shape

        h = self.input_proj(signal)  # (B, T, N, d_hidden)

        if self._dynamic_graph:
            for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
                h_new = gcn(h, adjacency)  # (B, T, N, d_hidden)
                h = norm(F.gelu(h_new) + h)
        else:
            adj_mean = adjacency.mean(dim=1)  # (B, N, N, C)
            h_bt = h.reshape(B * T, N, -1)
            adj_bt = (
                adj_mean.unsqueeze(1)
                .expand(-1, T, -1, -1, -1)
                .reshape(B * T, N, N, -1)
            )
            for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
                h_bt = norm(F.gelu(gcn(h_bt, adj_bt)) + h_bt)
            h = h_bt.reshape(B, T, N, -1)

        h_bn = h.permute(0, 2, 3, 1).reshape(B * N, -1, T)
        for tcn in self.tcn_blocks:
            h_bn = tcn(h_bn) + h_bn
        h = h_bn.reshape(B, N, -1, T).permute(0, 3, 1, 2)  # (B, T, N, d)

        # ---- Masked mean pool over T (and unmasked over N) ----
        if lengths is not None:
            lengths = lengths.to(device=h.device, dtype=torch.long)
            max_T = h.shape[1]
            time_idx = torch.arange(max_T, device=h.device).unsqueeze(0)  # (1, T)
            mask = (time_idx < lengths.unsqueeze(1)).float()              # (B, T)
            mask = mask.unsqueeze(-1).unsqueeze(-1)                       # (B, T, 1, 1)
            denom = mask.sum(dim=1).clamp(min=1.0)                        # (B, 1, 1)
            z_t = (h * mask).sum(dim=1) / denom                           # (B, N, d)
            z = z_t.mean(dim=1)                                           # (B, d)
        else:
            z = h.mean(dim=(1, 2))

        return self.head(z)
