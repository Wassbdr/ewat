"""EWAT Step 1 — STGCN Encoder.

Maps an episode window to a fixed-size embedding:

    z_e = Enc_θ(S̃_{[t-W, t+δ]}, A(t)) ∈ ℝ^{d_e}

Architecture
============

For each timestep t:
  1. **Spatial GCN**: X_t ∈ ℝ^{N×d_in}  →  H_t ∈ ℝ^{N×d_hidden}
     using the time-averaged adjacency A_bar = mean_t(A(t)).
     Adjacency is normalised D^{-1/2} A D^{-1/2} (symmetric).
     Multi-channel: 3 adjacency channels combined via learned weights.

  2. **Temporal TCN**: H ∈ ℝ^{T×N×d_hidden}  →  z ∈ ℝ^{T'×N×d_hidden}
     1-D causal convolution over the time axis per node.

  3. **Readout**: mean-pool over T' and N  →  ℝ^{d_hidden}

  4. **MLP head**: ℝ^{d_hidden}  →  ℝ^{d_e}

Input shapes
============
- signal:    (B, T, N, d_feat)   — normalised S(t)
- adjacency: (B, T, N, N, C)     — raw adjacency tensor (3 channels)

Output shape
============
- z_e: (B, d_e)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpatialGCNLayer(nn.Module):
    """Single-hop spectral GCN with multi-channel adjacency.

    Combines C adjacency channels via learned scalar weights, then applies
    D^{-1/2} A_combined D^{-1/2} normalisation before the linear transform.
    """

    def __init__(self, in_features: int, out_features: int, n_adj_channels: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Learnable weights to combine adjacency channels
        self.ch_weights = nn.Parameter(torch.ones(n_adj_channels) / n_adj_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, N, in_features)
        adj : (B, N, N, C)  — C adjacency channels (e.g. volume, latency, error_rate)

        Returns
        -------
        (B, N, out_features)
        """
        # Combine channels: (B, N, N)
        ch_w = F.softmax(self.ch_weights, dim=0)  # (C,)
        A = (adj * ch_w).sum(dim=-1)               # (B, N, N)

        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N, 1)
        d_inv_sqrt = deg.pow(-0.5)
        A_norm = d_inv_sqrt * A * d_inv_sqrt.transpose(-1, -2)  # (B, N, N)

        # GCN step: A_norm @ x @ W + b
        out = torch.bmm(A_norm, x)       # (B, N, in_features)
        out = self.linear(out) + self.bias  # (B, N, out_features)
        return out


class _TemporalBlock(nn.Module):
    """Causal 1-D convolution over the time axis, applied node-wise.

    Input/output shape: (B, N, T, d) with causal padding on the left so
    that each output timestep only sees past inputs.
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
        out = out[..., : -self._pad] if self._pad else out  # remove causal padding
        return self.drop(F.gelu(out))


class STGCNEncoder(nn.Module):
    """Spatio-temporal encoder: (B, T, N, 17) + (B, T, N, N, 3) → (B, d_e).

    Parameters
    ----------
    d_feat:       Number of input features per node (17 by default).
    n_nodes:      Number of nodes N (6 by default).
    d_hidden:     Hidden dimension of the GCN layers.
    d_embed:      Output embedding dimension d_e (64 by default).
    n_gcn_layers: Number of stacked spatial GCN layers.
    tcn_kernel:   Kernel size for the temporal convolution.
    tcn_layers:   Number of stacked TCN blocks.
    n_adj_ch:     Number of adjacency channels (3: volume, latency, error_rate).
    dropout:      Dropout rate applied after each block.
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
    ) -> None:
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(d_feat, d_hidden)

        # Spatial GCN stack
        gcn_dims = [d_hidden] * (n_gcn_layers + 1)
        self.gcn_layers = nn.ModuleList([
            _SpatialGCNLayer(gcn_dims[i], gcn_dims[i + 1], n_adj_ch)
            for i in range(n_gcn_layers)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(gcn_dims[i + 1]) for i in range(n_gcn_layers)
        ])

        # Temporal TCN stack (each node processed independently)
        self.tcn_blocks = nn.ModuleList([
            _TemporalBlock(
                channels=d_hidden,
                kernel_size=tcn_kernel,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(tcn_layers)
        ])

        # MLP readout head: mean-pooled d_hidden → d_embed
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_embed),
        )

        self._d_embed = d_embed

    @property
    def embedding_dim(self) -> int:
        return self._d_embed

    def forward(
        self,
        signal: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        signal    : (B, T, N, d_feat)
        adjacency : (B, T, N, N, C)  — raw (non-normalised) adjacency

        Returns
        -------
        z_e : (B, d_embed)
        """
        B, T, N, _ = signal.shape

        # Time-average adjacency across the episode window: (B, N, N, C)
        adj_mean = adjacency.mean(dim=1)

        # Input projection: (B, T, N, d_hidden)
        h = self.input_proj(signal)

        # Spatial GCN: apply to each timestep
        # Reshape to (B*T, N, d) for batched matmul
        h_bt = h.reshape(B * T, N, -1)
        adj_bt = adj_mean.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, N, N, -1)
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_bt = norm(F.gelu(gcn(h_bt, adj_bt)) + h_bt)  # residual
        h = h_bt.reshape(B, T, N, -1)  # (B, T, N, d_hidden)

        # Temporal TCN: process each node independently over time
        # Reshape to (B*N, d, T) for Conv1d
        h_bn = h.permute(0, 2, 3, 1).reshape(B * N, -1, T)  # (B*N, d, T)
        for tcn in self.tcn_blocks:
            h_bn = tcn(h_bn) + h_bn  # residual
        h = h_bn.reshape(B, N, -1, T).permute(0, 3, 1, 2)  # (B, T, N, d)

        # Global mean pool over T and N
        z = h.mean(dim=(1, 2))  # (B, d_hidden)

        return self.head(z)  # (B, d_embed)
