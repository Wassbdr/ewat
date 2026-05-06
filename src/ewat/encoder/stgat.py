"""ST-GAT encoder — Graph Attention variant of :class:`STGCNEncoder`.

The layout mirrors ``stgcn.py`` so this module is a drop-in replacement:

* same ``(B, T, N, d_feat)`` + ``(B, T, N, N, C)`` input shape;
* same ``(B, d_embed)`` output;
* same temporal block stack and masked pooling;
* same ``arch`` metadata serialised into checkpoints.

The only architectural difference is the spatial layer: instead of the
spectral GCN with degree normalisation (``D^{-1/2} A D^{-1/2}``), each
node attends to its neighbours through a multi-head Graph Attention
mechanism (Veličković et al., GAT, 2018), with the raw multi-channel
adjacency used as a soft mask: edges with zero weight across all
channels receive ``-∞`` attention logits, effectively pruning them.

Provide ``architecture: "gat"`` to the encoder factory or pass a
:class:`STGATEncoder` instance to the typing pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpatialGATLayer(nn.Module):
    """Multi-head Graph Attention layer with soft adjacency masking.

    Parameters
    ----------
    in_features:
        Input feature dim per node.
    out_features:
        Output dim per head. The total output dim is ``out_features * n_heads``
        when ``concat=True``, else ``out_features``.
    n_heads:
        Number of attention heads.
    n_adj_channels:
        Number of adjacency channels. The C channels are summed into a soft
        mask used to gate the attention logits (no edge ⇒ ``-∞``).
    concat:
        If ``True`` (default), head outputs are concatenated. If ``False``,
        averaged.
    dropout:
        Attention dropout (Veličković et al.).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        n_adj_channels: int = 3,
        concat: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_features))
        self.bias = nn.Parameter(
            torch.zeros(out_features * n_heads if concat else out_features)
        )
        self.ch_weights = nn.Parameter(torch.ones(n_adj_channels) / n_adj_channels)
        self.leaky_slope = 0.2

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    @property
    def output_dim(self) -> int:
        return self.out_features * self.n_heads if self.concat else self.out_features

    def _adj_mask(self, adj: torch.Tensor) -> torch.Tensor:
        """Collapse C adjacency channels into a single soft mask in ``[0, 1]``.

        ``adj`` may be ``(B, N, N, C)`` or ``(B, T, N, N, C)``. The output
        has the trailing channel removed.
        """
        ch_w = F.softmax(self.ch_weights, dim=0)        # (C,)
        return (adj * ch_w).sum(dim=-1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:   ``(B, N, in_features)`` (static) or ``(B, T, N, in_features)``.
        adj: ``(B, N, N, C)`` (static) or ``(B, T, N, N, C)`` (dynamic).
        """
        dynamic = (x.dim() == 4 and adj.dim() == 5)

        if dynamic:
            B, T, N, _ = x.shape
            h = self.W(x).view(B, T, N, self.n_heads, self.out_features)
            adj_mask = self._adj_mask(adj)              # (B, T, N, N)
        else:
            B, N, _ = x.shape
            h = self.W(x).view(B, N, self.n_heads, self.out_features)
            adj_mask = self._adj_mask(adj)              # (B, N, N)

        # Per-head attention scores e_ij = LeakyReLU(a_src·h_i + a_dst·h_j).
        e_src = (h * self.a_src).sum(dim=-1)            # (..., N, H)
        e_dst = (h * self.a_dst).sum(dim=-1)            # (..., N, H)
        e_src = e_src.unsqueeze(-2)                     # (..., N, 1, H)
        e_dst = e_dst.unsqueeze(-3)                     # (..., 1, N, H)
        logits = F.leaky_relu(e_src + e_dst, negative_slope=self.leaky_slope)
        # logits shape: (..., N, N, H)

        # Apply soft adjacency mask: zero-weight edges → −∞ logits.
        mask_bool = adj_mask > 0                        # (..., N, N)
        mask_bool = mask_bool.unsqueeze(-1)             # (..., N, N, 1)
        logits = logits.masked_fill(~mask_bool, float("-inf"))

        # Add the soft mask in log space so attention is also modulated by
        # edge magnitude (not just presence).
        logits = logits + torch.log(adj_mask.unsqueeze(-1).clamp(min=1e-12))

        # Replace fully-masked rows (no neighbours, e.g. isolated node at this
        # timestep) with a uniform self-attention to avoid NaN softmax.
        all_neg_inf = torch.isneginf(logits).all(dim=-2, keepdim=True)
        if all_neg_inf.any():
            uniform = torch.zeros_like(logits)
            logits = torch.where(all_neg_inf.expand_as(logits), uniform, logits)

        attn = F.softmax(logits, dim=-2)               # softmax over source N
        attn = self.dropout(attn)

        # Aggregate: out[..., i, h, d] = sum_j attn[..., j, i, h] * h[..., j, h, d]
        # Reshape for matmul: stack heads as a new batch axis.
        if dynamic:
            B, T, N, H, D = h.shape
            # attn: (B, T, N_src, N_dst, H) → (B*T*H, N_dst, N_src)
            attn_t = attn.permute(0, 1, 4, 3, 2).reshape(B * T * H, N, N)
            h_t = h.permute(0, 1, 3, 2, 4).reshape(B * T * H, N, D)
            out = torch.bmm(attn_t, h_t)                # (B*T*H, N, D)
            out = out.view(B, T, H, N, D).permute(0, 1, 3, 2, 4)
            if self.concat:
                out = out.reshape(B, T, N, H * D)
            else:
                out = out.mean(dim=-2)
        else:
            B, N, H, D = h.shape
            attn_t = attn.permute(0, 3, 2, 1).reshape(B * H, N, N)
            h_t = h.permute(0, 2, 1, 3).reshape(B * H, N, D)
            out = torch.bmm(attn_t, h_t).view(B, H, N, D).permute(0, 2, 1, 3)
            if self.concat:
                out = out.reshape(B, N, H * D)
            else:
                out = out.mean(dim=-2)

        return out + self.bias


class _TemporalBlock(nn.Module):
    """Identical to ``stgcn._TemporalBlock`` — duplicated to avoid a runtime
    import cycle and keep this module self-contained.
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
        out = self.conv(x)
        out = out[..., : -self._pad] if self._pad else out
        out = F.gelu(out)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.drop(out)


class STGATEncoder(nn.Module):
    """Spatio-temporal encoder using Graph Attention.

    Drop-in replacement for :class:`ewat.encoder.stgcn.STGCNEncoder`. The
    ``embedding_dim`` property and ``forward(signal, adjacency, lengths)``
    signature are identical so callers (typing, alerts, etc.) work
    transparently when handed an :class:`STGATEncoder`.
    """

    def __init__(
        self,
        d_feat: int = 17,
        n_nodes: int = 6,
        d_hidden: int = 64,
        d_embed: int = 64,
        n_gat_layers: int = 2,
        n_heads: int = 4,
        tcn_kernel: int = 3,
        tcn_layers: int = 2,
        n_adj_ch: int = 3,
        dropout: float = 0.1,
        dynamic_graph: bool = True,
        concat_heads: bool = True,
    ) -> None:
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if concat_heads and d_hidden % n_heads != 0:
            raise ValueError(
                "d_hidden must be divisible by n_heads when concat_heads=True"
            )

        head_dim = d_hidden // n_heads if concat_heads else d_hidden
        self.input_proj = nn.Linear(d_feat, d_hidden)

        self.gat_layers = nn.ModuleList([
            _SpatialGATLayer(
                in_features=d_hidden,
                out_features=head_dim,
                n_heads=n_heads,
                n_adj_channels=n_adj_ch,
                concat=concat_heads,
                dropout=dropout,
            )
            for _ in range(n_gat_layers)
        ])
        self.gat_norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_gat_layers)])

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
        self._n_heads = n_heads

    @property
    def embedding_dim(self) -> int:
        return self._d_embed

    @property
    def dynamic_graph(self) -> bool:
        return self._dynamic_graph

    @property
    def n_heads(self) -> int:
        return self._n_heads

    def forward(
        self,
        signal: torch.Tensor,
        adjacency: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, _ = signal.shape

        h = self.input_proj(signal)  # (B, T, N, d_hidden)

        if self._dynamic_graph:
            for gat, norm in zip(self.gat_layers, self.gat_norms):
                h_new = gat(h, adjacency)
                h = norm(F.gelu(h_new) + h)
        else:
            adj_mean = adjacency.mean(dim=1)
            h_bt = h.reshape(B * T, N, -1)
            adj_bt = (
                adj_mean.unsqueeze(1)
                .expand(-1, T, -1, -1, -1)
                .reshape(B * T, N, N, -1)
            )
            for gat, norm in zip(self.gat_layers, self.gat_norms):
                h_bt = norm(F.gelu(gat(h_bt, adj_bt)) + h_bt)
            h = h_bt.reshape(B, T, N, -1)

        h_bn = h.permute(0, 2, 3, 1).reshape(B * N, -1, T)
        for tcn in self.tcn_blocks:
            h_bn = tcn(h_bn) + h_bn
        h = h_bn.reshape(B, N, -1, T).permute(0, 3, 1, 2)

        if lengths is not None:
            lengths = lengths.to(device=h.device, dtype=torch.long)
            max_T = h.shape[1]
            time_idx = torch.arange(max_T, device=h.device).unsqueeze(0)
            mask = (time_idx < lengths.unsqueeze(1)).float()
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            z_t = (h * mask).sum(dim=1) / denom
            z = z_t.mean(dim=1)
        else:
            z = h.mean(dim=(1, 2))

        return self.head(z)
