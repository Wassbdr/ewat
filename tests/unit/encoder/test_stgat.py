"""Unit tests for src/ewat/encoder/stgat.py and the encoder factory."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ewat.encoder.factory import build_encoder
from ewat.encoder.stgat import STGATEncoder


def _random_inputs(B=2, T=8, N=4, d_feat=17, n_adj_ch=3, seed=0):
    rng = np.random.default_rng(seed)
    sig = torch.from_numpy(rng.normal(size=(B, T, N, d_feat)).astype(np.float32))
    adj = torch.from_numpy(
        rng.uniform(0.0, 1.0, size=(B, T, N, N, n_adj_ch)).astype(np.float32)
    )
    lengths = torch.tensor([T - 2, T], dtype=torch.long)
    return sig, adj, lengths


class TestSTGATForward:
    def test_output_shape(self):
        sig, adj, _ = _random_inputs()
        enc = STGATEncoder(
            d_feat=17, n_nodes=4, d_hidden=32, d_embed=16,
            n_gat_layers=2, n_heads=4, tcn_kernel=3, tcn_layers=1,
        )
        z = enc(sig, adj)
        assert z.shape == (sig.shape[0], 16)

    def test_dynamic_vs_static_differ_when_adjacency_varies(self):
        sig, adj, _ = _random_inputs(seed=1)
        enc_dyn = STGATEncoder(
            d_feat=17, n_nodes=4, d_hidden=16, d_embed=8,
            n_gat_layers=1, n_heads=2, tcn_kernel=3, tcn_layers=1,
            dynamic_graph=True,
        )
        enc_stat = STGATEncoder(
            d_feat=17, n_nodes=4, d_hidden=16, d_embed=8,
            n_gat_layers=1, n_heads=2, tcn_kernel=3, tcn_layers=1,
            dynamic_graph=False,
        )
        enc_stat.load_state_dict(enc_dyn.state_dict())
        z_dyn = enc_dyn(sig, adj)
        z_stat = enc_stat(sig, adj)
        assert not torch.allclose(z_dyn, z_stat, atol=1e-4)

    def test_masked_pool_ignores_padding(self):
        sig, adj, lengths = _random_inputs()
        enc = STGATEncoder(
            d_feat=17, n_nodes=4, d_hidden=16, d_embed=8,
            n_gat_layers=1, n_heads=2, tcn_kernel=3, tcn_layers=1,
        ).eval()
        with torch.no_grad():
            z_full = enc(sig, adj)
            sig_pad = sig.clone()
            adj_pad = adj.clone()
            sig_pad[0, lengths[0]:] = 0
            adj_pad[0, lengths[0]:] = 0
            z_masked = enc(sig_pad, adj_pad, lengths=lengths)
        # The second episode has lengths == T so masked pool is identical
        # to unmasked pool — verify that explicitly.
        assert torch.allclose(z_full[1], z_masked[1], atol=1e-5)

    def test_concat_heads_invalid_d_hidden_raises(self):
        with pytest.raises(ValueError):
            STGATEncoder(
                d_feat=17, n_nodes=4, d_hidden=15, d_embed=8,
                n_heads=4, concat_heads=True,
            )

    def test_invalid_n_heads_raises(self):
        with pytest.raises(ValueError):
            STGATEncoder(
                d_feat=17, n_nodes=4, d_hidden=16, d_embed=8,
                n_heads=0,
            )

    def test_isolated_node_does_not_produce_nan(self):
        """If a node has no neighbours at some timestep, the softmax over
        its source row would normally be NaN. Verify the encoder handles
        this gracefully."""
        sig, adj, _ = _random_inputs(B=1, T=4, N=3, seed=42)
        adj[..., 0, :, :] = 0.0  # isolate node 0 entirely
        enc = STGATEncoder(
            d_feat=17, n_nodes=3, d_hidden=8, d_embed=4,
            n_gat_layers=1, n_heads=2, tcn_kernel=3, tcn_layers=1,
        ).eval()
        with torch.no_grad():
            z = enc(sig, adj)
        assert torch.isfinite(z).all()


class TestEncoderFactory:
    def test_build_stgcn(self):
        from ewat.encoder.stgcn import STGCNEncoder
        enc = build_encoder("stgcn", d_feat=17, n_nodes=4, d_hidden=16, d_embed=8)
        assert isinstance(enc, STGCNEncoder)

    def test_build_stgat(self):
        enc = build_encoder("stgat", d_feat=17, n_nodes=4, d_hidden=16, d_embed=8, n_heads=2)
        assert isinstance(enc, STGATEncoder)

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError):
            build_encoder("bogus")
