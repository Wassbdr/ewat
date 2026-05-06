"""Unit tests for src/ewat/encoder/stgcn.py."""

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def encoder():
    from ewat.encoder.stgcn import STGCNEncoder
    return STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=32, d_embed=64, n_gcn_layers=2,
                        tcn_kernel=3, tcn_layers=2, n_adj_ch=3, dropout=0.0)


def _batch(B=2, T=8, N=6, d=17, C=3):
    sig = torch.randn(B, T, N, d)
    adj = torch.rand(B, T, N, N, C).abs()  # non-negative
    return sig, adj


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape(encoder):
    sig, adj = _batch(B=2, T=8)
    with torch.no_grad():
        z = encoder(sig, adj)
    assert z.shape == (2, 64), f"Expected (2, 64), got {z.shape}"


def test_output_shape_single_sample(encoder):
    sig, adj = _batch(B=1, T=10)
    with torch.no_grad():
        z = encoder(sig, adj)
    assert z.shape == (1, 64)


def test_output_shape_batch_8(encoder):
    sig, adj = _batch(B=8, T=5)
    with torch.no_grad():
        z = encoder(sig, adj)
    assert z.shape == (8, 64)


# ---------------------------------------------------------------------------
# Embedding dimension property
# ---------------------------------------------------------------------------

def test_embedding_dim_property(encoder):
    assert encoder.embedding_dim == 64


def test_custom_embedding_dim():
    from ewat.encoder.stgcn import STGCNEncoder
    enc = STGCNEncoder(d_embed=128)
    assert enc.embedding_dim == 128
    sig, adj = _batch(B=1, T=4)
    with torch.no_grad():
        z = enc(sig, adj)
    assert z.shape == (1, 128)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow(encoder):
    sig, adj = _batch(B=2, T=6)
    sig.requires_grad_(True)
    z = encoder(sig, adj)
    loss = z.sum()
    loss.backward()
    assert sig.grad is not None
    assert not torch.isnan(sig.grad).any(), "NaN gradient in signal"


def test_parameter_gradients(encoder):
    sig, adj = _batch(B=2, T=6)
    z = encoder(sig, adj)
    z.sum().backward()
    for name, param in encoder.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_eval(encoder):
    encoder.eval()
    sig, adj = _batch(B=2, T=8)
    with torch.no_grad():
        z1 = encoder(sig, adj)
        z2 = encoder(sig, adj)
    assert torch.allclose(z1, z2, atol=1e-6), "Encoder not deterministic in eval mode"


# ---------------------------------------------------------------------------
# Input invariances
# ---------------------------------------------------------------------------

def test_different_T_same_N(encoder):
    """Model should handle variable T (episode length)."""
    encoder.eval()
    for T in [3, 8, 15, 20]:
        sig, adj = _batch(B=1, T=T)
        with torch.no_grad():
            z = encoder(sig, adj)
        assert z.shape == (1, 64), f"Wrong shape at T={T}"


def test_zero_adjacency(encoder):
    """Zero adjacency = no edges; model must still produce a finite embedding."""
    encoder.eval()
    sig, _ = _batch(B=2, T=6)
    adj_zero = torch.zeros(2, 6, 6, 6, 3)
    with torch.no_grad():
        z = encoder(sig, adj_zero)
    assert not torch.isnan(z).any(), "NaN embedding with zero adjacency"
    assert z.shape == (2, 64)


def test_nan_free_output_on_normal_input(encoder):
    encoder.eval()
    sig, adj = _batch(B=4, T=8)
    with torch.no_grad():
        z = encoder(sig, adj)
    assert not torch.isnan(z).any()
    assert not torch.isinf(z).any()


# ---------------------------------------------------------------------------
# Adjacency channel combination
# ---------------------------------------------------------------------------

def test_gcn_channel_weights_sum_to_one(encoder):
    """Softmax over adjacency channel weights should sum to 1."""
    import torch.nn.functional as F
    for gcn in encoder.gcn_layers:
        w = F.softmax(gcn.ch_weights, dim=0)
        assert abs(w.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Residual connections (output should differ across GCN layers)
# ---------------------------------------------------------------------------

def test_residual_connection_is_active(encoder):
    """With non-zero adjacency the GCN residual should make output ≠ input proj."""
    encoder.eval()
    sig = torch.ones(1, 8, 6, 17) * 0.5
    adj = torch.ones(1, 8, 6, 6, 3) * 0.1
    with torch.no_grad():
        z = encoder(sig, adj)
    z_trivial = encoder.head(encoder.input_proj(sig).mean(dim=(1, 2)))
    assert not torch.allclose(z, z_trivial, atol=1e-4), \
        "GCN residual has no effect — check spatial processing"


# ---------------------------------------------------------------------------
# Dynamic graph variant + masked pool
# ---------------------------------------------------------------------------


def test_dynamic_graph_default():
    from ewat.encoder.stgcn import STGCNEncoder
    enc = STGCNEncoder()
    assert enc.dynamic_graph is True


def test_static_graph_variant_runs():
    from ewat.encoder.stgcn import STGCNEncoder
    enc = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=16, d_embed=8,
                       dynamic_graph=False, dropout=0.0)
    sig, adj = _batch(B=2, T=6)
    with torch.no_grad():
        z = enc(sig, adj)
    assert z.shape == (2, 8)


def test_dynamic_vs_static_differ_when_adjacency_varies(encoder):
    """When the adjacency varies in time, dynamic vs static encoders differ."""
    from ewat.encoder.stgcn import STGCNEncoder
    enc_static = STGCNEncoder(
        d_feat=17, n_nodes=6, d_hidden=32, d_embed=64,
        n_gcn_layers=2, tcn_kernel=3, tcn_layers=2,
        n_adj_ch=3, dropout=0.0, dynamic_graph=False,
    )
    enc_static.load_state_dict(encoder.state_dict())  # share weights
    encoder.eval()
    enc_static.eval()

    sig = torch.randn(1, 8, 6, 17)
    adj = torch.zeros(1, 8, 6, 6, 3)
    # Time-varying connectivity that averages to a constant block.
    adj[0, :4, 0, 1, :] = 1.0
    adj[0, 4:, 1, 2, :] = 1.0
    with torch.no_grad():
        z_dyn = encoder(sig, adj)
        z_stat = enc_static(sig, adj)
    assert not torch.allclose(z_dyn, z_stat, atol=1e-4), (
        "Dynamic and static graphs produce identical embeddings on time-varying A(t)"
    )


def test_layernorm_used_in_temporal_block():
    from ewat.encoder.stgcn import _TemporalBlock
    block = _TemporalBlock(channels=16, kernel_size=3, dilation=1, dropout=0.0)
    block.eval()
    x = torch.randn(2, 16, 8)
    with torch.no_grad():
        y = block(x)
    # The LayerNorm should produce roughly zero-mean / unit-variance
    # outputs along the channel axis (after the gelu+conv).
    means = y.mean(dim=1)
    stds = y.std(dim=1, unbiased=False)
    assert torch.all(means.abs() < 0.5)
    assert torch.all(stds < 5.0)


def test_masked_pool_ignores_padding():
    from ewat.encoder.stgcn import STGCNEncoder
    enc = STGCNEncoder(d_feat=4, n_nodes=3, d_hidden=8, d_embed=4,
                       n_gcn_layers=1, tcn_layers=1, dropout=0.0)
    enc.eval()

    sig = torch.zeros(1, 6, 3, 4)
    sig[0, :3] = torch.randn(3, 3, 4)
    adj = torch.zeros(1, 6, 3, 3, 3)
    adj[0, :3] = torch.rand(3, 3, 3, 3)

    sig_short = sig[:, :3]
    adj_short = adj[:, :3]

    with torch.no_grad():
        z_padded = enc(sig, adj, lengths=torch.tensor([3]))
        z_short = enc(sig_short, adj_short)
    assert torch.allclose(z_padded, z_short, atol=1e-5)


def test_unmasked_pool_includes_padding():
    """Without lengths, padded zeros bias the embedding (regression)."""
    from ewat.encoder.stgcn import STGCNEncoder
    enc = STGCNEncoder(d_feat=4, n_nodes=3, d_hidden=8, d_embed=4,
                       n_gcn_layers=1, tcn_layers=1, dropout=0.0)
    enc.eval()
    sig = torch.zeros(1, 6, 3, 4)
    sig[0, :3] = torch.randn(3, 3, 4)
    adj = torch.zeros(1, 6, 3, 3, 3)
    adj[0, :3] = torch.rand(3, 3, 3, 3)
    with torch.no_grad():
        z_padded = enc(sig, adj)
        z_short = enc(sig[:, :3], adj[:, :3])
    assert not torch.allclose(z_padded, z_short, atol=1e-5)
