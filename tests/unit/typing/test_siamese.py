"""Unit tests for src/ewat/typing/siamese.py."""

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def encoder():
    from ewat.encoder.stgcn import STGCNEncoder
    return STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=32, d_embed=64,
                        n_gcn_layers=2, tcn_kernel=3, tcn_layers=2, n_adj_ch=3,
                        dropout=0.0)


@pytest.fixture()
def typer(encoder):
    from ewat.typing.siamese import SiameseTyper
    return SiameseTyper(encoder=encoder, d_proj=32, freeze_encoder=False)


@pytest.fixture()
def frozen_typer(encoder):
    from ewat.typing.siamese import SiameseTyper
    return SiameseTyper(encoder=encoder, d_proj=32, freeze_encoder=True)


def _batch(B=2, T=8, N=6, d=17, C=3):
    sig = torch.randn(B, T, N, d)
    adj = torch.rand(B, T, N, N, C).abs()
    return sig, adj


# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------

def test_projection_head_output_shape():
    from ewat.typing.siamese import ProjectionHead
    head = ProjectionHead(d_in=64, d_proj=32)
    z = torch.randn(4, 64)
    out = head(z)
    assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"


def test_projection_head_l2_normalized():
    from ewat.typing.siamese import ProjectionHead
    head = ProjectionHead(d_in=64, d_proj=32)
    head.eval()
    z = torch.randn(8, 64)
    with torch.no_grad():
        out = head(z)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), \
        f"Projections not L2-normalised: norms={norms}"


def test_projection_head_different_d_proj():
    from ewat.typing.siamese import ProjectionHead
    head = ProjectionHead(d_in=64, d_proj=16)
    out = head(torch.randn(3, 64))
    assert out.shape == (3, 16)


# ---------------------------------------------------------------------------
# SiameseTyper.embed
# ---------------------------------------------------------------------------

def test_siamese_embed_shape(typer):
    typer.eval()
    sig, adj = _batch(B=4)
    with torch.no_grad():
        z = typer.embed(sig, adj)
    assert z.shape == (4, 32), f"Expected (4, 32), got {z.shape}"


def test_siamese_embed_l2_normalized(typer):
    typer.eval()
    sig, adj = _batch(B=3)
    with torch.no_grad():
        z = typer.embed(sig, adj)
    norms = z.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# SiameseTyper.distance
# ---------------------------------------------------------------------------

def test_siamese_distance_same_input_zero(typer):
    """d(z, z) must be 0 (or very close) for identical projections."""
    z = torch.randn(5, 32)
    z = torch.nn.functional.normalize(z, p=2, dim=-1)
    dist = typer.distance(z, z)
    assert dist.shape == (5,)
    assert torch.allclose(dist, torch.zeros(5), atol=1e-5), \
        f"d(z,z) should be 0, got {dist}"


def test_siamese_distance_nonnegative(typer):
    z1 = torch.nn.functional.normalize(torch.randn(10, 32), p=2, dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(10, 32), p=2, dim=-1)
    dist = typer.distance(z1, z2)
    assert (dist >= -1e-6).all(), f"Distance has negative values: {dist.min()}"


def test_siamese_distance_symmetric(typer):
    z1 = torch.nn.functional.normalize(torch.randn(6, 32), p=2, dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(6, 32), p=2, dim=-1)
    d12 = typer.distance(z1, z2)
    d21 = typer.distance(z2, z1)
    assert torch.allclose(d12, d21, atol=1e-5), "Distance must be symmetric"


def test_siamese_distance_range(typer):
    """Cosine distance ∈ [0, 2] for L2-normalised vectors."""
    z1 = torch.nn.functional.normalize(torch.randn(20, 32), p=2, dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(20, 32), p=2, dim=-1)
    dist = typer.distance(z1, z2)
    assert (dist >= -1e-5).all() and (dist <= 2.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# ContrastiveLoss
# ---------------------------------------------------------------------------

def test_contrastive_loss_positive_pair_zero_distance():
    """If dist=0 for a same pair, loss should be 0."""
    from ewat.typing.siamese import ContrastiveLoss
    loss_fn = ContrastiveLoss(margin=1.0)
    dist = torch.zeros(4)
    is_same = torch.ones(4, dtype=torch.bool)
    loss = loss_fn(dist, is_same)
    assert abs(loss.item()) < 1e-6, f"Expected 0 loss, got {loss.item()}"


def test_contrastive_loss_negative_at_margin_zero():
    """If dist=margin for a diff pair, loss should be 0."""
    from ewat.typing.siamese import ContrastiveLoss
    loss_fn = ContrastiveLoss(margin=1.0)
    dist = torch.ones(4) * 1.0   # dist == margin
    is_same = torch.zeros(4, dtype=torch.bool)
    loss = loss_fn(dist, is_same)
    assert abs(loss.item()) < 1e-6, f"Expected 0 loss, got {loss.item()}"


def test_contrastive_loss_hard_negative_penalised():
    """Hard negative (dist=0) should receive max penalty: margin²."""
    from ewat.typing.siamese import ContrastiveLoss
    margin = 1.0
    loss_fn = ContrastiveLoss(margin=margin)
    dist = torch.zeros(1)        # hard negative: dist=0 < margin
    is_same = torch.zeros(1, dtype=torch.bool)
    loss = loss_fn(dist, is_same)
    assert abs(loss.item() - margin ** 2) < 1e-5, \
        f"Expected {margin**2}, got {loss.item()}"


def test_contrastive_loss_positive_at_large_distance():
    """Same pair at large distance → large loss (proportional to d²)."""
    from ewat.typing.siamese import ContrastiveLoss
    loss_fn = ContrastiveLoss(margin=1.0)
    dist = torch.ones(1) * 1.5    # same pair far apart
    is_same = torch.ones(1, dtype=torch.bool)
    loss = loss_fn(dist, is_same)
    assert abs(loss.item() - 1.5 ** 2) < 1e-5


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_through_encoder(typer):
    sig, adj = _batch(B=2)
    sig.requires_grad_(True)
    z = typer.embed(sig, adj)
    z.sum().backward()
    assert sig.grad is not None
    assert not torch.isnan(sig.grad).any()


def test_freeze_encoder_no_encoder_grad(frozen_typer):
    """When freeze_encoder=True, encoder params must have requires_grad=False."""
    for name, param in frozen_typer.encoder.named_parameters():
        assert not param.requires_grad, \
            f"Encoder param '{name}' should be frozen"
    # Head params must still have grad
    for name, param in frozen_typer.head.named_parameters():
        assert param.requires_grad, \
            f"Head param '{name}' should require grad"
