"""Tests self-loops STGCN + validation de shapes (M4+M5, audit 2026-06)."""

from __future__ import annotations

import pytest
import torch

from ewat.encoder.factory import build_encoder_from_checkpoint
from ewat.encoder.stgcn import STGCNEncoder, _SpatialGCNLayer


def _toy_batch(b=2, t=5, n=4, d=17, c=3):
    sig = torch.randn(b, t, n, d)
    adj = torch.rand(b, t, n, n, c)
    return sig, adj


# ---------------------------------------------------------------------------
# M4 — self-loops
# ---------------------------------------------------------------------------


def test_isolated_node_gets_zero_message_without_self_loops():
    layer = _SpatialGCNLayer(8, 8, use_self_loops=False)
    x = torch.randn(1, 3, 8)
    adj = torch.zeros(1, 3, 3, 3)
    adj[0, 0, 1, :] = 1.0  # nœud 2 isolé
    a_norm = layer._normalised_adj(adj)
    assert torch.allclose(a_norm[0, 2], torch.zeros(3)), \
        "ligne d'un nœud isolé doit être nulle sans self-loops"
    out = layer(x, adj)
    # sans self-loop, la sortie du nœud isolé = biais seul (msg nul)
    assert torch.allclose(out[0, 2], layer.bias, atol=1e-6)


def test_isolated_node_keeps_identity_with_self_loops():
    layer = _SpatialGCNLayer(8, 8, use_self_loops=True)
    x = torch.randn(1, 3, 8)
    adj = torch.zeros(1, 3, 3, 3)
    adj[0, 0, 1, :] = 1.0  # nœud 2 isolé
    a_norm = layer._normalised_adj(adj)
    # self-loop unitaire : le nœud isolé s'agrège lui-même (poids 1)
    assert float(a_norm[0, 2, 2]) == pytest.approx(1.0, abs=1e-5)
    out = layer(x, adj)
    expected = layer.linear(x[0, 2]) + layer.bias
    assert torch.allclose(out[0, 2], expected, atol=1e-5)


def test_self_loops_default_off_backward_compat():
    # Le défaut de la couche ET de l'encodeur doit rester False : le flag n'a
    # pas d'empreinte state_dict, les anciens checkpoints gardent leur forward.
    assert _SpatialGCNLayer(4, 4).use_self_loops is False
    assert STGCNEncoder()._use_self_loops is False


def test_encoder_forward_with_self_loops():
    enc = STGCNEncoder(use_self_loops=True)
    sig, adj = _toy_batch(n=6)
    z = enc(sig, adj)
    assert z.shape == (2, 64)
    assert torch.isfinite(z).all()


def test_factory_reads_self_loops_from_arch_meta():
    enc = STGCNEncoder(use_self_loops=True)
    ckpt = {
        "encoder_state": enc.state_dict(),
        "arch": {"architecture": "stgcn", "d_feat": 17, "n_nodes": 6,
                 "d_hidden": 64, "d_embed": 64, "use_self_loops": True},
    }
    rebuilt = build_encoder_from_checkpoint(ckpt)
    assert rebuilt._use_self_loops is True
    rebuilt.load_state_dict(ckpt["encoder_state"])

    # checkpoint legacy sans la clé → False (forward d'entraînement préservé)
    ckpt_legacy = {
        "encoder_state": enc.state_dict(),
        "arch": {"architecture": "stgcn", "d_feat": 17, "n_nodes": 6,
                 "d_hidden": 64, "d_embed": 64},
    }
    assert build_encoder_from_checkpoint(ckpt_legacy)._use_self_loops is False


# ---------------------------------------------------------------------------
# M5 — validation de shapes au forward
# ---------------------------------------------------------------------------


def test_forward_rejects_t_mismatch():
    enc = STGCNEncoder()
    sig = torch.randn(2, 5, 6, 17)
    adj = torch.rand(2, 6, 6, 6, 3)  # T=6 ≠ 5
    with pytest.raises(ValueError, match="incompatible"):
        enc(sig, adj)


def test_forward_rejects_n_mismatch():
    enc = STGCNEncoder()
    sig = torch.randn(2, 5, 6, 17)
    adj = torch.rand(2, 5, 4, 4, 3)  # N=4 ≠ 6
    with pytest.raises(ValueError, match="incompatible"):
        enc(sig, adj)


def test_forward_rejects_wrong_ndim():
    enc = STGCNEncoder()
    sig = torch.randn(5, 6, 17)  # pas de dim batch
    adj = torch.rand(2, 5, 6, 6, 3)
    with pytest.raises(ValueError, match="expected signal"):
        enc(sig, adj)


def test_forward_valid_shapes_unchanged():
    enc = STGCNEncoder()
    sig, adj = _toy_batch(n=6)
    z = enc(sig, adj)
    assert z.shape == (2, 64)
