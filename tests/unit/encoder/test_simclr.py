"""Unit tests for SimCLR pre-training utilities (src/ewat/encoder/simclr.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ewat.encoder.simclr import (
    AugmentationConfig,
    SimCLRHead,
    SimCLRTrainer,
    augment_temporal_pair,
    collate_simclr_views,
    nt_xent_loss,
)
from ewat.encoder.stgcn import STGCNEncoder


# --------------------------------------------------------------------------- #
# nt_xent_loss
# --------------------------------------------------------------------------- #

class TestNTXent:
    def test_scalar_output(self):
        z1 = F.normalize(torch.randn(4, 8), dim=1)
        z2 = F.normalize(torch.randn(4, 8), dim=1)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_low_loss_when_views_match(self):
        """Identical views give the lowest possible loss."""
        torch.manual_seed(0)
        z = F.normalize(torch.randn(8, 16), dim=1)
        same = nt_xent_loss(z, z.clone(), temperature=0.1)
        random = nt_xent_loss(z, F.normalize(torch.randn(8, 16), dim=1), temperature=0.1)
        assert same.item() < random.item()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            nt_xent_loss(torch.randn(4, 8), torch.randn(4, 7))

    def test_returns_zero_for_batch_size_one(self):
        loss = nt_xent_loss(torch.randn(1, 8), torch.randn(1, 8))
        assert float(loss) == 0.0

    def test_gradient_flows_through_loss(self):
        raw1 = torch.randn(4, 8, requires_grad=True)
        raw2 = torch.randn(4, 8, requires_grad=True)
        z1 = F.normalize(raw1, dim=1)
        z2 = F.normalize(raw2, dim=1)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        loss.backward()
        assert raw1.grad is not None
        assert raw2.grad is not None
        assert torch.isfinite(raw1.grad).all()
        assert torch.isfinite(raw2.grad).all()


# --------------------------------------------------------------------------- #
# Augmentations
# --------------------------------------------------------------------------- #

class TestAugmentations:
    def _episode(self, T: int = 12, N: int = 4, d: int = 17, seed: int = 0):
        rng = np.random.default_rng(seed)
        sig = torch.from_numpy(rng.normal(size=(T, N, d)).astype(np.float32))
        adj = torch.from_numpy(rng.uniform(0, 1, size=(T, N, N, 3)).astype(np.float32))
        return sig, adj

    def test_two_views_have_expected_shapes(self):
        sig, adj = self._episode()
        cfg = AugmentationConfig(crop_min=0.7, crop_max=0.9, seed=1)
        gen = torch.Generator().manual_seed(1)
        (s1, a1), (s2, a2) = augment_temporal_pair(sig, adj, cfg=cfg, rng=gen)
        assert s1.shape[1:] == sig.shape[1:]
        assert a1.shape[1:] == adj.shape[1:]
        assert s2.shape[1:] == sig.shape[1:]

    def test_views_differ_under_stochastic_augmentations(self):
        sig, adj = self._episode(seed=10)
        cfg = AugmentationConfig(seed=2)
        gen = torch.Generator().manual_seed(2)
        (s1, _), (s2, _) = augment_temporal_pair(sig, adj, cfg=cfg, rng=gen)
        assert not torch.equal(s1, s2) or s1.shape != s2.shape

    def test_seed_reproducibility(self):
        sig, adj = self._episode(seed=7)
        cfg = AugmentationConfig(seed=42)
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        (sa1, _), (sa2, _) = augment_temporal_pair(sig, adj, cfg=cfg, rng=gen_a)
        (sb1, _), (sb2, _) = augment_temporal_pair(sig, adj, cfg=cfg, rng=gen_b)
        assert sa1.shape == sb1.shape and sa2.shape == sb2.shape
        assert torch.equal(sa1, sb1)
        assert torch.equal(sa2, sb2)

    def test_collate_pads_to_longest(self):
        sig_a = torch.randn(8, 4, 17)
        sig_b = torch.randn(12, 4, 17)
        adj_a = torch.randn(8, 4, 4, 3)
        adj_b = torch.randn(12, 4, 4, 3)
        sigs, adjs, lens = collate_simclr_views([(sig_a, adj_a), (sig_b, adj_b)])
        assert sigs.shape == (2, 12, 4, 17)
        assert adjs.shape == (2, 12, 4, 4, 3)
        assert lens.tolist() == [8, 12]


# --------------------------------------------------------------------------- #
# End-to-end mini training
# --------------------------------------------------------------------------- #

class TestSimCLRTrainer:
    def _episode(self, T: int = 10, N: int = 4, d: int = 17, seed: int = 0):
        rng = np.random.default_rng(seed)
        return {
            "signal": torch.from_numpy(
                rng.normal(size=(T, N, d)).astype(np.float32)
            ),
            "adjacency": torch.from_numpy(
                rng.uniform(0, 1, size=(T, N, N, 3)).astype(np.float32)
            ),
        }

    def test_one_epoch_reduces_loss(self):
        torch.manual_seed(0)
        encoder = STGCNEncoder(
            d_feat=17, n_nodes=4, d_hidden=16, d_embed=16,
            n_gcn_layers=2, tcn_kernel=3, tcn_layers=1,
        )
        head = SimCLRHead(d_in=16, d_proj=8)
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()), lr=1e-3,
        )
        eps = [self._episode(seed=i) for i in range(8)]

        class _ListLoader:
            def __init__(self, eps, batch_size):
                self.eps, self.bs = eps, batch_size
            def __iter__(self):
                for i in range(0, len(self.eps), self.bs):
                    yield self.eps[i: i + self.bs]
            def __len__(self):
                return (len(self.eps) + self.bs - 1) // self.bs

        loader = _ListLoader(eps, batch_size=4)
        trainer = SimCLRTrainer(encoder, head, opt, temperature=0.5, seed=1)
        s1 = trainer.run_epoch(loader, epoch=1)
        s2 = trainer.run_epoch(loader, epoch=2)
        s3 = trainer.run_epoch(loader, epoch=3)
        assert s1.train_loss > 0
        # Loss should generally decrease over the first few epochs on this
        # tiny synthetic fixture (allow a small amount of slack).
        assert s3.train_loss <= s1.train_loss + 1e-3

    def test_simclr_head_output_is_unit_norm(self):
        head = SimCLRHead(d_in=8, d_proj=4)
        z = head(torch.randn(3, 8))
        norms = torch.linalg.norm(z, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
