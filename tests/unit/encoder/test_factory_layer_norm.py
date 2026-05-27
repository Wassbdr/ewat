"""Tests for build_encoder factory + auto-detection of use_layer_norm.

Covers Step 5 fix 5.3 (audit 2026-05-26).
"""

import torch
import pytest

from ewat.encoder.factory import (
    build_encoder,
    build_encoder_from_checkpoint,
    detect_use_layer_norm,
)
from ewat.encoder.stgcn import STGCNEncoder


def test_detect_use_layer_norm_true_when_norm_keys_present():
    """A state_dict with tcn_blocks.*.norm.weight indicates LN was enabled."""
    state_dict = {
        "input_proj.weight": torch.zeros(64, 17),
        "tcn_blocks.0.conv.weight": torch.zeros(64, 64, 3),
        "tcn_blocks.0.norm.weight": torch.zeros(64),
        "tcn_blocks.0.norm.bias": torch.zeros(64),
        "head.0.weight": torch.zeros(64, 64),
    }
    assert detect_use_layer_norm(state_dict) is True


def test_detect_use_layer_norm_false_when_no_norm_keys():
    state_dict = {
        "input_proj.weight": torch.zeros(64, 17),
        "tcn_blocks.0.conv.weight": torch.zeros(64, 64, 3),
        "tcn_blocks.0.conv.bias": torch.zeros(64),
        "head.0.weight": torch.zeros(64, 64),
    }
    assert detect_use_layer_norm(state_dict) is False


def test_detect_ignores_other_norm_layers():
    """gcn_norms.* must not falsely trigger detection (those are always present)."""
    state_dict = {
        "input_proj.weight": torch.zeros(64, 17),
        "gcn_norms.0.weight": torch.zeros(64),
        "gcn_norms.0.bias": torch.zeros(64),
        "tcn_blocks.0.conv.weight": torch.zeros(64, 64, 3),
    }
    assert detect_use_layer_norm(state_dict) is False


def test_build_encoder_from_checkpoint_roundtrip_with_ln():
    """Save → load via build_encoder_from_checkpoint should reproduce exact weights.

    Both encoders are set to eval() to disable stochastic dropout layers.
    """
    enc = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64,
                       use_layer_norm=True)
    enc.eval()
    ckpt = {"encoder_state": enc.state_dict(),
            "arch": {"architecture": "stgcn", "d_feat": 17, "n_nodes": 6,
                     "d_hidden": 64, "d_embed": 64}}
    loaded = build_encoder_from_checkpoint(ckpt)
    loaded.load_state_dict(ckpt["encoder_state"])
    loaded.eval()
    sig = torch.randn(2, 5, 6, 17)
    adj = torch.randn(2, 5, 6, 6, 3).abs()
    with torch.no_grad():
        out_orig = enc(sig, adj, lengths=torch.tensor([5, 5]))
        out_loaded = loaded(sig, adj, lengths=torch.tensor([5, 5]))
    torch.testing.assert_close(out_orig, out_loaded)


def test_build_encoder_from_checkpoint_roundtrip_no_ln():
    enc = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64,
                       use_layer_norm=False)
    enc.eval()
    ckpt = {"encoder_state": enc.state_dict()}
    loaded = build_encoder_from_checkpoint(ckpt)
    loaded.load_state_dict(ckpt["encoder_state"])
    loaded.eval()
    sig = torch.randn(1, 3, 6, 17)
    adj = torch.randn(1, 3, 6, 6, 3).abs()
    with torch.no_grad():
        out_orig = enc(sig, adj, lengths=torch.tensor([3]))
        out_loaded = loaded(sig, adj, lengths=torch.tensor([3]))
    torch.testing.assert_close(out_orig, out_loaded)


def test_stgcn_forward_rejects_zero_length():
    """Step 5 fix 5.2: forward must reject lengths=0 with explicit ValueError."""
    enc = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    sig = torch.randn(2, 5, 6, 17)
    adj = torch.randn(2, 5, 6, 6, 3).abs()
    with pytest.raises(ValueError, match="lengths must be >= 1"):
        enc(sig, adj, lengths=torch.tensor([3, 0]))


def test_stgcn_forward_warns_on_batch_without_lengths():
    """Step 5 fix 5.2: forward without lengths on batched input warns user."""
    enc = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    sig = torch.randn(3, 5, 6, 17)
    adj = torch.randn(3, 5, 6, 6, 3).abs()
    with pytest.warns(UserWarning, match="lengths=None on a batch"):
        enc(sig, adj, lengths=None)


def test_stgcn_forward_no_warning_for_single_sample_without_lengths():
    """Batch size 1 without lengths should not warn (no padding ambiguity)."""
    enc = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    sig = torch.randn(1, 5, 6, 17)
    adj = torch.randn(1, 5, 6, 6, 3).abs()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning would raise
        enc(sig, adj, lengths=None)


def test_stgcn_masked_pool_correct_for_padded_sample():
    """Padded positions must be excluded from the mean."""
    enc = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    enc.eval()
    # Same valid prefix, different padding
    valid = torch.randn(1, 4, 6, 17)
    pad = torch.zeros(1, 6, 6, 17)
    sig = torch.cat([valid, pad], dim=1)   # (1, 10, 6, 17)
    adj = torch.randn(1, 10, 6, 6, 3).abs()
    with torch.no_grad():
        z_padded = enc(sig, adj, lengths=torch.tensor([4]))
        z_valid_only = enc(valid, adj[:, :4], lengths=torch.tensor([4]))
    # Both should produce nearly the same embedding (padded ignored)
    torch.testing.assert_close(z_padded, z_valid_only, atol=1e-4, rtol=1e-4)
