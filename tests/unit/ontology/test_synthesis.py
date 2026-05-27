"""Tests for the composite episode synthesis module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ewat.ontology.synthesis import (
    COMPOSITE_TRANSITION_REGIME,
    EpisodeBundle,
    audit_realism_corpus,
    cascade_episodes,
    clip_to_p99,
    load_episode,
    overlay_episodes,
    realism_envelope,
    write_episode,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = REPO_ROOT / "data/features/v3"
EP_A_NAME = "episode_cpu_starvation_000_20260430T022941Z"
EP_B_NAME = "episode_memory_pressure_000_20260430T063620Z"


# ---------------------------------------------------------------------------
# Synthetic fixtures (no I/O)
# ---------------------------------------------------------------------------


def _toy_bundle(t: int = 20, scenario: str = "fake_scen") -> EpisodeBundle:
    rng = np.random.default_rng(0)
    n_services, n_features = 6, 17
    signal = rng.standard_normal((t, n_services, n_features)).astype(np.float32)
    adjacency = rng.standard_normal((t, n_services, n_services, 3)).astype(np.float32)
    labels = pd.DataFrame({
        "timestamp": np.arange(t) * 30.0,
        "regime": (["normal"] * (t // 2)) + (["injection"] * (t - t // 2)),
        "scenario": [scenario] * t,
        "episode_id": [f"toy_{scenario}"] * t,
    })
    metadata = {
        "episode_id": f"toy_{scenario}",
        "scenario": {"name": scenario, "category": "test", "targets": ["frontend"]},
        "canonical_services": [
            "ad", "cart", "frontend", "load-generator",
            "product-catalog", "recommendation",
        ],
    }
    return EpisodeBundle(
        episode_id=f"toy_{scenario}",
        signal=signal,
        adjacency=adjacency,
        labels=labels,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# clip_to_p99
# ---------------------------------------------------------------------------


def test_clip_to_p99_caps_overshoots():
    sig = np.array([[[1.0, 5.0]]], dtype=np.float32)  # (1, 1, 2)
    p99 = np.array([[3.0, 4.0]], dtype=np.float32)
    p01 = np.array([[0.0, 0.0]], dtype=np.float32)
    out, frac = clip_to_p99(sig, p99, p01)
    assert out[0, 0, 1] == pytest.approx(4.0)
    assert frac == pytest.approx(0.5)


def test_clip_to_p99_preserves_in_envelope():
    sig = np.array([[[2.0, 3.0]]], dtype=np.float32)
    p99 = np.array([[5.0, 5.0]], dtype=np.float32)
    p01 = np.array([[0.0, 0.0]], dtype=np.float32)
    out, frac = clip_to_p99(sig, p99, p01)
    np.testing.assert_array_equal(out, sig)
    assert frac == 0.0


# ---------------------------------------------------------------------------
# realism_envelope
# ---------------------------------------------------------------------------


def test_realism_envelope_shapes():
    eps = [_toy_bundle(20, "a"), _toy_bundle(20, "b")]
    p01, p99 = realism_envelope(eps)
    assert p01.shape == (6, 17)
    assert p99.shape == (6, 17)
    assert np.all(p01 <= p99)


def test_realism_envelope_empty_raises():
    with pytest.raises(ValueError):
        realism_envelope([])


# ---------------------------------------------------------------------------
# overlay_episodes (toy)
# ---------------------------------------------------------------------------


def test_overlay_truncates_to_shortest():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(12, "b")
    out, _ = overlay_episodes(ep_a, ep_b, alpha=1.0)
    assert out.n_steps == 12


def test_overlay_preserves_n_services_and_features():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    out, _ = overlay_episodes(ep_a, ep_b, alpha=0.5)
    assert out.signal.shape == ep_a.signal.shape


def test_overlay_records_composite_metadata():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    out, _ = overlay_episodes(ep_a, ep_b, alpha=0.5)
    assert out.metadata["composite"]["kind"] == "overlay"
    assert out.metadata["composite"]["scenario_a"] == "a"
    assert out.metadata["composite"]["scenario_b"] == "b"
    assert out.metadata["composite"]["alpha"] == 0.5


def test_overlay_alpha_zero_is_identity():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    out, _ = overlay_episodes(ep_a, ep_b, alpha=0.0)
    np.testing.assert_allclose(out.signal, ep_a.signal, atol=1e-5)


# ---------------------------------------------------------------------------
# cascade_episodes (toy)
# ---------------------------------------------------------------------------


def test_cascade_length_is_sum_plus_gap():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    out, _ = cascade_episodes(ep_a, ep_b, gap_steps=5)
    assert out.n_steps == 45


def test_cascade_first_segment_equals_a():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    out, _ = cascade_episodes(ep_a, ep_b, gap_steps=5)
    np.testing.assert_allclose(out.signal[:20], ep_a.signal, atol=1e-5)


def test_cascade_last_segment_equals_b():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    out, _ = cascade_episodes(ep_a, ep_b, gap_steps=5)
    np.testing.assert_allclose(out.signal[-20:], ep_b.signal, atol=1e-5)


def test_cascade_bridge_uses_composite_transition_label():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    out, _ = cascade_episodes(ep_a, ep_b, gap_steps=5)
    bridge_rows = out.labels.iloc[20:25]
    assert (bridge_rows["regime"] == COMPOSITE_TRANSITION_REGIME).all()


def test_cascade_with_zero_gap_concatenates_directly():
    ep_a = _toy_bundle(15, "a")
    ep_b = _toy_bundle(15, "b")
    out, _ = cascade_episodes(ep_a, ep_b, gap_steps=0)
    assert out.n_steps == 30


def test_cascade_negative_gap_raises():
    ep_a = _toy_bundle(20, "a")
    ep_b = _toy_bundle(20, "b")
    with pytest.raises(ValueError):
        cascade_episodes(ep_a, ep_b, gap_steps=-1)


# ---------------------------------------------------------------------------
# write_episode round-trip
# ---------------------------------------------------------------------------


def test_write_episode_creates_canonical_layout(tmp_path: Path):
    bundle = _toy_bundle(15, "rt")
    out_dir = write_episode(bundle, tmp_path)
    assert (out_dir / "signal.npz").exists()
    assert (out_dir / "adjacency.npz").exists()
    assert (out_dir / "labels.parquet").exists()
    assert (out_dir / "metadata.json").exists()


def test_write_episode_round_trip(tmp_path: Path):
    bundle = _toy_bundle(15, "rt")
    out_dir = write_episode(bundle, tmp_path)
    reloaded = load_episode(out_dir)
    np.testing.assert_allclose(reloaded.signal, bundle.signal)
    assert reloaded.scenario == "rt"


# ---------------------------------------------------------------------------
# Integration with real ewat_v3 episodes
# ---------------------------------------------------------------------------


REAL_DATA_AVAILABLE = (FEATURES_DIR / EP_A_NAME / "signal.npz").exists()


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="ewat_v3 features absent")
def test_real_cascade_passes_garde_fous():
    ep_a = load_episode(FEATURES_DIR / EP_A_NAME)
    ep_b = load_episode(FEATURES_DIR / EP_B_NAME)
    p01, p99 = realism_envelope([ep_a, ep_b])
    _, check = cascade_episodes(
        ep_a, ep_b, gap_steps=5, p01_table=p01, p99_table=p99,
    )
    # Cascade preserves A's first segment unchanged → median rank ≈ 1
    assert check.spearman_median >= 0.95
    assert check.passed


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="ewat_v3 features absent")
def test_real_overlay_alpha_0_3_passes():
    ep_a = load_episode(FEATURES_DIR / EP_A_NAME)
    ep_b = load_episode(FEATURES_DIR / EP_B_NAME)
    p01, p99 = realism_envelope([ep_a, ep_b])
    _, check = overlay_episodes(
        ep_a, ep_b, alpha=0.3, p01_table=p01, p99_table=p99, spearman_min=0.85,
    )
    assert check.spearman_median >= 0.85
    assert check.passed


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="ewat_v3 features absent")
def test_real_overlay_alpha_1_0_rejected():
    """Garde-fou correctly rejects aggressive overlay alpha=1.0."""
    ep_a = load_episode(FEATURES_DIR / EP_A_NAME)
    ep_b = load_episode(FEATURES_DIR / EP_B_NAME)
    p01, p99 = realism_envelope([ep_a, ep_b])
    _, check = overlay_episodes(
        ep_a, ep_b, alpha=1.0, p01_table=p01, p99_table=p99, spearman_min=0.85,
    )
    assert not check.passed
    assert any("spearman" in r for r in check.reasons)


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="ewat_v3 features absent")
def test_discriminator_auc_below_threshold():
    """At corpus level, synthetic episodes must be hard to distinguish."""
    cpu_eps = sorted(FEATURES_DIR.glob("episode_cpu*"))[:5]
    mem_eps = sorted(FEATURES_DIR.glob("episode_memory*"))[:5]
    real = [load_episode(p) for p in cpu_eps + mem_eps]
    p01, p99 = realism_envelope(real)
    synthetic = []
    for i in range(len(cpu_eps)):
        o, _ = overlay_episodes(real[i], real[i + 5], alpha=0.3,
                                p01_table=p01, p99_table=p99)
        c, _ = cascade_episodes(real[i], real[i + 5], gap_steps=5,
                                p01_table=p01, p99_table=p99)
        synthetic += [o, c]
    auc = audit_realism_corpus(real, synthetic)
    assert auc < 0.75
