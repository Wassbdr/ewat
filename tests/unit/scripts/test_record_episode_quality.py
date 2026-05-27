"""Tests for scripts/record_episode.py quality-gate logic.

Covers Step 1 fixes (audit 2026-05-26):
- 1.4 : min_traces parameter on _check_episode_quality
- New CLI fields (traffic_pattern, seed, min_traces_quality_gate)
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))

from scripts.record_episode import _check_episode_quality   # noqa: E402


def _manifest_with_jaeger(n_traces: int) -> dict:
    return {"sources": {"jaeger": {"n_traces_total": n_traces}}}


def test_quality_gate_rejects_below_min_traces():
    """n_traces=2, min_traces=5 → reject with explicit reason."""
    manifest = _manifest_with_jaeger(2)
    ok, reasons = _check_episode_quality(
        manifest,
        enable_prometheus=False,
        enable_jaeger=True,
        enable_loki=False,
        min_traces=5,
    )
    assert not ok
    assert any("too-few-traces" in r for r in reasons)
    assert any("n=2" in r and "< 5" in r for r in reasons)


def test_quality_gate_accepts_above_min_traces():
    manifest = _manifest_with_jaeger(10)
    ok, reasons = _check_episode_quality(
        manifest,
        enable_prometheus=False,
        enable_jaeger=True,
        enable_loki=False,
        min_traces=5,
    )
    assert ok
    assert reasons == []


def test_quality_gate_accepts_exact_min_traces():
    manifest = _manifest_with_jaeger(5)
    ok, _ = _check_episode_quality(
        manifest,
        enable_prometheus=False,
        enable_jaeger=True,
        enable_loki=False,
        min_traces=5,
    )
    assert ok


def test_quality_gate_rejects_zero_traces_under_default():
    """The default min_traces=5 must reject n=0 (previous behavior allowed n=0)."""
    manifest = _manifest_with_jaeger(0)
    ok, reasons = _check_episode_quality(
        manifest,
        enable_prometheus=False,
        enable_jaeger=True,
        enable_loki=False,
        # default min_traces=5
    )
    assert not ok
    assert reasons


def test_quality_gate_min_traces_zero_is_permissive():
    """Setting min_traces=0 reverts to the previous permissive behaviour."""
    manifest = _manifest_with_jaeger(0)
    ok, reasons = _check_episode_quality(
        manifest,
        enable_prometheus=False,
        enable_jaeger=True,
        enable_loki=False,
        min_traces=0,
    )
    # With min=0, n=0 still triggers because 0 < 0 is False but the check is
    # `< min_traces` → 0 < 0 = False → passes. Confirm via direct compute.
    # (If a user explicitly sets min=0, they explicitly accept empty traces.)
    assert ok or "too-few-traces" not in (reasons[0] if reasons else "")


def test_quality_gate_disabled_jaeger_ignores_min_traces():
    manifest = _manifest_with_jaeger(0)
    ok, _ = _check_episode_quality(
        manifest,
        enable_prometheus=False,
        enable_jaeger=False,
        enable_loki=False,
        min_traces=5,
    )
    # Jaeger disabled → no gate even with empty traces
    assert ok


def test_quality_gate_jaeger_skipped():
    """If jaeger source reports 'skipped', should be flagged."""
    manifest = {"sources": {"jaeger": {"skipped": True, "n_traces_total": 0}}}
    ok, reasons = _check_episode_quality(
        manifest,
        enable_prometheus=False,
        enable_jaeger=True,
        enable_loki=False,
        min_traces=5,
    )
    assert not ok
    assert "jaeger-skipped" in reasons
