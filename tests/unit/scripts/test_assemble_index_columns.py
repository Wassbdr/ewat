"""Tests for the assemble_dataset CLI defaults and index columns.

Covers Step 3 fixes (audit 2026-05-26):
- 3.1 : --stratified opt-in, --no-stratified opt-out, warning when temporal
- 3.3 : --copy-episodes default=True, --symlink-episodes opt-out
- 3.4 : index.parquet exposes target_services + chaos_resource
"""

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.assemble_dataset import FeaturedEpisode, _build_index   # noqa: E402


def _make_ep(
    episode_id: str = "ep_001",
    scenario: str = "fail_slow_cpu",
    category: str = "anomaly",
    targets: list[str] | None = None,
    chaos_resource: str = "chaos/fail_slow_cpu.yaml",
    nan_total: float = 0.05,
) -> FeaturedEpisode:
    return FeaturedEpisode(
        path=Path(f"/tmp/{episode_id}"),
        episode_id=episode_id,
        scenario=scenario,
        category=category,
        n_timesteps=48,
        services=["frontend", "cart", "checkout"],
        nan_ratio_total=nan_total,
        nan_ratio_metrics=0.03,
        nan_ratio_traces=0.10,
        nan_ratio_logs=0.02,
        baseline_start=1700000000.0,
        recovery_end=1700001000.0,
        target_services=targets or ["frontend"],
        chaos_resource=chaos_resource,
    )


def test_index_includes_target_services_and_chaos_resource():
    eps = [
        _make_ep("ep_001", scenario="fail_slow_cpu",
                 targets=["frontend"], chaos_resource="fail_slow_cpu.yaml"),
        _make_ep("ep_002", scenario="oom",
                 targets=["cart", "checkout"], chaos_resource="oom.yaml"),
    ]
    split = {"train": ["ep_001"], "val": [], "test": ["ep_002"]}
    df = _build_index(eps, split)
    assert "target_services" in df.columns
    assert "chaos_resource" in df.columns

    # JSON-serialised list of targets
    import json
    row1 = df[df["episode_id"] == "ep_001"].iloc[0]
    assert json.loads(row1["target_services"]) == ["frontend"]
    assert row1["chaos_resource"] == "fail_slow_cpu.yaml"
    row2 = df[df["episode_id"] == "ep_002"].iloc[0]
    assert json.loads(row2["target_services"]) == ["cart", "checkout"]


def test_index_preserves_pre_existing_columns():
    eps = [_make_ep("ep_001")]
    split = {"train": ["ep_001"], "val": [], "test": []}
    df = _build_index(eps, split)
    for col in [
        "episode_id", "scenario", "category", "split", "n_timesteps",
        "baseline_start", "recovery_end",
        "nan_ratio_total", "nan_ratio_metrics", "nan_ratio_traces", "nan_ratio_logs",
    ]:
        assert col in df.columns, f"missing column {col}"


def test_featured_episode_defaults():
    """A FeaturedEpisode without target_services/chaos_resource keeps backward-compat defaults."""
    ep = FeaturedEpisode(
        path=Path("/tmp"),
        episode_id="x",
        scenario="s",
        category="c",
        n_timesteps=1,
        services=["a"],
        nan_ratio_total=0.0,
        nan_ratio_metrics=0.0,
        nan_ratio_traces=0.0,
        nan_ratio_logs=0.0,
        baseline_start=0.0,
        recovery_end=0.0,
    )
    assert ep.target_services == []
    assert ep.chaos_resource == ""


def test_cli_parser_defaults():
    """--copy-episodes defaults to True; --stratified defaults to False; --no-stratified opt-out exists."""
    from scripts.assemble_dataset import _cli

    # Inject argv via monkeypatching sys.argv
    old_argv = sys.argv
    try:
        sys.argv = ["assemble_dataset.py", "--features-root", "/tmp/f", "--output", "/tmp/o"]
        args = _cli()
        assert args.copy_episodes is True, "copy_episodes default should be True (Step 3 fix 3.3)"
        assert args.stratified is False
        assert args.no_stratified is False
    finally:
        sys.argv = old_argv


def test_cli_symlink_opts_out_of_copy():
    from scripts.assemble_dataset import _cli

    old_argv = sys.argv
    try:
        sys.argv = ["assemble_dataset.py", "--features-root", "/tmp/f",
                    "--output", "/tmp/o", "--symlink-episodes"]
        args = _cli()
        assert args.copy_episodes is False
    finally:
        sys.argv = old_argv


def test_cli_no_stratified_flag():
    from scripts.assemble_dataset import _cli

    old_argv = sys.argv
    try:
        sys.argv = ["assemble_dataset.py", "--features-root", "/tmp/f",
                    "--output", "/tmp/o", "--no-stratified"]
        args = _cli()
        assert args.no_stratified is True
    finally:
        sys.argv = old_argv
