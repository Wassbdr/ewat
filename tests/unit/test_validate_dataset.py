"""Tests for dataset validation checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.validate_dataset import (
    check_metadata_contract,
    check_signal_mask,
    check_temporal_split,
    run_checks,
)


def test_temporal_split_detects_episode_leakage() -> None:
    labels = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "episode_id": ["ep1", "ep1", "ep2", "ep2", "ep2"],
            "regime": ["normal", "injection", "normal", "injection", "recovery"],
        }
    )

    result = check_temporal_split(labels)

    assert not result.passed
    assert "crossing split boundary" in result.details.lower()


def test_temporal_split_passes_without_episode_leakage() -> None:
    labels = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "episode_id": ["ep1", "ep1", "ep1", "ep1", "ep2"],
            "regime": ["normal", "injection", "recovery", "normal", "injection"],
        }
    )

    result = check_temporal_split(labels)

    assert result.passed


def test_run_checks_skips_quality_for_dry_run(tmp_path) -> None:
    run_dir = tmp_path / "run_dry"
    run_dir.mkdir()

    np.savez_compressed(run_dir / "signal.npz", signal=np.zeros((0, 0, 17), dtype=np.float32))
    np.savez_compressed(
        run_dir / "adjacency.npz",
        adjacency=np.zeros((0, 0, 0, 3), dtype=np.float32),
    )

    labels = pd.DataFrame(
        columns=[
            "timestamp",
            "regime",
            "category",
            "scenario",
            "target_services",
            "target_service",
            "chaos_resource",
            "episode_id",
            "is_injection",
        ]
    )
    labels.to_parquet(run_dir / "labels.parquet", index=False)

    graph_stats = pd.DataFrame(
        columns=[
            "timestamp",
            "n_nodes",
            "n_edges",
            "density",
            "avg_degree",
            "max_degree",
            "n_connected_components",
            "diameter",
            "largest_component_size",
            "total_volume",
            "mean_latency",
            "mean_error_rate",
            "regime",
            "scenario",
            "category",
            "episode_id",
        ]
    )
    graph_stats.to_csv(run_dir / "graph_stats.csv", index=False)

    metadata = {"dry_run": True}
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    checks, failures = run_checks(
        run_dir=run_dir,
        min_coverage_episodes=20,
        min_distribution_episodes=15,
        max_nan_ratio=0.2,
        min_baseline_edges=5,
        strict_dry_run=False,
    )

    assert failures == 0
    assert len(checks) == 1
    assert checks[0].name == "dry_run_mode"
    assert checks[0].passed


def test_run_checks_strict_dry_run_fails_on_empty_data(tmp_path) -> None:
    run_dir = tmp_path / "run_dry_strict"
    run_dir.mkdir()

    np.savez_compressed(run_dir / "signal.npz", signal=np.zeros((0, 0, 17), dtype=np.float32))
    np.savez_compressed(
        run_dir / "adjacency.npz",
        adjacency=np.zeros((0, 0, 0, 3), dtype=np.float32),
    )

    labels = pd.DataFrame(columns=["timestamp", "regime", "scenario", "episode_id"])
    labels.to_parquet(run_dir / "labels.parquet", index=False)

    graph_stats = pd.DataFrame(columns=["regime", "n_edges"])
    graph_stats.to_csv(run_dir / "graph_stats.csv", index=False)

    metadata = {"dry_run": True}
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    checks, failures = run_checks(
        run_dir=run_dir,
        min_coverage_episodes=20,
        min_distribution_episodes=15,
        max_nan_ratio=0.2,
        min_baseline_edges=5,
        strict_dry_run=True,
    )

    assert failures > 0
    assert any(not c.passed for c in checks)


def test_check_signal_mask_detects_shape_mismatch() -> None:
    signal = np.array([[1.0, np.nan]], dtype=np.float32)
    mask = np.array([[False, True, False]], dtype=bool)

    result = check_signal_mask(signal, mask)

    assert not result.passed
    assert "shape mismatch" in result.details


def test_check_signal_mask_detects_value_mismatch() -> None:
    signal = np.array([[1.0, np.nan]], dtype=np.float32)
    mask = np.array([[False, False]], dtype=bool)

    result = check_signal_mask(signal, mask)

    assert not result.passed
    assert "mismatch" in result.details


def test_check_signal_mask_passes_when_consistent() -> None:
    signal = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
    mask = np.isnan(signal)

    result = check_signal_mask(signal, mask)

    assert result.passed


def test_check_metadata_contract_passes_with_complete_schema() -> None:
    signal = np.zeros((2, 3, 17), dtype=np.float32)
    signal_mask = np.zeros_like(signal, dtype=bool)
    adjacency = np.zeros((2, 3, 3, 3), dtype=np.float32)

    metadata = {
        "dataset_schema_version": "1.2.0",
        "signal_dim_expected": 17,
        "artifacts": {
            "signal": {},
            "signal_mask": {},
            "adjacency": {},
            "labels": {},
            "graph_stats": {},
            "services": {},
        },
        "hashes": {"services_sha256": "abc"},
    }

    result = check_metadata_contract(metadata, signal, signal_mask, adjacency)
    assert result.passed
