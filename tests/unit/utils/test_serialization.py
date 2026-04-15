"""Tests for utils.serialization LabelRecord drift metadata."""

from __future__ import annotations

import json

import numpy as np

from graph.diagnostics import GraphStats
from graph.types import ServiceGraph, WeightedEdge
from utils.serialization import (
    DATASET_SCHEMA_VERSION,
    LabelRecord,
    _labels_to_dataframe,
    save_run_dataset,
)


def test_labelrecord_accepts_drift_anomaly_regime() -> None:
    label = LabelRecord(
        timestamp=1_700_000_000.0,
        regime="drift_anomaly",
        category="config",
        scenario="bad_env_config",
        target_services=["payment"],
        chaos_resource="config/bad_env_config.sh",
        episode_id="ep_001",
        drift_flag=True,
    )

    assert label.regime == "drift_anomaly"
    assert label.drift_flag is True


def test_labels_dataframe_keeps_drift_columns() -> None:
    labels = [
        LabelRecord(
            timestamp=1.0,
            regime="normal",
            category="normal",
            scenario="normal",
            target_services=[],
            chaos_resource="",
            drift_flag=False,
        ),
        LabelRecord(
            timestamp=2.0,
            regime="drift_anomaly",
            category="gray",
            scenario="fail_slow_cpu",
            target_services=["checkout"],
            chaos_resource="gray/fail_slow_cpu.yaml",
            drift_flag=True,
        ),
    ]

    df = _labels_to_dataframe(labels)

    assert "drift_flag" in df.columns
    assert list(df["regime"]) == ["normal", "drift_anomaly"]
    assert list(df["drift_flag"]) == [False, True]


def test_save_run_dataset_writes_schema_contract_and_mask(tmp_path) -> None:
    run_dir = tmp_path / "run_contract"

    signal = np.array(
        [
            [[1.0, np.nan], [0.5, 0.2]],
            [[np.nan, 0.1], [0.3, 0.4]],
        ],
        dtype=np.float32,
    )
    # Expand to expected signal dim for realistic contract fields
    signal = np.pad(
        signal,
        pad_width=((0, 0), (0, 0), (0, 15)),
        mode="constant",
        constant_values=np.nan,
    )

    graph = ServiceGraph(
        services=["svc-a", "svc-b"],
        edges=[
            WeightedEdge(
                source="svc-a",
                target="svc-b",
                volume=3,
                latency_median_s=0.2,
                error_rate=0.0,
            )
        ],
        timestamp=1.0,
    )
    labels = [
        LabelRecord(
            timestamp=1.0,
            regime="normal",
            category="normal",
            scenario="normal",
            target_services=[],
            chaos_resource="",
            episode_id="ep_001",
        )
    ]
    graph_stats = [
        GraphStats(
            n_nodes=2,
            n_edges=1,
            density=0.5,
            avg_degree=0.5,
            max_degree=1,
            n_connected_components=1,
            diameter=1,
            largest_component_size=2,
            total_volume=3,
            mean_latency=0.2,
            mean_error_rate=0.0,
            timestamp=1.0,
        )
    ]

    save_run_dataset(
        run_dir=run_dir,
        metadata={"run_id": "run_test", "dry_run": False, "config": {"collection": {"x": 1}}},
        signal_tensor=signal,
        graph_sequence=[graph, graph],
        labels=labels,
        graph_stats=graph_stats,
        services=["svc-a", "svc-b"],
    )

    assert (run_dir / "signal_mask.npz").exists()

    with np.load(run_dir / "signal_mask.npz") as payload:
        missing_mask = payload["missing_mask"]
    assert missing_mask.shape == signal.shape
    assert np.array_equal(missing_mask, np.isnan(signal))

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_schema_version"] == DATASET_SCHEMA_VERSION
    assert metadata["artifacts"]["signal_mask"]["path"] == "signal_mask.npz"
    assert metadata["hashes"]["services_sha256"]
