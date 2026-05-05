"""Tests for temporal relation computation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ewat.ontology.temporal import compute_temporal_relations


def _make_manifest_and_labels(tmpdir: Path) -> tuple[dict[str, dict], list]:
    """Create a synthetic manifest and feature store for testing."""
    manifest = {}
    records = []
    # Simulate 3 clusters and 12 episodes with known timestamps and scenarios
    rng = np.random.default_rng(0)
    base_time = 1_700_000_000.0

    for i, (cluster, scenario) in enumerate([
        (0, "cpu_stress"), (1, "mem_pressure"), (0, "cpu_stress"),
        (1, "mem_pressure"), (2, "net_loss"), (0, "cpu_stress"),
        (1, "mem_pressure"), (2, "net_loss"), (0, "cpu_stress"),
        (1, "mem_pressure"), (2, "net_loss"), (0, "cpu_stress"),
    ]):
        ep_id = f"ep_{i:03d}"
        t0 = base_time + i * 1000.0 + rng.uniform(0, 100)

        # Write labels.parquet
        ep_dir = tmpdir / ep_id
        ep_dir.mkdir()
        timestamps = np.linspace(t0, t0 + 600, 12)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "regime": ["normal"] * 12,
            "category": [scenario] * 12,
            "scenario": [scenario] * 12,
            "target_services": ["svc"] * 12,
            "chaos_resource": ["cpu"] * 12,
            "episode_id": [ep_id] * 12,
            "drift_flag": [False] * 12,
            "target_service": ["svc"] * 12,
            "is_injection": [False] * 12,
        })
        df.to_parquet(ep_dir / "labels.parquet", index=False)

        manifest[ep_id] = {"cluster": cluster, "split": "train", "scenario": scenario}
        records.append((t0, ep_id, cluster))

    return manifest, records


def test_temporal_returns_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, _ = _make_manifest_and_labels(Path(tmpdir))
        rels = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=2,
            max_delta_seconds=5000.0,
        )
    assert isinstance(rels, list)


def test_temporal_relation_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, _ = _make_manifest_and_labels(Path(tmpdir))
        rels = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=2,
            max_delta_seconds=5000.0,
        )
    for r in rels:
        assert r.relation_type == "temporal"


def test_temporal_support_geq_min():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, _ = _make_manifest_and_labels(Path(tmpdir))
        min_sup = 3
        rels = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=min_sup,
            max_delta_seconds=5000.0,
        )
    for r in rels:
        assert r.support >= min_sup


def test_temporal_delta_t_nonnegative():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, _ = _make_manifest_and_labels(Path(tmpdir))
        rels = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=1,
            max_delta_seconds=5000.0,
        )
    for r in rels:
        assert r.delta_t_mean is not None
        assert r.delta_t_mean >= 0
        assert r.delta_t_std is not None
        assert r.delta_t_std >= 0


def test_temporal_max_delta_filters():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, _ = _make_manifest_and_labels(Path(tmpdir))
        rels_large = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=1,
            max_delta_seconds=1e9,
        )
        rels_small = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=1,
            max_delta_seconds=0.001,
        )
    # With tiny window, no consecutive pairs → no relations
    assert len(rels_large) >= len(rels_small)


def test_temporal_source_target_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, _ = _make_manifest_and_labels(Path(tmpdir))
        n_clusters = 3
        rels = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=1,
            max_delta_seconds=5000.0,
        )
    for r in rels:
        assert 0 <= r.source < n_clusters
        assert 0 <= r.target < n_clusters


def test_temporal_high_min_support_returns_fewer():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, _ = _make_manifest_and_labels(Path(tmpdir))
        rels_low = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=1,
            max_delta_seconds=5000.0,
        )
        rels_high = compute_temporal_relations(
            cluster_manifest=manifest,
            features_root=Path(tmpdir),
            min_support=100,
            max_delta_seconds=5000.0,
        )
    assert len(rels_high) <= len(rels_low)
    assert len(rels_high) == 0
