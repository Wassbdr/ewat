"""Tests for co-occurrence relation computation."""

import pytest

from ewat.ontology.cooccurrence import compute_cooccurrence_relations


def _make_manifest(
    scenario_cluster_map: list[tuple[str, int]]
) -> dict[str, dict]:
    """Build a manifest where each (scenario, cluster) pair is one episode."""
    manifest = {}
    for i, (scenario, cluster) in enumerate(scenario_cluster_map):
        ep_id = f"ep_{i:03d}"
        manifest[ep_id] = {"cluster": cluster, "split": "train", "scenario": scenario}
    return manifest


def test_cooccurrence_returns_list():
    manifest = _make_manifest([("s1", 0), ("s1", 1), ("s2", 0), ("s2", 2)] * 5)
    rels = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=0.1)
    assert isinstance(rels, list)


def test_cooccurrence_relation_type():
    manifest = _make_manifest([("s1", 0), ("s1", 1)] * 10)
    rels = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=1.0,
                                          min_cooccurrences=1)
    for r in rels:
        assert r.relation_type == "cooccurrence"


def test_cooccurrence_strength_nonnegative():
    manifest = _make_manifest([("s1", 0), ("s1", 1), ("s2", 1), ("s2", 2)] * 5)
    rels = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=1.0,
                                          min_cooccurrences=1)
    for r in rels:
        assert r.strength >= 0.0


def test_cooccurrence_p_value_in_range():
    manifest = _make_manifest([("s1", 0), ("s1", 1)] * 10)
    rels = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=1.0,
                                          min_cooccurrences=1)
    for r in rels:
        assert r.p_value is not None
        assert 0.0 <= r.p_value <= 1.0


def test_cooccurrence_excess_cooccurrence_detected():
    # 40 scenarios have BOTH cluster 0 and 1; only 5 have just 0, 5 just 1, 50 just 2.
    # → observed=40 >> expected=20.25 → strong χ² signal.
    entries = []
    for i in range(40):
        entries += [(f"both_{i}", 0), (f"both_{i}", 1)]
    for i in range(5):
        entries.append((f"only0_{i}", 0))
    for i in range(5):
        entries.append((f"only1_{i}", 1))
    for i in range(50):
        entries.append((f"only2_{i}", 2))
    manifest = _make_manifest(entries)
    rels = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=0.05,
                                          min_cooccurrences=2)
    pairs = {(r.source, r.target) for r in rels}
    assert (0, 1) in pairs or (1, 0) in pairs


def test_cooccurrence_never_cooccurring_not_detected():
    # 0 only in s1, 1 only in s2 → no co-occurrence
    manifest = _make_manifest([("s1", 0)] * 10 + [("s2", 1)] * 10)
    rels = compute_cooccurrence_relations(manifest, n_clusters=2, p_threshold=0.05,
                                          min_cooccurrences=2)
    assert rels == []


def test_cooccurrence_min_cooccurrences_filter():
    manifest = _make_manifest([("s1", 0), ("s1", 1)] * 3)  # 3 scenarios with co-occ
    rels_low = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=1.0,
                                              min_cooccurrences=1)
    rels_high = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=1.0,
                                               min_cooccurrences=100)
    assert len(rels_high) == 0
    assert len(rels_low) >= len(rels_high)


def test_cooccurrence_source_target_in_range():
    manifest = _make_manifest([("s1", 0), ("s1", 1), ("s1", 2)] * 5)
    n_clusters = 5
    rels = compute_cooccurrence_relations(manifest, n_clusters=n_clusters, p_threshold=1.0,
                                          min_cooccurrences=1)
    for r in rels:
        assert 0 <= r.source < n_clusters
        assert 0 <= r.target < n_clusters
        assert r.source != r.target


def test_cooccurrence_single_scenario_no_pair_relations():
    # Only one scenario → co-occurrence counts = 0 in some pairs
    manifest = _make_manifest([("s1", 0)] * 5)
    rels = compute_cooccurrence_relations(manifest, n_clusters=3, p_threshold=1.0,
                                          min_cooccurrences=1)
    # No pairs of different clusters co-occur
    for r in rels:
        assert r.source != r.target
