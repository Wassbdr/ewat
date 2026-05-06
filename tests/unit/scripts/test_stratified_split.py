"""Tests for the stratified split logic in scripts.assemble_dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from scripts.assemble_dataset import _stratified_temporal_split


@dataclass
class _FE:
    episode_id: str
    scenario: str
    baseline_start: float


def _make(scenarios: dict[str, int]) -> list[_FE]:
    """Create episodes with monotonically increasing baseline_start per scenario."""
    out: list[_FE] = []
    counter = 0
    for sc, n in scenarios.items():
        for i in range(n):
            out.append(
                _FE(episode_id=f"{sc}_{i:02d}", scenario=sc, baseline_start=float(counter))
            )
            counter += 1
    return out


def test_split_respects_ratios_when_groups_large():
    eps = _make({"sc1": 20, "sc2": 20})
    split = _stratified_temporal_split(eps, 0.7, 0.15)
    assert len(split["train"]) >= 26
    assert len(split["val"]) >= 4
    assert len(split["test"]) >= 4


def test_min_test_per_group_floor():
    eps = _make({"big": 50, "small": 3})
    split = _stratified_temporal_split(
        eps, 0.7, 0.15, min_test_per_group=1, min_val_per_group=1,
    )
    test_eps = set(split["test"])
    big_test = [e for e in eps if e.scenario == "big" and e.episode_id in test_eps]
    small_test = [e for e in eps if e.scenario == "small" and e.episode_id in test_eps]
    assert len(big_test) >= 1
    assert len(small_test) >= 1


def test_min_test_per_group_strong_floor():
    eps = _make({"sc": 12})
    split = _stratified_temporal_split(
        eps, 0.7, 0.15, min_test_per_group=3, min_val_per_group=2,
    )
    test_eps = set(split["test"])
    val_eps = set(split["val"])
    assert len(test_eps) == 3
    assert len(val_eps) == 2
    assert len(split["train"]) == 12 - 3 - 2


def test_grouping_by_cluster_keeps_each_cluster_in_test():
    """Two clusters spread across episodes — both should appear in test."""
    eps = _make({"sc1": 30})
    grouping = {ep.episode_id: ("c0" if i < 15 else "c1") for i, ep in enumerate(eps)}
    split = _stratified_temporal_split(
        eps, 0.7, 0.15, grouping=grouping,
        min_test_per_group=1, min_val_per_group=1,
    )
    test_set = set(split["test"])
    clusters_in_test = {grouping[eid] for eid in test_set}
    assert clusters_in_test == {"c0", "c1"}


def test_temporal_order_per_group():
    eps = _make({"sc1": 10})
    split = _stratified_temporal_split(eps, 0.7, 0.15)
    test_ids = split["test"]
    indices = [int(ep_id.split("_")[1]) for ep_id in test_ids]
    assert indices == sorted(indices)


def test_negative_min_raises():
    eps = _make({"sc": 5})
    with pytest.raises(SystemExit):
        _stratified_temporal_split(eps, 0.7, 0.15, min_test_per_group=-1)


def test_no_overlap_between_splits():
    eps = _make({"sc1": 20, "sc2": 15})
    split = _stratified_temporal_split(eps, 0.7, 0.15)
    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert (train | val | test) == {ep.episode_id for ep in eps}


def test_distribution_counts_consistent():
    eps = _make({"sc1": 10, "sc2": 6})
    split = _stratified_temporal_split(eps, 0.7, 0.15)
    counts = Counter()
    for split_name in ("train", "val", "test"):
        for eid in split[split_name]:
            sc = next(e.scenario for e in eps if e.episode_id == eid)
            counts[(sc, split_name)] += 1
    # Every scenario contributes to every split.
    for sc in ("sc1", "sc2"):
        assert counts[(sc, "train")] >= 1
        assert counts[(sc, "test")] >= 1
