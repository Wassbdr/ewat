"""Unit tests for src/ewat/typing/pairs.py."""

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal EpisodeDataset-like object for testing."""

    def __init__(self, scenario_map: dict[int, str]) -> None:
        self._items = {i: {"scenario": s, "episode_id": f"ep_{i}"} for i, s in scenario_map.items()}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


@pytest.fixture()
def small_ds():
    # 3 scenarios × 4 episodes each = 12 episodes
    mapping = {}
    for sc_i, sc in enumerate(["cpu_stress", "network_delay", "oom"]):
        for ep_i in range(4):
            mapping[sc_i * 4 + ep_i] = sc
    return _FakeDataset(mapping)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_positive_pairs_have_same_scenario(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    sampler = EpisodePairSampler(small_ds, n_neg_per_anchor=1, seed=0)
    positive_pairs = [(i, j) for i, j, same in sampler if same]
    assert len(positive_pairs) > 0
    for i, j in positive_pairs:
        assert small_ds[i]["scenario"] == small_ds[j]["scenario"]


def test_negative_pairs_have_different_scenario(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    sampler = EpisodePairSampler(small_ds, n_neg_per_anchor=2, seed=0)
    negative_pairs = [(i, j) for i, j, same in sampler if not same]
    assert len(negative_pairs) > 0
    for i, j in negative_pairs:
        assert small_ds[i]["scenario"] != small_ds[j]["scenario"]


def test_reproducible_with_same_seed(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    pairs_1 = list(EpisodePairSampler(small_ds, n_neg_per_anchor=3, seed=42))
    pairs_2 = list(EpisodePairSampler(small_ds, n_neg_per_anchor=3, seed=42))
    assert pairs_1 == pairs_2, "Same seed must give identical pair sequence"


def test_different_seeds_different_order(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    pairs_1 = list(EpisodePairSampler(small_ds, n_neg_per_anchor=3, seed=0))
    pairs_2 = list(EpisodePairSampler(small_ds, n_neg_per_anchor=3, seed=99))
    assert pairs_1 != pairs_2


def test_n_negatives_per_anchor_respected(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    n_neg = 3
    sampler = EpisodePairSampler(small_ds, n_neg_per_anchor=n_neg, seed=7)
    pairs = list(sampler)
    # Count negatives per anchor
    from collections import Counter
    neg_counts = Counter()
    for i, j, same in pairs:
        if not same:
            neg_counts[i] += 1
    # Each anchor should have at most n_neg negatives
    for anchor, count in neg_counts.items():
        assert count <= n_neg, f"Anchor {anchor} has {count} negatives (max {n_neg})"


def test_all_episodes_appear_as_anchors(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    sampler = EpisodePairSampler(small_ds, n_neg_per_anchor=2, seed=0)
    pairs = list(sampler)
    anchors_seen = {i for i, j, same in pairs}
    all_idxs = set(range(len(small_ds)))
    assert anchors_seen == all_idxs, \
        f"Missing anchors: {all_idxs - anchors_seen}"


def test_no_self_pairs(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    sampler = EpisodePairSampler(small_ds, n_neg_per_anchor=3, seed=0)
    for i, j, same in sampler:
        assert i != j, f"Self-pair found: idx {i}"


def test_scenario_index_built_correctly(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    sampler = EpisodePairSampler(small_ds, n_neg_per_anchor=1, seed=0)
    idx_map = sampler.scenario_to_idxs
    assert set(idx_map.keys()) == {"cpu_stress", "network_delay", "oom"}
    for sc, idxs in idx_map.items():
        assert len(idxs) == 4
        for i in idxs:
            assert small_ds[i]["scenario"] == sc
