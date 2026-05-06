"""Unit tests for src/ewat/typing/pairs.py."""

import numpy as np
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


# ---------------------------------------------------------------------------
# Hard / semi-hard mining
# ---------------------------------------------------------------------------

def test_unknown_mining_strategy_raises(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    with pytest.raises(ValueError):
        EpisodePairSampler(small_ds, mining="bogus")


def test_hard_mining_falls_back_to_random_without_embeddings(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    s_random = EpisodePairSampler(small_ds, n_neg_per_anchor=2, seed=42, mining="random")
    s_hard = EpisodePairSampler(small_ds, n_neg_per_anchor=2, seed=42, mining="hard")
    assert list(s_random) == list(s_hard)


def test_hard_mining_picks_closest_negatives(small_ds):
    """When embeddings are provided, the hardest negatives are the ones with
    the smallest L2 distance to the anchor — *not* random."""
    from ewat.typing.pairs import EpisodePairSampler

    sampler = EpisodePairSampler(
        small_ds, n_neg_per_anchor=2, seed=0, mining="hard",
    )

    n = len(small_ds)
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(n, 8)).astype(np.float32)
    sampler.update_embeddings(embeddings)

    pairs = list(sampler)
    negatives_by_anchor: dict[int, list[int]] = {}
    for i, j, same in pairs:
        if not same:
            negatives_by_anchor.setdefault(i, []).append(j)

    for anchor, negs in negatives_by_anchor.items():
        anchor_scenario = small_ds[anchor]["scenario"]
        all_negs = [k for k in range(n) if small_ds[k]["scenario"] != anchor_scenario]
        anchor_vec = embeddings[anchor]
        all_dists = np.linalg.norm(embeddings[all_negs] - anchor_vec[None, :], axis=1)
        sorted_negs = [all_negs[i] for i in np.argsort(all_dists)]
        expected = sorted_negs[: len(negs)]
        assert sorted(negs) == sorted(expected), (
            f"hard mining for anchor {anchor} should pick {expected}, got {negs}"
        )


def test_semi_hard_mining_respects_margin(small_ds):
    """Semi-hard negatives are farther than the positive but within margin
    of the positive distance."""
    from ewat.typing.pairs import EpisodePairSampler

    n = len(small_ds)
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(n, 4)).astype(np.float32)

    sampler = EpisodePairSampler(
        small_ds, n_neg_per_anchor=1, seed=0, mining="semi-hard", margin=1.0,
    )
    sampler.update_embeddings(embeddings)

    pairs = list(sampler)
    pos_by_anchor = {i: j for i, j, same in pairs if same}
    neg_by_anchor: dict[int, list[int]] = {}
    for i, j, same in pairs:
        if not same:
            neg_by_anchor.setdefault(i, []).append(j)

    for anchor, negs in neg_by_anchor.items():
        if anchor not in pos_by_anchor:
            continue
        a_vec = embeddings[anchor]
        d_pos = float(np.linalg.norm(embeddings[pos_by_anchor[anchor]] - a_vec))
        for n_idx in negs:
            d_neg = float(np.linalg.norm(embeddings[n_idx] - a_vec))
            anchor_scenario = small_ds[anchor]["scenario"]
            all_negs = [
                k for k in range(n) if small_ds[k]["scenario"] != anchor_scenario
            ]
            all_dists = np.linalg.norm(
                embeddings[all_negs] - a_vec[None, :], axis=1
            )
            semi_mask = (all_dists > d_pos) & (all_dists < d_pos + sampler.margin)
            if semi_mask.any():
                # When semi-hard candidates exist, the picked negative must
                # belong to that set.
                assert d_neg > d_pos, (
                    f"semi-hard pick {n_idx} (d={d_neg:.3f}) is closer than "
                    f"the positive (d={d_pos:.3f})"
                )
                assert d_neg < d_pos + sampler.margin


def test_update_embeddings_validates_shape(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    sampler = EpisodePairSampler(small_ds, mining="hard")
    with pytest.raises(ValueError):
        sampler.update_embeddings(np.zeros((len(small_ds) - 1, 4)))
    with pytest.raises(ValueError):
        sampler.update_embeddings(np.zeros((len(small_ds), 4, 1)))
    sampler.update_embeddings(np.zeros((len(small_ds), 4)))
    assert sampler.has_embeddings
    sampler.update_embeddings(None)
    assert not sampler.has_embeddings


def test_candidate_pool_size_bounds_compute(small_ds):
    """When ``candidate_pool_size`` is set, mining only scans that many
    candidates — useful for very large datasets."""
    from ewat.typing.pairs import EpisodePairSampler

    n = len(small_ds)
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(n, 6)).astype(np.float32)

    sampler = EpisodePairSampler(
        small_ds,
        n_neg_per_anchor=2,
        seed=1,
        mining="hard",
        candidate_pool_size=3,
    )
    sampler.update_embeddings(embeddings)

    pairs = list(sampler)
    assert any(not same for _, _, same in pairs)


def test_invalid_candidate_pool_size_raises(small_ds):
    from ewat.typing.pairs import EpisodePairSampler
    with pytest.raises(ValueError):
        EpisodePairSampler(small_ds, candidate_pool_size=0)
