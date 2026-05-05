"""Episode pair sampler for contrastive siamese training.

Constructs positive and negative episode pairs from scenario labels:
- **Positive** : two episodes with the same Chaos Mesh scenario name.
- **Negative** : two episodes with different scenario names.

For each episode (anchor), we sample `n_neg_per_anchor` negatives
and 1 positive (if available), yielding a balanced stream.

The sampler is iterable and reproducible given a fixed seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

from ewat.encoder.dataset import EpisodeDataset


class EpisodePairSampler:
    """Yields (idx_i, idx_j, is_same) tuples from an EpisodeDataset.

    Parameters
    ----------
    dataset:          EpisodeDataset (any split).
    n_neg_per_anchor: Number of negative pairs per anchor episode.
    seed:             Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: EpisodeDataset,
        n_neg_per_anchor: int = 5,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.n_neg_per_anchor = n_neg_per_anchor
        self.seed = seed

        # Build scenario → [idx] index and reverse map idx → scenario
        self._scenario_to_idxs: dict[str, list[int]] = defaultdict(list)
        self._idx_to_scenario: dict[int, str] = {}
        for i in range(len(dataset)):
            item = dataset[i]
            sc = item["scenario"]
            self._scenario_to_idxs[sc].append(i)
            self._idx_to_scenario[i] = sc

        self._scenarios: list[str] = sorted(self._scenario_to_idxs.keys())
        self._all_idxs: list[int] = list(range(len(dataset)))

    @property
    def scenario_to_idxs(self) -> dict[str, list[int]]:
        return dict(self._scenario_to_idxs)

    @property
    def scenarios(self) -> list[str]:
        return list(self._scenarios)

    def __iter__(self) -> Iterator[tuple[int, int, bool]]:
        """Yield (anchor_idx, other_idx, is_same) pairs.

        For each anchor:
        - 1 positive pair (same scenario), if ≥2 episodes in scenario.
        - n_neg_per_anchor negative pairs (different scenario).
        """
        rng = random.Random(self.seed)

        anchors = list(self._all_idxs)
        rng.shuffle(anchors)

        for anchor_idx in anchors:
            anchor_scenario = self._idx_to_scenario[anchor_idx]
            same_idxs = self._scenario_to_idxs[anchor_scenario]

            # Positive pair (skip if only 1 episode in this scenario)
            positives = [i for i in same_idxs if i != anchor_idx]
            if positives:
                pos_idx = rng.choice(positives)
                yield (anchor_idx, pos_idx, True)

            # Negative pairs
            diff_scenarios = [s for s in self._scenarios if s != anchor_scenario]
            neg_pool: list[int] = []
            for s in diff_scenarios:
                neg_pool.extend(self._scenario_to_idxs[s])

            n_neg = min(self.n_neg_per_anchor, len(neg_pool))
            neg_idxs = rng.sample(neg_pool, n_neg)
            for neg_idx in neg_idxs:
                yield (anchor_idx, neg_idx, False)
