"""Episode pair sampler for contrastive siamese training.

Constructs positive and negative episode pairs from scenario labels:

- **Positive** : two episodes with the same Chaos Mesh scenario name.
- **Negative** : two episodes with different scenario names.

Three negative-mining strategies are supported:

- ``"random"`` (default, backwards-compatible) — uniform random negatives.
- ``"hard"`` — pick the negatives that are *closest* to the anchor in the
  current embedding space. These are the most informative pairs for the
  contrastive loss but are also the most prone to overfitting and noisy
  labels. Following Schroff et al. (FaceNet, 2015) we recommend warming up
  with ``"random"`` for a few epochs before enabling hard mining.
- ``"semi-hard"`` — Schroff et al.'s default: among all negatives that are
  *farther than the positive* (so they do not violate the margin yet) pick
  the **closest** to the anchor. Falls back to a random hard negative when
  no semi-hard candidate exists.

For ``hard`` and ``semi-hard``, the caller is responsible for supplying the
latest embeddings via :meth:`update_embeddings` — typically once per epoch.
Without an embedding cache the sampler silently degrades to ``"random"``
sampling for that pass.

The sampler is iterable and reproducible given a fixed seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Literal

import numpy as np

from ewat.encoder.dataset import EpisodeDataset

MiningStrategy = Literal["random", "hard", "semi-hard"]
_VALID_STRATEGIES: tuple[MiningStrategy, ...] = ("random", "hard", "semi-hard")


class EpisodePairSampler:
    """Yields ``(idx_i, idx_j, is_same)`` tuples from an :class:`EpisodeDataset`.

    Parameters
    ----------
    dataset:
        The :class:`EpisodeDataset` to sample from.
    n_neg_per_anchor:
        Number of negative pairs emitted per anchor episode.
    seed:
        Random seed for reproducibility.
    mining:
        Negative-mining strategy. One of ``"random"`` (default),
        ``"hard"`` or ``"semi-hard"``. Hard / semi-hard need an embedding
        cache (see :meth:`update_embeddings`).
    margin:
        Margin used to identify semi-hard negatives. Should match the
        margin of the contrastive loss.
    candidate_pool_size:
        Maximum number of candidates to scan when picking the
        hardest / semi-hardest negative. ``None`` (default) means scan all
        negatives. Large datasets can pass a small integer to bound the
        cost of distance computations.
    """

    def __init__(
        self,
        dataset: EpisodeDataset,
        n_neg_per_anchor: int = 5,
        seed: int = 42,
        mining: MiningStrategy = "random",
        margin: float = 1.0,
        candidate_pool_size: int | None = None,
    ) -> None:
        if mining not in _VALID_STRATEGIES:
            raise ValueError(
                f"unknown mining strategy {mining!r}; "
                f"expected one of {_VALID_STRATEGIES}"
            )
        if candidate_pool_size is not None and candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive or None")

        self.dataset = dataset
        self.n_neg_per_anchor = n_neg_per_anchor
        self.seed = seed
        self.mining: MiningStrategy = mining
        self.margin = float(margin)
        self.candidate_pool_size = candidate_pool_size

        self._embeddings: np.ndarray | None = None

        self._scenario_to_idxs: dict[str, list[int]] = defaultdict(list)
        self._idx_to_scenario: dict[int, str] = {}
        for i in range(len(dataset)):
            item = dataset[i]
            sc = item["scenario"]
            self._scenario_to_idxs[sc].append(i)
            self._idx_to_scenario[i] = sc

        self._scenarios: list[str] = sorted(self._scenario_to_idxs.keys())
        self._all_idxs: list[int] = list(range(len(dataset)))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def scenario_to_idxs(self) -> dict[str, list[int]]:
        return dict(self._scenario_to_idxs)

    @property
    def scenarios(self) -> list[str]:
        return list(self._scenarios)

    @property
    def has_embeddings(self) -> bool:
        return self._embeddings is not None

    # ------------------------------------------------------------------ #
    # Embedding cache (driven by the training loop)
    # ------------------------------------------------------------------ #

    def update_embeddings(self, embeddings: np.ndarray | None) -> None:
        """Refresh the embedding cache used for hard / semi-hard mining.

        Parameters
        ----------
        embeddings:
            ``(N, d)`` array where ``N == len(dataset)``. Pass ``None`` to
            invalidate the cache (the next iteration will fall back to
            random mining).
        """
        if embeddings is None:
            self._embeddings = None
            return
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.dataset):
            raise ValueError(
                f"embeddings must have shape (N={len(self.dataset)}, d); "
                f"got {embeddings.shape}"
            )
        self._embeddings = embeddings

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[tuple[int, int, bool]]:
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        anchors = list(self._all_idxs)
        rng.shuffle(anchors)

        active_strategy: MiningStrategy = self.mining
        if active_strategy != "random" and self._embeddings is None:
            active_strategy = "random"

        for anchor_idx in anchors:
            anchor_scenario = self._idx_to_scenario[anchor_idx]
            same_idxs = self._scenario_to_idxs[anchor_scenario]

            positives = [i for i in same_idxs if i != anchor_idx]
            pos_idx: int | None = None
            if positives:
                pos_idx = rng.choice(positives)
                yield (anchor_idx, pos_idx, True)

            diff_scenarios = [s for s in self._scenarios if s != anchor_scenario]
            neg_pool: list[int] = []
            for s in diff_scenarios:
                neg_pool.extend(self._scenario_to_idxs[s])
            if not neg_pool:
                continue

            n_neg = min(self.n_neg_per_anchor, len(neg_pool))
            neg_idxs = self._select_negatives(
                anchor_idx=anchor_idx,
                pos_idx=pos_idx,
                neg_pool=neg_pool,
                n_neg=n_neg,
                strategy=active_strategy,
                rng=rng,
                np_rng=np_rng,
            )
            for neg_idx in neg_idxs:
                yield (anchor_idx, neg_idx, False)

    # ------------------------------------------------------------------ #
    # Negative selection
    # ------------------------------------------------------------------ #

    def _select_negatives(
        self,
        *,
        anchor_idx: int,
        pos_idx: int | None,
        neg_pool: list[int],
        n_neg: int,
        strategy: MiningStrategy,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> list[int]:
        if strategy == "random" or self._embeddings is None:
            return rng.sample(neg_pool, n_neg)

        if self.candidate_pool_size is not None and len(neg_pool) > self.candidate_pool_size:
            candidates = rng.sample(neg_pool, self.candidate_pool_size)
        else:
            candidates = list(neg_pool)

        emb = self._embeddings
        anchor_vec = emb[anchor_idx]
        cand_arr = np.asarray(candidates, dtype=np.int64)
        dist_neg = np.linalg.norm(emb[cand_arr] - anchor_vec[None, :], axis=1)

        if strategy == "hard":
            order = np.argsort(dist_neg)
            picked = [int(cand_arr[order[k]]) for k in range(min(n_neg, len(order)))]
            return picked

        if strategy == "semi-hard":
            d_pos = (
                float(np.linalg.norm(emb[pos_idx] - anchor_vec))
                if pos_idx is not None
                else None
            )
            mask = np.ones(len(cand_arr), dtype=bool)
            if d_pos is not None:
                mask = (dist_neg > d_pos) & (dist_neg < d_pos + self.margin)
            semi_idx = np.where(mask)[0]
            if semi_idx.size == 0:
                order = np.argsort(dist_neg)
                return [int(cand_arr[order[k]]) for k in range(min(n_neg, len(order)))]
            order = semi_idx[np.argsort(dist_neg[semi_idx])]
            picked = [int(cand_arr[order[k]]) for k in range(min(n_neg, len(order)))]
            if len(picked) < n_neg:
                remaining_pool = [c for c in candidates if c not in set(picked)]
                if remaining_pool:
                    extra = rng.sample(
                        remaining_pool, min(n_neg - len(picked), len(remaining_pool))
                    )
                    picked.extend(extra)
            return picked

        raise AssertionError(f"unreachable mining strategy {strategy!r}")
