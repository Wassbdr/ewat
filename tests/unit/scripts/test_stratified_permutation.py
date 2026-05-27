"""Tests for the stratified permutation logic used in h3_robustness/permutation_test.

Step 10 fix 10.7 (audit 2026-05-26): permute labels WITHIN each scenario
group rather than across the full train set, preserving scenario marginals.
"""

import numpy as np


def _stratified_permute(y: np.ndarray, scenarios: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Inline implementation mirroring the loop in permutation_test.py."""
    y_perm = y.copy()
    for sc in np.unique(scenarios):
        idx = np.where(scenarios == sc)[0]
        if len(idx) > 1:
            y_perm[idx] = rng.permutation(y[idx])
    return y_perm


def test_stratified_permutation_preserves_marginal_per_scenario():
    """The count of each cluster label within a scenario must be identical
    before and after stratified permutation."""
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1])
    scenarios = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "B"])
    y_perm = _stratified_permute(y, scenarios, rng)
    # For each scenario, the multiset of labels is unchanged
    for sc in np.unique(scenarios):
        idx = scenarios == sc
        before = sorted(y[idx].tolist())
        after = sorted(y_perm[idx].tolist())
        assert before == after, f"scenario {sc!r} marginal broken"


def test_stratified_permutation_actually_permutes():
    """For a sufficiently large scenario, the permutation should produce a
    different order at least some of the time."""
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    scenarios = np.array(["A"] * 8)
    different_count = 0
    for _ in range(50):
        y_perm = _stratified_permute(y, scenarios, rng)
        if not np.array_equal(y, y_perm):
            different_count += 1
    assert different_count >= 40, "permutation rarely changes order — bug?"


def test_stratified_permutation_singleton_scenario_unchanged():
    """A scenario with only 1 episode cannot be permuted; label stays."""
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 2, 3])
    scenarios = np.array(["A", "B", "C", "D"])
    y_perm = _stratified_permute(y, scenarios, rng)
    np.testing.assert_array_equal(y, y_perm)


def test_stratified_permutation_global_distribution_preserved():
    """The overall label histogram is preserved (a corollary of per-scenario)."""
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 1, 2, 0, 1, 2, 2, 0, 0, 1, 2])
    scenarios = np.array(["A"] * 6 + ["B"] * 6)
    y_perm = _stratified_permute(y, scenarios, rng)
    np.testing.assert_array_equal(
        np.bincount(y, minlength=3),
        np.bincount(y_perm, minlength=3),
    )
