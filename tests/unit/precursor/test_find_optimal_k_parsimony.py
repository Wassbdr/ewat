"""Tests règle k* parcimonieuse (M12, audit 2026-06)."""

from __future__ import annotations

import numpy as np

from ewat.precursor.model import find_optimal_k


def test_flat_curve_prefers_smallest_k():
    # courbe plate à tolérance près : l'argmax legacy aurait pris k=10
    table = {2: {0: 0.95}, 6: {0: 0.955}, 10: {0: 0.96}}
    assert find_optimal_k(table, n_clusters=1)[0] == 2


def test_clear_peak_still_selected():
    table = {2: {0: 0.60}, 6: {0: 0.95}, 10: {0: 0.70}}
    assert find_optimal_k(table, n_clusters=1)[0] == 6


def test_zero_tolerance_restores_argmax():
    table = {2: {0: 0.95}, 10: {0: 0.96}}
    assert find_optimal_k(table, n_clusters=1, parsimony_tol=0.0)[0] == 10


def test_nan_aurocs_skipped():
    table = {2: {0: float("nan")}, 6: {0: 0.9}, 10: {0: float("nan")}}
    assert find_optimal_k(table, n_clusters=1)[0] == 6


def test_all_nan_falls_back_to_smallest_k():
    table = {2: {0: float("nan")}, 6: {0: float("nan")}}
    assert find_optimal_k(table, n_clusters=1)[0] == 2


def test_per_cluster_independent():
    table = {
        2: {0: 0.95, 1: 0.50},
        10: {0: 0.96, 1: 0.90},
    }
    res = find_optimal_k(table, n_clusters=2)
    assert res[0] == 2   # plat → parcimonie
    assert res[1] == 10  # pic net → argmax


def test_stability_under_noise():
    """La parcimonie absorbe le bruit ±0.005 qui faisait sauter l'argmax."""
    rng = np.random.default_rng(0)
    picks = set()
    for _ in range(20):
        base = {2: 0.950, 6: 0.952, 10: 0.953}
        table = {k: {0: v + rng.normal(scale=0.005)} for k, v in base.items()}
        picks.add(find_optimal_k(table, n_clusters=1)[0])
    assert picks == {2}, f"k* instable: {picks}"
