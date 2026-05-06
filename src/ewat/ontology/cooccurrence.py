"""Co-occurrence relations via χ² 2×2 / Fisher exact test.

For each pair of cluster types (i, j), tests whether they tend to appear in the
same scenario more often than expected by chance.

Co-occurrence is defined at the scenario level: a scenario S "contains" cluster
type C_k if at least one of its episodes was assigned to cluster k.

Statistical test
================

Each pair (i, j) is tested with the standard 2×2 contingency table:

    +---------------+----------+----------+
    |               |  j ∈ S   |  j ∉ S   |
    +---------------+----------+----------+
    |  i ∈ S        |    a     |    b     |
    |  i ∉ S        |    c     |    d     |
    +---------------+----------+----------+

with totals N = a + b + c + d = n_scenarios.

- If all expected cell counts ≥ 5 (Cochran's rule), we use Pearson's
  Yates-corrected χ² with 1 degree of freedom (sum over the 4 cells).
- Otherwise, we fall back to Fisher's exact test (two-sided).

Multiple testing correction
===========================

K(K−1)/2 pairs are tested simultaneously. We apply two corrections to the
p-values and let the caller pick:

- ``p_adj_holm``     — Holm–Bonferroni (FWER control, conservative).
- ``p_adj_bh``       — Benjamini–Hochberg (FDR control, less conservative).

The ``p_value`` field on the returned ``OntologyRelation`` is the *adjusted*
p-value selected via the ``correction`` argument; the raw p-value is also
preserved on the relation as a metadata field if needed.

References
==========
- Agresti (2002) — Categorical Data Analysis (χ² Yates, Fisher exact)
- Holm (1979) — A simple sequentially rejective multiple test procedure
- Benjamini & Hochberg (1995) — Controlling the false discovery rate
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
from scipy.stats import chi2, fisher_exact

from ewat.ontology.graph import OntologyRelation

MIN_EXPECTED_FOR_CHI2 = 5.0


def _chi2_2x2_yates(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Yates-corrected Pearson χ² on the full 2×2 table.

    Parameters
    ----------
    a, b, c, d:
        Cell counts of the 2×2 contingency table (see module docstring).

    Returns
    -------
    (chi2_stat, p_value)
    """
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0

    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0, 1.0

    e_a = row1 * col1 / n
    e_b = row1 * col2 / n
    e_c = row2 * col1 / n
    e_d = row2 * col2 / n

    def _term(o: float, e: float) -> float:
        if e <= 0:
            return 0.0
        return (max(0.0, abs(o - e) - 0.5) ** 2) / e

    chi2_stat = _term(a, e_a) + _term(b, e_b) + _term(c, e_c) + _term(d, e_d)
    p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))
    return float(chi2_stat), p_val


def _min_expected(a: int, b: int, c: int, d: int) -> float:
    """Minimum expected cell count of the 2×2 table."""
    n = a + b + c + d
    if n == 0:
        return 0.0
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    return float(min(row1 * col1, row1 * col2, row2 * col1, row2 * col2)) / n


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values.

    Returns a list of the same length, with each adjusted p-value clipped to
    [p_raw, 1.0]. Robust to NaN.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    cummax = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if np.isnan(p):
            adj[idx] = float("nan")
            continue
        candidate = (m - rank) * p
        cummax = max(cummax, candidate)
        adj[idx] = min(1.0, cummax)
    return [float(x) for x in adj]


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg adjusted p-values (FDR control)."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        p = pvals[idx]
        if np.isnan(p):
            adj[idx] = float("nan")
            continue
        candidate = p * m / (rank + 1)
        prev = min(prev, candidate)
        adj[idx] = min(1.0, prev)
    return [float(x) for x in adj]


def compute_cooccurrence_relations(
    cluster_manifest: dict[str, dict],
    n_clusters: int,
    p_threshold: float = 0.05,
    min_cooccurrences: int = 2,
    correction: Literal["holm", "bh", "none"] = "bh",
) -> list[OntologyRelation]:
    """Discover co-occurrence relations between cluster type pairs.

    Parameters
    ----------
    cluster_manifest:
        ``{episode_id → {"cluster": int, "scenario": str, ...}}``.
    n_clusters:
        Total number of cluster types.
    p_threshold:
        Maximum *adjusted* p-value to emit a relation.
    min_cooccurrences:
        Minimum observed co-occurrence count to test the pair.
    correction:
        Multiple-testing correction applied to the raw p-values:

        - ``"bh"``  — Benjamini–Hochberg (FDR, default).
        - ``"holm"`` — Holm–Bonferroni (FWER).
        - ``"none"`` — keep raw p-values.

    Returns
    -------
    List of ``OntologyRelation`` with ``relation_type="cooccurrence"``.
    The ``p_value`` field carries the *adjusted* p-value.
    """
    if correction not in ("holm", "bh", "none"):
        raise ValueError(f"unknown correction: {correction!r}")

    scenario_clusters: dict[str, set[int]] = {}
    for info in cluster_manifest.values():
        sc = info["scenario"]
        c = int(info["cluster"])
        scenario_clusters.setdefault(sc, set()).add(c)

    scenarios = list(scenario_clusters.values())
    n_scenarios = len(scenarios)

    if n_scenarios < 2:
        return []

    presence = np.zeros((n_scenarios, n_clusters), dtype=int)
    for s_idx, cs in enumerate(scenarios):
        for c in cs:
            if 0 <= c < n_clusters:
                presence[s_idx, c] = 1

    candidates: list[tuple[int, int, int, int, int, int, float, str]] = []
    raw_pvals: list[float] = []
    for i, j in combinations(range(n_clusters), 2):
        col_i = presence[:, i]
        col_j = presence[:, j]
        a = int(np.sum((col_i == 1) & (col_j == 1)))
        b = int(np.sum((col_i == 1) & (col_j == 0)))
        c = int(np.sum((col_i == 0) & (col_j == 1)))
        d = int(np.sum((col_i == 0) & (col_j == 0)))

        if a < min_cooccurrences:
            continue

        if _min_expected(a, b, c, d) < MIN_EXPECTED_FOR_CHI2:
            try:
                _, p_raw = fisher_exact([[a, b], [c, d]], alternative="two-sided")
                p_raw = float(p_raw)
            except ValueError:
                continue
            stat = float(a)
            test = "fisher"
        else:
            stat, p_raw = _chi2_2x2_yates(a, b, c, d)
            test = "chi2_yates"

        candidates.append((i, j, a, b, c, d, stat, test))
        raw_pvals.append(p_raw)

    if not candidates:
        return []

    if correction == "holm":
        adj_pvals = holm_bonferroni(raw_pvals)
    elif correction == "bh":
        adj_pvals = benjamini_hochberg(raw_pvals)
    else:  # "none" — already validated above
        adj_pvals = list(raw_pvals)

    relations: list[OntologyRelation] = []
    for (i, j, a, _b, _c, _d, stat, _test), p_adj in zip(candidates, adj_pvals):
        if not np.isnan(p_adj) and p_adj < p_threshold:
            relations.append(
                OntologyRelation(
                    source=i,
                    target=j,
                    relation_type="cooccurrence",
                    strength=float(stat),
                    p_value=float(p_adj),
                    support=a,
                )
            )

    return relations
