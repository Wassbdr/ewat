"""Pairwise causal / co-occurrence extraction from synthetic composite episodes.

This module bridges Phase 4 (synthesis) and Phase 5 (reasoning): it consumes
the cascade and overlay episodes produced by
:func:`ewat.ontology.synthesis.cascade_episodes` and
:func:`ewat.ontology.synthesis.overlay_episodes`, then estimates per-pair
multivariate Transfer Entropy (causality, on cascades) and χ² co-occurrence
(on overlays).

The pair labels are read from the synthetic episodes' ``metadata.json``
``composite`` block (``scenario_a``, ``scenario_b``), which the synthesis
script writes. Scenarios are mapped to cluster ids via the empirical
``cluster_manifest.json``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from ewat.ontology.causal import _multivariate_te, _permutation_p_value
from ewat.ontology.cooccurrence import benjamini_hochberg
from ewat.ontology.graph import OntologyRelation
from ewat.ontology.synthesis import load_episode


log = logging.getLogger(__name__)


def _scenario_to_cluster_map(cluster_manifest_path: Path) -> dict[str, int]:
    """Build a *dominant* scenario → cluster_id map from the manifest.

    For each scenario we pick the cluster that occurs most frequently.
    """
    manifest = json.loads(cluster_manifest_path.read_text())
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for info in manifest.values():
        counts[info["scenario"]][int(info["cluster"])] += 1
    out: dict[str, int] = {}
    for scen, by_cluster in counts.items():
        out[scen] = max(by_cluster.items(), key=lambda kv: kv[1])[0]
    return out


def _aggregate_pair_signal(episodes: list[Path]) -> np.ndarray | None:
    """Stack per-step service-averaged signals across episodes.

    Returns a (T, F) array averaged over services, or None if the episodes
    have insufficient length.
    """
    if not episodes:
        return None
    sigs = []
    for ep in episodes:
        bundle = load_episode(ep)
        # Reduce services by mean → (T, F)
        sigs.append(bundle.signal.mean(axis=1))
    min_t = min(s.shape[0] for s in sigs)
    if min_t < 10:
        return None
    stacked = np.stack([s[:min_t] for s in sigs])
    # Episode-averaged trajectory in feature space.
    return stacked.mean(axis=0)


def _split_cascade(
    episode: Path, gap_steps: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (signal_A, signal_B) halves of a cascade episode.

    The two halves are the segments before and after the
    ``composite_transition`` bridge.
    """
    bundle = load_episode(episode)
    regimes = bundle.labels["regime"].to_numpy()
    bridge_mask = regimes == "composite_transition"
    if not bridge_mask.any():
        return None
    bridge_start = int(np.argmax(bridge_mask))
    bridge_end = bridge_start + int(bridge_mask.sum())
    s = bundle.signal.mean(axis=1)  # (T, F)
    return s[:bridge_start], s[bridge_end:]


def extract_pairwise_causal(
    synthetic_root: Path,
    cluster_manifest_path: Path,
    k_knn: int = 4,
    n_permutations: int = 200,
    p_threshold: float = 0.05,
    feature_variance_floor: float = 1e-6,
    seed: int = 42,
) -> list[OntologyRelation]:
    """Compute multivariate KSG-1 TE per cluster pair from cascade episodes.

    Returns ``OntologyRelation`` objects with ``relation_type='causal'`` and
    adjusted p-values (BH-FDR).
    """
    rng = np.random.default_rng(seed)
    scen_to_cluster = _scenario_to_cluster_map(cluster_manifest_path)

    # Group cascade episodes by (cid_a, cid_b)
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for ep_dir in sorted(synthetic_root.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("synth_cascade_"):
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        composite = meta.get("composite", {})
        scen_a = composite.get("scenario_a")
        scen_b = composite.get("scenario_b")
        if scen_a not in scen_to_cluster or scen_b not in scen_to_cluster:
            continue
        cid_a = scen_to_cluster[scen_a]
        cid_b = scen_to_cluster[scen_b]
        if cid_a == cid_b:
            continue
        grouped[(cid_a, cid_b)].append(ep_dir)

    log.info("Extracting causal TE from %d cluster pairs", len(grouped))

    candidates: list[tuple[int, int, float]] = []
    raw_pvals: list[float] = []
    for (cid_a, cid_b), eps in grouped.items():
        if len(eps) < 3:
            continue
        halves = [_split_cascade(ep, gap_steps=5) for ep in eps]
        halves = [h for h in halves if h is not None]
        if len(halves) < 3:
            continue
        # Stack and average per episode → (T_a, F) and (T_b, F)
        len_a = min(h[0].shape[0] for h in halves)
        len_b = min(h[1].shape[0] for h in halves)
        if len_a < 5 or len_b < 5:
            continue
        sig_a = np.stack([h[0][:len_a] for h in halves]).mean(axis=0)
        sig_b = np.stack([h[1][:len_b] for h in halves]).mean(axis=0)

        # Filter low-variance features
        keep = (sig_a.var(axis=0) > feature_variance_floor) & \
               (sig_b.var(axis=0) > feature_variance_floor)
        if keep.sum() < 3:
            continue
        sig_a = sig_a[:, keep]
        sig_b = sig_b[:, keep]

        # Align time axes by truncation
        t_min = min(sig_a.shape[0], sig_b.shape[0])
        sig_a = sig_a[:t_min]
        sig_b = sig_b[:t_min]

        te_obs = _multivariate_te(sig_a, sig_b, lag=1, k=k_knn)
        if te_obs <= 0:
            continue
        # Permutation null on the source
        null_tes = []
        for _ in range(n_permutations):
            perm = rng.permutation(sig_a.shape[0])
            te_perm = _multivariate_te(sig_a[perm], sig_b, lag=1, k=k_knn)
            null_tes.append(te_perm)
        p_raw = _permutation_p_value(te_obs, np.asarray(null_tes))
        candidates.append((cid_a, cid_b, te_obs))
        raw_pvals.append(p_raw)

    if not candidates:
        return []

    p_adj = benjamini_hochberg(raw_pvals)
    out: list[OntologyRelation] = []
    for (cid_a, cid_b, te), p in zip(candidates, p_adj):
        if p <= p_threshold:
            out.append(OntologyRelation(
                source=cid_a, target=cid_b,
                relation_type="causal",
                strength=float(te),
                p_value=float(p),
                support=0,
            ))
    return out


def extract_pairwise_cooccurrence(
    synthetic_root: Path,
    cluster_manifest_path: Path,
    min_overlay_count: int = 2,
) -> list[OntologyRelation]:
    """Co-occurrence between cluster pairs based on overlay episodes.

    Each overlay episode is, by construction, direct evidence of A ↔ B
    co-occurrence on disjoint target services. Significance testing here
    would be circular (the data is engineered to co-occur). We therefore
    emit a symmetric relation for every pair (cid_a, cid_b) whose overlay
    count meets ``min_overlay_count``.

    The :attr:`OntologyRelation.support` field carries the actual count.
    Downstream consumers can re-rank by ``support`` if needed.
    """
    scen_to_cluster = _scenario_to_cluster_map(cluster_manifest_path)

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for ep_dir in sorted(synthetic_root.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("synth_overlay_"):
            continue
        meta = json.loads((ep_dir / "metadata.json").read_text())
        comp = meta.get("composite", {})
        scen_a = comp.get("scenario_a")
        scen_b = comp.get("scenario_b")
        if scen_a not in scen_to_cluster or scen_b not in scen_to_cluster:
            continue
        cid_a = scen_to_cluster[scen_a]
        cid_b = scen_to_cluster[scen_b]
        if cid_a == cid_b:
            continue
        pair = (min(cid_a, cid_b), max(cid_a, cid_b))
        pair_counts[pair] += 1

    out: list[OntologyRelation] = []
    for (cid_a, cid_b), count in pair_counts.items():
        if count < min_overlay_count:
            continue
        out.append(OntologyRelation(
            source=cid_a, target=cid_b,
            relation_type="cooccurrence",
            strength=float(count),
            p_value=None,  # No statistical test — by-construction evidence
            support=count,
        ))
    return out
