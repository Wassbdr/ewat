"""Synthesis of composite episodes from single-scenario ewat_v3 episodes.

The mono-scenario design of ewat_v3 makes it structurally impossible to
observe co-occurrence or causality between cluster types. This module
generates synthetic *composite* episodes by combining two real episodes
through:

- **overlay**: residual addition that simulates two anomalies present at the
  same time on disjoint service targets (co-occurrence proxy).
- **cascade**: temporal concatenation A → gap → B that simulates a
  precedence chain (causality proxy). The resulting episodes are long
  enough (~50 steps) for the multivariate KSG estimator on d=17 features
  to satisfy the empirical T ≥ 5·d rule of thumb.

Realism garde-fous
------------------
Three checks gate every emitted synthetic episode:

1. **Density clip**: each feature is clipped to the ``p99_clip_quantile``
   observed on the source dataset, so additive overlays cannot escape the
   empirical envelope.
2. **Rank preservation**: on the A-aligned segment, Spearman rank correlation
   with the original A must exceed ``spearman_min`` (default 0.85).
3. **Discriminator AUC**: at corpus level, a logistic-regression classifier
   trained to separate real vs. synthetic on a feature subset must stay
   below ``discriminator_auc_max`` (default 0.75). Implementation lives in
   :func:`audit_realism_corpus`.

Episodes are emitted in the *same on-disk layout* as the real ewat_v3
episodes (``signal.npz`` / ``adjacency.npz`` / ``labels.parquet`` /
``metadata.json``) so that downstream code (encoder, typing,
``compute_causal_relations``) can consume them transparently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


SIGNAL_FILENAME = "signal.npz"
ADJ_FILENAME = "adjacency.npz"
LABELS_FILENAME = "labels.parquet"
METADATA_FILENAME = "metadata.json"
MASK_FILENAME = "signal_mask.npz"

COMPOSITE_TRANSITION_REGIME = "composite_transition"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass
class EpisodeBundle:
    """In-memory view of one ewat_v3 episode."""

    episode_id: str
    signal: np.ndarray             # (T, N, F)
    adjacency: np.ndarray          # (T, N, N, 3)
    labels: pd.DataFrame
    metadata: dict
    mask: np.ndarray | None = None  # (T, N) bool, may be absent

    @property
    def n_steps(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_services(self) -> int:
        return int(self.signal.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.signal.shape[2])

    @property
    def scenario(self) -> str:
        return self.metadata["scenario"]["name"]

    @property
    def target_services(self) -> set[str]:
        return set(self.metadata["scenario"].get("targets", []))

    def normal_mean(self) -> np.ndarray:
        """Mean of the signal over the ``regime == 'normal'`` steps."""
        normal_idx = np.where(self.labels["regime"].to_numpy() == "normal")[0]
        if normal_idx.size == 0:
            return self.signal.mean(axis=0)
        return self.signal[normal_idx].mean(axis=0)


def load_episode(episode_dir: Path) -> EpisodeBundle:
    """Load one episode from disk."""
    episode_dir = Path(episode_dir)
    signal = np.load(episode_dir / SIGNAL_FILENAME)["signal"].astype(np.float32)
    adjacency = np.load(episode_dir / ADJ_FILENAME)["adjacency"].astype(np.float32)
    labels = pd.read_parquet(episode_dir / LABELS_FILENAME)
    metadata = json.loads((episode_dir / METADATA_FILENAME).read_text())
    mask_path = episode_dir / MASK_FILENAME
    mask = None
    if mask_path.exists():
        raw = np.load(mask_path)
        # ewat_v3 stores the mask under "missing_mask"; tolerate both keys.
        for key in ("mask", "missing_mask"):
            if key in raw:
                mask = raw[key]
                break
    return EpisodeBundle(
        episode_id=episode_dir.name,
        signal=signal,
        adjacency=adjacency,
        labels=labels,
        metadata=metadata,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Realism garde-fous
# ---------------------------------------------------------------------------


@dataclass
class RealismCheck:
    """Outcome of the per-episode realism garde-fous.

    ``spearman_min`` / ``spearman_median`` are computed over the subset of
    (service, feature) pairs that have non-degenerate variance in both
    series. The ``passed`` flag is governed by the median (more robust to
    isolated constant features such as queue_depth = 0).
    """

    spearman_min: float
    spearman_median: float
    clip_fraction: float
    n_active_pairs: int
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _spearman_rank_per_feature(
    a: np.ndarray, b: np.ndarray, variance_floor: float = 1e-6,
) -> tuple[float, float, int]:
    """Spearman ρ per (service, feature) over time.

    Returns ``(min, median, n_active_pairs)`` where ``n_active_pairs`` is
    the count of (service, feature) pairs with variance above
    ``variance_floor`` in both series. Degenerate pairs are excluded.
    """
    from scipy.stats import spearmanr

    n_steps, n_services, n_features = a.shape
    values: list[float] = []
    for s in range(n_services):
        for f in range(n_features):
            x = a[:, s, f]
            y = b[:, s, f]
            if x.var() < variance_floor or y.var() < variance_floor:
                continue
            rho, _ = spearmanr(x, y)
            if np.isnan(rho):
                continue
            values.append(float(rho))
    if not values:
        # No discriminating feature in either side — neutral pass.
        return 1.0, 1.0, 0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.min()), float(np.median(arr)), int(arr.size)


def clip_to_p99(
    signal: np.ndarray,
    p99_table: np.ndarray,
    p01_table: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Clip each (service, feature) to its (p01, p99) envelope.

    Returns the clipped signal and the fraction of entries actually clipped.
    """
    above = signal > p99_table[None, :, :]
    below = signal < p01_table[None, :, :]
    fraction = float((above | below).mean())
    out = np.minimum(signal, p99_table[None, :, :])
    out = np.maximum(out, p01_table[None, :, :])
    return out.astype(signal.dtype), fraction


def realism_envelope(
    episodes: list[EpisodeBundle],
    quantile_high: float = 0.99,
    quantile_low: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the empirical (low, high) quantile per (service, feature)
    across a corpus of episodes."""
    if not episodes:
        raise ValueError("realism_envelope requires at least one episode")
    stacked = np.concatenate([ep.signal for ep in episodes], axis=0)
    p99 = np.quantile(stacked, quantile_high, axis=0)
    p01 = np.quantile(stacked, quantile_low, axis=0)
    return p01.astype(np.float32), p99.astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay (co-occurrence proxy)
# ---------------------------------------------------------------------------


def overlay_episodes(
    ep_a: EpisodeBundle,
    ep_b: EpisodeBundle,
    alpha: float = 1.0,
    p01_table: np.ndarray | None = None,
    p99_table: np.ndarray | None = None,
    spearman_min: float = 0.85,
) -> tuple[EpisodeBundle, RealismCheck]:
    """Overlay episode B onto A by adding B's residual against its normal mean.

    ``S_overlay[t, s, f] = S_A[t, s, f] + alpha * (S_B[t, s, f] - μ_B_normal[s, f])``

    The two episodes are truncated to ``min(T_A, T_B)``. The result is
    optionally clipped to a (p01, p99) envelope and validated against
    Spearman rank preservation vs. A.
    """
    if ep_a.n_services != ep_b.n_services:
        raise ValueError("services mismatch")
    if ep_a.n_features != ep_b.n_features:
        raise ValueError("features mismatch")
    t = min(ep_a.n_steps, ep_b.n_steps)
    if t < 5:
        raise ValueError(f"too few aligned steps ({t}) for overlay")

    s_a = ep_a.signal[:t]
    s_b = ep_b.signal[:t]
    mu_b = ep_b.normal_mean()                  # (N, F)
    residual = s_b - mu_b[None, :, :]
    s_overlay = s_a + alpha * residual

    reasons: list[str] = []
    clip_fraction = 0.0
    if p01_table is not None and p99_table is not None:
        s_overlay, clip_fraction = clip_to_p99(s_overlay, p99_table, p01_table)
        if clip_fraction > 0.30:
            reasons.append(f"clip_fraction={clip_fraction:.2f}>0.30")

    rho_min, rho_median, n_active = _spearman_rank_per_feature(s_a, s_overlay)
    if n_active > 0 and rho_median < spearman_min:
        reasons.append(
            f"spearman_median={rho_median:.2f}<{spearman_min}"
        )

    check = RealismCheck(
        spearman_min=rho_min,
        spearman_median=rho_median,
        clip_fraction=clip_fraction,
        n_active_pairs=n_active,
        passed=not reasons,
        reasons=reasons,
    )

    # Adjacency: pick A's (the structural graph dominates over additive signal).
    adj_overlay = ep_a.adjacency[:t]

    # Labels: take A's first t rows; add a synthetic column marking the overlay
    labels = ep_a.labels.iloc[:t].copy()
    labels["composite_kind"] = "overlay"
    labels["composite_with"] = ep_b.scenario

    metadata = _composite_metadata(
        ep_a, ep_b, kind="overlay", alpha=alpha, gap_steps=0,
    )

    new_ep_id = f"synth_overlay_{ep_a.scenario}_x_{ep_b.scenario}_a{int(alpha * 10)}"
    return EpisodeBundle(
        episode_id=new_ep_id,
        signal=s_overlay,
        adjacency=adj_overlay,
        labels=labels.reset_index(drop=True),
        metadata=metadata,
    ), check


# ---------------------------------------------------------------------------
# Cascade (causality proxy)
# ---------------------------------------------------------------------------


def cascade_episodes(
    ep_a: EpisodeBundle,
    ep_b: EpisodeBundle,
    gap_steps: int = 5,
    interpolation: Literal["linear"] = "linear",
    p01_table: np.ndarray | None = None,
    p99_table: np.ndarray | None = None,
) -> tuple[EpisodeBundle, RealismCheck]:
    """Concatenate A then B with a ``gap_steps`` linear-interpolated bridge.

    The bridge regime is ``composite_transition``.
    """
    if ep_a.n_services != ep_b.n_services:
        raise ValueError("services mismatch")
    if ep_a.n_features != ep_b.n_features:
        raise ValueError("features mismatch")
    if interpolation != "linear":
        raise NotImplementedError(f"interpolation {interpolation!r} not supported")
    if gap_steps < 0:
        raise ValueError(f"gap_steps must be >= 0, got {gap_steps}")

    last_a = ep_a.signal[-1]
    first_b = ep_b.signal[0]
    if gap_steps > 0:
        alphas = np.linspace(0.0, 1.0, gap_steps + 2)[1:-1]  # exclude endpoints
        bridge = (1.0 - alphas[:, None, None]) * last_a[None, :, :] \
            + alphas[:, None, None] * first_b[None, :, :]
        bridge = bridge.astype(ep_a.signal.dtype)
    else:
        bridge = np.zeros(
            (0, ep_a.n_services, ep_a.n_features), dtype=ep_a.signal.dtype,
        )

    s_cascade = np.concatenate([ep_a.signal, bridge, ep_b.signal], axis=0)

    reasons: list[str] = []
    clip_fraction = 0.0
    if p01_table is not None and p99_table is not None:
        s_cascade, clip_fraction = clip_to_p99(s_cascade, p99_table, p01_table)
        if clip_fraction > 0.30:
            reasons.append(f"clip_fraction={clip_fraction:.2f}>0.30")

    rho_min, rho_median, n_active = _spearman_rank_per_feature(
        ep_a.signal, s_cascade[: ep_a.n_steps],
    )
    if n_active > 0 and rho_median < 0.85:
        reasons.append(f"spearman_median={rho_median:.2f}<0.85")

    check = RealismCheck(
        spearman_min=rho_min,
        spearman_median=rho_median,
        clip_fraction=clip_fraction,
        n_active_pairs=n_active,
        passed=not reasons,
        reasons=reasons,
    )

    # Adjacency: A's then mean-bridge then B's.
    last_adj = ep_a.adjacency[-1]
    first_adj = ep_b.adjacency[0]
    if gap_steps > 0:
        alphas_e = np.linspace(0.0, 1.0, gap_steps + 2)[1:-1]
        bridge_adj = (1.0 - alphas_e[:, None, None, None]) * last_adj[None, :, :, :] \
            + alphas_e[:, None, None, None] * first_adj[None, :, :, :]
        bridge_adj = bridge_adj.astype(ep_a.adjacency.dtype)
    else:
        bridge_adj = np.zeros(
            (0,) + ep_a.adjacency.shape[1:], dtype=ep_a.adjacency.dtype,
        )
    adj_cascade = np.concatenate(
        [ep_a.adjacency, bridge_adj, ep_b.adjacency], axis=0,
    )

    # Labels: concat A's labels, gap rows (composite_transition), B's labels
    labels_a = ep_a.labels.copy()
    labels_b = ep_b.labels.copy()
    labels_b = labels_b.assign(episode_id=ep_a.labels.iloc[0]["episode_id"])

    if gap_steps > 0:
        template = labels_a.iloc[-1].copy()
        gap_rows = pd.DataFrame([template.to_dict()] * gap_steps)
        gap_rows["regime"] = COMPOSITE_TRANSITION_REGIME
        gap_rows["scenario"] = f"{ep_a.scenario}__to__{ep_b.scenario}"
        labels = pd.concat([labels_a, gap_rows, labels_b], ignore_index=True)
    else:
        labels = pd.concat([labels_a, labels_b], ignore_index=True)
    labels["composite_kind"] = "cascade"
    labels["composite_with"] = ep_b.scenario

    metadata = _composite_metadata(
        ep_a, ep_b, kind="cascade", alpha=1.0, gap_steps=gap_steps,
    )

    new_ep_id = (
        f"synth_cascade_{ep_a.scenario}__to__{ep_b.scenario}_g{gap_steps}"
    )
    return EpisodeBundle(
        episode_id=new_ep_id,
        signal=s_cascade,
        adjacency=adj_cascade,
        labels=labels,
        metadata=metadata,
    ), check


# ---------------------------------------------------------------------------
# Metadata + on-disk writer
# ---------------------------------------------------------------------------


def _composite_metadata(
    ep_a: EpisodeBundle,
    ep_b: EpisodeBundle,
    kind: str,
    alpha: float,
    gap_steps: int,
) -> dict:
    """Synthesize a metadata dict for a composite episode based on A's."""
    md = json.loads(json.dumps(ep_a.metadata))  # deep copy
    md.setdefault("composite", {})
    md["composite"] = {
        "kind": kind,
        "source_a": ep_a.episode_id,
        "source_b": ep_b.episode_id,
        "scenario_a": ep_a.scenario,
        "scenario_b": ep_b.scenario,
        "alpha": alpha,
        "gap_steps": gap_steps,
    }
    # Step 7 fix 7.3 (audit 2026-05-26): explicit flag so downstream ontology
    # callers (compute_causal_relations, OWL export) can mark relations
    # derived from synthetic episodes as such (is_from_synthetic=True).
    md["is_synthetic"] = True
    md["scenario"] = {
        **md.get("scenario", {}),
        "name": f"composite_{kind}_{ep_a.scenario}_x_{ep_b.scenario}",
        "category": "composite",
        "description": (
            f"Synthetic composite ({kind}) of {ep_a.scenario} and "
            f"{ep_b.scenario} (alpha={alpha}, gap={gap_steps})"
        ),
    }
    md["episode_id"] = (
        f"synth_{kind}_{ep_a.scenario}_x_{ep_b.scenario}_a{int(alpha * 10)}_g{gap_steps}"
    )
    return md


def write_episode(bundle: EpisodeBundle, output_root: Path) -> Path:
    """Write a bundle to disk in the canonical ewat_v3 layout."""
    out = Path(output_root) / bundle.episode_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / SIGNAL_FILENAME, signal=bundle.signal)
    np.savez_compressed(out / ADJ_FILENAME, adjacency=bundle.adjacency)
    bundle.labels.to_parquet(out / LABELS_FILENAME, index=False)
    (out / METADATA_FILENAME).write_text(json.dumps(bundle.metadata, indent=2))
    if bundle.mask is not None:
        np.savez_compressed(out / MASK_FILENAME, mask=bundle.mask)
    return out


# ---------------------------------------------------------------------------
# Corpus-level realism audit
# ---------------------------------------------------------------------------


def audit_realism_corpus(
    real_episodes: list[EpisodeBundle],
    synthetic_episodes: list[EpisodeBundle],
    feature_subset_size: int = 5,
    random_state: int = 42,
) -> float:
    """Train a logistic regression classifier on per-step rows to separate
    real from synthetic. Returns the held-out AUC.

    A value below 0.75 indicates the synthetic corpus is hard to distinguish
    from the real corpus (good); above 0.85 means the synthesis is producing
    obvious artefacts that need parameter tuning.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    if not real_episodes or not synthetic_episodes:
        raise ValueError("audit_realism_corpus requires both corpora non-empty")

    rng = np.random.default_rng(random_state)

    def flatten(episodes: list[EpisodeBundle]) -> np.ndarray:
        rows = []
        for ep in episodes:
            sig = ep.signal.reshape(ep.n_steps, -1)  # (T, N*F)
            rows.append(sig)
        return np.concatenate(rows, axis=0)

    x_real = flatten(real_episodes)
    x_synth = flatten(synthetic_episodes)
    n_total_features = x_real.shape[1]
    feature_idx = rng.choice(
        n_total_features,
        size=min(feature_subset_size, n_total_features),
        replace=False,
    )
    x_real = x_real[:, feature_idx]
    x_synth = x_synth[:, feature_idx]

    x = np.concatenate([x_real, x_synth], axis=0)
    y = np.concatenate(
        [np.zeros(len(x_real)), np.ones(len(x_synth))], axis=0,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=random_state, stratify=y,
    )
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores))
