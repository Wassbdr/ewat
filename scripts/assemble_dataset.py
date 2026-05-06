"""EWAT — Phase 3: assemble per-episode features into one dataset.

Consumes the artefacts emitted by ``scripts/build_features.py`` and produces
a single unified dataset under ``data/datasets/<name>/``:

::

    data/datasets/<name>/
    ├── episodes/                      # symlinks (or copies) of source episode dirs
    ├── index.parquet                  # one row per (episode_id, split, category, ...)
    ├── split.json                     # temporal split definition
    ├── services.json                  # canonical service set (verified across episodes)
    ├── summary.csv                    # per-scenario counts + quality summary
    └── dataset.json                   # top-level manifest

The split is strictly temporal: episodes are ordered by their collection
timestamp and the first ``train_ratio`` go to train, the next
``val_ratio`` to val, the remainder to test. This prevents temporal
leakage between splits — a requirement of the formalisation's evaluation
protocol.

Usage
=====

::

    python -m scripts.assemble_dataset \
        --features-root data/features/v1 \
        --output data/datasets/ewat_v1 \
        --train-ratio 0.7 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-episode view after Phase 2
# ---------------------------------------------------------------------------


@dataclass
class FeaturedEpisode:
    path: Path
    episode_id: str
    scenario: str
    category: str
    n_timesteps: int
    services: list[str]
    nan_ratio_total: float
    nan_ratio_metrics: float
    nan_ratio_traces: float
    nan_ratio_logs: float
    baseline_start: float
    recovery_end: float
    metadata: dict = field(default_factory=dict)


def _load_featured_episodes(root: Path) -> list[FeaturedEpisode]:
    episodes: list[FeaturedEpisode] = []
    for ep_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            logger.warning("skip %s (no metadata.json)", ep_dir.name)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to parse %s", meta_path)
            continue
        services = json.loads((ep_dir / "services.json").read_text(encoding="utf-8"))
        with np.load(ep_dir / "signal.npz") as z:
            signal = z["signal"]
        quality = meta.get("quality_snapshot", {})
        bounds = meta.get("boundaries", {}) or {}
        episodes.append(
            FeaturedEpisode(
                path=ep_dir,
                episode_id=meta.get("episode_id", ep_dir.name),
                scenario=(meta.get("scenario") or {}).get("name", ""),
                category=(meta.get("scenario") or {}).get("category", ""),
                n_timesteps=int(signal.shape[0]),
                services=list(services),
                nan_ratio_total=float(quality.get("signal_nan_ratio", float("nan"))),
                nan_ratio_metrics=float(quality.get("metrics_nan_ratio", float("nan"))),
                nan_ratio_traces=float(quality.get("traces_nan_ratio", float("nan"))),
                nan_ratio_logs=float(quality.get("logs_nan_ratio", float("nan"))),
                baseline_start=float(bounds.get("baseline_start", 0.0)),
                recovery_end=float(bounds.get("recovery_end", 0.0)),
                metadata=meta,
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def _temporal_split(
    episodes: list[FeaturedEpisode],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[str]]:
    """Return a dict mapping split name → list of episode_ids.

    Episodes are sorted by ``baseline_start`` to enforce a strict temporal
    partition: all train episodes end before any val episode begins (modulo
    the cool-down between chunks).
    """
    if train_ratio + val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1.0")
    by_time = sorted(episodes, key=lambda e: e.baseline_start)
    n = len(by_time)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise SystemExit(f"too few episodes ({n}) for requested ratios")
    return {
        "train": [e.episode_id for e in by_time[:n_train]],
        "val": [e.episode_id for e in by_time[n_train:n_train + n_val]],
        "test": [e.episode_id for e in by_time[n_train + n_val:]],
    }


def _stratified_temporal_split(
    episodes: list[FeaturedEpisode],
    train_ratio: float,
    val_ratio: float,
    grouping: dict[str, str] | None = None,
    min_test_per_group: int = 1,
    min_val_per_group: int = 1,
) -> dict[str, list[str]]:
    """Stratified temporal split with a configurable grouping key.

    Within each group, episodes are sorted by ``baseline_start`` and cut
    at the ratio boundaries while *guaranteeing* at least
    ``min_test_per_group`` episodes go to test (and ``min_val_per_group``
    to val) when the group has enough episodes. This eliminates the
    ``NaN AUROC`` rows on under-represented groups.

    Parameters
    ----------
    grouping:
        ``{episode_id → group_key}``. When ``None`` (default) episodes are
        grouped by ``scenario``. Pass a cluster manifest's
        ``{eid → cluster_id}`` map for cluster-aware splitting after a
        first typing pass.
    min_test_per_group:
        Minimum number of episodes guaranteed in the test set for each
        group, capped by the group size.
    min_val_per_group:
        Minimum number of episodes guaranteed in the val set for each
        group, capped by the group size.
    """
    if train_ratio + val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1.0")
    if min_test_per_group < 0 or min_val_per_group < 0:
        raise SystemExit("min_test_per_group and min_val_per_group must be ≥ 0")

    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    grouped: dict[str, list[FeaturedEpisode]] = defaultdict(list)
    for ep in episodes:
        if grouping is not None:
            key = str(grouping.get(ep.episode_id, "__unassigned__"))
        else:
            key = ep.scenario
        grouped[key].append(ep)

    for _, group_eps in sorted(grouped.items()):
        by_time = sorted(group_eps, key=lambda e: e.baseline_start)
        n = len(by_time)
        # Honour minimum quotas, but never exceed the group size.
        min_test = min(min_test_per_group, max(0, n - 1))
        min_val = min(min_val_per_group, max(0, n - 1 - min_test))
        n_train = max(1, round(n * train_ratio))
        n_val = max(min_val, round(n * val_ratio))
        # Ensure at least min_test episodes in test.
        if n - n_train - n_val < min_test:
            shortage = min_test - (n - n_train - n_val)
            n_train = max(1, n_train - shortage)
        # Re-validate group sizing.
        n_test = n - n_train - n_val
        if n_test < min_test:
            n_val = max(0, n - n_train - min_test)
            n_test = n - n_train - n_val
        split["train"].extend(e.episode_id for e in by_time[:n_train])
        split["val"].extend(e.episode_id for e in by_time[n_train:n_train + n_val])
        split["test"].extend(e.episode_id for e in by_time[n_train + n_val:])

    return split


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def _filter_on_quality(
    episodes: list[FeaturedEpisode],
    max_nan_total: float,
    max_nan_metrics: float,
    max_nan_traces: float,
    max_nan_logs: float,
) -> tuple[list[FeaturedEpisode], list[tuple[str, str]]]:
    kept: list[FeaturedEpisode] = []
    rejected: list[tuple[str, str]] = []
    for ep in episodes:
        if not np.isnan(ep.nan_ratio_total) and ep.nan_ratio_total > max_nan_total:
            rejected.append((ep.episode_id, f"nan_total={ep.nan_ratio_total:.2f}"))
            continue
        if not np.isnan(ep.nan_ratio_metrics) and ep.nan_ratio_metrics > max_nan_metrics:
            rejected.append((ep.episode_id, f"nan_M={ep.nan_ratio_metrics:.2f}"))
            continue
        if not np.isnan(ep.nan_ratio_traces) and ep.nan_ratio_traces > max_nan_traces:
            rejected.append((ep.episode_id, f"nan_T={ep.nan_ratio_traces:.2f}"))
            continue
        if not np.isnan(ep.nan_ratio_logs) and ep.nan_ratio_logs > max_nan_logs:
            rejected.append((ep.episode_id, f"nan_L={ep.nan_ratio_logs:.2f}"))
            continue
        kept.append(ep)
    return kept, rejected


# ---------------------------------------------------------------------------
# Service set consistency
# ---------------------------------------------------------------------------


def _verify_services(episodes: list[FeaturedEpisode]) -> list[str]:
    """Ensure all episodes share the same canonical service list."""
    if not episodes:
        raise SystemExit("no episodes to assemble")
    reference = list(episodes[0].services)
    for ep in episodes[1:]:
        if list(ep.services) != reference:
            raise SystemExit(
                f"inconsistent services between {episodes[0].episode_id} and {ep.episode_id}:\n"
                f"  ref={reference}\n  got={ep.services}"
            )
    return reference


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_dir():
            try:
                if dst.is_symlink():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            except Exception:
                pass
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _episode_category_for_summary(scenario: str, category: str) -> str:
    return category or "unknown"


def _build_index(
    episodes: list[FeaturedEpisode],
    split: dict[str, list[str]],
) -> pd.DataFrame:
    split_of: dict[str, str] = {}
    for name, ids in split.items():
        for eid in ids:
            split_of[eid] = name
    rows = []
    for ep in episodes:
        rows.append({
            "episode_id": ep.episode_id,
            "scenario": ep.scenario,
            "category": ep.category,
            "split": split_of.get(ep.episode_id, ""),
            "n_timesteps": ep.n_timesteps,
            "baseline_start": ep.baseline_start,
            "recovery_end": ep.recovery_end,
            "nan_ratio_total": ep.nan_ratio_total,
            "nan_ratio_metrics": ep.nan_ratio_metrics,
            "nan_ratio_traces": ep.nan_ratio_traces,
            "nan_ratio_logs": ep.nan_ratio_logs,
        })
    return pd.DataFrame(rows).sort_values(["split", "baseline_start"]).reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(path, index=False, engine=engine)
            return
        except Exception:
            continue
    df.to_csv(path.with_suffix(".csv"), index=False)


def _build_summary(episodes: list[FeaturedEpisode]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for ep in episodes:
        counter[_episode_category_for_summary(ep.scenario, ep.category)] += 1
    scenario_counts: dict[tuple[str, str], int] = Counter()
    for ep in episodes:
        scenario_counts[(ep.category, ep.scenario)] += 1
    rows = [
        {"category": cat, "scenario": sc, "n_episodes": n}
        for (cat, sc), n in sorted(scenario_counts.items())
    ]
    return pd.DataFrame(rows)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    features_root = Path(args.features_root)
    if not features_root.is_absolute():
        features_root = REPO_ROOT / features_root
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    logger.info("assemble_dataset: features_root=%s output=%s", features_root, output_root)

    episodes = _load_featured_episodes(features_root)
    if not episodes:
        raise SystemExit(f"no featured episodes under {features_root}")
    logger.info("discovered %d episodes", len(episodes))

    kept, rejected = _filter_on_quality(
        episodes,
        max_nan_total=args.max_nan_total,
        max_nan_metrics=args.max_nan_metrics,
        max_nan_traces=args.max_nan_traces,
        max_nan_logs=args.max_nan_logs,
    )
    if rejected:
        logger.warning("rejected %d episodes on quality gates:", len(rejected))
        for eid, reason in rejected:
            logger.warning("  - %s: %s", eid, reason)
    if not kept:
        raise SystemExit("all episodes rejected by quality filters")

    services = _verify_services(kept)

    grouping: dict[str, str] | None = None
    if args.cluster_manifest:
        manifest_path = Path(args.cluster_manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SystemExit(f"cluster manifest not found: {manifest_path}") from exc
        grouping = {
            eid: str(info.get("cluster", "__unassigned__"))
            for eid, info in manifest_data.items()
        }
        logger.info(
            "cluster-aware split: %d groups from %s",
            len(set(grouping.values())), manifest_path,
        )

    if args.stratified or grouping is not None:
        split = _stratified_temporal_split(
            kept, args.train_ratio, args.val_ratio,
            grouping=grouping,
            min_test_per_group=args.min_test_per_cluster,
            min_val_per_group=args.min_val_per_cluster,
        )
        logger.info(
            "stratified temporal split: train=%d  val=%d  test=%d "
            "(min_test_per_group=%d, min_val_per_group=%d)",
            len(split["train"]), len(split["val"]), len(split["test"]),
            args.min_test_per_cluster, args.min_val_per_cluster,
        )
    else:
        split = _temporal_split(kept, args.train_ratio, args.val_ratio)
        logger.info(
            "temporal split: train=%d  val=%d  test=%d",
            len(split["train"]), len(split["val"]), len(split["test"]),
        )

    if output_root.exists():
        if not args.force:
            raise SystemExit(f"{output_root} already exists (use --force to overwrite)")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    ep_dst_root = output_root / "episodes"
    for ep in kept:
        _link_or_copy(ep.path, ep_dst_root / ep.episode_id, copy=args.copy_episodes)

    index_df = _build_index(kept, split)
    _write_parquet(index_df, output_root / "index.parquet")

    summary_df = _build_summary(kept)
    summary_df.to_csv(output_root / "summary.csv", index=False)

    (output_root / "services.json").write_text(json.dumps(services, indent=2), encoding="utf-8")
    (output_root / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    dataset_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "features_root": str(features_root),
        "n_services": len(services),
        "n_episodes_total": len(episodes),
        "n_episodes_kept": len(kept),
        "n_episodes_rejected": len(rejected),
        "rejected": [{"episode_id": e, "reason": r} for e, r in rejected],
        "quality_filters": {
            "max_nan_total": args.max_nan_total,
            "max_nan_metrics": args.max_nan_metrics,
            "max_nan_traces": args.max_nan_traces,
            "max_nan_logs": args.max_nan_logs,
        },
        "split": {k: len(v) for k, v in split.items()},
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 4),
        },
        "index_sha256": _file_sha(output_root / "index.parquet") if
            (output_root / "index.parquet").exists()
            else _file_sha(output_root / "index.csv"),
    }
    (output_root / "dataset.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )

    logger.info("wrote dataset manifest to %s", output_root)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EWAT Phase 3 — temporal split & dataset assembly")
    p.add_argument("--features-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-nan-total", type=float, default=0.50)
    p.add_argument("--max-nan-metrics", type=float, default=0.50)
    p.add_argument("--max-nan-traces", type=float, default=0.80)
    p.add_argument("--max-nan-logs", type=float, default=0.80)
    p.add_argument("--copy-episodes", action="store_true",
                   help="copy episode dirs instead of symlinking (needed when the dataset "
                        "will be moved to another filesystem)")
    p.add_argument("--stratified", action="store_true",
                   help="use stratified temporal split (per-scenario 70/15/15) instead of "
                        "global temporal split — required for H2 validation")
    p.add_argument(
        "--cluster-manifest",
        type=str,
        default=None,
        help=(
            "optional path to cluster_manifest.json produced by typing/train.py. "
            "When provided, overrides scenario-stratification by cluster-stratification. "
            "Implies --stratified."
        ),
    )
    p.add_argument(
        "--min-test-per-cluster", type=int, default=1,
        help=(
            "minimum number of episodes guaranteed in the test split for each "
            "stratification group (scenario or cluster). Eliminates NaN AUROC rows "
            "on under-represented clusters."
        ),
    )
    p.add_argument(
        "--min-val-per-cluster", type=int, default=1,
        help="minimum number of episodes guaranteed in the val split per group.",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
