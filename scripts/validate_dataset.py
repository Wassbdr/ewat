"""EWAT — quality checks on Phase 2 featured episodes (or a full assembled dataset).

Usage
=====

Validate a single featured episode::

    python -m scripts.validate_dataset --episode data/features/v1/episode_crash_000_XXX

Validate all episodes under one feature set::

    python -m scripts.validate_dataset --features-root data/features/v1

Validate a full assembled dataset::

    python -m scripts.validate_dataset --dataset data/datasets/ewat_v1

Checks
------
- ``shape``           : signal is (T, N, 17), adjacency is (T, N, N, 3), matching services.json.
- ``nan_ratios``      : per-modality NaN ratios below the supplied thresholds.
- ``labels``          : labels.parquet covers every timestep; regime values are valid.
- ``graph_non_empty`` : fraction of graph snapshots with at least one edge ≥ threshold.
- ``service_stability``: services across episodes are consistent (dataset-level only).
- ``temporal_split``  : episodes in later splits start strictly after train/val (dataset-level).

All checks are non-fatal by default; set ``--strict`` to exit non-zero on the first
failure so the script can be used as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _load_episode(ep_dir: Path) -> dict:
    meta = json.loads((ep_dir / "metadata.json").read_text(encoding="utf-8"))
    services = json.loads((ep_dir / "services.json").read_text(encoding="utf-8"))
    with np.load(ep_dir / "signal.npz") as z:
        signal = z["signal"]
    with np.load(ep_dir / "signal_mask.npz") as z:
        mask = z["missing_mask"]
    with np.load(ep_dir / "adjacency.npz") as z:
        adj = z["adjacency"]
    labels = pd.read_parquet(ep_dir / "labels.parquet")
    return {
        "path": ep_dir,
        "meta": meta,
        "services": services,
        "signal": signal,
        "mask": mask,
        "adjacency": adj,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Per-episode checks
# ---------------------------------------------------------------------------


def check_shape(ep: dict) -> CheckResult:
    sig = ep["signal"]
    adj = ep["adjacency"]
    n = len(ep["services"])
    if sig.ndim != 3:
        return CheckResult("shape", False, f"signal.ndim={sig.ndim} (expected 3)")
    if sig.shape[1] != n:
        return CheckResult("shape", False, f"signal.shape[1]={sig.shape[1]} ≠ N={n}")
    if sig.shape[2] != 17:
        return CheckResult("shape", False, f"signal.shape[2]={sig.shape[2]} ≠ 17")
    if adj.ndim != 4:
        return CheckResult("shape", False, f"adjacency.ndim={adj.ndim} (expected 4)")
    if adj.shape[0] != sig.shape[0]:
        return CheckResult("shape", False,
                           f"T mismatch signal={sig.shape[0]} adj={adj.shape[0]}")
    if adj.shape[1] != n or adj.shape[2] != n:
        return CheckResult("shape", False,
                           f"adj spatial shape ({adj.shape[1]},{adj.shape[2]}) ≠ ({n},{n})")
    if adj.shape[3] != 3:
        return CheckResult("shape", False, f"adj channels={adj.shape[3]} (expected 3)")
    return CheckResult("shape", True, f"T={sig.shape[0]} N={n}")


def check_nan_ratios(ep: dict, thresholds: dict[str, float]) -> CheckResult:
    mask = ep["mask"]
    total = float(mask.mean()) if mask.size else 0.0
    m_r = float(mask[:, :, 0:7].mean())
    t_r = float(mask[:, :, 7:13].mean())
    l_r = float(mask[:, :, 13:17].mean())
    if total > thresholds["total"]:
        return CheckResult("nan_ratios", False,
                           f"nan_total={total:.2f} > {thresholds['total']}")
    if m_r > thresholds["metrics"]:
        return CheckResult("nan_ratios", False, f"nan_M={m_r:.2f} > {thresholds['metrics']}")
    if t_r > thresholds["traces"]:
        return CheckResult("nan_ratios", False, f"nan_T={t_r:.2f} > {thresholds['traces']}")
    if l_r > thresholds["logs"]:
        return CheckResult("nan_ratios", False, f"nan_L={l_r:.2f} > {thresholds['logs']}")
    return CheckResult(
        "nan_ratios", True,
        f"total={total:.2f} M={m_r:.2f} T={t_r:.2f} L={l_r:.2f}",
    )


def check_labels(ep: dict) -> CheckResult:
    sig = ep["signal"]
    labels = ep["labels"]
    if len(labels) != sig.shape[0]:
        return CheckResult("labels", False,
                           f"rows={len(labels)} ≠ T={sig.shape[0]}")
    valid = {"normal", "injection", "recovery", "drift_anomaly"}
    invalid = set(labels["regime"].unique()) - valid
    if invalid:
        return CheckResult("labels", False, f"invalid regimes: {invalid}")
    return CheckResult("labels", True,
                       f"rows={len(labels)} regimes={sorted(labels['regime'].unique())}")


def check_graph_non_empty(ep: dict, min_fraction: float) -> CheckResult:
    adj = ep["adjacency"]
    if adj.size == 0:
        return CheckResult("graph_non_empty", False, "adjacency empty")
    vol = adj[..., 0]  # channel 0 = volume
    nonempty = float((vol.sum(axis=(1, 2)) > 0).mean())
    if nonempty < min_fraction:
        return CheckResult("graph_non_empty", False,
                           f"nonempty_fraction={nonempty:.2f} < {min_fraction}")
    return CheckResult("graph_non_empty", True, f"nonempty_fraction={nonempty:.2f}")


# ---------------------------------------------------------------------------
# Dataset-level checks
# ---------------------------------------------------------------------------


def check_service_stability(episode_dicts: list[dict]) -> CheckResult:
    ref = tuple(episode_dicts[0]["services"])
    for ep in episode_dicts[1:]:
        if tuple(ep["services"]) != ref:
            return CheckResult("service_stability", False,
                               f"services differ between {episode_dicts[0]['path'].name} "
                               f"and {ep['path'].name}")
    return CheckResult(
        "service_stability", True,
        f"N={len(ref)} across {len(episode_dicts)} episodes",
    )


def check_temporal_split(dataset_dir: Path) -> CheckResult:
    split_path = dataset_dir / "split.json"
    index_path = dataset_dir / "index.parquet"
    if not split_path.exists() or not index_path.exists():
        return CheckResult("temporal_split", False, "missing split.json or index.parquet")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    try:
        index = pd.read_parquet(index_path)
    except Exception as exc:
        return CheckResult("temporal_split", False, f"cannot read index.parquet: {exc}")
    by_id = index.set_index("episode_id")
    for earlier, later in [("train", "val"), ("val", "test"), ("train", "test")]:
        if not split.get(earlier) or not split.get(later):
            continue
        earlier_end = max(by_id.loc[eid, "recovery_end"] for eid in split[earlier])
        later_start = min(by_id.loc[eid, "baseline_start"] for eid in split[later])
        if later_start < earlier_end:
            return CheckResult(
                "temporal_split", False,
                f"{later} starts (t={later_start:.0f}) before end of "
                f"{earlier} (t={earlier_end:.0f})",
            )
    return CheckResult("temporal_split", True, "all splits are strictly ordered in time")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_DEFAULT_THRESHOLDS = {
    "total": 0.60,
    "metrics": 0.60,
    "traces": 0.90,  # trace traffic can be sparse
    "logs": 0.90,
}


def _print_results(label: str, results: list[CheckResult]) -> bool:
    all_pass = True
    logger.info("=== %s ===", label)
    for r in results:
        status = "OK" if r.passed else "FAIL"
        logger.info("  [%s] %-20s  %s", status, r.name, r.details)
        if not r.passed:
            all_pass = False
    return all_pass


def run_on_episode(ep_dir: Path, thresholds: dict[str, float], min_graph_fraction: float,
                   ) -> tuple[bool, list[CheckResult]]:
    ep = _load_episode(ep_dir)
    results = [
        check_shape(ep),
        check_nan_ratios(ep, thresholds),
        check_labels(ep),
        check_graph_non_empty(ep, min_graph_fraction),
    ]
    return all(r.passed for r in results), results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    thresholds = {
        "total": args.max_nan_total,
        "metrics": args.max_nan_metrics,
        "traces": args.max_nan_traces,
        "logs": args.max_nan_logs,
    }

    any_fail = False
    loaded: list[dict] = []

    if args.episode:
        ep_dir = Path(args.episode)
        ok, results = run_on_episode(ep_dir, thresholds, args.min_graph_fraction)
        any_fail = any_fail or not ok
        _print_results(ep_dir.name, results)

    elif args.features_root:
        root = Path(args.features_root)
        eps = sorted(p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists())
        if not eps:
            raise SystemExit(f"no featured episodes under {root}")
        for ep_dir in eps:
            ok, results = run_on_episode(ep_dir, thresholds, args.min_graph_fraction)
            any_fail = any_fail or not ok
            _print_results(ep_dir.name, results)
            try:
                loaded.append(_load_episode(ep_dir))
            except Exception:
                pass
        if loaded:
            result = check_service_stability(loaded)
            any_fail = any_fail or not result.passed
            _print_results("dataset-wide", [result])

    elif args.dataset:
        ds_dir = Path(args.dataset)
        ep_root = ds_dir / "episodes"
        eps = sorted(p for p in ep_root.iterdir() if p.is_dir())
        for ep_dir in eps:
            ok, results = run_on_episode(ep_dir, thresholds, args.min_graph_fraction)
            any_fail = any_fail or not ok
            _print_results(ep_dir.name, results)
            try:
                loaded.append(_load_episode(ep_dir))
            except Exception:
                pass
        ds_results = []
        if loaded:
            ds_results.append(check_service_stability(loaded))
        ds_results.append(check_temporal_split(ds_dir))
        for r in ds_results:
            if not r.passed:
                any_fail = True
        _print_results("dataset-wide", ds_results)

    else:
        raise SystemExit("provide --episode, --features-root, or --dataset")

    if any_fail and args.strict:
        sys.exit(1)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EWAT Phase 2/3 validator")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--episode", help="single featured episode dir")
    src.add_argument("--features-root", help="feature set root, e.g. data/features/v1")
    src.add_argument("--dataset", help="assembled dataset root, e.g. data/datasets/ewat_v1")
    p.add_argument("--max-nan-total", type=float, default=_DEFAULT_THRESHOLDS["total"])
    p.add_argument("--max-nan-metrics", type=float, default=_DEFAULT_THRESHOLDS["metrics"])
    p.add_argument("--max-nan-traces", type=float, default=_DEFAULT_THRESHOLDS["traces"])
    p.add_argument("--max-nan-logs", type=float, default=_DEFAULT_THRESHOLDS["logs"])
    p.add_argument("--min-graph-fraction", type=float, default=0.10,
                   help="minimum fraction of timesteps with a non-empty graph")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any check fails (useful in CI)")
    return p.parse_args()


if __name__ == "__main__":
    main()
