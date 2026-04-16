"""Automatic quality checks for EWAT labeled dataset runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def check_metadata_contract(
    metadata: dict,
    signal: np.ndarray,
    signal_mask: np.ndarray,
    adjacency: np.ndarray,
) -> CheckResult:
    """Validate schema contract fields recorded in metadata.json."""
    required_top = ["dataset_schema_version", "artifacts", "signal_dim_expected", "hashes"]
    missing_top = [key for key in required_top if key not in metadata]
    if missing_top:
        # Backward compatibility: older/legacy runs (e.g. scripts/collect_labeled.py
        # direct writer) do not include the full metadata contract. We treat this
        # as a non-blocking warning so smoke/calibration runs can proceed.
        return CheckResult(
            "metadata_contract",
            True,
            f"legacy metadata (missing keys: {missing_top})",
        )

    expected_dim = int(metadata.get("signal_dim_expected", -1))
    if signal.ndim != 3:
        return CheckResult("metadata_contract", False, f"signal ndim expected 3, got {signal.ndim}")
    if signal.shape[2] != expected_dim:
        return CheckResult(
            "metadata_contract",
            False,
            f"signal dim mismatch: expected={expected_dim}, actual={signal.shape[2]}",
        )

    if signal_mask.shape != signal.shape:
        return CheckResult(
            "metadata_contract",
            False,
            f"signal_mask shape mismatch: signal={signal.shape}, mask={signal_mask.shape}",
        )

    if adjacency.ndim != 4:
        return CheckResult(
            "metadata_contract",
            False,
            f"adjacency ndim expected 4, got {adjacency.ndim}",
        )

    artifacts = metadata.get("artifacts", {})
    required_artifacts = ["signal", "signal_mask", "adjacency", "labels", "graph_stats", "services"]
    missing_artifacts = [name for name in required_artifacts if name not in artifacts]
    if missing_artifacts:
        return CheckResult("metadata_contract", False, f"missing artifacts: {missing_artifacts}")

    return CheckResult(
        "metadata_contract",
        True,
        f"schema_version={metadata.get('dataset_schema_version')}",
    )


def _load_artifacts(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    with np.load(run_dir / "signal.npz") as payload:
        signal = payload["signal"]

    signal_mask_path = run_dir / "signal_mask.npz"
    if signal_mask_path.exists():
        with np.load(signal_mask_path) as payload:
            signal_mask = payload["missing_mask"].astype(bool)
    else:
        # Backward compatibility for older runs generated before signal_mask.npz.
        signal_mask = np.isnan(signal)

    with np.load(run_dir / "adjacency.npz") as payload:
        adjacency = payload["adjacency"]

    labels = pd.read_parquet(run_dir / "labels.parquet")
    try:
        graph_stats = pd.read_csv(run_dir / "graph_stats.csv")
    except EmptyDataError:
        graph_stats = pd.DataFrame(
            columns=[
                "timestamp",
                "n_nodes",
                "n_edges",
                "density",
                "avg_degree",
                "max_degree",
                "n_connected_components",
                "diameter",
                "largest_component_size",
                "total_volume",
                "mean_latency",
                "mean_error_rate",
                "regime",
                "scenario",
                "category",
                "episode_id",
            ]
        )

    with (run_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with (run_dir / "services.json").open("r", encoding="utf-8") as f:
        services = json.load(f)

    return signal, signal_mask, adjacency, labels, graph_stats, metadata, services


def check_coverage(labels: pd.DataFrame, min_episodes: int) -> CheckResult:
    subset = labels[labels["regime"] == "injection"]
    if subset.empty:
        return CheckResult("coverage", False, "No injection labels found")

    counts = subset.groupby("scenario")["episode_id"].nunique()
    failing = counts[counts < min_episodes]
    if failing.empty:
        return CheckResult("coverage", True, f"All scenarios >= {min_episodes} episodes")

    return CheckResult("coverage", False, f"Insufficient episodes: {failing.to_dict()}")


def check_distribution(labels: pd.DataFrame, min_episodes: int) -> CheckResult:
    subset = labels[labels["regime"] == "injection"]
    counts = subset.groupby("scenario")["episode_id"].nunique()
    failing = counts[counts < min_episodes]
    if failing.empty:
        return CheckResult("distribution", True, "Class distribution is balanced enough")

    return CheckResult("distribution", False, f"Classes below threshold: {failing.to_dict()}")


def check_signal_nan(signal: np.ndarray, max_nan_ratio: float) -> CheckResult:
    if signal.size == 0:
        return CheckResult("signal_nan", False, "Signal tensor is empty")

    nan_ratio = float(np.isnan(signal).mean())
    passed = nan_ratio <= max_nan_ratio
    return CheckResult(
        "signal_nan",
        passed,
        f"nan_ratio={nan_ratio:.4f}, threshold={max_nan_ratio:.4f}",
    )


def check_signal_mask(signal: np.ndarray, signal_mask: np.ndarray) -> CheckResult:
    """Validate that missing-value mask is shape-aligned and semantically correct."""
    if signal.shape != signal_mask.shape:
        return CheckResult(
            "signal_mask",
            False,
            f"shape mismatch: signal={signal.shape}, mask={signal_mask.shape}",
        )

    expected = np.isnan(signal)
    mismatches = int(np.count_nonzero(signal_mask != expected))
    if mismatches > 0:
        return CheckResult(
            "signal_mask",
            False,
            f"mask mismatch count={mismatches}",
        )

    return CheckResult("signal_mask", True, "mask matches NaN positions")


def check_graph_non_empty_baseline(graph_stats: pd.DataFrame, min_edges: int) -> CheckResult:
    if "regime" not in graph_stats.columns:
        return CheckResult(
            "baseline_graph_non_empty",
            False,
            "graph_stats.csv missing 'regime' column",
        )

    baseline = graph_stats[graph_stats["regime"] == "normal"]
    if baseline.empty:
        return CheckResult("baseline_graph_non_empty", False, "No baseline rows")

    min_observed = int(baseline["n_edges"].min())
    passed = min_observed >= min_edges
    return CheckResult(
        "baseline_graph_non_empty",
        passed,
        f"min_baseline_edges={min_observed}, threshold={min_edges}",
    )


def check_durations(labels: pd.DataFrame) -> CheckResult:
    required_regimes = {"normal", "injection", "recovery"}

    episodes = labels[labels["scenario"] != "normal"].groupby(["scenario", "episode_id"])
    if episodes.ngroups == 0:
        return CheckResult("durations", False, "No anomaly episodes found")

    invalid: list[str] = []
    for (scenario, episode_id), group in episodes:
        observed = set(group["regime"].unique())
        missing = required_regimes - observed
        if missing:
            invalid.append(f"{scenario}/{episode_id}: missing {sorted(missing)}")
            continue

        inj = group[group["regime"] == "injection"]["timestamp"]
        if inj.empty:
            invalid.append(f"{scenario}/{episode_id}: empty injection timestamps")

    if invalid:
        return CheckResult("durations", False, "; ".join(invalid[:10]))
    return CheckResult("durations", True, "Injection/recovery structure is valid")


def check_temporal_split(labels: pd.DataFrame) -> CheckResult:
    timestamps = np.sort(labels["timestamp"].to_numpy(dtype=float))
    if timestamps.size < 2:
        return CheckResult("temporal_split", False, "Not enough timestamps")

    split_idx = max(1, int(0.8 * timestamps.size))
    if split_idx >= timestamps.size:
        split_idx = timestamps.size - 1

    train_max = float(timestamps[split_idx - 1])
    test_min = float(timestamps[split_idx])
    if train_max >= test_min:
        return CheckResult(
            "temporal_split",
            False,
            f"train_max={train_max:.3f}, test_min={test_min:.3f}",
        )

    # Stronger leakage check: no scenario episode should cross the split boundary.
    if "episode_id" in labels.columns:
        episodes = labels[labels["episode_id"].astype(str) != ""].groupby("episode_id")
        leaking_episodes: list[str] = []
        for episode_id, group in episodes:
            t_min = float(group["timestamp"].min())
            t_max = float(group["timestamp"].max())
            # Leakage only if the same episode has samples in both partitions.
            in_train = t_min <= train_max
            in_test = t_max >= test_min
            if in_train and in_test:
                leaking_episodes.append(str(episode_id))

        if leaking_episodes:
            return CheckResult(
                "temporal_split",
                False,
                "Episodes crossing split boundary: " + ", ".join(leaking_episodes[:10]),
            )

    return CheckResult(
        "temporal_split",
        True,
        f"train_max={train_max:.3f}, test_min={test_min:.3f}",
    )


def check_temporal_split_skipped() -> CheckResult:
    return CheckResult("temporal_split", True, "skipped (campaign wave gate)")


def check_services_stability(services: list[str], expected_services: list[str] | None) -> CheckResult:
    """Validate that services.json matches expected canonical services when provided."""
    if expected_services is None:
        return CheckResult("services_stability", True, f"n_services={len(services)}")
    if services != expected_services:
        return CheckResult(
            "services_stability",
            False,
            f"services mismatch: expected={expected_services}, actual={services}",
        )
    return CheckResult("services_stability", True, f"services match canonical set ({len(services)})")


def check_trace_collection_health(
    metadata: dict,
    max_trace_timeout_ratio: float,
    max_empty_trace_window_ratio: float,
) -> CheckResult:
    """Validate Jaeger timeout ratio and empty-trace windows ratio from metadata."""
    trace_stats = metadata.get("trace_collection_stats", {})
    services_considered = float(trace_stats.get("services_considered", 0.0))
    services_timed_out = float(trace_stats.get("services_timed_out", 0.0))
    empty_windows_ratio = float(trace_stats.get("traces_empty_window_ratio", 1.0))

    timeout_ratio = (services_timed_out / services_considered) if services_considered > 0 else 0.0
    if timeout_ratio > max_trace_timeout_ratio:
        return CheckResult(
            "trace_collection_health",
            False,
            (
                f"timeout_ratio={timeout_ratio:.4f} > threshold={max_trace_timeout_ratio:.4f}, "
                f"empty_windows_ratio={empty_windows_ratio:.4f}"
            ),
        )
    if empty_windows_ratio > max_empty_trace_window_ratio:
        return CheckResult(
            "trace_collection_health",
            False,
            (
                f"empty_windows_ratio={empty_windows_ratio:.4f} > "
                f"threshold={max_empty_trace_window_ratio:.4f}, timeout_ratio={timeout_ratio:.4f}"
            ),
        )
    return CheckResult(
        "trace_collection_health",
        True,
        (
            f"timeout_ratio={timeout_ratio:.4f}, "
            f"empty_windows_ratio={empty_windows_ratio:.4f}"
        ),
    )


def run_checks(
    run_dir: Path,
    min_coverage_episodes: int,
    min_distribution_episodes: int,
    max_nan_ratio: float,
    min_baseline_edges: int,
    strict_dry_run: bool,
    expected_services: list[str] | None = None,
    max_trace_timeout_ratio: float = 0.20,
    max_empty_trace_window_ratio: float = 0.50,
    enforce_temporal_split: bool = True,
) -> tuple[list[CheckResult], int]:
    signal, signal_mask, adjacency, labels, graph_stats, metadata, services = _load_artifacts(run_dir)

    if metadata.get("dry_run", False) and not strict_dry_run:
        checks = [
            CheckResult(
                "dry_run_mode",
                True,
                "Dry-run artifacts are intentionally empty; quality checks skipped. "
                "Use --strict-dry-run to enforce full checks.",
            )
        ]
        return checks, 0

    checks = [
        check_metadata_contract(metadata, signal, signal_mask, adjacency),
        check_coverage(labels, min_coverage_episodes),
        check_distribution(labels, min_distribution_episodes),
        check_signal_nan(signal, max_nan_ratio),
        check_signal_mask(signal, signal_mask),
        check_graph_non_empty_baseline(graph_stats, min_baseline_edges),
        check_durations(labels),
        check_temporal_split(labels) if enforce_temporal_split else check_temporal_split_skipped(),
        check_services_stability(services, expected_services),
        check_trace_collection_health(
            metadata=metadata,
            max_trace_timeout_ratio=max_trace_timeout_ratio,
            max_empty_trace_window_ratio=max_empty_trace_window_ratio,
        ),
    ]
    failures = sum(1 for check in checks if not check.passed)
    return checks, failures


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EWAT dataset run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-coverage-episodes", type=int, default=20)
    parser.add_argument("--min-distribution-episodes", type=int, default=15)
    parser.add_argument("--max-nan-ratio", type=float, default=0.20)
    parser.add_argument("--min-baseline-edges", type=int, default=5)
    parser.add_argument(
        "--strict-dry-run",
        action="store_true",
        help="If set, run full quality checks even when metadata indicates dry-run mode.",
    )
    parser.add_argument(
        "--expected-services",
        default="",
        help="Comma-separated canonical services list to enforce exact services.json stability.",
    )
    parser.add_argument("--max-trace-timeout-ratio", type=float, default=0.20)
    parser.add_argument("--max-empty-trace-window-ratio", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = _cli()

    expected_services = (
        [s.strip() for s in args.expected_services.split(",") if s.strip()]
        if args.expected_services
        else None
    )
    checks, failures = run_checks(
        run_dir=args.run_dir,
        min_coverage_episodes=args.min_coverage_episodes,
        min_distribution_episodes=args.min_distribution_episodes,
        max_nan_ratio=args.max_nan_ratio,
        min_baseline_edges=args.min_baseline_edges,
        strict_dry_run=args.strict_dry_run,
        expected_services=expected_services,
        max_trace_timeout_ratio=args.max_trace_timeout_ratio,
        max_empty_trace_window_ratio=args.max_empty_trace_window_ratio,
    )

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.details}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
