"""Run collection in scenario chunks with wave-based quality gates."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from scripts.collect_labeled import collect_once
from scripts.merge_collection_runs import merge_runs
from scripts.validate_dataset import run_checks


def _chunked(seq: list[str], chunk_size: int) -> list[list[str]]:
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def _expected_services_csv(cfg: OmegaConf) -> str:
    return ",".join(list(cfg.collection.get("canonical_services", [])))


def _validate_or_raise(
    run_dir: Path,
    *,
    expected_services: list[str] | None,
    min_coverage_episodes: int,
    min_distribution_episodes: int,
    max_nan_ratio: float,
    min_baseline_edges: int,
    max_trace_timeout_ratio: float,
    max_empty_trace_window_ratio: float,
    enforce_temporal_split: bool,
) -> None:
    checks, failures = run_checks(
        run_dir=run_dir,
        min_coverage_episodes=min_coverage_episodes,
        min_distribution_episodes=min_distribution_episodes,
        max_nan_ratio=max_nan_ratio,
        min_baseline_edges=min_baseline_edges,
        strict_dry_run=False,
        expected_services=expected_services,
        max_trace_timeout_ratio=max_trace_timeout_ratio,
        max_empty_trace_window_ratio=max_empty_trace_window_ratio,
        enforce_temporal_split=enforce_temporal_split,
    )
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {run_dir.name} {check.name}: {check.details}")
    if failures:
        raise RuntimeError(f"Validation failed for {run_dir} with {failures} check(s)")


def _scenarios_under_covered(run_dir: Path, target_episodes: int) -> list[str]:
    labels = pd.read_parquet(run_dir / "labels.parquet")
    subset = labels[labels["regime"] == "injection"]
    if subset.empty:
        return []
    counts = subset.groupby("scenario")["episode_id"].nunique()
    return sorted(counts[counts < target_episodes].index.tolist())


def _run_chunked_campaign(
    cfg: OmegaConf,
    args: argparse.Namespace,
    *,
    scenarios: list[str],
    repetitions: int,
    gate_profile: dict[str, float | int],
    wave_name: str,
    enforce_temporal_split: bool,
) -> list[Path]:
    groups = _chunked(scenarios, args.chunk_size)
    run_dirs: list[Path] = []
    expected_services = list(cfg.collection.get("canonical_services", [])) or None
    for i, chunk in enumerate(groups):
        cfg.collection.scenarios = chunk
        cfg.collection.repetitions = repetitions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            OmegaConf.save(config=cfg, f=tmp.name)
        run_dir = collect_once(
            config_path=tmp_path,
            base_config_path=Path(args.base_config),
            dry_run=args.dry_run,
            endpoint_mode=args.endpoint_mode,
        )
        _validate_or_raise(
            run_dir,
            expected_services=expected_services,
            min_coverage_episodes=int(gate_profile["min_coverage_episodes"]),
            min_distribution_episodes=int(gate_profile["min_distribution_episodes"]),
            max_nan_ratio=float(gate_profile["max_nan_ratio"]),
            min_baseline_edges=int(gate_profile["min_baseline_edges"]),
            max_trace_timeout_ratio=float(gate_profile["max_trace_timeout_ratio"]),
            max_empty_trace_window_ratio=float(gate_profile["max_empty_trace_window_ratio"]),
            enforce_temporal_split=enforce_temporal_split,
        )
        run_dirs.append(run_dir)
        print(
            json.dumps(
                {
                    "wave": wave_name,
                    "campaign_chunk": i,
                    "scenarios": chunk,
                    "repetitions": repetitions,
                    "run_dir": str(run_dir),
                }
            )
        )
    return run_dirs


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run collection by scenario campaigns")
    parser.add_argument("--config", default="configs/collection.yaml")
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--endpoint-mode", choices=["cluster", "local-portforward"], default="cluster")
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-output-dir", default="")
    parser.add_argument("--wave-a-repetitions", type=int, default=2)
    parser.add_argument("--wave-b-repetitions", type=int, default=12)
    parser.add_argument("--wave-c-target-repetitions", type=int, default=20)
    parser.add_argument(
        "--enforce-temporal-split",
        action="store_true",
        help="Enforce temporal split leakage check during campaign gates (recommended only on final merged runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = _cli()
    cfg = OmegaConf.load(args.config)
    all_scenarios = list(cfg.collection.scenarios)
    campaign_runs: list[Path] = []

    # Progressive quality gates: A lenient, B/C strict.
    # Notes:
    # - coverage/distribution thresholds are set relative to repetitions for wave A
    #   because smoke/calibration runs are often 1–2 reps.
    # - temporal split leakage check is disabled by default during campaign waves;
    #   enforce it on the final merged dataset instead.
    enforce_temporal_split = bool(args.enforce_temporal_split)
    gate_a = {
        "min_coverage_episodes": max(1, min(2, int(args.wave_a_repetitions))),
        "min_distribution_episodes": max(1, min(2, int(args.wave_a_repetitions))),
        "max_nan_ratio": 0.80,
        "min_baseline_edges": 1,
        "max_trace_timeout_ratio": 0.35,
        "max_empty_trace_window_ratio": 0.75,
    }
    gate_bc = {
        "min_coverage_episodes": 12,
        "min_distribution_episodes": 12,
        "max_nan_ratio": 0.20,
        "min_baseline_edges": 5,
        "max_trace_timeout_ratio": 0.20,
        "max_empty_trace_window_ratio": 0.50,
    }

    wave_a_runs = _run_chunked_campaign(
        cfg=cfg,
        args=args,
        scenarios=all_scenarios,
        repetitions=args.wave_a_repetitions,
        gate_profile=gate_a,
        wave_name="A",
        enforce_temporal_split=enforce_temporal_split,
    )
    campaign_runs.extend(wave_a_runs)

    wave_b_runs = _run_chunked_campaign(
        cfg=cfg,
        args=args,
        scenarios=all_scenarios,
        repetitions=args.wave_b_repetitions,
        gate_profile=gate_bc,
        wave_name="B",
        enforce_temporal_split=enforce_temporal_split,
    )
    campaign_runs.extend(wave_b_runs)

    wave_b_merged = None
    if wave_b_runs:
        wave_b_merged = merge_runs(
            run_dirs=wave_b_runs,
            output_dir=Path(args.merge_output_dir) / "wave_b_merged"
            if args.merge_output_dir
            else Path(wave_b_runs[-1]).parent / "wave_b_merged",
        )

    wave_c_scenarios = (
        _scenarios_under_covered(wave_b_merged, args.wave_c_target_repetitions)
        if wave_b_merged is not None
        else []
    )
    if wave_c_scenarios:
        wave_c_runs = _run_chunked_campaign(
            cfg=cfg,
            args=args,
            scenarios=wave_c_scenarios,
            repetitions=args.wave_c_target_repetitions,
            gate_profile=gate_bc,
            wave_name="C",
            enforce_temporal_split=enforce_temporal_split,
        )
        campaign_runs.extend(wave_c_runs)
    else:
        print(json.dumps({"wave": "C", "skipped": True, "reason": "No under-covered scenarios"}))

    if args.merge_output_dir and campaign_runs:
        merged = merge_runs(run_dirs=campaign_runs, output_dir=Path(args.merge_output_dir))
        print(json.dumps({"merged_run_dir": str(merged), "expected_services": _expected_services_csv(cfg)}))


if __name__ == "__main__":
    main()
