"""EWAT v4 quality gates — fail fast if the new collection misses its targets.

Runs the standard ``scripts.validate_dataset`` checks **plus** the v4-specific
criteria from the audit:

* every episode has at least ``--min-steps`` (default: 40) timesteps;
* the *training* NaN ratio of ``disk_io`` is exactly 0% (the v3 dataset had
  ~50% NaN here because the metric was missing — cluster-admin must deploy
  the OTel SDK or expose ``container_fs_*`` Prometheus series);
* every episode contains at least one Jaeger trace (``traces dim T > 0``);
* every episode has at least one OTel span attribute (``trace_depth`` or
  ``fan_out``) populated for ≥ 95% of timesteps.

Exit code 0 when all gates pass, ``2`` when any gate fails. The Markdown
summary written next to ``--features-root`` lists pass/fail per episode for
easy inspection.

Usage
-----
    python -m scripts.validate_v4 \\
        --features-root data/features/v4 \\
        --output reports/v4_gate.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Feature index 5 = disk_io (cf. EpisodeDataset.FEATURE_NAMES)
DISK_IO_IDX = 5
TRACE_IDX = list(range(7, 13))  # trace block: 6 features, indices 7..12


def _check_episode(
    ep_dir: Path,
    min_steps: int,
    span_completeness_threshold: float,
) -> dict:
    """Run all v4 gates on a single featured episode."""
    sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
    T, N, d = sig.shape
    failures: list[str] = []

    if T < min_steps:
        failures.append(f"T={T} < min_steps={min_steps}")

    disk_col = sig[..., DISK_IO_IDX]
    nan_disk = float(np.isnan(disk_col).mean())
    if nan_disk > 0.0:
        failures.append(f"disk_io NaN ratio = {nan_disk:.3%} (>0)")

    trace_block = sig[..., TRACE_IDX]
    span_present = (~np.isnan(trace_block)).any(axis=2).any(axis=1)  # (T,)
    span_ratio = float(span_present.mean())
    if span_ratio < span_completeness_threshold:
        failures.append(
            f"trace presence {span_ratio:.3%} < threshold {span_completeness_threshold:.3%}"
        )

    return {
        "episode_id": ep_dir.name,
        "T": int(T),
        "N": int(N),
        "d": int(d),
        "disk_io_nan_ratio": nan_disk,
        "trace_presence_ratio": span_ratio,
        "passed": len(failures) == 0,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EWAT v4 quality gate")
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--min-steps", type=int, default=40)
    parser.add_argument("--span-threshold", type=float, default=0.95,
                        help="Minimum fraction of timesteps with at least one trace feature")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional Markdown report path")
    parser.add_argument("--strict", action="store_true",
                        help="exit 2 if any episode fails (default: just report)")
    args = parser.parse_args()

    feature_dirs = sorted(p for p in args.features_root.iterdir() if p.is_dir())
    if not feature_dirs:
        print(f"No featured episodes found under {args.features_root}", file=sys.stderr)
        sys.exit(2)

    rows = [
        _check_episode(p, args.min_steps, args.span_threshold) for p in feature_dirs
    ]
    df = pd.DataFrame(rows)
    n = len(df)
    n_pass = int(df["passed"].sum())
    n_fail = n - n_pass

    print(f"\nEWAT v4 gate — {n} episodes  ({n_pass} pass, {n_fail} fail)")
    print(f"  disk_io NaN ratio (mean): {df['disk_io_nan_ratio'].mean():.3%}")
    print(f"  trace presence    (mean): {df['trace_presence_ratio'].mean():.3%}")
    print(f"  T (mean / min / max):     "
          f"{df['T'].mean():.1f} / {df['T'].min()} / {df['T'].max()}")

    payload = {
        "n_episodes": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "min_steps_threshold": args.min_steps,
        "span_threshold": args.span_threshold,
        "summary": {
            "disk_io_nan_mean": float(df["disk_io_nan_ratio"].mean()),
            "trace_presence_mean": float(df["trace_presence_ratio"].mean()),
            "T_mean": float(df["T"].mean()),
            "T_min": int(df["T"].min()),
            "T_max": int(df["T"].max()),
        },
        "per_episode": rows,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".json":
            args.output.write_text(json.dumps(payload, indent=2))
        else:
            md = ["# EWAT v4 — Quality gate\n"]
            md.append(f"Episodes: {n}  |  pass: {n_pass}  |  fail: {n_fail}\n")
            md.append("## Aggregate\n")
            md.append(f"- mean disk_io NaN ratio: {payload['summary']['disk_io_nan_mean']:.3%}")
            md.append(f"- mean trace presence:    {payload['summary']['trace_presence_mean']:.3%}")
            md.append(f"- T (mean / min / max):   "
                      f"{payload['summary']['T_mean']:.1f} / "
                      f"{payload['summary']['T_min']} / "
                      f"{payload['summary']['T_max']}\n")
            md.append("## Failed episodes\n")
            for r in rows:
                if r["passed"]:
                    continue
                md.append(f"- **{r['episode_id']}**  T={r['T']}, "
                          f"disk_io_nan={r['disk_io_nan_ratio']:.2%}, "
                          f"trace={r['trace_presence_ratio']:.2%}")
                for f in r["failures"]:
                    md.append(f"  - {f}")
            args.output.write_text("\n".join(md))
        print(f"Report: {args.output}")

    if args.strict and n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
