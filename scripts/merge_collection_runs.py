"""Merge multiple partial collection runs into one dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_services(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(output_path, index=False, engine=engine)
            return
        except Exception:
            continue
    raise RuntimeError("Unable to write parquet file. Install pyarrow or fastparquet.")


def merge_runs(run_dirs: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    signals: list[np.ndarray] = []
    adjs: list[np.ndarray] = []
    labels: list[pd.DataFrame] = []
    stats: list[pd.DataFrame] = []
    raw_logs: list[str] = []
    services_ref: list[str] | None = None
    metadata_payloads: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        services = _load_services(run_dir / "services.json")
        if services_ref is None:
            services_ref = services
        elif services != services_ref:
            raise ValueError(f"Service ordering mismatch in {run_dir}")

        signals.append(np.load(run_dir / "signal.npz")["signal"].astype(np.float32))
        adjs.append(np.load(run_dir / "adjacency.npz")["adjacency"].astype(np.float32))
        labels.append(pd.read_parquet(run_dir / "labels.parquet"))
        stats.append(pd.read_csv(run_dir / "graph_stats.csv"))

        raw_logs_path = run_dir / "raw_logs.jsonl"
        if raw_logs_path.exists():
            raw_logs.extend(raw_logs_path.read_text(encoding="utf-8").splitlines())
        metadata_payloads.append(json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")))

    merged_signal = np.concatenate(signals, axis=0) if signals else np.zeros((0, 0, 17), dtype=np.float32)
    merged_adj = np.concatenate(adjs, axis=0) if adjs else np.zeros((0, 0, 0, 3), dtype=np.float32)
    merged_labels = (
        pd.concat(labels, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if labels
        else pd.DataFrame()
    )
    merged_stats = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()

    np.savez_compressed(output_dir / "signal.npz", signal=merged_signal)
    np.savez_compressed(output_dir / "signal_mask.npz", missing_mask=np.isnan(merged_signal))
    np.savez_compressed(output_dir / "adjacency.npz", adjacency=merged_adj)
    (output_dir / "services.json").write_text(json.dumps(services_ref or [], indent=2), encoding="utf-8")
    _write_parquet(merged_labels, output_dir / "labels.parquet")
    merged_stats.to_csv(output_dir / "graph_stats.csv", index=False)
    if raw_logs:
        (output_dir / "raw_logs.jsonl").write_text("\n".join(raw_logs) + "\n", encoding="utf-8")

    merged_meta = {
        "merged_from": [str(p) for p in run_dirs],
        "n_input_runs": len(run_dirs),
        "n_timestamps": int(merged_signal.shape[0]),
        "n_services": int(merged_signal.shape[1]) if merged_signal.ndim >= 2 else 0,
        "signal_dim": int(merged_signal.shape[2]) if merged_signal.ndim == 3 else 0,
        "semantic_mode": "offline",
        "semantic_postprocessed": any(m.get("semantic_postprocessed", False) for m in metadata_payloads),
    }
    (output_dir / "metadata.json").write_text(json.dumps(merged_meta, indent=2), encoding="utf-8")
    return output_dir


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple run_* collection folders")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to merge")
    parser.add_argument("--output-dir", required=True, help="Merged output directory")
    return parser.parse_args()


def main() -> None:
    args = _cli()
    merged = merge_runs(
        run_dirs=[Path(run) for run in args.runs],
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({"merged_run_dir": str(merged)}, indent=2))


if __name__ == "__main__":
    main()
