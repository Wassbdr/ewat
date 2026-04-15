"""Offline semantic anomaly post-processing for collected EWAT runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from telemetry.feature_names import L_SEMANTIC_ANOMALY
from telemetry.features.semantic import SemanticAnomalyScorer


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_service_scorers(
    records: list[dict],
    baseline_timestamps: set[float],
    model_name: str,
    batch_size: int,
) -> dict[str, SemanticAnomalyScorer]:
    per_service_lines: dict[str, list[str]] = {}
    for rec in records:
        ts = float(rec.get("timestamp", 0.0))
        service = rec.get("service_name", "")
        body = rec.get("body", "")
        if ts in baseline_timestamps and service and body:
            per_service_lines.setdefault(service, []).append(body)

    scorers: dict[str, SemanticAnomalyScorer] = {}
    for service, lines in per_service_lines.items():
        if not lines:
            continue
        scorer = SemanticAnomalyScorer(model_name=model_name, batch_size=batch_size)
        scorer.fit(lines)
        scorers[service] = scorer
    return scorers


def _score_timestamps(
    labels: pd.DataFrame,
    services: list[str],
    records: list[dict],
    scorers: dict[str, SemanticAnomalyScorer],
    window_s: float,
) -> np.ndarray:
    n_rows = len(labels)
    n_services = len(services)
    semantic = np.full((n_rows, n_services), np.nan, dtype=np.float32)
    svc_idx = {service: i for i, service in enumerate(services)}

    for row in range(n_rows):
        ts = float(labels.iloc[row]["timestamp"])
        start_ts = ts - window_s
        lines_by_service: dict[str, list[str]] = {}
        for rec in records:
            rec_ts = float(rec.get("timestamp", 0.0))
            if start_ts <= rec_ts <= ts:
                service = rec.get("service_name", "")
                body = rec.get("body", "")
                if service and body:
                    lines_by_service.setdefault(service, []).append(body)

        for service, lines in lines_by_service.items():
            scorer = scorers.get(service)
            idx = svc_idx.get(service)
            if scorer is None or idx is None or not lines:
                continue
            semantic[row, idx] = np.float32(scorer.score(lines))
    return semantic


def postprocess_run(
    run_dir: Path,
    *,
    window_s: float,
    model_name: str,
    batch_size: int,
) -> None:
    signal_path = run_dir / "signal.npz"
    labels_path = run_dir / "labels.parquet"
    services_path = run_dir / "services.json"
    logs_path = run_dir / "raw_logs.jsonl"
    metadata_path = run_dir / "metadata.json"

    signal = np.load(signal_path)["signal"].astype(np.float32)
    labels = pd.read_parquet(labels_path)
    services = json.loads(services_path.read_text(encoding="utf-8"))
    records = _read_jsonl(logs_path)

    baseline_ts = set(
        labels.loc[
            (labels["regime"] == "normal") & (labels["scenario"] == "normal"),
            "timestamp",
        ].astype(float).tolist()
    )
    scorers = _build_service_scorers(
        records=records,
        baseline_timestamps=baseline_ts,
        model_name=model_name,
        batch_size=batch_size,
    )
    semantic_values = _score_timestamps(
        labels=labels,
        services=services,
        records=records,
        scorers=scorers,
        window_s=window_s,
    )

    signal[:, :, L_SEMANTIC_ANOMALY] = semantic_values
    np.savez_compressed(signal_path, signal=signal)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["semantic_postprocessed"] = True
    metadata["semantic_mode"] = "offline"
    metadata["semantic_postprocess"] = {
        "window_s": window_s,
        "model_name": model_name,
        "batch_size": batch_size,
        "n_services_fitted": len(scorers),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process semantic anomaly offline")
    parser.add_argument("--run-dir", required=True, help="Path to run directory (data/raw/run_*)")
    parser.add_argument("--window-s", type=float, default=120.0, help="Scoring window in seconds")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _cli()
    postprocess_run(
        run_dir=Path(args.run_dir),
        window_s=args.window_s,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
