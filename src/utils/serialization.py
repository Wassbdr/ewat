"""Dataset serialization helpers for labeled EWAT collection runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from graph.diagnostics import GraphStats, stats_to_dict
from graph.types import ServiceGraph
from telemetry.feature_names import (
    FEATURE_NAMES,
    LOGS_SLICE,
    METRICS_SLICE,
    SIGNAL_DIM,
    TRACES_SLICE,
)

logger = logging.getLogger(__name__)

DATASET_SCHEMA_VERSION = "1.2.0"


@dataclass
class LabelRecord:
    """Structured label attached to one timestep."""

    timestamp: float
    regime: Literal["normal", "injection", "recovery", "drift_anomaly"]
    category: str
    scenario: str
    target_services: list[str]
    chaos_resource: str
    episode_id: str = ""
    drift_flag: bool = False  # set post-hoc by MMD-RFF (Step 0) when drift∩anomaly


def save_run_dataset(
    run_dir: str | Path,
    metadata: dict,
    signal_tensor: npt.NDArray[np.float32],
    graph_sequence: list[ServiceGraph],
    labels: list[LabelRecord],
    graph_stats: list[GraphStats],
    services: list[str],
) -> None:
    """Persist one labeled run to disk atomically.

    All files are first written to ``{run_dir}.tmp/``, then the directory is
    renamed to ``run_dir`` in a single ``os.rename`` call (atomic on Linux
    when both paths are on the same filesystem).  This guarantees that either
    the full episode directory exists on disk or nothing does — a partially
    written episode is never silently left behind.

    Output files:
        - signal.npz
        - signal_mask.npz
        - adjacency.npz
        - services.json
        - labels.parquet
        - graph_stats.csv
        - metadata.json
    """
    run_path = Path(run_dir)
    tmp_path = run_path.parent / (run_path.name + ".tmp")

    # Clean up any leftover tmp dir from a previous interrupted run.
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        signal = signal_tensor.astype(np.float32)
        missing_mask = np.isnan(signal)

        np.savez_compressed(tmp_path / "signal.npz", signal=signal)
        np.savez_compressed(tmp_path / "signal_mask.npz", missing_mask=missing_mask)

        adjacency = _stack_adjacency(graph_sequence, services)
        np.savez_compressed(tmp_path / "adjacency.npz", adjacency=adjacency)

        with (tmp_path / "services.json").open("w", encoding="utf-8") as f:
            json.dump(services, f, indent=2)

        labels_df = _labels_to_dataframe(labels)
        _write_parquet(labels_df, tmp_path / "labels.parquet")

        stats_df = _stats_to_dataframe(graph_stats, labels)
        stats_df.to_csv(tmp_path / "graph_stats.csv", index=False)

        metadata_out = _build_metadata_contract(
            metadata=metadata,
            signal=signal,
            missing_mask=missing_mask,
            adjacency=adjacency,
            labels_df=labels_df,
            stats_df=stats_df,
            services=services,
        )

        with (tmp_path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata_out, f, indent=2)

    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    # Atomic promotion: replace any existing run_dir with the new one.
    if run_path.exists():
        shutil.rmtree(run_path)
    os.rename(tmp_path, run_path)

    _log_quality(metadata_out["quality_snapshot"], run_path)


def _build_metadata_contract(
    metadata: dict,
    signal: npt.NDArray[np.float32],
    missing_mask: npt.NDArray[np.bool_],
    adjacency: npt.NDArray[np.float32],
    labels_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    services: list[str],
) -> dict:
    """Build metadata enriched with dataset schema and artifact contract."""
    metadata_out = dict(metadata)
    metadata_out["dataset_schema_version"] = DATASET_SCHEMA_VERSION
    metadata_out["signal_feature_names"] = FEATURE_NAMES
    metadata_out["signal_dim_expected"] = SIGNAL_DIM
    metadata_out["artifacts"] = {
        "signal": {
            "path": "signal.npz",
            "key": "signal",
            "shape": list(signal.shape),
            "dtype": str(signal.dtype),
        },
        "signal_mask": {
            "path": "signal_mask.npz",
            "key": "missing_mask",
            "shape": list(missing_mask.shape),
            "dtype": str(missing_mask.dtype),
        },
        "adjacency": {
            "path": "adjacency.npz",
            "key": "adjacency",
            "shape": list(adjacency.shape),
            "dtype": str(adjacency.dtype),
        },
        "labels": {
            "path": "labels.parquet",
            "columns": list(labels_df.columns),
            "n_rows": int(len(labels_df)),
        },
        "graph_stats": {
            "path": "graph_stats.csv",
            "columns": list(stats_df.columns),
            "n_rows": int(len(stats_df)),
        },
        "services": {
            "path": "services.json",
            "n_services": len(services),
        },
    }

    def _nan_ratio(sl: slice) -> float:
        sub = missing_mask[:, :, sl] if missing_mask.ndim == 3 else missing_mask[:, sl]
        return float(sub.mean()) if sub.size else 0.0

    metadata_out["quality_snapshot"] = {
        "signal_nan_ratio": float(missing_mask.mean()) if missing_mask.size else 0.0,
        "signal_nan_count": int(missing_mask.sum()),
        "metrics_nan_ratio": _nan_ratio(METRICS_SLICE),
        "traces_nan_ratio": _nan_ratio(TRACES_SLICE),
        "logs_nan_ratio": _nan_ratio(LOGS_SLICE),
    }

    config_payload = metadata.get("config")
    metadata_out["hashes"] = {
        "services_sha256": _sha256_json(services),
        "feature_names_sha256": _sha256_json(FEATURE_NAMES),
        "labels_columns_sha256": _sha256_json(list(labels_df.columns)),
        "graph_stats_columns_sha256": _sha256_json(list(stats_df.columns)),
        "config_sha256": _sha256_json(config_payload) if config_payload is not None else None,
    }
    return metadata_out


def _sha256_json(payload: object) -> str:
    """Return a deterministic SHA-256 digest for a JSON-serializable payload."""
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stack_adjacency(
    graphs: list[ServiceGraph],
    services: list[str],
) -> npt.NDArray[np.float32]:
    if not graphs:
        n = len(services)
        return np.zeros((0, n, n, 3), dtype=np.float32)

    tensors: list[npt.NDArray[np.float32]] = []
    for graph in graphs:
        if graph.services != services:
            msg = "Graph services are inconsistent with canonical service ordering"
            raise ValueError(msg)
        tensors.append(graph.adjacency_tensor())

    return np.stack(tensors, axis=0).astype(np.float32)


def _labels_to_dataframe(labels: list[LabelRecord]) -> pd.DataFrame:
    rows: list[dict] = []
    for label in labels:
        row = asdict(label)
        row["target_service"] = label.target_services[0] if label.target_services else ""
        row["target_services"] = json.dumps(label.target_services)
        row["is_injection"] = label.regime == "injection"
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "regime",
                "category",
                "scenario",
                "target_services",
                "target_service",
                "chaos_resource",
                "episode_id",
                "drift_flag",
                "is_injection",
            ]
        )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _stats_to_dataframe(
    stats: list[GraphStats],
    labels: list[LabelRecord],
) -> pd.DataFrame:
    rows = [stats_to_dict(s) for s in stats]
    base_columns = [
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
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=base_columns)

    if len(stats) == len(labels) and not df.empty:
        df["regime"] = [label.regime for label in labels]
        df["scenario"] = [label.scenario for label in labels]
        df["category"] = [label.category for label in labels]
        df["episode_id"] = [label.episode_id for label in labels]
    elif df.empty:
        df["regime"] = pd.Series(dtype="object")
        df["scenario"] = pd.Series(dtype="object")
        df["category"] = pd.Series(dtype="object")
        df["episode_id"] = pd.Series(dtype="object")

    return df


def _log_quality(quality: dict, run_path: Path) -> None:
    """Emit INFO-level quality summary after a successful atomic write."""
    pct = quality["signal_nan_ratio"] * 100
    m_pct = quality["metrics_nan_ratio"] * 100
    t_pct = quality["traces_nan_ratio"] * 100
    l_pct = quality["logs_nan_ratio"] * 100
    logger.info(
        "Saved %s  NaN: total=%.1f%%  M=%.1f%%  T=%.1f%%  L=%.1f%%",
        run_path.name,
        pct,
        m_pct,
        t_pct,
        l_pct,
    )
    if t_pct == 100.0:
        logger.warning(
            "%s: T(t) is entirely NaN — Jaeger data not collected for this run",
            run_path.name,
        )
    if l_pct == 100.0:
        logger.warning(
            "%s: L(t) is entirely NaN — Loki data not collected for this run",
            run_path.name,
        )


def _write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(output_path, index=False, engine=engine)
            return
        except Exception:
            continue

    msg = (
        "Unable to write parquet file. Install either 'pyarrow' or 'fastparquet' "
        "in the active environment."
    )
    raise RuntimeError(msg)
