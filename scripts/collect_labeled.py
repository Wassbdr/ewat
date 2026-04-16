"""Main orchestrator for labeled EWAT data collection runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph.builder import ServiceGraphBuilder  # noqa: E402
from graph.diagnostics import stats_to_dict  # noqa: E402
from scripts.chaos_injector import ChaosInjector  # noqa: E402
from scripts.snapshot_collector import SnapshotBatch, SnapshotCollector  # noqa: E402
from telemetry.feature_names import TRACES_SLICE  # noqa: E402
from telemetry.signal_builder import SignalBuilder  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


_LOCAL_PORT_FORWARD_ENDPOINTS: dict[str, str] = {
    "prometheus": "http://127.0.0.1:19090",
    "jaeger": "http://127.0.0.1:16686",
    "loki": "http://127.0.0.1:13100",
}


def _safe_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_git_commit(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _parse_duration(duration: str | int | float) -> float:
    if isinstance(duration, (int, float)):
        return float(duration)

    text = str(duration).strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(text[:-1])
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    if text.endswith("h"):
        return float(text[:-1]) * 3600.0
    return float(text)


def _persist_batch_chunk(chunk_dir: Path, chunk_idx: int, batch: SnapshotBatch) -> dict[str, Any]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = f"chunk_{chunk_idx:06d}"
    signal_path = chunk_dir / f"{chunk_name}.signal.npy"
    adjacency_path = chunk_dir / f"{chunk_name}.adj.npy"
    labels_path = chunk_dir / f"{chunk_name}.labels.jsonl"
    stats_path = chunk_dir / f"{chunk_name}.stats.jsonl"
    manifest_path = chunk_dir / f"{chunk_name}.manifest.json"

    signal = batch.signal.astype(np.float32)
    np.save(signal_path, signal)

    if batch.graphs:
        adjacency = np.stack([g.adjacency_tensor() for g in batch.graphs], axis=0).astype(np.float32)
    else:
        n = len(batch.services)
        adjacency = np.zeros((0, n, n, 3), dtype=np.float32)
    np.save(adjacency_path, adjacency)

    with labels_path.open("w", encoding="utf-8") as f:
        for label in batch.labels:
            f.write(json.dumps(asdict(label), ensure_ascii=True) + "\n")

    with stats_path.open("w", encoding="utf-8") as f:
        for stat in batch.graph_stats:
            f.write(json.dumps(stats_to_dict(stat), ensure_ascii=True) + "\n")

    manifest = {
        "chunk_name": chunk_name,
        "n_rows": int(signal.shape[0]),
        "signal_path": signal_path.name,
        "adjacency_path": adjacency_path.name,
        "labels_path": labels_path.name,
        "stats_path": stats_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _materialize_from_chunks(
    run_dir: Path,
    services: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_dir = run_dir / "chunks"
    manifests = sorted(chunk_dir.glob("chunk_*.manifest.json"))
    total_rows = 0
    manifest_payloads: list[dict[str, Any]] = []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_payloads.append(payload)
        total_rows += int(payload["n_rows"])

    n_services = len(services)
    signal = np.zeros((total_rows, n_services, 17), dtype=np.float32)
    adjacency = np.zeros((total_rows, n_services, n_services, 3), dtype=np.float32)
    labels_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    cursor = 0
    for payload in manifest_payloads:
        n_rows = int(payload["n_rows"])
        if n_rows <= 0:
            continue
        signal_chunk = np.load(chunk_dir / payload["signal_path"])
        adj_chunk = np.load(chunk_dir / payload["adjacency_path"])
        signal[cursor : cursor + n_rows] = signal_chunk
        adjacency[cursor : cursor + n_rows] = adj_chunk
        labels_rows.extend(_read_jsonl(chunk_dir / payload["labels_path"]))
        stats_rows.extend(_read_jsonl(chunk_dir / payload["stats_path"]))
        cursor += n_rows

    return signal, adjacency, labels_rows, stats_rows


def _write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(output_path, index=False, engine=engine)
            return
        except Exception:
            continue
    raise RuntimeError("Unable to write parquet file. Install pyarrow or fastparquet.")


def _load_yaml(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _apply_endpoint_mode(base_cfg: Any, endpoint_mode: str) -> None:
    if endpoint_mode != "local-portforward":
        return

    base_cfg.telemetry.prometheus.endpoint = _LOCAL_PORT_FORWARD_ENDPOINTS["prometheus"]
    base_cfg.telemetry.jaeger.endpoint = _LOCAL_PORT_FORWARD_ENDPOINTS["jaeger"]
    base_cfg.telemetry.loki.endpoint = _LOCAL_PORT_FORWARD_ENDPOINTS["loki"]
    logger.info(
        "Endpoint mode local-portforward enabled "
        "(prometheus=%s, jaeger=%s, loki=%s)",
        base_cfg.telemetry.prometheus.endpoint,
        base_cfg.telemetry.jaeger.endpoint,
        base_cfg.telemetry.loki.endpoint,
    )


def _warmup_semantic_model(signal_builder: "SignalBuilder") -> None:
    """Pre-load SentenceBERT into memory before the collection loop starts.

    Lazy loading mid-run allocates ~1.5 GB of PyTorch memory at an unpredictable
    point (end of first baseline window), which can crash WSL under memory pressure.
    Loading eagerly at startup makes the failure visible immediately and avoids a
    mid-run OOM.
    """
    logs = getattr(signal_builder, "_logs", None)
    if logs is None:
        return
    scorers = getattr(logs, "_semantic_scorers", {})
    # Scorers dict may be empty before the first fit; instantiate one temporarily
    # just to trigger the PyTorch import and model download/load.
    from telemetry.features.semantic import SemanticAnomalyScorer

    probe = scorers.get(next(iter(scorers), None)) if scorers else None
    if probe is None:
        probe = SemanticAnomalyScorer()
    try:
        probe.warmup()
    except Exception:
        logger.warning("SentenceBERT pre-load failed; model will load lazily", exc_info=True)


def collect_once(
    config_path: Path,
    base_config_path: Path,
    dry_run: bool = False,
    endpoint_mode: str = "cluster",
    no_traces: bool = False,
) -> Path:
    repo_root = REPO_ROOT
    cfg = _load_yaml(config_path)
    base_cfg = OmegaConf.load(base_config_path)
    _apply_endpoint_mode(base_cfg, endpoint_mode)

    seed = int(base_cfg.get("random", {}).get("seed", 42))
    seed_everything(seed)

    collection_cfg = cfg["collection"]
    semantic_cfg = collection_cfg.get("semantic", {})
    canonical_services = list(collection_cfg.get("canonical_services", [])) or None
    semantic_mode = str(semantic_cfg.get("mode", "online")).lower()
    if semantic_mode not in {"online", "offline"}:
        msg = f"Invalid collection.semantic.mode='{semantic_mode}', expected online|offline"
        raise ValueError(msg)
    semantic_enabled = semantic_mode == "online"
    traces_enabled = not no_traces

    if no_traces:
        logger.info("Trace collection disabled via --no-traces.")

    signal_builder = SignalBuilder.from_config(
        base_cfg,
        semantic_enabled=semantic_enabled,
        traces_enabled=traces_enabled,
        services=canonical_services,
    )
    graph_builder = ServiceGraphBuilder.from_config(base_cfg)

    # Pre-load SentenceBERT (PyTorch ~1.5 GB) at startup so it doesn't OOM WSL
    # mid-run when port-forwards are already consuming memory.
    if semantic_enabled:
        _warmup_semantic_model(signal_builder)

    injector = ChaosInjector(
        namespace=collection_cfg.get("namespace", "ewat"),
        dry_run=dry_run,
    )

    registry = {scenario.name: scenario for scenario in injector.list_scenarios()}

    scenarios = collection_cfg.get("scenarios", [])
    repetitions = int(collection_cfg.get("repetitions", 20))

    baseline_s = _parse_duration(collection_cfg.get("baseline_s", "5m"))
    pre_injection_s = _parse_duration(collection_cfg.get("pre_injection_s", "1m"))
    recovery_s = _parse_duration(collection_cfg.get("recovery_s", "2m"))
    cool_down_s = _parse_duration(collection_cfg.get("cool_down_s", "5m"))

    if dry_run:
        logger.info("Dry-run mode enabled: forcing phase durations to 0s")
        baseline_s = 0.0
        pre_injection_s = 0.0
        recovery_s = 0.0
        cool_down_s = 0.0

    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    output_root = repo_root / collection_cfg.get("output_root", "data/raw")
    run_dir = output_root / run_id

    logger.info("Starting collection run: %s", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = run_dir / "chunks"
    raw_logs_path = run_dir / "raw_logs.jsonl"

    def _append_raw_log(payload: dict[str, Any]) -> None:
        with raw_logs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    sample_interval_s = float(collection_cfg["sample_interval_s"])
    collector = SnapshotCollector(
        signal_builder=signal_builder,
        graph_builder=graph_builder,
        sample_interval_s=sample_interval_s,
        semantic_fit_enabled=semantic_enabled,
        raw_logs_hook=_append_raw_log if semantic_mode == "offline" else None,
    )

    run_services: list[str] | None = canonical_services
    chunk_idx = 0

    for scenario_name in scenarios:
        if scenario_name not in registry:
            raise ValueError(f"Scenario '{scenario_name}' not found in registry")

        scenario_spec = registry[scenario_name]
        inject_s = 0.0 if dry_run else _parse_duration(scenario_spec.duration)

        for rep in range(repetitions):
            episode_id = f"{scenario_name}_ep_{rep:03d}"
            logger.info("Episode %s", episode_id)

            baseline_batch = collector.collect_for_duration(
                duration_s=baseline_s,
                regime="normal",
                category="normal",
                scenario="normal",
                target_services=[],
                chaos_resource="",
                episode_id=episode_id,
                services=run_services,
            )
            run_services = baseline_batch.services
            _persist_batch_chunk(chunk_dir, chunk_idx, baseline_batch)
            chunk_idx += 1

            pre_batch = collector.collect_for_duration(
                duration_s=pre_injection_s,
                regime="normal",
                category=scenario_spec.category,
                scenario=scenario_spec.name,
                target_services=scenario_spec.targets,
                chaos_resource=scenario_spec.file,
                episode_id=episode_id,
                services=run_services,
            )
            _persist_batch_chunk(chunk_dir, chunk_idx, pre_batch)
            chunk_idx += 1

            injector.apply(scenario_name)
            try:
                inj_batch = collector.collect_for_duration(
                    duration_s=inject_s,
                    regime="injection",
                    category=scenario_spec.category,
                    scenario=scenario_spec.name,
                    target_services=scenario_spec.targets,
                    chaos_resource=scenario_spec.file,
                    episode_id=episode_id,
                    services=run_services,
                )
                _persist_batch_chunk(chunk_dir, chunk_idx, inj_batch)
                chunk_idx += 1
            finally:
                injector.delete(scenario_name)

            recovery_batch = collector.collect_for_duration(
                duration_s=recovery_s,
                regime="recovery",
                category=scenario_spec.category,
                scenario=scenario_spec.name,
                target_services=scenario_spec.targets,
                chaos_resource=scenario_spec.file,
                episode_id=episode_id,
                services=run_services,
            )
            _persist_batch_chunk(chunk_dir, chunk_idx, recovery_batch)
            chunk_idx += 1

            if cool_down_s > 0:
                logger.info("Cooling down %.1fs", cool_down_s)
                time.sleep(cool_down_s)

    services = run_services or []
    signal_tensor, adjacency_tensor, labels_rows, stats_rows = _materialize_from_chunks(run_dir, services)

    trace_collector = getattr(signal_builder, "_traces", None)
    trace_backend = getattr(trace_collector, "_backend", None) if trace_collector is not None else None
    trace_fetch_stats = (
        trace_backend.get_last_fetch_stats()
        if trace_backend is not None and hasattr(trace_backend, "get_last_fetch_stats")
        else {}
    )
    traces_empty_window_ratio = (
        float(np.isnan(signal_tensor[:, :, TRACES_SLICE]).all(axis=(1, 2)).mean())
        if signal_tensor.size
        else 1.0
    )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": cfg,
        "base_config_path": str(base_config_path),
        "n_timestamps": int(signal_tensor.shape[0]),
        "n_services": len(services),
        "signal_dim": int(signal_tensor.shape[2]) if signal_tensor.ndim == 3 else 0,
        "dry_run": dry_run,
        "semantic_mode": semantic_mode,
        "traces_enabled": traces_enabled,
        "semantic_postprocessed": False,
        "canonical_services": services,
        "trace_collection_stats": {
            **trace_fetch_stats,
            "traces_empty_window_ratio": traces_empty_window_ratio,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": {
                "numpy": _safe_package_version("numpy"),
                "pandas": _safe_package_version("pandas"),
                "hydra-core": _safe_package_version("hydra-core"),
                "scikit-learn": _safe_package_version("scikit-learn"),
                "torch": _safe_package_version("torch"),
            },
        },
        "git_commit": _safe_git_commit(repo_root),
    }

    np.savez_compressed(run_dir / "signal.npz", signal=signal_tensor)
    np.savez_compressed(run_dir / "signal_mask.npz", missing_mask=np.isnan(signal_tensor))
    np.savez_compressed(run_dir / "adjacency.npz", adjacency=adjacency_tensor)
    with (run_dir / "services.json").open("w", encoding="utf-8") as f:
        json.dump(services, f, indent=2)

    labels_df = pd.DataFrame(labels_rows)
    if not labels_df.empty:
        labels_df = labels_df.sort_values("timestamp").reset_index(drop=True)
    else:
        labels_df = pd.DataFrame(
            columns=[
                "timestamp",
                "regime",
                "category",
                "scenario",
                "target_services",
                "chaos_resource",
                "episode_id",
                "drift_flag",
            ]
        )
    if not labels_df.empty:
        labels_df["target_service"] = labels_df["target_services"].apply(
            lambda xs: xs[0] if isinstance(xs, list) and xs else ""
        )
        labels_df["target_services"] = labels_df["target_services"].apply(json.dumps)
        labels_df["is_injection"] = labels_df["regime"] == "injection"
    else:
        labels_df["target_service"] = pd.Series(dtype="object")
        labels_df["target_services"] = pd.Series(dtype="object")
        labels_df["is_injection"] = pd.Series(dtype="bool")
    _write_parquet(labels_df, run_dir / "labels.parquet")

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty and len(stats_df) == len(labels_df):
        stats_df["regime"] = labels_df["regime"]
        stats_df["scenario"] = labels_df["scenario"]
        stats_df["category"] = labels_df["category"]
        stats_df["episode_id"] = labels_df["episode_id"]
    stats_df.to_csv(run_dir / "graph_stats.csv", index=False)
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Run saved to %s", run_dir)
    return run_dir


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled EWAT dataset")
    parser.add_argument(
        "--config",
        default="configs/collection.yaml",
        help="Collection config path",
    )
    parser.add_argument(
        "--base-config",
        default="configs/default.yaml",
        help="EWAT base config path",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=["cluster", "local-portforward"],
        default="cluster",
        help="Endpoint selection mode for telemetry backends",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Disable trace collection; T(t) and graph edges remain empty/NaN",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()
    run_dir = collect_once(
        config_path=Path(args.config),
        base_config_path=Path(args.base_config),
        dry_run=args.dry_run,
        endpoint_mode=args.endpoint_mode,
        no_traces=args.no_traces,
    )
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
