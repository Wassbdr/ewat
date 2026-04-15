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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph.builder import ServiceGraphBuilder  # noqa: E402
from scripts.chaos_injector import ChaosInjector  # noqa: E402
from scripts.snapshot_collector import SnapshotBatch, SnapshotCollector  # noqa: E402
from telemetry.signal_builder import SignalBuilder  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402
from utils.serialization import save_run_dataset  # noqa: E402

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


def _concat_batches(
    batches: list[SnapshotBatch],
    services: list[str],
) -> tuple[np.ndarray, list, list, list]:
    signal_parts = [batch.signal for batch in batches if batch.signal.size > 0]
    signal_tensor = (
        np.concatenate(signal_parts, axis=0).astype(np.float32)
        if signal_parts
        else np.zeros((0, len(services), 17), dtype=np.float32)
    )

    graphs = [graph for batch in batches for graph in batch.graphs]
    labels = [label for batch in batches for label in batch.labels]
    stats = [stat for batch in batches for stat in batch.graph_stats]
    return signal_tensor, graphs, labels, stats


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


def collect_once(
    config_path: Path,
    base_config_path: Path,
    dry_run: bool = False,
    endpoint_mode: str = "cluster",
) -> Path:
    repo_root = REPO_ROOT
    cfg = _load_yaml(config_path)
    base_cfg = OmegaConf.load(base_config_path)
    _apply_endpoint_mode(base_cfg, endpoint_mode)

    seed = int(base_cfg.get("random", {}).get("seed", 42))
    seed_everything(seed)

    collection_cfg = cfg["collection"]

    signal_builder = SignalBuilder.from_config(base_cfg)
    graph_builder = ServiceGraphBuilder.from_config(base_cfg)

    sample_interval_s = float(collection_cfg["sample_interval_s"])
    collector = SnapshotCollector(
        signal_builder=signal_builder,
        graph_builder=graph_builder,
        sample_interval_s=sample_interval_s,
    )

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

    all_batches: list[SnapshotBatch] = []
    canonical_services: list[str] | None = None

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
                services=canonical_services,
            )
            canonical_services = baseline_batch.services
            all_batches.append(baseline_batch)

            pre_batch = collector.collect_for_duration(
                duration_s=pre_injection_s,
                regime="normal",
                category=scenario_spec.category,
                scenario=scenario_spec.name,
                target_services=scenario_spec.targets,
                chaos_resource=scenario_spec.file,
                episode_id=episode_id,
                services=canonical_services,
            )
            all_batches.append(pre_batch)

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
                    services=canonical_services,
                )
                all_batches.append(inj_batch)
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
                services=canonical_services,
            )
            all_batches.append(recovery_batch)

            if cool_down_s > 0:
                logger.info("Cooling down %.1fs", cool_down_s)
                time.sleep(cool_down_s)

    services = canonical_services or []
    signal_tensor, graphs, labels, stats = _concat_batches(all_batches, services)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": cfg,
        "base_config_path": str(base_config_path),
        "n_timestamps": int(signal_tensor.shape[0]),
        "n_services": len(services),
        "signal_dim": int(signal_tensor.shape[2]) if signal_tensor.ndim == 3 else 0,
        "dry_run": dry_run,
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

    save_run_dataset(
        run_dir=run_dir,
        metadata=metadata,
        signal_tensor=signal_tensor,
        graph_sequence=graphs,
        labels=labels,
        graph_stats=stats,
        services=services,
    )

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
    )
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
