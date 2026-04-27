"""EWAT — Phase 2: build per-episode feature tensors from raw dumps.

Reads the raw telemetry dumps produced by ``scripts/record_episode.py`` and
materialises the canonical dataset artefacts:

::

    data/features/<feature_set>/<episode_id>/
    ├── signal.npz           # signal: (T, N, 17)  float32, NaN-filled where unavailable
    ├── signal_mask.npz      # missing_mask: (T, N, 17)  bool
    ├── adjacency.npz        # adjacency: (T, N, N, 3)  float32
    ├── labels.parquet       # one row per timestep: regime, scenario, category, ...
    ├── services.json        # canonical service ordering V
    ├── graph_stats.csv      # per-timestep graph diagnostics
    ├── metadata.json        # feature config + provenance (sha256, git commit, …)
    └── feature_provenance.json

No cluster access, no HTTP. Re-run freely with different grids or aggregation
tweaks — the raw dumps in ``data/raw/`` are the ground truth.

Usage
=====

::

    python -m scripts.build_features \
        --raw-root data/raw \
        --base-config configs/default.yaml \
        --config configs/collection.yaml \
        --feature-set v1 \
        --grid-step-s 30 \
        --trace-window-s 120
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
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

from graph.diagnostics import compute_stats as compute_graph_stats  # noqa: E402
from graph.types import ServiceGraph  # noqa: E402
from telemetry.collectors.log_collector import LogCollector  # noqa: E402
from telemetry.collectors.trace_collector import TraceCollector  # noqa: E402
from telemetry.extractors.logs_file import (  # noqa: E402
    InMemoryLogBackend,
    apply_aliases as apply_log_aliases,
    parse_loki_dump,
)
from telemetry.extractors.prom_file import FilePrometheusCollector  # noqa: E402
from telemetry.extractors.traces_file import (  # noqa: E402
    InMemorySpanBackend,
    apply_aliases as apply_span_aliases,
    parse_jaeger_dump,
)
from telemetry.feature_names import (  # noqa: E402
    LOGS_SLICE,
    METRICS_SLICE,
    SIGNAL_DIM,
    TRACES_SLICE,
)
from utils.serialization import LabelRecord, save_run_dataset  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EpisodeBundle:
    """In-memory view of one Phase 1 episode directory."""

    episode_dir: Path
    episode_id: str
    scenario: dict[str, Any]
    boundaries: dict[str, float]
    canonical_services: list[str]
    prom_dump: dict[str, Any] | None
    jaeger_dump: dict[str, Any] | None
    loki_dump: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def load_episode(episode_dir: Path) -> EpisodeBundle:
    """Read all dumps from one episode directory."""
    ep_json = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    prom = _load_json_gz(episode_dir / "prometheus_range.json.gz") if (
        episode_dir / "prometheus_range.json.gz"
    ).exists() else None
    jaeger = _load_json_gz(episode_dir / "jaeger_spans.json.gz") if (
        episode_dir / "jaeger_spans.json.gz"
    ).exists() else None
    loki = _load_json_gz(episode_dir / "loki_logs.json.gz") if (
        episode_dir / "loki_logs.json.gz"
    ).exists() else None
    return EpisodeBundle(
        episode_dir=episode_dir,
        episode_id=ep_json["episode_id"],
        scenario=ep_json["scenario"],
        boundaries=ep_json["boundaries"],
        canonical_services=ep_json.get("canonical_services", []),
        prom_dump=prom,
        jaeger_dump=jaeger,
        loki_dump=loki,
    )


# ---------------------------------------------------------------------------
# Phase boundaries → label regime
# ---------------------------------------------------------------------------


def _regime_for(ts: float, boundaries: dict[str, float]) -> str:
    """Return the regime name for a given timestamp inside the episode.

    Order (exclusive upper bound, inclusive lower):
      baseline → pre → injection → recovery
    """
    if ts < boundaries["baseline_end"]:
        return "normal"
    if ts < boundaries["pre_end"]:
        return "normal"  # pre-injection is still nominally normal
    if ts < boundaries["injection_end"]:
        return "injection"
    return "recovery"


# ---------------------------------------------------------------------------
# Feature construction per episode
# ---------------------------------------------------------------------------


def build_features(
    bundle: EpisodeBundle,
    *,
    grid_step_s: float,
    metric_window_s: float,  # unused at the moment (Prometheus step already handles)
    trace_window_s: float,
    log_window_s: float,
    aliases: dict[str, str],
    graph_threshold: int,
    services: list[str] | None = None,
) -> tuple[np.ndarray, list[ServiceGraph], list[LabelRecord]]:
    """Build signal + graphs + labels for one episode on a uniform grid."""
    t_start = float(bundle.boundaries["baseline_start"])
    t_end = float(bundle.boundaries["recovery_end"])
    grid = np.arange(t_start, t_end + 1e-9, grid_step_s, dtype=np.float64)
    svc = services or bundle.canonical_services
    if not svc:
        raise ValueError(f"episode {bundle.episode_id} has no canonical_services")
    service_index = {s: i for i, s in enumerate(svc)}
    n = len(svc)

    # ------ Prometheus ------------------------------------------------
    prom_cfg = bundle.prom_dump or {}
    prom_collector = FilePrometheusCollector(
        range_results=prom_cfg.get("results", {}) or {},
        fallback_used=prom_cfg.get("fallback_used", {}) or {},
        namespace=prom_cfg.get("namespace", "ewat"),
        services=svc,
        aliases=aliases,
    )

    # ------ Jaeger → spans --------------------------------------------
    jaeger_cfg = bundle.jaeger_dump or {}
    spans = parse_jaeger_dump(jaeger_cfg) if jaeger_cfg else []
    apply_span_aliases(spans, aliases)
    span_backend = InMemorySpanBackend(spans)
    trace_collector = TraceCollector(
        backend=span_backend,
        window_s=trace_window_s,
        services=svc,
        cache_ttl_s=grid_step_s,
        aliases={},
    )

    # ------ Loki → records --------------------------------------------
    loki_cfg = bundle.loki_dump or {}
    timed_records = parse_loki_dump(loki_cfg) if loki_cfg else []
    apply_log_aliases(timed_records, aliases)
    log_backend = InMemoryLogBackend(timed_records)
    log_collector = LogCollector(
        backend=log_backend,
        window_s=log_window_s,
        semantic_scorers=None,
        services=svc,
        semantic_enabled=False,  # filled in by optional Phase 2b post-processor
        aliases={},
    )

    # ------ Loop over grid --------------------------------------------
    signal = np.full((grid.size, n, SIGNAL_DIM), float("nan"), dtype=np.float32)
    graphs: list[ServiceGraph] = []
    labels: list[LabelRecord] = []
    episode_id = bundle.episode_id

    from graph.builder import ServiceGraphBuilder

    gbuilder = ServiceGraphBuilder(edge_presence_threshold=graph_threshold)

    scenario_name = bundle.scenario.get("name", "")
    category = bundle.scenario.get("category", "")
    chaos_resource = bundle.scenario.get("file", "")
    targets = list(bundle.scenario.get("targets", []) or [])

    t_loop_start = time.time()
    for i, ts in enumerate(grid):
        M_t, _ = prom_collector.collect(timestamp=float(ts), service_index=service_index)
        T_t, _ = trace_collector.collect(timestamp=float(ts), service_index=service_index)
        L_t, _ = log_collector.collect(timestamp=float(ts), service_index=service_index)

        signal[i, :, METRICS_SLICE] = M_t
        signal[i, :, TRACES_SLICE] = T_t
        signal[i, :, LOGS_SLICE] = L_t

        window_spans = span_backend.fetch_spans(float(ts) - trace_window_s, float(ts))
        graph = gbuilder.build(window_spans, services=svc, timestamp=float(ts))
        graphs.append(graph)

        regime = _regime_for(float(ts), bundle.boundaries)
        # Four-regime encoding (EWAT §2, formalisation.md):
        #   (regime_label="normal",       drift_flag=False) → θ_normal
        #   (regime_label="normal",       drift_flag=True)  → θ_drift   (benign drift, not an anomaly)
        #   (regime_label="injection",    drift_flag=False) → θ_anomaly
        #   (regime_label="drift_anomaly",drift_flag=True)  → θ_{drift∩anomaly}
        if regime == "injection" and category in ("drift", "overlap"):
            regime_label = "drift_anomaly" if category == "overlap" else "normal"
        else:
            regime_label = regime

        labels.append(
            LabelRecord(
                timestamp=float(ts),
                regime=regime_label,  # type: ignore[arg-type]
                category=category,
                scenario=scenario_name,
                target_services=targets,
                chaos_resource=chaos_resource,
                episode_id=episode_id,
                drift_flag=(category in ("drift", "overlap")),
            )
        )

    logger.info(
        "  [%s] built %d timesteps in %.1fs (services=%d, dim=%d)",
        episode_id,
        grid.size,
        time.time() - t_loop_start,
        n,
        SIGNAL_DIM,
    )
    return signal, graphs, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _discover_episodes(raw_root: Path, pattern: str | None) -> list[Path]:
    candidates = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if pattern:
        candidates = [p for p in candidates if pattern in p.name]
    return [p for p in candidates if (p / "episode.json").exists()]


def _write_feature_provenance(
    out_dir: Path,
    *,
    bundle: EpisodeBundle,
    grid_step_s: float,
    trace_window_s: float,
    log_window_s: float,
    feature_set: str,
    base_cfg_path: Path,
    collection_cfg_path: Path,
) -> None:
    payload = {
        "feature_set": feature_set,
        "built_at": datetime.now(UTC).isoformat(),
        "source_episode_dir": str(bundle.episode_dir),
        "source_episode_id": bundle.episode_id,
        "grid_step_s": grid_step_s,
        "trace_window_s": trace_window_s,
        "log_window_s": log_window_s,
        "signal_feature_names_source": "src/telemetry/feature_names.py",
        "extractors": {
            "metrics": "telemetry.extractors.prom_file.FilePrometheusCollector",
            "traces": "telemetry.collectors.trace_collector.TraceCollector + InMemorySpanBackend",
            "logs": "telemetry.collectors.log_collector.LogCollector + InMemoryLogBackend "
                    "(semantic offline)",
        },
        "base_config_sha256": _file_sha(base_cfg_path),
        "collection_config_sha256": _file_sha(collection_cfg_path),
    }
    with (out_dir / "feature_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _file_sha(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = REPO_ROOT / raw_root

    base_cfg = OmegaConf.load(args.base_config)
    collection_cfg_path = Path(args.config) if not Path(args.config).is_absolute() \
        else Path(args.config)
    if not collection_cfg_path.is_absolute():
        collection_cfg_path = REPO_ROOT / collection_cfg_path
    collection_cfg = OmegaConf.load(collection_cfg_path)

    aliases = dict(base_cfg.telemetry.get("service_name_aliases", {}) or {})
    graph_threshold = int(base_cfg.graph.get("edge_presence_threshold", 0))

    canonical_services = [str(s) for s in collection_cfg.collection.canonical_services]

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    feature_out_root = out_root / args.feature_set
    feature_out_root.mkdir(parents=True, exist_ok=True)

    episodes = _discover_episodes(raw_root, args.only)
    if not episodes:
        raise SystemExit(f"no episode directories found under {raw_root}")

    logger.info(
        "build_features: found %d episodes under %s  feature_set=%s  grid_step_s=%.0f",
        len(episodes), raw_root, args.feature_set, args.grid_step_s,
    )

    for ep_dir in episodes:
        out_dir = feature_out_root / ep_dir.name
        if out_dir.exists() and not args.force:
            logger.info("  skip %s (already built; use --force to overwrite)", ep_dir.name)
            continue
        try:
            bundle = load_episode(ep_dir)
        except Exception:
            logger.exception("failed to load episode %s", ep_dir)
            continue

        try:
            signal, graphs, labels = build_features(
                bundle,
                grid_step_s=args.grid_step_s,
                metric_window_s=args.metric_window_s,
                trace_window_s=args.trace_window_s,
                log_window_s=args.log_window_s,
                aliases=aliases,
                graph_threshold=graph_threshold,
                services=canonical_services,
            )
        except Exception:
            logger.exception("failed to build features for %s", bundle.episode_id)
            continue

        graph_stats = [compute_graph_stats(g) for g in graphs]

        metadata = {
            "episode_id": bundle.episode_id,
            "scenario": bundle.scenario,
            "boundaries": bundle.boundaries,
            "canonical_services": canonical_services,
            "feature_set": args.feature_set,
            "grid_step_s": args.grid_step_s,
            "trace_window_s": args.trace_window_s,
            "log_window_s": args.log_window_s,
            "config": OmegaConf.to_container(collection_cfg, resolve=True),
            "base_config": OmegaConf.to_container(base_cfg, resolve=True),
        }

        save_run_dataset(
            run_dir=out_dir,
            metadata=metadata,
            signal_tensor=signal,
            graph_sequence=graphs,
            labels=labels,
            graph_stats=graph_stats,
            services=canonical_services,
        )

        _write_feature_provenance(
            out_dir,
            bundle=bundle,
            grid_step_s=args.grid_step_s,
            trace_window_s=args.trace_window_s,
            log_window_s=args.log_window_s,
            feature_set=args.feature_set,
            base_cfg_path=Path(args.base_config),
            collection_cfg_path=collection_cfg_path,
        )


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EWAT Phase 2 — offline feature builder")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--output-root", default="data/features")
    p.add_argument("--feature-set", default="v1",
                   help="feature-set name; outputs go under data/features/<feature_set>/")
    p.add_argument("--base-config", default="configs/default.yaml")
    p.add_argument("--config", default="configs/collection.yaml")
    p.add_argument("--grid-step-s", type=float, default=30.0)
    p.add_argument("--metric-window-s", type=float, default=120.0,
                   help="(reserved) future use for rolling windows on M(t)")
    p.add_argument("--trace-window-s", type=float, default=120.0)
    p.add_argument("--log-window-s", type=float, default=120.0)
    p.add_argument("--only", default="",
                   help="only process episode dirs whose name contains this substring")
    p.add_argument("--force", action="store_true", help="re-build even if output exists")
    return p.parse_args()


if __name__ == "__main__":
    main()
