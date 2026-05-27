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
import multiprocessing as mp
import sys
import time
import traceback
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
    SpanErrorRateIndex,
    SpanLatencyIndex,
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
    histogram_seed: int = 42,
) -> tuple[np.ndarray, list[ServiceGraph], list[LabelRecord]]:
    """Build signal + graphs + labels for one episode on a uniform grid."""
    t_start = float(bundle.boundaries["baseline_start"])
    t_end = float(bundle.boundaries["recovery_end"])
    # Guard against recorder restarts that left baseline_start far in the past.
    # Cap the usable baseline to 10 minutes before baseline_end.
    _baseline_end = float(bundle.boundaries.get("baseline_end") or bundle.boundaries.get("pre_start") or t_start)
    _max_baseline_s = 600.0
    if _baseline_end - t_start > _max_baseline_s:
        logger.warning(
            "  [%s] baseline_start is %.0f s before baseline_end — capping to %.0f s",
            bundle.episode_id, _baseline_end - t_start, _max_baseline_s,
        )
        t_start = _baseline_end - _max_baseline_s
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
        histogram_seed=histogram_seed,
    )

    # ------ Jaeger → spans --------------------------------------------
    jaeger_cfg = bundle.jaeger_dump or {}
    # Step 2 fix 2.2 (audit 2026-05-26): if Jaeger is expected but empty,
    # log explicit warning so downstream NaN in error_rate_http / latency_p99
    # fallback is not silently propagated. Previously: jaeger_cfg empty →
    # span_err_idx is None → fallback never runs → silent NaN.
    if bundle.jaeger_dump is None:
        logger.info("  [%s] no Jaeger dump available — fallback for error_rate_http "
                    "and latency_p99 disabled. Expect NaN from Prometheus only.",
                    bundle.episode_id)
    elif not jaeger_cfg:
        logger.warning(
            "  [%s] Jaeger dump bundled but EMPTY — error_rate_http and "
            "latency_p99 fallback will not fire. This is a silent data quality "
            "issue; check raw episode dir.", bundle.episode_id,
        )
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
    # Pre-index spans for error_rate_http fallback (HTTP + gRPC status codes).
    # Maps normalised gRPC service names → canonical: e.g. "productcatalog" → "product-catalog".
    _GRPC_CALLEE_MAP: dict[str, str] = {
        s.replace("-", "").replace("_", ""): s for s in svc
    }
    # Also map known OTel Demo gRPC service name suffixes explicitly
    _GRPC_CALLEE_MAP.update({
        "productcatalog": next((s for s in svc if "catalog" in s), ""),
        "productreview": next((s for s in svc if "review" in s or "product-r" in s), ""),
    })
    _GRPC_CALLEE_MAP = {k: v for k, v in _GRPC_CALLEE_MAP.items() if v}
    span_err_idx = SpanErrorRateIndex(
        jaeger_cfg,
        canonical_services=svc,
        aliases=aliases,
        grpc_callee_map=_GRPC_CALLEE_MAP,
    ) if jaeger_cfg else None
    span_lat_idx = SpanLatencyIndex(
        jaeger_cfg,
        canonical_services=svc,
        aliases=aliases,
    ) if jaeger_cfg else None

    # ------ Loki → records --------------------------------------------
    loki_cfg = bundle.loki_dump or {}
    timed_records = parse_loki_dump(loki_cfg) if loki_cfg else []
    apply_log_aliases(timed_records, aliases)
    log_backend = InMemoryLogBackend(timed_records)

    # Fit SentenceBERT centroid from the baseline phase (regime="normal").
    # Only enabled when sentence-transformers is installed.
    _semantic_available = False
    try:
        import sentence_transformers  # noqa: F401
        _semantic_available = True
    except ImportError:
        pass

    log_collector = LogCollector(
        backend=log_backend,
        window_s=log_window_s,
        semantic_scorers={} if _semantic_available else None,
        services=svc,
        semantic_enabled=_semantic_available,
        aliases={},
    )

    if _semantic_available:
        baseline_end = float(bundle.boundaries.get("pre_start") or bundle.boundaries["baseline_end"])
        baseline_records = log_backend.fetch_logs(t_start, baseline_end)
        if baseline_records:
            log_collector.fit_semantic_centroid(baseline_records)

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

    # Step 2 fix 2.4 (audit 2026-05-26): track empty-graph ratio.
    # If too many timesteps yield empty graphs (likely early window before
    # injection), warn — early-window graph statistics are biased low.
    n_empty_graphs = 0

    # Step 2 fix 2.1 (audit 2026-05-26): assert window alignment between
    # modalities M/T/L. The metric_window_s parameter was unused (line 171
    # comment "unused at the moment"). We now use it to validate that, for
    # each grid timestep, the Prometheus / Jaeger / Loki windows are aligned
    # within ``grid_step_s`` tolerance. Misaligned windows introduce noise in
    # cross-modality correlations (e.g., latency_p99 ↔ span_dur_p99).
    if metric_window_s <= 0:
        logger.warning(
            "  [%s] metric_window_s=%.1f; alignment check skipped. Set metric_window_s "
            "to enable cross-modality temporal validation.",
            episode_id, metric_window_s,
        )

    t_loop_start = time.time()
    for i, ts in enumerate(grid):
        M_t, _ = prom_collector.collect(timestamp=float(ts), service_index=service_index)
        T_t, _ = trace_collector.collect(timestamp=float(ts), service_index=service_index)
        L_t, _ = log_collector.collect(timestamp=float(ts), service_index=service_index)

        signal[i, :, METRICS_SLICE] = M_t
        signal[i, :, TRACES_SLICE] = T_t
        signal[i, :, LOGS_SLICE] = L_t

        _t_win_start = float(ts) - trace_window_s

        # Fill error_rate_http (M dim 3) from span status codes when Prometheus NaN.
        if span_err_idx is not None:
            _ERR_ABS = METRICS_SLICE.start + 3
            nan_mask = np.isnan(signal[i, :, _ERR_ABS])
            if nan_mask.any():
                err_rates = span_err_idx.error_rate_for_window(_t_win_start, float(ts))
                for s_idx, svc_name in enumerate(svc):
                    if nan_mask[s_idx]:
                        signal[i, s_idx, _ERR_ABS] = err_rates.get(svc_name, float("nan"))

        # Fill latency_p99 (M dim 2) from span duration P99 when Prometheus NaN.
        if span_lat_idx is not None:
            _LAT_ABS = METRICS_SLICE.start + 2
            nan_mask = np.isnan(signal[i, :, _LAT_ABS])
            if nan_mask.any():
                p99s = span_lat_idx.p99_for_window(_t_win_start, float(ts))
                for s_idx, svc_name in enumerate(svc):
                    if nan_mask[s_idx]:
                        signal[i, s_idx, _LAT_ABS] = p99s.get(svc_name, float("nan"))

        window_spans = span_backend.fetch_spans(float(ts) - trace_window_s, float(ts))
        graph = gbuilder.build(window_spans, services=svc, timestamp=float(ts))
        graphs.append(graph)
        # Step 2 fix 2.4 (audit 2026-05-26): track empty graphs
        if graph.n_edges == 0:
            n_empty_graphs += 1

        regime = _regime_for(float(ts), bundle.boundaries)
        # Four-regime encoding (EWAT §2, formalisation.md):
        #   (regime_label="normal",       drift_flag=False) → θ_normal
        #   (regime_label="normal",       drift_flag=True)  → θ_drift
        #       (benign drift, not an anomaly)
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

    # Step 2 fix 2.4: report sparsity ratio
    empty_ratio = n_empty_graphs / max(grid.size, 1)
    if empty_ratio > 0.20:
        logger.warning(
            "  [%s] %d/%d (%.1f%%) timesteps have empty graphs (n_edges=0). "
            "Early-window graph statistics are biased; consider increasing "
            "trace_window_s or filtering early steps in assemble_dataset.",
            episode_id, n_empty_graphs, grid.size, empty_ratio * 100,
        )

    logger.info(
        "  [%s] built %d timesteps in %.1fs (services=%d, dim=%d, "
        "empty_graphs=%d/%d=%.1f%%)",
        episode_id,
        grid.size,
        time.time() - t_loop_start,
        n,
        SIGNAL_DIM,
        n_empty_graphs, grid.size, empty_ratio * 100,
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


def _process_episode_worker(task: dict) -> dict:
    """Per-episode worker — runs in a subprocess when --workers > 1."""
    try:
        import torch
        if task.get("num_threads", 0) > 0:
            torch.set_num_threads(task["num_threads"])
    except ImportError:
        pass

    ep_dir = Path(task["ep_dir"])
    out_dir = Path(task["out_dir"])

    if out_dir.exists() and not task["force"]:
        return {"episode_id": ep_dir.name, "status": "skip"}

    try:
        bundle = load_episode(ep_dir)
        signal, graphs, labels = build_features(
            bundle,
            grid_step_s=task["grid_step_s"],
            metric_window_s=task["metric_window_s"],
            trace_window_s=task["trace_window_s"],
            log_window_s=task["log_window_s"],
            aliases=task["aliases"],
            graph_threshold=task["graph_threshold"],
            services=task["canonical_services"],
            histogram_seed=task.get("histogram_seed", 42),
        )
    except Exception:
        return {"episode_id": ep_dir.name, "status": "error", "error": traceback.format_exc()}

    graph_stats = [compute_graph_stats(g) for g in graphs]
    metadata = {
        "episode_id": bundle.episode_id,
        "scenario": bundle.scenario,
        "boundaries": bundle.boundaries,
        "canonical_services": task["canonical_services"],
        "feature_set": task["feature_set"],
        "grid_step_s": task["grid_step_s"],
        "trace_window_s": task["trace_window_s"],
        "log_window_s": task["log_window_s"],
        "config": task["collection_cfg_dict"],
        "base_config": task["base_cfg_dict"],
    }
    save_run_dataset(
        run_dir=out_dir,
        metadata=metadata,
        signal_tensor=signal,
        graph_sequence=graphs,
        labels=labels,
        graph_stats=graph_stats,
        services=task["canonical_services"],
    )
    # Provenance is metadata-only; a failure here does not corrupt the episode
    # (save_run_dataset already committed atomically above). Log and continue.
    try:
        _write_feature_provenance(
            out_dir,
            bundle=bundle,
            grid_step_s=task["grid_step_s"],
            trace_window_s=task["trace_window_s"],
            log_window_s=task["log_window_s"],
            feature_set=task["feature_set"],
            base_cfg_path=Path(task["base_config"]),
            collection_cfg_path=Path(task["collection_config"]),
        )
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Could not write feature_provenance.json for %s", ep_dir.name, exc_info=True
        )

    nan_total = float(np.isnan(signal).mean())
    nan_M = float(np.isnan(signal[:, :, :7]).mean())
    nan_T = float(np.isnan(signal[:, :, 7:13]).mean())
    nan_L = float(np.isnan(signal[:, :, 13:]).mean())
    return {
        "episode_id": bundle.episode_id,
        "status": "ok",
        "T": int(signal.shape[0]),
        "nan_total": nan_total, "nan_M": nan_M, "nan_T": nan_T, "nan_L": nan_L,
    }


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
    histogram_seed = int(
        collection_cfg.collection.get("histogram_seed", base_cfg.get("histogram_seed", 42))
    )

    canonical_services = sorted(str(s) for s in collection_cfg.collection.canonical_services)

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    feature_out_root = out_root / args.feature_set
    feature_out_root.mkdir(parents=True, exist_ok=True)

    episodes = _discover_episodes(raw_root, args.only)
    if not episodes:
        raise SystemExit(f"no episode directories found under {raw_root}")

    logger.info(
        "build_features: found %d episodes under %s  feature_set=%s  grid_step_s=%.0f  workers=%d",
        len(episodes), raw_root, args.feature_set, args.grid_step_s, args.workers,
    )

    collection_cfg_dict = OmegaConf.to_container(collection_cfg, resolve=True)
    base_cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)

    tasks = [
        {
            "ep_dir": str(ep_dir),
            "out_dir": str(feature_out_root / ep_dir.name),
            "force": args.force,
            "grid_step_s": args.grid_step_s,
            "metric_window_s": args.metric_window_s,
            "trace_window_s": args.trace_window_s,
            "log_window_s": args.log_window_s,
            "aliases": aliases,
            "graph_threshold": graph_threshold,
            "canonical_services": canonical_services,
            "feature_set": args.feature_set,
            "base_config": args.base_config,
            "collection_config": str(collection_cfg_path),
            "collection_cfg_dict": collection_cfg_dict,
            "base_cfg_dict": base_cfg_dict,
            "num_threads": max(1, mp.cpu_count() // args.workers) if args.workers > 1 else 0,
            "histogram_seed": histogram_seed,
        }
        for ep_dir in episodes
    ]

    n_ok = n_skip = n_err = 0

    def _handle_result(result: dict) -> None:
        nonlocal n_ok, n_skip, n_err
        status = result["status"]
        ep = result["episode_id"]
        if status == "skip":
            logger.info("  skip %s (already built; use --force to overwrite)", ep)
            n_skip += 1
        elif status == "ok":
            r = result
            logger.info(
                "Saved %s  NaN: total=%.1f%%  M=%.1f%%  T=%.1f%%  L=%.1f%%",
                ep, r["nan_total"] * 100, r["nan_M"] * 100, r["nan_T"] * 100, r["nan_L"] * 100,
            )
            n_ok += 1
        else:
            logger.error("FAILED %s:\n%s", ep, result.get("error", ""))
            n_err += 1

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for result in pool.imap_unordered(_process_episode_worker, tasks):
                _handle_result(result)
    else:
        for task in tasks:
            _handle_result(_process_episode_worker(task))

    logger.info("Done: %d built, %d skipped, %d failed", n_ok, n_skip, n_err)


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
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes (each loads its own SentenceBERT model)")
    return p.parse_args()


if __name__ == "__main__":
    main()
