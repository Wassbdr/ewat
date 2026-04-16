"""Periodic collection of signal and graph snapshots for one regime segment."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph.builder import ServiceGraphBuilder
from graph.diagnostics import GraphStats, compute_stats
from graph.types import ServiceGraph
from telemetry.signal_builder import SignalBuilder
from utils.serialization import LabelRecord

logger = logging.getLogger(__name__)


@dataclass
class SnapshotBatch:
    signal: npt.NDArray[np.float32]
    graphs: list[ServiceGraph]
    labels: list[LabelRecord]
    graph_stats: list[GraphStats]
    services: list[str]


class SnapshotCollector:
    """Collect S(t) and G(t) snapshots at a fixed interval."""

    def __init__(
        self,
        signal_builder: SignalBuilder,
        graph_builder: ServiceGraphBuilder,
        sample_interval_s: float = 15.0,
        semantic_fit_enabled: bool = True,
        raw_logs_hook: Callable[[dict], None] | None = None,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        self._signal_builder = signal_builder
        self._graph_builder = graph_builder
        self._sample_interval_s = sample_interval_s
        self._semantic_fit_enabled = semantic_fit_enabled
        self._raw_logs_hook = raw_logs_hook

    def collect_for_duration(
        self,
        duration_s: float,
        regime: str,
        category: str,
        scenario: str,
        target_services: list[str],
        chaos_resource: str,
        episode_id: str,
        services: list[str] | None = None,
    ) -> SnapshotBatch:
        """Collect a segment of snapshots for one regime.

        Parameters
        ----------
        duration_s:
            Segment length in seconds.
        regime:
            One of normal|injection|recovery.
        category:
            Scenario category.
        scenario:
            Scenario name.
        target_services:
            Services targeted by the scenario.
        chaos_resource:
            Chaos Mesh resource name or shell script id.
        episode_id:
            Episode identifier used for dataset validation.
        services:
            Optional canonical service ordering.
        """
        if duration_s <= 0:
            n = len(services or [])
            return SnapshotBatch(
                signal=np.zeros((0, n, 17), dtype=np.float32),
                graphs=[],
                labels=[],
                graph_stats=[],
                services=services or [],
            )

        start = time.time()
        deadline = start + duration_s
        next_tick = start

        signal_rows: list[npt.NDArray[np.float32]] = []
        graphs: list[ServiceGraph] = []
        labels: list[LabelRecord] = []
        stats: list[GraphStats] = []

        canonical_services = list(services) if services else None

        n_samples = 0
        while True:
            now = time.time()
            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.time()

            elapsed = now - start
            remaining = max(0.0, deadline - now)
            logger.info(
                "  [%s] sample=%d elapsed=%.0fs remaining=%.0fs",
                regime,
                n_samples,
                elapsed,
                remaining,
            )
            sample_start = time.time()
            snapshot = self._signal_builder.build(timestamp=now)
            signal_build_s = time.time() - sample_start
            n_samples += 1
            if canonical_services is None:
                canonical_services = list(snapshot.services)

            aligned_signal = _align_signal(snapshot.S, snapshot.services, canonical_services)
            graph_start = time.time()
            graph = self._build_graph(now, canonical_services)
            graph_build_s = time.time() - graph_start
            sample_elapsed_s = time.time() - sample_start
            tick_drift_s = now - next_tick
            logger.info(
                "  [%s] timing drift=%.2fs signal=%.2fs graph=%.2fs sample=%.2fs interval=%.2fs",
                regime,
                tick_drift_s,
                signal_build_s,
                graph_build_s,
                sample_elapsed_s,
                self._sample_interval_s,
            )

            signal_rows.append(aligned_signal)
            graphs.append(graph)
            stats.append(compute_stats(graph))
            labels.append(
                LabelRecord(
                    timestamp=now,
                    regime=regime,
                    category=category,
                    scenario=scenario,
                    target_services=list(target_services),
                    chaos_resource=chaos_resource,
                    episode_id=episode_id,
                )
            )
            if self._raw_logs_hook and snapshot.log_records:
                for record in snapshot.log_records:
                    self._raw_logs_hook(
                        {
                            "timestamp": now,
                            "service_name": record.service_name,
                            "pod_name": record.pod_name,
                            "level": record.level,
                            "body": record.body,
                            "regime": regime,
                            "category": category,
                            "scenario": scenario,
                            "episode_id": episode_id,
                            "chaos_resource": chaos_resource,
                        }
                    )

            next_tick += self._sample_interval_s
            if now >= deadline:
                break

        signal_tensor = np.stack(signal_rows, axis=0).astype(np.float32)
        batch = SnapshotBatch(
            signal=signal_tensor,
            graphs=graphs,
            labels=labels,
            graph_stats=stats,
            services=canonical_services,
        )

        # Auto-fit semantic centroids from the baseline window (Fix 2.3).
        # The SentenceBERT scorer only needs a reference fit once — we do it
        # at the end of the first normal segment so it is ready for all
        # subsequent regimes without requiring manual intervention.
        if self._semantic_fit_enabled and regime == "normal":
            t_start = start
            t_end = time.time()
            self._maybe_fit_semantic_centroid(t_start, t_end, canonical_services)

        return batch

    def _maybe_fit_semantic_centroid(
        self,
        t_start: float,
        t_end: float,
        expected_services: list[str] | None = None,
    ) -> None:
        """Fit per-service SentenceBERT centroids from a baseline window.

        Only runs when:
        1. The signal builder has a :class:`LogCollector` attached.
        2. At least one service still lacks a fitted centroid.

        The log records for the window [t_start, t_end] are fetched fresh
        from the log backend (they are cheap to re-fetch compared to not
        having a fitted scorer for the entire run).
        """
        log_collector = getattr(self._signal_builder, "_logs", None)
        if log_collector is None:
            return

        # Skip only when all expected services already have fitted centroids.
        scorers = getattr(log_collector, "_semantic_scorers", {})
        if expected_services is not None:
            missing = [
                service
                for service in expected_services
                if service not in scorers or scorers[service].centroid is None
            ]
            if not missing:
                return
        elif scorers and all(s.centroid is not None for s in scorers.values()):
            return

        try:
            reference_records = log_collector._backend.fetch_logs(t_start, t_end)
        except Exception:
            logger.warning("_maybe_fit_semantic_centroid: log fetch failed; skipping centroid fit")
            return

        if not reference_records:
            logger.debug("_maybe_fit_semantic_centroid: no log records in baseline window")
            return

        try:
            log_collector.fit_semantic_centroid(reference_records)
        except Exception:
            logger.warning(
                "_maybe_fit_semantic_centroid: centroid fit failed; "
                "L_SEMANTIC_ANOMALY will be NaN"
            )

    def _build_graph(self, timestamp: float, services: list[str]) -> ServiceGraph:
        trace_collector = getattr(self._signal_builder, "_traces", None)
        if trace_collector is None:
            return ServiceGraph(services=services, edges=[], timestamp=timestamp)

        return self._graph_builder.build_from_collector(
            trace_collector=trace_collector,
            timestamp=timestamp,
            services=services,
        )


def _align_signal(
    signal: npt.NDArray[np.float32],
    current_services: list[str],
    canonical_services: list[str],
) -> npt.NDArray[np.float32]:
    """Align one S(t) matrix to canonical service ordering with NaN padding."""
    if current_services == canonical_services:
        return signal

    idx = {name: i for i, name in enumerate(current_services)}
    aligned = np.full((len(canonical_services), signal.shape[1]), np.nan, dtype=np.float32)
    for row, service in enumerate(canonical_services):
        current_row = idx.get(service)
        if current_row is not None:
            aligned[row] = signal[current_row]
    return aligned
