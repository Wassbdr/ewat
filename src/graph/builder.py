"""Service graph builder — constructs G(t) from OTel trace spans.

Consumes ``Span`` objects (from ``telemetry.collectors.trace_collector``) and
aggregates inter-service calls into a weighted directed graph.

Design decisions
----------------
- Nodes = Kubernetes services (identified by ``span.service_name``), not pods.
- Edge (i → j) exists if a span from service i has a child span in service j
  (CHILD_OF relationship).
- Edge weights: w_E = (volume, latency_median, error_rate) ∈ ℝ³.
- Presence threshold: ``volume > threshold`` (configurable, default 0).

Usage
-----
>>> from telemetry.collectors.trace_collector import Span
>>> builder = ServiceGraphBuilder(edge_presence_threshold=0)
>>> graph = builder.build(spans, services=["gateway", "order-svc", "payment-svc"])
>>> A = graph.adjacency_tensor()  # shape (3, 3, 3)
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from typing import Any

from graph.types import ServiceGraph, WeightedEdge
from telemetry.collectors.trace_collector import Span

logger = logging.getLogger(__name__)


# Type alias for a directed edge key (source_service, target_service)
_EdgeKey = tuple[str, str]


def _span_start_time(span: Span) -> float | None:
    """Return span start timestamp in seconds when available.

    The ``Span`` model exposes ``start_time_s`` for real backends, while
    older synthetic test spans may not define it.
    """
    return getattr(span, "start_time_s", None)


class ServiceGraphBuilder:
    """Build G(t) from a list of OTel spans.

    Parameters
    ----------
    edge_presence_threshold:
        Minimum call volume for an edge to be included. Default 0 = include
        all observed edges.
    """

    def __init__(self, edge_presence_threshold: int = 0) -> None:
        self._threshold = edge_presence_threshold

    @classmethod
    def from_config(cls, cfg: Any) -> ServiceGraphBuilder:
        """Construct from a Hydra/OmegaConf config.

        Expected key: ``graph.edge_presence_threshold``.
        """
        threshold = cfg.graph.get("edge_presence_threshold", 0)
        return cls(edge_presence_threshold=threshold)

    def build(
        self,
        spans: list[Span],
        services: list[str] | None = None,
        timestamp: float | None = None,
    ) -> ServiceGraph:
        """Construct G(t) from a flat list of spans.

        Parameters
        ----------
        spans:
            All spans in the observation window [t-W, t]. Must include both
            parent and child spans to detect inter-service edges.
        services:
            Canonical service list. When ``None``, auto-discovered from spans.
        timestamp:
            Unix timestamp for the snapshot. Defaults to now.

        Returns
        -------
        ServiceGraph
            The constructed graph with weighted edges.
        """
        ts = timestamp or time.time()

        # Build span lookup: span_id → Span (for resolving parent-child)
        span_by_id: dict[str, Span] = {}
        for span in spans:
            span_by_id[span.span_id] = span

        # Discover services if not provided
        if services is None:
            services = sorted({s.service_name for s in spans if s.service_name})

        svc_set = set(services)

        # Collect per-edge call records
        edge_calls: dict[_EdgeKey, list[_CallRecord]] = defaultdict(list)

        for span in spans:
            if not span.parent_span_id:
                continue  # root span — no parent → no inter-service edge

            parent = span_by_id.get(span.parent_span_id)
            if parent is None:
                continue  # parent not in this window (cross-window trace)

            parent_svc = parent.service_name
            child_svc = span.service_name

            # Skip intra-service calls (same service → same service)
            if parent_svc == child_svc:
                continue

            # Skip services not in our canonical list
            if parent_svc not in svc_set or child_svc not in svc_set:
                continue

            edge_calls[(parent_svc, child_svc)].append(
                _CallRecord(
                    duration_s=span.duration_s,
                    is_error=span.status_code == "ERROR",
                )
            )

        # Aggregate into weighted edges
        edges: list[WeightedEdge] = []
        for (source, target), calls in edge_calls.items():
            volume = len(calls)
            if volume <= self._threshold:
                continue

            durations = [c.duration_s for c in calls]
            n_errors = sum(1 for c in calls if c.is_error)

            edges.append(
                WeightedEdge(
                    source=source,
                    target=target,
                    volume=volume,
                    latency_median_s=float(statistics.median(durations)),
                    error_rate=n_errors / volume,
                )
            )

        graph = ServiceGraph(services=sorted(services), edges=edges, timestamp=ts)

        logger.debug(
            "ServiceGraphBuilder: G(t) with %d nodes, %d edges at t=%.0f",
            graph.n_services,
            graph.n_edges,
            ts,
        )
        return graph

    def build_windowed(
        self,
        spans: list[Span],
        window_s: float = 60.0,
        stride_s: float = 15.0,
        services: list[str] | None = None,
    ) -> list[ServiceGraph]:
        """Build a temporal sequence of graphs from spans.

        Partitions spans into overlapping time windows and builds one
        G(t) per window.

        Parameters
        ----------
        spans:
            All spans across the full observation period.
        window_s:
            Window duration in seconds.
        stride_s:
            Stride between consecutive windows in seconds.
        services:
            Canonical service list.

        Returns
        -------
        list[ServiceGraph]
            Sequence G(t_1), ..., G(t_T).
        """
        if window_s <= 0:
            msg = f"window_s must be > 0, got {window_s}"
            raise ValueError(msg)
        if stride_s <= 0:
            msg = f"stride_s must be > 0, got {stride_s}"
            raise ValueError(msg)

        if not spans:
            return []

        if services is None:
            services = sorted({s.service_name for s in spans if s.service_name})

        timed_spans: list[Span] = []
        span_times: list[float] = []
        for span in spans:
            start_time_s = _span_start_time(span)
            if start_time_s is not None:
                timed_spans.append(span)
                span_times.append(start_time_s)

        if not timed_spans:
            logger.warning(
                "ServiceGraphBuilder.build_windowed: spans have no start timestamps; "
                "falling back to a single graph snapshot"
            )
            return [self.build(spans, services=services, timestamp=time.time())]

        untimed_count = len(spans) - len(timed_spans)
        if untimed_count > 0:
            logger.warning(
                "ServiceGraphBuilder.build_windowed: ignoring %d/%d spans without start_time_s",
                untimed_count,
                len(spans),
            )

        min_ts = min(span_times)
        max_ts = max(span_times)

        # If the observation range is shorter than the window, build one snapshot.
        if max_ts <= min_ts + window_s:
            return [
                self.build(
                    timed_spans,
                    services=services,
                    timestamp=max_ts,
                )
            ]

        window_ends: list[float] = []
        t_end = min_ts + window_s
        while t_end <= max_ts:
            window_ends.append(t_end)
            t_end += stride_s

        # Always include the final window ending at max_ts.
        if not window_ends or window_ends[-1] < max_ts:
            window_ends.append(max_ts)

        graphs: list[ServiceGraph] = []
        for end_ts in window_ends:
            start_ts = end_ts - window_s
            window_spans = [
                span
                for span in timed_spans
                if (span_ts := _span_start_time(span)) is not None
                and start_ts <= span_ts <= end_ts
            ]
            graphs.append(
                self.build(
                    window_spans,
                    services=services,
                    timestamp=end_ts,
                )
            )

        return graphs

    def build_from_collector(
        self,
        trace_collector: Any,
        timestamp: float | None = None,
        services: list[str] | None = None,
        window_s: float = 120.0,
    ) -> ServiceGraph:
        """Build G(t) directly from a TraceCollector.

        Convenience method that fetches spans and builds the graph in one call.

        Parameters
        ----------
        trace_collector:
            A ``TraceCollector`` instance with a ``_backend.fetch_spans()`` method.
        timestamp:
            Target timestamp. Defaults to now.
        services:
            Canonical service list.
        window_s:
            Look-back window in seconds.

        Returns
        -------
        ServiceGraph
        """
        ts = timestamp or time.time()
        spans = self._fetch_spans_from_collector(trace_collector, ts - window_s, ts)
        return self.build(spans, services=services, timestamp=ts)

    def _fetch_spans_from_collector(
        self,
        trace_collector: Any,
        start_unix_s: float,
        end_unix_s: float,
    ) -> list[Span]:
        """Fetch spans from collector-compatible objects.

        Supports either:
        - ``trace_collector._backend.fetch_spans(start, end)``
        - ``trace_collector.fetch_spans(start, end)``
        """
        backend = getattr(trace_collector, "_backend", None)
        if backend is not None and hasattr(backend, "fetch_spans"):
            return backend.fetch_spans(start_unix_s, end_unix_s)

        fetch_spans = getattr(trace_collector, "fetch_spans", None)
        if callable(fetch_spans):
            return fetch_spans(start_unix_s, end_unix_s)

        msg = (
            "trace_collector must provide _backend.fetch_spans(start, end) "
            "or fetch_spans(start, end)"
        )
        raise AttributeError(msg)


class _CallRecord:
    """Lightweight record for a single inter-service call."""

    __slots__ = ("duration_s", "is_error")

    def __init__(self, duration_s: float, is_error: bool) -> None:
        self.duration_s = duration_s
        self.is_error = is_error
