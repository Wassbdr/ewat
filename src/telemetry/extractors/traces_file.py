"""Offline trace extractor — spans → T(t) and G(t) over arbitrary windows.

Reuses :class:`telemetry.collectors.trace_collector.TraceCollector` via a
file-backed :class:`SpanQueryBackend` so that all the existing feature
logic (abnormal rate, depth, fan-out, retry rate, latency CV, span
duration P99 union) applies unchanged.

The dump is the compressed JSON produced by
:class:`telemetry.recorder.TelemetryRecorder.record_jaeger`. We parse
every span once, sort them by start time, and serve time-window
sub-ranges to the collector by bisection.
"""

from __future__ import annotations

import bisect
from typing import Any

import numpy as np

from telemetry.collectors.trace_collector import Span, SpanQueryBackend


class InMemorySpanBackend(SpanQueryBackend):
    """Serve spans from a pre-parsed in-memory list by start time.

    The backend is stateless wrt the collector's allowlist — the
    collector will filter via its canonical service index.
    """

    def __init__(self, spans: list[Span]) -> None:
        # Keep only spans with a usable start timestamp so bisection is meaningful.
        usable = [sp for sp in spans if sp.start_time_s is not None]
        usable.sort(key=lambda sp: sp.start_time_s or 0.0)
        self._spans = usable
        self._keys = [sp.start_time_s or 0.0 for sp in usable]
        # `_service_allowlist` is read by TraceCollector._sync_backend_allowlist
        # via hasattr() reflection; the attribute must exist even if unused.
        self._service_allowlist: set[str] | None = None

    def fetch_spans(self, start_unix_s: float, end_unix_s: float) -> list[Span]:
        if not self._spans:
            return []
        lo = bisect.bisect_left(self._keys, start_unix_s)
        hi = bisect.bisect_right(self._keys, end_unix_s)
        subset = self._spans[lo:hi]
        if self._service_allowlist is None:
            return list(subset)
        return [sp for sp in subset if sp.service_name in self._service_allowlist]


def parse_jaeger_dump(dump: dict[str, Any]) -> list[Span]:
    """Flatten a Jaeger dump into a :class:`Span` list.

    Mirrors :meth:`JaegerBackend._parse_response` so that the offline and
    online parsers stay in lock-step. Duplicated traces returned by
    multiple per-service queries are deduplicated by ``(trace_id, span_id)``.
    """
    seen: set[tuple[str, str]] = set()
    spans: list[Span] = []
    for trace in dump.get("traces", []) or []:
        trace_id = trace.get("traceID", "")
        processes = trace.get("processes", {}) or {}
        for sp in trace.get("spans", []) or []:
            span_id = sp.get("spanID", "")
            if (trace_id, span_id) in seen:
                continue
            seen.add((trace_id, span_id))

            duration_s = float(sp.get("duration", 0)) / 1_000_000.0
            start_us = sp.get("startTime")
            start_time_s = float(start_us) / 1_000_000.0 if start_us is not None else None

            parent_span_id = ""
            for ref in sp.get("references", []) or []:
                if ref.get("refType") == "CHILD_OF":
                    parent_span_id = ref.get("spanID", "")
                    break

            process_id = sp.get("processID", "")
            process = processes.get(process_id, {}) or {}
            service_name = process.get("serviceName", "") or ""

            status_code = "UNSET"
            is_retry = False
            for tag in sp.get("tags", []) or []:
                key = tag.get("key", "")
                value = tag.get("value")
                if key == "error" and value is True:
                    status_code = "ERROR"
                if key in ("retry", "http.retry") and value is True:
                    is_retry = True
                if key in ("http.retry_count", "http.resend_count"):
                    try:
                        is_retry = int(value) > 0
                    except (TypeError, ValueError):
                        pass

            spans.append(
                Span(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    service_name=service_name,
                    duration_s=duration_s,
                    start_time_s=start_time_s,
                    status_code=status_code,
                    is_retry=is_retry,
                )
            )
    return spans


def apply_aliases(spans: list[Span], aliases: dict[str, str]) -> None:
    """Rewrite ``Span.service_name`` in place using the aliases map."""
    if not aliases:
        return
    for sp in spans:
        if sp.service_name in aliases:
            sp.service_name = aliases[sp.service_name]


def bin_spans_by_service(
    spans: list[Span], services: list[str]
) -> dict[str, list[Span]]:
    """Return spans grouped by canonical service; unknowns are dropped."""
    svc_set = set(services)
    out: dict[str, list[Span]] = {s: [] for s in services}
    for sp in spans:
        if sp.service_name in svc_set:
            out[sp.service_name].append(sp)
    return out


def counts_in_window(
    spans_by_service: dict[str, list[Span]],
    t_start: float,
    t_end: float,
) -> dict[str, int]:
    """Return the number of spans per service whose start lies in ``[t_start, t_end]``."""
    out: dict[str, int] = {}
    for svc, sp_list in spans_by_service.items():
        out[svc] = _bisect_count(sp_list, t_start, t_end)
    return out


def _bisect_count(spans: list[Span], t_start: float, t_end: float) -> int:
    keys = [sp.start_time_s or 0.0 for sp in spans]
    lo = bisect.bisect_left(keys, t_start)
    hi = bisect.bisect_right(keys, t_end)
    return max(0, hi - lo)


def compute_graph_for_window(
    spans: list[Span],
    services: list[str],
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Compute the N×N×3 weighted adjacency for spans in ``[t_start, t_end]``.

    Channels: (volume, latency_median_s, error_rate). Mirrors
    :func:`graph.builder.build_service_graph` but operates on already-filtered
    spans to avoid a second allowlist pass.
    """
    from graph.builder import ServiceGraphBuilder

    subset = [sp for sp in spans if sp.start_time_s is not None
              and t_start <= sp.start_time_s <= t_end]
    builder = ServiceGraphBuilder(edge_presence_threshold=0)
    graph = builder.build(subset, services=services, timestamp=t_end)
    return graph.adjacency_tensor()
