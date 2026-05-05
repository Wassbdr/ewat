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
from dataclasses import dataclass
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


@dataclass
class _ErrRecord:
    """Lightweight record for error-rate computation from span tags."""

    start_time_s: float
    service: str  # canonical name (after alias + grpc callee mapping)
    is_error: bool


class SpanErrorRateIndex:
    """Pre-index all error-bearing spans for O(log n + k) per-window queries.

    Supports two evidence sources:
    * **Server-side HTTP spans** — spans owned by the service that carry
      ``http.status_code`` or ``http.response.status_code`` tags.
    * **Client-side gRPC spans** — spans that carry ``rpc.grpc.status_code``
      and a ``rpc.service`` tag identifying the callee service.  These
      attribute error counts to the callee (e.g. a frontend span calling
      ``hipstershop.AdService`` → counted toward ``ad``).

    The index is built once per episode from the raw Jaeger dump; per-window
    queries use binary search on ``start_time_s``.

    Parameters
    ----------
    dump:
        Parsed Jaeger dump (``{'traces': [...], ...}``).
    canonical_services:
        Ordered list of canonical service names.
    aliases:
        Service name aliases applied to span owners (e.g. ``frontend-proxy`` →
        ``frontend``).
    grpc_callee_map:
        Maps normalised gRPC service names to canonical service names, e.g.
        ``{'productcatalog': 'product-catalog', 'ad': 'ad', ...}``.
        The normalisation rule is: take the last ``.``-separated component of
        ``rpc.service``, lowercase, strip trailing ``"service"``.
    """

    # gRPC status codes that are considered non-error (OK/not set)
    _GRPC_OK = frozenset({"0", "OK", "ok", "", "UNSET"})
    # HTTP status key preference order
    _HTTP_KEYS = ("http.status_code", "http.response.status_code")

    def __init__(
        self,
        dump: dict[str, Any],
        canonical_services: list[str],
        aliases: dict[str, str],
        grpc_callee_map: dict[str, str] | None = None,
    ) -> None:
        self._svc_set = set(canonical_services)
        self._aliases = aliases
        self._grpc_map = grpc_callee_map or {}

        records: list[_ErrRecord] = []
        for trace in dump.get("traces", []) or []:
            processes = {
                pid: p.get("serviceName", "")
                for pid, p in (trace.get("processes", {}) or {}).items()
            }
            for sp in trace.get("spans", []) or []:
                start_us = sp.get("startTime")
                if start_us is None:
                    continue
                start_s = float(start_us) / 1_000_000.0
                tags = {
                    t["key"]: t["value"]
                    for t in (sp.get("tags", []) or [])
                    if isinstance(t, dict) and "key" in t
                }

                svc_raw = processes.get(sp.get("processID", ""), "")
                svc = aliases.get(svc_raw, svc_raw)

                # Server-side HTTP
                for hk in self._HTTP_KEYS:
                    if hk in tags:
                        if svc in self._svc_set:
                            v = str(tags[hk])
                            records.append(_ErrRecord(
                                start_time_s=start_s,
                                service=svc,
                                is_error=v[:1] in ("4", "5"),
                            ))
                        break  # only count once per span

                # Client-side gRPC → attribute to callee
                if "rpc.grpc.status_code" in tags:
                    rpc_svc_raw = str(tags.get("rpc.service", "") or "")
                    short = rpc_svc_raw.split(".")[-1].lower().removesuffix("service")
                    callee = self._grpc_map.get(short, short)
                    if callee in self._svc_set:
                        v = str(tags["rpc.grpc.status_code"])
                        records.append(_ErrRecord(
                            start_time_s=start_s,
                            service=callee,
                            is_error=v not in self._GRPC_OK,
                        ))

        records.sort(key=lambda r: r.start_time_s)
        self._records = records
        self._keys = [r.start_time_s for r in records]

    def error_rate_for_window(
        self, t_start: float, t_end: float
    ) -> dict[str, float]:
        """Return per-service HTTP/gRPC error rate in ``[t_start, t_end]``.

        Returns NaN for services with no spans in the window.
        """
        lo = bisect.bisect_left(self._keys, t_start)
        hi = bisect.bisect_right(self._keys, t_end)
        subset = self._records[lo:hi]

        totals: dict[str, int] = {}
        errors: dict[str, int] = {}
        for rec in subset:
            totals[rec.service] = totals.get(rec.service, 0) + 1
            errors[rec.service] = errors.get(rec.service, 0) + int(rec.is_error)

        return {
            svc: (errors.get(svc, 0) / totals[svc]) if svc in totals else float("nan")
            for svc in self._svc_set
        }


class SpanLatencyIndex:
    """Pre-index span durations for O(log n + k) per-service P99 queries.

    Only spans that are directly owned by a canonical service (i.e. their
    ``processID`` resolves to a canonical service name via aliases) are indexed.
    Services without direct spans (e.g. gRPC-only backends not instrumented
    with OTel SDK) will return NaN for all windows.

    Used to fill ``latency_p99`` (M dim 2) for services whose Prometheus
    histogram is unavailable (no Istio, no OTel HTTP server metrics).
    """

    def __init__(
        self,
        dump: dict[str, Any],
        canonical_services: list[str],
        aliases: dict[str, str],
    ) -> None:
        self._svc_set = set(canonical_services)
        # Per-service sorted arrays: start_time_s, duration_s
        svc_starts: dict[str, list[float]] = {s: [] for s in canonical_services}
        svc_durs: dict[str, list[float]] = {s: [] for s in canonical_services}

        for trace in dump.get("traces", []) or []:
            processes = {
                pid: p.get("serviceName", "")
                for pid, p in (trace.get("processes", {}) or {}).items()
            }
            for sp in trace.get("spans", []) or []:
                start_us = sp.get("startTime")
                if start_us is None:
                    continue
                dur_us = sp.get("duration", 0)
                if dur_us <= 0:
                    continue
                start_s = float(start_us) / 1_000_000.0
                dur_s = float(dur_us) / 1_000_000.0

                svc_raw = processes.get(sp.get("processID", ""), "")
                svc = aliases.get(svc_raw, svc_raw)
                if svc in self._svc_set:
                    svc_starts[svc].append(start_s)
                    svc_durs[svc].append(dur_s)

        # Sort each service's records by start time
        self._starts: dict[str, list[float]] = {}
        self._durs: dict[str, list[float]] = {}
        for svc in canonical_services:
            order = sorted(range(len(svc_starts[svc])), key=lambda i: svc_starts[svc][i])
            self._starts[svc] = [svc_starts[svc][i] for i in order]
            self._durs[svc] = [svc_durs[svc][i] for i in order]

    def p99_for_window(self, t_start: float, t_end: float) -> dict[str, float]:
        """Return per-service P99 span duration (seconds) in ``[t_start, t_end]``.

        Returns NaN for services with fewer than 2 spans in the window
        (P99 unreliable on tiny samples).
        """
        result: dict[str, float] = {}
        for svc in self._svc_set:
            starts = self._starts[svc]
            durs = self._durs[svc]
            if not starts:
                result[svc] = float("nan")
                continue
            lo = bisect.bisect_left(starts, t_start)
            hi = bisect.bisect_right(starts, t_end)
            window_durs = durs[lo:hi]
            if len(window_durs) < 2:
                result[svc] = float("nan")
            else:
                result[svc] = float(np.percentile(window_durs, 99))
        return result


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
