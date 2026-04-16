"""Trace collector — extracts T(t) ∈ ℝ^{N×6} from OTel spans via Jaeger HTTP API.

The collector queries a Jaeger-compatible backend for OTLP spans in the window
[t-W, t] and aggregates per service.  The backend is pluggable via the
`SpanQueryBackend` ABC.

Cluster backend: rca-jaeger.rca-sandbox.svc.cluster.local:16686

Features (columns 7–12 in S(t), columns 0–5 in T(t)):
    0  span_dur_med       Median span duration (P99 on union, seconds)
    1  abnormal_span_rate Fraction of error/abnormal spans
    2  trace_depth        Median max depth of trace trees
    3  fan_out            Median fan-out (children per span)
    4  retry_rate         Fraction of retry spans (retried / total)
    5  latency_cv         Latency coefficient of variation (std / mean)

Aggregation (pod → service):
    span_dur_med     → p99_union on raw durations
    abnormal_rate    → volume_weighted
    retry_rate       → volume_weighted
    trace_depth      → median
    fan_out          → median
    latency_cv       → median
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from telemetry.feature_names import TRACES_DIM
from telemetry.features.aggregation import (
    aggregate_median,
    aggregate_p99_union,
)

# Local column indices within T_t (shape N×6) — NOT global S(t) indices
_T_SPAN_DUR_MED = 0
_T_ABNORMAL_RATE = 1
_T_TRACE_DEPTH = 2
_T_FAN_OUT = 3
_T_RETRY_RATE = 4
_T_LATENCY_CV = 5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """Minimal span representation extracted from OTLP.

    Parameters
    ----------
    trace_id:
        16-byte hex trace identifier.
    span_id:
        8-byte hex span identifier.
    parent_span_id:
        Parent span id, or empty string for root spans.
    service_name:
        ``service.name`` resource attribute.
    duration_s:
        Span duration in seconds.
    start_time_s:
        Span start timestamp in Unix seconds when available.
    status_code:
        OTel status code: ``"OK"``, ``"ERROR"``, or ``"UNSET"``.
    is_retry:
        True if the span carries a retry semantic attribute
        (e.g. ``http.resend_count > 0`` or ``rpc.retry = true``).
    """

    trace_id: str
    span_id: str
    parent_span_id: str
    service_name: str
    duration_s: float
    start_time_s: float | None = None
    status_code: str = "UNSET"
    is_retry: bool = False


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class SpanQueryBackend(ABC):
    """Abstract backend for fetching spans in a time window."""

    @abstractmethod
    def fetch_spans(self, start_unix_s: float, end_unix_s: float) -> list[Span]:
        """Return all spans whose *start time* falls in [start_unix_s, end_unix_s].

        Parameters
        ----------
        start_unix_s:
            Window start (Unix epoch, seconds).
        end_unix_s:
            Window end (Unix epoch, seconds).

        Returns
        -------
        list[Span]
            All spans in the window across all services.
        """
        ...


# ---------------------------------------------------------------------------
# Jaeger HTTP backend (reference implementation)
# ---------------------------------------------------------------------------


class JaegerBackend(SpanQueryBackend):
    """Jaeger HTTP query API backend.

    Uses the Jaeger HTTP API v1:
        GET /api/services                → list of instrumented service names
        GET /api/traces?service=<svc>&start=<µs>&end=<µs>&limit=<n>
                                         → traces with embedded spans

    Jaeger durations are in **microseconds**; we convert to seconds.

    Parameters
    ----------
    endpoint:
        Jaeger query service base URL.
        Cluster value: ``http://rca-jaeger.rca-sandbox.svc.cluster.local:16686``
    namespace:
        Kubernetes namespace whose services we want. Jaeger has no native
        namespace filter — we restrict to services whose name matches a
        known prefix or an explicit allow-list (``services`` param of
        :class:`TraceCollector`).
    timeout:
        HTTP request timeout in seconds.
    limit:
        Maximum number of traces to fetch per service per call.
    service_allowlist:
        When provided, only fetch traces for services in this set.
        Prevents fetching from services in other namespaces that Jaeger
        also instruments, which would pollute G(t) and S(t).
    """

    def __init__(
        self,
        endpoint: str,
        namespace: str = "ewat",
        timeout: float = 15.0,
        limit: int = 100,
        service_allowlist: set[str] | None = None,
        fetch_total_timeout_s: float = 10.0,  # MUST be < sample_interval_s (15s) to prevent segment drift
        max_parallel: int = 8,
    ) -> None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self._endpoint = endpoint.rstrip("/")
        self._namespace = namespace
        # Per-socket read timeout — caps idle time between bytes on one request.
        # Not a total-response-time cap (use fetch_total_timeout_s for that).
        self._timeout = min(timeout, 15.0)
        # Limit traces per service: keeps payload small over slow port-forwards.
        self._limit = min(limit, 20)
        self._service_allowlist = service_allowlist
        # Hard wall-clock budget for the entire fetch_spans() call.
        self._fetch_total_timeout_s = fetch_total_timeout_s
        self._max_parallel = max_parallel
        self._last_fetch_stats: dict[str, float] = {
            "services_considered": 0.0,
            "services_skipped_budget": 0.0,
            "services_timed_out": 0.0,
            "services_request_error": 0.0,
            "elapsed_s": 0.0,
        }

        # No retries on read/connect timeouts — fail fast, don't compound delays.
        _retry = Retry(
            total=1,
            read=0,
            connect=0,
            backoff_factor=0.0,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        _adapter = HTTPAdapter(max_retries=_retry)
        self._session = requests.Session()
        self._session.mount("http://", _adapter)
        self._session.mount("https://", _adapter)

    # ------------------------------------------------------------------

    def fetch_spans(self, start_unix_s: float, end_unix_s: float) -> list[Span]:
        """Query Jaeger for all spans across known services in the time window.

        Steps:
        1. GET /api/services to discover instrumented service names.
        2. For each service: GET /api/traces with the time window.
        3. Parse and flatten all spans into a :class:`Span` list.

        Parameters
        ----------
        start_unix_s:
            Window start (Unix epoch, seconds).
        end_unix_s:
            Window end (Unix epoch, seconds).

        Returns
        -------
        list[Span]
        """
        # If the canonical service allowlist is already populated (set by
        # _sync_backend_allowlist on every collect() call), skip the
        # /api/services roundtrip entirely — it adds one slow HTTP request
        # per tick for no benefit when the service list is already known.
        t0 = time.time()
        if self._service_allowlist:
            services = list(self._service_allowlist)
        else:
            services = self._get_services()
            if not services:
                self._last_fetch_stats = {
                    "services_considered": 0.0,
                    "services_skipped_budget": 0.0,
                    "services_timed_out": 0.0,
                    "services_request_error": 0.0,
                    "elapsed_s": time.time() - t0,
                }
                return []

        # Jaeger API expects microseconds
        start_us = int(start_unix_s * 1_000_000)
        end_us = int(end_unix_s * 1_000_000)

        # Parallel queries with a hard wall-clock budget.
        # `requests` timeout= only measures idle time between bytes, NOT total
        # response time. Over a slow port-forward a response can stream one byte
        # every few seconds and never trigger the socket timeout. We solve this
        # by running each service query in a thread and calling fut.result(timeout=)
        # which is a real wall-clock deadline enforced by the GIL-releasing I/O wait.
        deadline = time.time() + self._fetch_total_timeout_s
        workers = min(self._max_parallel, len(services))
        spans: list[Span] = []
        n_budget_skips = 0
        n_timeouts = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_svc = {
                pool.submit(self._fetch_one_service, svc, start_us, end_us): svc
                for svc in services
            }
            for fut, svc in future_to_svc.items():
                remaining = deadline - time.time()
                if remaining <= 0:
                    logger.warning("JaegerBackend: budget exhausted, skipping '%s'", svc)
                    n_budget_skips += 1
                    continue
                try:
                    spans.extend(fut.result(timeout=remaining))
                except concurrent.futures.TimeoutError:
                    n_timeouts += 1
                    logger.warning(
                        "JaegerBackend: wall-clock timeout for '%s' (budget=%.0fs)",
                        svc,
                        self._fetch_total_timeout_s,
                    )

        self._last_fetch_stats = {
            "services_considered": float(len(services)),
            "services_skipped_budget": float(n_budget_skips),
            "services_timed_out": float(n_timeouts),
            "services_request_error": float(getattr(self, "_request_errors_last_fetch", 0)),
            "elapsed_s": time.time() - t0,
        }
        self._request_errors_last_fetch = 0
        return spans

    def _fetch_one_service(self, svc: str, start_us: int, end_us: int) -> list[Span]:
        """Fetch spans for a single service — runs inside a thread pool worker."""
        import requests

        params = {"service": svc, "start": start_us, "end": end_us, "limit": self._limit}
        t0 = time.time()
        try:
            resp = self._session.get(
                f"{self._endpoint}/api/traces",
                params=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            self._request_errors_last_fetch = getattr(self, "_request_errors_last_fetch", 0) + 1
            logger.warning("JaegerBackend: /api/traces error for '%s': %s", svc, exc)
            return []
        try:
            result = self._parse_response(resp.json())
            logger.debug(
                "JaegerBackend: '%s' → %d spans in %.1fs", svc, len(result), time.time() - t0
            )
            return result
        except ValueError as exc:
            self._request_errors_last_fetch = getattr(self, "_request_errors_last_fetch", 0) + 1
            logger.warning("JaegerBackend: JSON parse error for '%s': %s", svc, exc)
            return []

    def get_last_fetch_stats(self) -> dict[str, float]:
        """Return diagnostics for the last fetch_spans() call."""
        return dict(self._last_fetch_stats)

    def _get_services(self) -> list[str]:
        """Return the list of service names from Jaeger /api/services."""
        import requests

        try:
            resp = self._session.get(
                f"{self._endpoint}/api/services",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            try:
                return resp.json().get("data", [])
            except ValueError as exc:
                logger.error("JaegerBackend: JSON parse error on /api/services: %s", exc)
                return []
        except requests.RequestException as exc:
            logger.error("JaegerBackend: /api/services error: %s", exc)
            return []

    def _parse_response(self, data: dict[str, Any]) -> list[Span]:
        """Parse a Jaeger /api/traces JSON response into a flat Span list.

        Jaeger JSON structure:
            data: list of trace objects, each with:
                traceID: str
                spans: list of span objects with:
                    spanID, traceID, references, duration (µs),
                    tags (list of {key, type, value}),
                    process: {serviceName, ...}
        """
        spans: list[Span] = []
        # Jaeger returns a process map keyed by processID per trace
        for trace in data.get("data", []):
            trace_id: str = trace.get("traceID", "")
            processes: dict[str, dict] = trace.get("processes", {})

            for sp in trace.get("spans", []):
                span_id: str = sp.get("spanID", "")
                duration_s: float = float(sp.get("duration", 0)) / 1_000_000.0
                start_us = sp.get("startTime")
                start_time_s = float(start_us) / 1_000_000.0 if start_us is not None else None

                # Parent span: first CHILD_OF reference
                parent_span_id = ""
                for ref in sp.get("references", []):
                    if ref.get("refType") == "CHILD_OF":
                        parent_span_id = ref.get("spanID", "")
                        break

                # Service name from the process map
                process_id: str = sp.get("processID", "")
                process = processes.get(process_id, {})
                service_name: str = process.get("serviceName", "")

                # Tags: error flag and retry flag
                status_code = "UNSET"
                is_retry = False
                for tag in sp.get("tags", []):
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


# ---------------------------------------------------------------------------
# Trace feature computation helpers
# ---------------------------------------------------------------------------


def _compute_trace_structures(
    spans: list[Span],
) -> dict[str, dict[str, Any]]:
    """Compute per-trace structural stats (depth, fan-out) from a flat span list.

    Returns a dict keyed by trace_id with:
        - "max_depth": int
        - "avg_fan_out": float
    """
    from collections import defaultdict

    # Build parent → children map per trace
    children: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for span in spans:
        if span.parent_span_id:
            children[span.trace_id][span.parent_span_id].append(span.span_id)

    span_by_id: dict[str, dict[str, Span]] = defaultdict(dict)
    for span in spans:
        span_by_id[span.trace_id][span.span_id] = span

    result: dict[str, dict[str, Any]] = {}
    for trace_id, smap in span_by_id.items():
        child_map = children[trace_id]

        # BFS from roots to compute depth.
        # Guard against cyclic span references in real Jaeger data (span A
        # lists B as parent, B lists A) — without a visited set the BFS
        # loops forever.
        roots = [sid for sid, sp in smap.items() if not sp.parent_span_id]
        max_depth = 0
        fan_outs: list[int] = []
        visited: set[str] = set()
        from collections import deque as _deque
        queue: _deque[tuple[str, int]] = _deque((r, 1) for r in roots)
        while queue:
            sid, depth = queue.popleft()  # FIFO = BFS
            if sid in visited:
                continue
            visited.add(sid)
            max_depth = max(max_depth, depth)
            kids = child_map.get(sid, [])
            fan_outs.append(len(kids))
            queue.extend((k, depth + 1) for k in kids)

        result[trace_id] = {
            "max_depth": max_depth,
            "avg_fan_out": float(np.mean(fan_outs)) if fan_outs else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------


class TraceCollector:
    """Fetch T(t) ∈ ℝ^{N×6} from a span query backend.

    Parameters
    ----------
    backend:
        Implementation of :class:`SpanQueryBackend`.
    window_s:
        Look-back window in seconds.
    services:
        Optional explicit list of services. Auto-discovered when ``None``.
    """

    def __init__(
        self,
        backend: SpanQueryBackend,
        window_s: float = 120.0,
        services: list[str] | None = None,
        cache_ttl_s: float = 30.0,
    ) -> None:
        self._backend = backend
        self._window_s = window_s
        self._services = services
        self._cache_ttl_s = cache_ttl_s
        # Cache last-fetched spans so the graph builder can reuse them without
        # a second Jaeger round-trip per sample tick.
        self._cached_spans: list[Span] | None = None
        self._cached_ts: float = 0.0

    def collect(
        self,
        timestamp: float | None = None,
        service_index: dict[str, int] | None = None,
    ) -> tuple[npt.NDArray[np.float32], list[str]]:
        """Return T(t) for the window ending at ``timestamp``.

        Parameters
        ----------
        timestamp:
            Unix timestamp (seconds). Defaults to now.
        service_index:
            Pre-defined mapping service → row index (shared with Prometheus
            collector so matrices can be concatenated).

        Returns
        -------
        T_t:
            Float32 array of shape (N, 6). NaN where no spans observed.
        services:
            List of N service names.
        """
        ts = timestamp or time.time()
        self._sync_backend_allowlist(service_index)
        spans = self._backend.fetch_spans(ts - self._window_s, ts)
        self._cached_spans = spans
        self._cached_ts = ts

        services, svc_idx = self._resolve_services(spans, service_index)
        n = len(services)
        T_t = np.full((n, TRACES_DIM), float("nan"), dtype=np.float32)

        if not spans:
            return T_t, services

        trace_structs = _compute_trace_structures(spans)
        self._fill_features(T_t, svc_idx, spans, trace_structs)
        return T_t, services

    def get_cached_spans(self, max_age_s: float | None = None) -> list[Span] | None:
        """Return spans from the last ``collect()`` call if still fresh.

        Parameters
        ----------
        max_age_s:
            Maximum age in seconds for the cache to be considered valid.
            Default 30 s covers one sample interval.

        Returns
        -------
        list[Span] or None if cache is empty or stale.
        """
        if self._cached_spans is None:
            return None
        max_age = self._cache_ttl_s if max_age_s is None else max_age_s
        if time.time() - self._cached_ts > max_age:
            return None
        return self._cached_spans

    def _sync_backend_allowlist(self, service_index: dict[str, int] | None) -> None:
        """Propagate canonical service names to backends that support allowlists.

        This is critical when Jaeger hosts traces from multiple namespaces:
        the collector should only fetch spans for our canonical service set.
        """
        backend_allowlist = getattr(self._backend, "_service_allowlist", None)
        if backend_allowlist is None and not hasattr(self._backend, "_service_allowlist"):
            return

        allowlist: set[str] | None = None
        if service_index is not None:
            allowlist = set(service_index.keys())
        elif self._services is not None:
            allowlist = set(self._services)

        if allowlist is not None:
            self._backend._service_allowlist = allowlist

    # ------------------------------------------------------------------

    def _resolve_services(
        self,
        spans: list[Span],
        service_index: dict[str, int] | None,
    ) -> tuple[list[str], dict[str, int]]:
        if service_index is not None:
            services = sorted(service_index, key=lambda s: service_index[s])
            return services, service_index

        discovered: set[str] = {s.service_name for s in spans if s.service_name}
        if self._services is not None:
            discovered |= set(self._services)
        services = sorted(discovered)
        return services, {s: i for i, s in enumerate(services)}

    def _fill_features(
        self,
        T_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        spans: list[Span],
        trace_structs: dict[str, dict[str, Any]],
    ) -> None:
        # Group spans by service
        svc_spans: dict[str, list[Span]] = {}
        for span in spans:
            if span.service_name in svc_idx:
                svc_spans.setdefault(span.service_name, []).append(span)

        # Group trace-level structural features by service (via trace's root service)
        # Heuristic: use service of the first span seen per trace
        trace_service: dict[str, str] = {}
        for span in spans:
            if span.trace_id not in trace_service and span.service_name in svc_idx:
                trace_service[span.trace_id] = span.service_name

        svc_depths: dict[str, list[float]] = {}
        svc_fanouts: dict[str, list[float]] = {}
        for trace_id, stats in trace_structs.items():
            svc = trace_service.get(trace_id, "")
            if svc and svc in svc_idx:
                svc_depths.setdefault(svc, []).append(stats["max_depth"])
                svc_fanouts.setdefault(svc, []).append(stats["avg_fan_out"])

        for svc, row in svc_idx.items():
            spans_s = svc_spans.get(svc, [])
            if not spans_s:
                continue

            durations = np.array([sp.duration_s for sp in spans_s], dtype=np.float64)
            n_spans = len(spans_s)
            n_error = sum(1 for sp in spans_s if sp.status_code == "ERROR")
            n_retry = sum(1 for sp in spans_s if sp.is_retry)

            # span_dur_med — P99 on union (single service = single pod group here)
            T_t[row, _T_SPAN_DUR_MED] = aggregate_p99_union([durations.astype(np.float32)])

            # abnormal_span_rate
            T_t[row, _T_ABNORMAL_RATE] = n_error / max(n_spans, 1)

            # trace_depth — median over traces attributed to this service
            depths = svc_depths.get(svc, [])
            T_t[row, _T_TRACE_DEPTH] = (
                aggregate_median(np.array(depths, dtype=np.float32)) if depths else float("nan")
            )

            # fan_out — median over traces
            fan_outs = svc_fanouts.get(svc, [])
            T_t[row, _T_FAN_OUT] = (
                aggregate_median(np.array(fan_outs, dtype=np.float32))
                if fan_outs
                else float("nan")
            )

            # retry_rate
            T_t[row, _T_RETRY_RATE] = n_retry / max(n_spans, 1)

            # latency_cv = std / mean
            mean_dur = float(np.mean(durations))
            std_dur = float(np.std(durations))
            T_t[row, _T_LATENCY_CV] = (
                std_dur / mean_dur if mean_dur > 0 else 0.0
            )
