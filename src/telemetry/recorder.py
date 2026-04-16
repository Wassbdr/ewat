"""Raw telemetry recorder used by Phase 1 of the EWAT pipeline.

For a given wall-clock window ``[t_start, t_end]`` the recorder issues bulk
range queries against Prometheus, Jaeger and Loki and returns the raw JSON
payloads as plain Python objects. No feature engineering happens here.

Design goals
------------
- **One fetch per source per episode**, not per sample tick. This removes
  the cadence pressure that caused the NaN/timeout cascade in the former
  online collector.
- **No dependency on NumPy/Pandas** so the recorder can be imported from
  lightweight CLI contexts (healthcheck scripts, quick probes).
- **Never raise on partial failures**: every source returns a payload plus
  a ``dict`` of diagnostics so Phase 2 can reason about data availability.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from telemetry.prom_queries import QUERIES, render

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(total_retries: int = 1, backoff: float = 0.3) -> requests.Session:
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


# ---------------------------------------------------------------------------
# Dump payloads
# ---------------------------------------------------------------------------


@dataclass
class PrometheusDump:
    """Raw range-query results keyed by logical query name."""

    start_unix_s: float
    end_unix_s: float
    step_s: int
    namespace: str
    window: str
    endpoint: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    fallback_used: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0


@dataclass
class JaegerDump:
    start_unix_s: float
    end_unix_s: float
    endpoint: str
    services_queried: list[str] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    per_service_counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0


@dataclass
class LokiDump:
    start_unix_s: float
    end_unix_s: float
    endpoint: str
    namespace: str
    streams: list[dict[str, Any]] = field(default_factory=list)
    n_lines: int = 0
    truncated: bool = False
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TelemetryRecorder:
    """Bulk-dump raw telemetry for a single episode window.

    Parameters
    ----------
    prometheus_endpoint, jaeger_endpoint, loki_endpoint:
        HTTP base URLs for the three backends. Any may be empty to skip
        the corresponding source.
    namespace:
        Kubernetes namespace filter.
    prom_step_s:
        Prometheus ``step`` parameter (seconds). Drives the resolution
        available to Phase 2 for M(t). 15 s matches typical scrape
        intervals.
    prom_rate_window:
        PromQL rate/irate window applied inside templates (e.g. ``"2m"``).
    prom_timeout_s, jaeger_timeout_s, loki_timeout_s:
        HTTP timeouts per request.
    jaeger_limit:
        Max number of traces per ``/api/traces`` call. Large enough to
        cover typical 10-minute episodes at load-generator traffic.
    loki_limit:
        Max log lines per Loki page. The recorder will paginate forward
        in time automatically.
    """

    def __init__(
        self,
        *,
        prometheus_endpoint: str,
        jaeger_endpoint: str,
        loki_endpoint: str,
        namespace: str = "ewat",
        prom_step_s: int = 15,
        prom_rate_window: str = "2m",
        prom_timeout_s: float = 30.0,
        jaeger_timeout_s: float = 30.0,
        loki_timeout_s: float = 30.0,
        jaeger_limit: int = 1500,
        loki_limit: int = 5000,
    ) -> None:
        self._prom = prometheus_endpoint.rstrip("/") if prometheus_endpoint else ""
        self._jaeger = jaeger_endpoint.rstrip("/") if jaeger_endpoint else ""
        self._loki = loki_endpoint.rstrip("/") if loki_endpoint else ""
        self._namespace = namespace
        self._prom_step = int(prom_step_s)
        self._prom_window = prom_rate_window
        self._prom_timeout = prom_timeout_s
        self._jaeger_timeout = jaeger_timeout_s
        self._loki_timeout = loki_timeout_s
        self._jaeger_limit = int(jaeger_limit)
        self._loki_limit = int(loki_limit)
        self._sess = _session(total_retries=1, backoff=0.3)

    # ------------------------------------------------------------------
    # Prometheus
    # ------------------------------------------------------------------

    def record_prometheus(self, t_start: float, t_end: float) -> PrometheusDump:
        """Run all PromQL range queries for the episode window."""
        dump = PrometheusDump(
            start_unix_s=t_start,
            end_unix_s=t_end,
            step_s=self._prom_step,
            namespace=self._namespace,
            window=self._prom_window,
            endpoint=self._prom,
        )
        if not self._prom:
            dump.errors["_global"] = "no endpoint"
            return dump

        t0 = time.time()
        for spec in QUERIES:
            primary, fallback = render(spec, self._namespace, self._prom_window)
            response, error = self._range_query(primary, t_start, t_end)
            used_fallback = False
            if (response is None or not _non_empty(response)) and fallback:
                response_fb, error_fb = self._range_query(fallback, t_start, t_end)
                if response_fb is not None and _non_empty(response_fb):
                    response = response_fb
                    used_fallback = True
                elif response is None and response_fb is not None:
                    response = response_fb
                elif error is None and error_fb is not None:
                    error = error_fb
            if response is None:
                dump.errors[spec.name] = error or "no response"
                continue
            dump.results[spec.name] = response
            dump.fallback_used[spec.name] = used_fallback
        dump.elapsed_s = time.time() - t0
        return dump

    def _range_query(
        self, query: str, t_start: float, t_end: float
    ) -> tuple[dict[str, Any] | None, str | None]:
        params = {
            "query": query,
            "start": f"{t_start:.3f}",
            "end": f"{t_end:.3f}",
            "step": str(self._prom_step),
        }
        try:
            resp = self._sess.get(
                f"{self._prom}/api/v1/query_range",
                params=params,
                timeout=self._prom_timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            return None, f"http: {exc}"
        except ValueError as exc:
            return None, f"json: {exc}"
        if payload.get("status") != "success":
            return None, f"prom status: {payload.get('error', 'unknown')}"
        return payload, None

    # ------------------------------------------------------------------
    # Jaeger
    # ------------------------------------------------------------------

    def record_jaeger(
        self, t_start: float, t_end: float, services: list[str]
    ) -> JaegerDump:
        """Fetch all traces for ``services`` between ``[t_start, t_end]``."""
        dump = JaegerDump(
            start_unix_s=t_start,
            end_unix_s=t_end,
            endpoint=self._jaeger,
            services_queried=list(services),
        )
        if not self._jaeger:
            dump.errors["_global"] = "no endpoint"
            return dump
        if not services:
            dump.errors["_global"] = "empty service list"
            return dump

        t0 = time.time()
        start_us = int(t_start * 1_000_000)
        end_us = int(t_end * 1_000_000)
        for svc in services:
            params = {
                "service": svc,
                "start": start_us,
                "end": end_us,
                "limit": self._jaeger_limit,
            }
            try:
                resp = self._sess.get(
                    f"{self._jaeger}/api/traces",
                    params=params,
                    timeout=self._jaeger_timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                dump.errors[svc] = f"http: {exc}"
                continue
            except ValueError as exc:
                dump.errors[svc] = f"json: {exc}"
                continue
            traces = payload.get("data", []) or []
            dump.traces.extend(traces)
            dump.per_service_counts[svc] = len(traces)
        dump.elapsed_s = time.time() - t0
        return dump

    # ------------------------------------------------------------------
    # Loki
    # ------------------------------------------------------------------

    def record_loki(self, t_start: float, t_end: float) -> LokiDump:
        """Fetch all log lines emitted from ``self._namespace`` in the window.

        Paginates forward in time when the number of returned lines equals
        the per-query limit (Loki truncation heuristic).
        """
        dump = LokiDump(
            start_unix_s=t_start,
            end_unix_s=t_end,
            endpoint=self._loki,
            namespace=self._namespace,
        )
        if not self._loki:
            dump.errors["_global"] = "no endpoint"
            return dump

        t0 = time.time()
        query = f'{{k8s_namespace_name="{self._namespace}"}}'
        cursor_ns = int(t_start * 1e9)
        end_ns = int(t_end * 1e9)
        max_pages = 50  # hard safety cap on pagination loops
        page = 0
        while cursor_ns < end_ns and page < max_pages:
            params = {
                "query": query,
                "start": cursor_ns,
                "end": end_ns,
                "limit": self._loki_limit,
                "direction": "forward",
            }
            try:
                resp = self._sess.get(
                    f"{self._loki}/loki/api/v1/query_range",
                    params=params,
                    timeout=self._loki_timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                dump.errors[f"page_{page}"] = f"http: {exc}"
                break
            except ValueError as exc:
                dump.errors[f"page_{page}"] = f"json: {exc}"
                break
            streams = payload.get("data", {}).get("result", []) or []
            if not streams:
                break
            n_added = 0
            last_ts_ns = cursor_ns
            for stream in streams:
                labels = stream.get("stream", {})
                values = stream.get("values", []) or []
                if not values:
                    continue
                dump.streams.append({"labels": labels, "values": values})
                n_added += len(values)
                try:
                    last_ts_ns = max(last_ts_ns, int(values[-1][0]))
                except (ValueError, IndexError):
                    continue
            dump.n_lines += n_added
            if n_added < self._loki_limit:
                break
            dump.truncated = True
            cursor_ns = last_ts_ns + 1
            page += 1
        dump.elapsed_s = time.time() - t0
        return dump


def _non_empty(response: dict[str, Any]) -> bool:
    return bool((response.get("data", {}) or {}).get("result", []))
