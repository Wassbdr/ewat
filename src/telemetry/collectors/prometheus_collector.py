"""Prometheus collector — extracts M(t) ∈ ℝ^{N×7}.

Queries Prometheus (HTTP API) for the 7 metrics features over a sliding
window [t-W, t] and returns one row per Kubernetes service in namespace ewat.

Features (columns 0–6):
    0  cpu_util         CPU utilisation (fraction of limit; NaN when limits are
                        missing; max over pods)
    1  ram_util         RAM utilisation (fraction of limit, max over pods)
    2  latency_p99      HTTP request latency P99 on union of histogram buckets
    3  error_rate_http  (4xx + 5xx) / total_requests, volume-weighted
    4  net_sat          Network saturation bytes/s (max over pods)
    5  disk_io          Disk IOPS (max over pods)
    6  queue_depth      Pod restart rate (restarts/s, proxy for queue saturation
                        when Envoy/Istio is absent; max over pods)

Aggregation (pod → service):
    Saturation:  max
    Rates:       volume-weighted
    Latency:     P99 on union of raw samples (reconstruct_from_histogram)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from telemetry.feature_names import (
    M_CPU_UTIL,
    M_DISK_IO,
    M_ERROR_RATE,
    M_LATENCY_P99,
    M_NET_SAT,
    M_QUEUE_DEPTH,
    M_RAM_UTIL,
    METRICS_DIM,
)
from telemetry.features.aggregation import (
    aggregate_max,
    aggregate_p99_union,
    aggregate_volume_weighted,
    reconstruct_from_histogram,
)

logger = logging.getLogger(__name__)

_DEPLOYMENT_HASH_RE = re.compile(r"^[a-f0-9]{8,12}$")
_POD_SUFFIX_RE = re.compile(r"^[a-z0-9]{5}$")
_STATEFULSET_ORDINAL_RE = re.compile(r"^\d+$")

# Prometheus range-query window (seconds) for rate() / irate() calls.
# Should match configs/default.yaml scrape_interval_s × some factor.
_RATE_WINDOW = "2m"

# PromQL templates — {namespace} and {window} are filled at query time.
# Service is identified via the `service` or `app` label (Kubernetes convention).
_QUERIES: dict[str, str] = {
    # CPU: sum of cores used / limit per pod, label: pod, service
    "cpu_usage": (
        "sum by (pod, service, namespace) ("
        "  rate(container_cpu_usage_seconds_total"
        "{{namespace='{namespace}', container!=''}}"
        "[{window}])"
        ")"
    ),
    "cpu_limit": (
        "sum by (pod, service, namespace) ("
        "  kube_pod_container_resource_limits"
        "{{namespace='{namespace}', resource='cpu', container!=''}}"
        ")"
    ),
    # RAM
    "ram_usage": (
        "sum by (pod, service, namespace) ("
        "  container_memory_working_set_bytes"
        "{{namespace='{namespace}', container!=''}}"
        ")"
    ),
    "ram_limit": (
        "sum by (pod, service, namespace) ("
        "  kube_pod_container_resource_limits"
        "{{namespace='{namespace}', resource='memory', container!=''}}"
        ")"
    ),
    # HTTP latency histogram (e.g. from Istio / Envoy sidecar)
    "http_request_duration_bucket": (
        "sum by (pod, service, namespace, le) ("
        "  rate(istio_request_duration_milliseconds_bucket"
        "{{namespace='{namespace}', reporter='destination'}}"
        "[{window}])"
        ")"
    ),
    # HTTP total and error counts
    "http_requests_total": (
        "sum by (pod, service, namespace) ("
        "  rate(istio_requests_total"
        "{{namespace='{namespace}', reporter='destination'}}"
        "[{window}])"
        ")"
    ),
    "http_requests_errors": (
        "sum by (pod, service, namespace) ("
        "  rate(istio_requests_total"
        "{{namespace='{namespace}', reporter='destination',"
        "   response_code=~'[45][0-9][0-9]'}}"
        "[{window}])"
        ")"
    ),
    # Network saturation (bytes transmitted per second)
    "net_transmit_bytes": (
        "sum by (pod, service, namespace) ("
        "  rate(container_network_transmit_bytes_total"
        "{{namespace='{namespace}'}}"
        "[{window}])"
        ")"
    ),
    # Disk I/O (reads + writes per second)
    "disk_iops": (
        "sum by (pod, service, namespace) ("
        "  rate(container_fs_reads_total{{namespace='{namespace}'}}[{window}])"
        "  + rate(container_fs_writes_total{{namespace='{namespace}'}}[{window}])"
        ")"
    ),
    # Queue depth — Envoy upstream_rq_pending gauge (NOT the _total counter)
    "queue_depth": (
        "sum by (pod, service, namespace) ("
        "  envoy_cluster_upstream_rq_pending"
        "{{namespace='{namespace}'}}"
        ")"
    ),
    # gRPC latency histogram — OTel SDK rpc.server.duration (ms unit, old semconv)
    # via gateway Prometheus exporter (namespace prefix "otel").
    "grpc_request_duration_bucket": (
        "sum by (k8s_pod_name, service_name, k8s_namespace_name, le) ("
        "  rate(otel_rpc_server_duration_milliseconds_bucket"
        "{{k8s_namespace_name='{namespace}'}}"
        "[{window}])"
        ")"
    ),
}

# Fallback PromQL queries for environments without Istio/Envoy.
# Tried when the primary query returns an empty result set.
#
# These target the OTel SDK old HTTP semconv (http.server.duration in ms),
# which is what the EWAT OTel gateway prometheus exporter exposes at port 9464
# (scraped by the monitoring-metrics Prometheus kubernetes-pods job).
#
# Key label differences vs Istio:
#   - Metric name: otel_http_server_duration_milliseconds (not istio_request_duration_ms)
#   - Namespace filter: k8s_namespace_name (not namespace)
#   - Service grouping: service_name (not service)
#   - Pod grouping: k8s_pod_name (not pod)
#   - Status code: http_status_code (not response_code)
#   - Buckets are in milliseconds (same as Istio → same /1000 conversion applies)
_FALLBACK_QUERIES: dict[str, str] = {
    "http_request_duration_bucket": (
        "sum by (k8s_pod_name, service_name, k8s_namespace_name, le) ("
        "  rate(otel_http_server_duration_milliseconds_bucket"
        "{{k8s_namespace_name='{namespace}'}}"
        "[{window}])"
        ")"
    ),
    "http_requests_total": (
        "sum by (k8s_pod_name, service_name, k8s_namespace_name) ("
        "  rate(otel_http_server_duration_milliseconds_count"
        "{{k8s_namespace_name='{namespace}'}}"
        "[{window}])"
        ")"
    ),
    "http_requests_errors": (
        "sum by (k8s_pod_name, service_name, k8s_namespace_name) ("
        "  rate(otel_http_server_duration_milliseconds_count"
        "{{k8s_namespace_name='{namespace}',"
        "   http_status_code=~'[45][0-9][0-9]'}}"
        "[{window}])"
        ")"
    ),
    # Envoy gauge not available and no Istio → fall back to pod restart rate
    # as a proxy for queue saturation / service instability.
    "queue_depth": (
        "sum by (pod, namespace) ("
        "  rate(kube_pod_container_status_restarts_total"
        "{{namespace='{namespace}'}}"
        "[{window}])"
        ")"
    ),
    # gRPC fallback: direct-scrape from app pods (no otel_ prefix, Prometheus-added
    # namespace label). rpc.server.duration → rpc_server_duration_milliseconds_bucket (ms).
    "grpc_request_duration_bucket": (
        "sum by (k8s_pod_name, service_name, namespace, le) ("
        "  rate(rpc_server_duration_milliseconds_bucket"
        "{{namespace='{namespace}'}}"
        "[{window}])"
        ")"
    ),
}


class PrometheusCollector:
    """Fetch M(t) ∈ ℝ^{N×7} from Prometheus for namespace ``ewat``.

    Parameters
    ----------
    endpoint:
        Prometheus HTTP API base URL, e.g.
        ``http://monitoring-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090``.
    namespace:
        Kubernetes namespace to query. Defaults to ``ewat``.
    rate_window:
        PromQL rate/irate window, e.g. ``"2m"``.
    timeout:
        HTTP request timeout in seconds.
    services:
        Optional explicit list of service names. When ``None`` the collector
        discovers services from the query labels.
    """

    def __init__(
        self,
        endpoint: str,
        namespace: str = "ewat",
        rate_window: str = _RATE_WINDOW,
        timeout: float = 10.0,
        services: list[str] | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._namespace = namespace
        self._rate_window = rate_window
        self._timeout = timeout
        self._services: list[str] | None = services
        self._aliases: dict[str, str] = aliases or {}

        _retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        _adapter = HTTPAdapter(max_retries=_retry)
        self._session = requests.Session()
        self._session.mount("http://", _adapter)
        self._session.mount("https://", _adapter)
        # Records whether each query key used its fallback path at last collect()
        self._query_used_fallback: dict[str, bool] = {}
        # Keys for which the fallback warning has already been emitted once.
        # Avoids flooding logs every 15 s when Istio metrics are absent.
        self._fallback_warned: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        timestamp: float | None = None,
        service_index: dict[str, int] | None = None,
    ) -> tuple[npt.NDArray[np.float32], list[str]]:
        """Query Prometheus and return M(t).

        Parameters
        ----------
        timestamp:
            Unix timestamp (seconds) for the instant query. Defaults to now.
        service_index:
            Mapping from service name → row index in the output matrix. When
            provided the returned matrix rows match this order. When ``None``
            the collector auto-discovers services and assigns indices
            alphabetically.

        Returns
        -------
        M_t:
            Float32 array of shape (N, 7). NaN where data is unavailable.
        services:
            List of N service names corresponding to rows of M_t.
        """
        ts = timestamp or time.time()

        raw = self._query_all(ts)
        services, svc_idx = self._resolve_services(raw, service_index)
        n = len(services)

        M_t = np.full((n, METRICS_DIM), float("nan"), dtype=np.float32)

        self._fill_cpu(M_t, svc_idx, raw)
        self._fill_ram(M_t, svc_idx, raw)
        self._fill_latency(M_t, svc_idx, raw)
        self._fill_error_rate(M_t, svc_idx, raw)
        self._fill_net_sat(M_t, svc_idx, raw)
        self._fill_disk_io(M_t, svc_idx, raw)
        self._fill_queue_depth(M_t, svc_idx, raw)

        return M_t, services

    # ------------------------------------------------------------------
    # Internal: PromQL execution
    # ------------------------------------------------------------------

    def _query_all(self, ts: float) -> dict[str, list[dict[str, Any]]]:
        """Execute all PromQL instant queries and return labelled result sets.

        When a primary query returns an empty result (e.g. because Istio is not
        installed), a fallback query (OTel SDK standard names) is tried and a
        WARNING is logged.
        """
        results: dict[str, list[dict[str, Any]]] = {}
        self._query_used_fallback = {}
        for key, tmpl in _QUERIES.items():
            query = tmpl.format(namespace=self._namespace, window=self._rate_window)
            data = self._instant_query(query, ts)
            used_fallback = False
            if not data and key in _FALLBACK_QUERIES:
                fallback = _FALLBACK_QUERIES[key].format(
                    namespace=self._namespace, window=self._rate_window
                )
                data = self._instant_query(fallback, ts)
                if data:
                    used_fallback = True
                    if key not in self._fallback_warned:
                        self._fallback_warned.add(key)
                        logger.warning(
                            "PrometheusCollector: primary query '%s' empty; "
                            "using OTel SDK fallback PromQL (logged once per session)",
                            key,
                        )
            results[key] = data
            self._query_used_fallback[key] = used_fallback
        return results

    def _instant_query(self, query: str, ts: float) -> list[dict[str, Any]]:
        """Execute a single PromQL instant query at ``ts``."""
        params: dict[str, Any] = {
            "query": query,
            "time": ts,
        }
        try:
            resp = self._session.get(
                f"{self._endpoint}/api/v1/query",
                params=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as exc:
                logger.error("Prometheus JSON parse error for query %.80s: %s", query, exc)
                return []
            if data.get("status") != "success":
                logger.warning("Prometheus query failed: %s", data.get("error", "unknown"))
                return []
            return data["data"]["result"]
        except requests.RequestException as exc:
            logger.error("Prometheus HTTP error for query %.80s: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # Internal: service resolution
    # ------------------------------------------------------------------

    def _resolve_services(
        self,
        raw: dict[str, list[dict[str, Any]]],
        service_index: dict[str, int] | None,
    ) -> tuple[list[str], dict[str, int]]:
        """Build or validate the service list."""
        if service_index is not None:
            services = sorted(service_index, key=lambda s: service_index[s])
            return services, service_index

        discovered: set[str] = set()
        for results in raw.values():
            for item in results:
                svc = self._service_label(item["metric"], self._aliases)
                if svc:
                    discovered.add(svc)

        if self._services is not None:
            # Merge explicit list with discovered
            discovered |= set(self._services)

        services = sorted(discovered)
        svc_idx = {s: i for i, s in enumerate(services)}
        return services, svc_idx

    @staticmethod
    def _service_label(
        metric: dict[str, str],
        aliases: dict[str, str] | None = None,
    ) -> str | None:
        """Extract service name from Prometheus metric labels.

        Handles both Istio/kubelet labels (``service``, ``pod``) and
        OTel SDK labels (``service_name``, ``k8s_pod_name``) that appear
        when the OTel gateway prometheus exporter is scraped.

        Applies ``self._aliases`` so that OTel SDK service names (e.g.
        ``productcatalogservice``) are normalised to canonical names
        (``productcatalog``) configured in ``telemetry.service_name_aliases``.
        """
        alias_map = aliases or {}
        direct = (
            metric.get("service")
            or metric.get("service_name")  # OTel SDK resource attribute
            or metric.get("app")
            or metric.get("app_kubernetes_io_name")
        )
        if direct:
            return alias_map.get(direct, direct)

        pod = metric.get("pod", "") or metric.get("k8s_pod_name", "")
        if not pod:
            return None

        parts = pod.split("-")

        # Deployment-like pod: <name>-<replicaset-hash>-<pod-suffix>
        if (
            len(parts) >= 3
            and _DEPLOYMENT_HASH_RE.match(parts[-2])
            and _POD_SUFFIX_RE.match(parts[-1])
        ):
            name = "-".join(parts[:-2]) or None
            return alias_map.get(name, name) if name else None

        # StatefulSet pod: <name>-<ordinal>
        if len(parts) >= 2 and _STATEFULSET_ORDINAL_RE.match(parts[-1]):
            name = "-".join(parts[:-1]) or None
            return alias_map.get(name, name) if name else None

        # DaemonSet-like pod: <name>-<pod-suffix>
        if len(parts) >= 2 and _POD_SUFFIX_RE.match(parts[-1]):
            name = "-".join(parts[:-1]) or None
            return alias_map.get(name, name) if name else None

        return alias_map.get(pod, pod)

    @staticmethod
    def _pod_value(item: dict[str, Any]) -> tuple[str, float]:
        """Return (pod_name, scalar_value) from an instant query result item.

        Accepts both ``pod`` (Istio/kubelet) and ``k8s_pod_name`` (OTel SDK).
        """
        pod = item["metric"].get("pod") or item["metric"].get("k8s_pod_name", "")
        value = float(item["value"][1])
        return pod, value

    # ------------------------------------------------------------------
    # Internal: feature filling
    # ------------------------------------------------------------------

    def _fill_cpu(
        self,
        M_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        raw: dict[str, list[dict[str, Any]]],
    ) -> None:
        """CPU utilisation = usage / limit, aggregated by max over pods."""
        # Build pod → (usage, limit) maps per service
        pod_usage: dict[str, dict[str, float]] = {}  # svc → {pod → usage}
        pod_limit: dict[str, dict[str, float]] = {}

        for item in raw.get("cpu_usage", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            pod, val = self._pod_value(item)
            pod_usage.setdefault(svc, {})[pod] = val

        for item in raw.get("cpu_limit", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            pod, val = self._pod_value(item)
            # Only store positive limits — pods without CPU limits are excluded
            # to avoid division by 1e-9 which inflates utilization to ~10^9.
            if val > 0:
                pod_limit.setdefault(svc, {})[pod] = val

        for svc, row in svc_idx.items():
            usage_map = pod_usage.get(svc, {})
            limit_map = pod_limit.get(svc, {})
            if not usage_map:
                continue
            # CPU util contract: only normalized usage/limit values are valid.
            # If no positive limit is available, keep NaN to avoid unit mixing.
            utils: list[float] = [
                usage_map[p] / limit_map[p]
                for p in usage_map
                if p in limit_map
            ]
            if utils:
                M_t[row, M_CPU_UTIL] = aggregate_max(np.array(utils, dtype=np.float32))

    def _fill_ram(
        self,
        M_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        raw: dict[str, list[dict[str, Any]]],
    ) -> None:
        """RAM utilisation = working_set / limit, aggregated by max over pods."""
        pod_usage: dict[str, dict[str, float]] = {}
        pod_limit: dict[str, dict[str, float]] = {}

        for item in raw.get("ram_usage", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            pod, val = self._pod_value(item)
            pod_usage.setdefault(svc, {})[pod] = val

        for item in raw.get("ram_limit", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            pod, val = self._pod_value(item)
            # Only store positive limits; pods without memory limits are skipped
            # from utilization to avoid silent inflation from arbitrary fallbacks.
            if val > 0:
                pod_limit.setdefault(svc, {})[pod] = val

        for svc, row in svc_idx.items():
            usage_map = pod_usage.get(svc, {})
            limit_map = pod_limit.get(svc, {})
            if not usage_map:
                continue
            util_per_pod = np.array(
                [usage_map[p] / limit_map[p] for p in usage_map if p in limit_map],
                dtype=np.float32,
            )
            if util_per_pod.size > 0:
                M_t[row, M_RAM_UTIL] = aggregate_max(util_per_pod)

    def _fill_latency(
        self,
        M_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        raw: dict[str, list[dict[str, Any]]],
    ) -> None:
        """HTTP and gRPC P99 latency via histogram reconstruction (union across pods).

        Collects from both ``http_request_duration_bucket`` (Istio or OTel HTTP)
        and ``grpc_request_duration_bucket`` (OTel gRPC rpc.server.duration) so
        that services instrumented only with gRPC still contribute a latency value.

        Both Istio and OTel SDK histograms use **millisecond** bucket boundaries.
        Conversion to seconds (÷1000) is applied unconditionally.
        """
        # Collect cumulative bucket rates per (svc, pod, le)
        # Each key is from a rate() query → already per-second incremental counts.
        bucket_data: dict[tuple[str, str], dict[float, float]] = {}

        for key in ("http_request_duration_bucket", "grpc_request_duration_bucket"):
            for item in raw.get(key, []):
                svc = self._service_label(item["metric"], self._aliases)
                if svc not in svc_idx:
                    continue
                pod = item["metric"].get("pod") or item["metric"].get("k8s_pod_name", "")
                le_str = item["metric"].get("le", "Inf")
                if le_str == "+Inf":
                    continue  # skip +Inf bucket
                le = float(le_str)
                _, val = self._pod_value(item)
                bucket_data.setdefault((svc, pod), {})[le] = val

        # Per service: reconstruct samples per pod, union them
        svc_samples: dict[str, list[np.ndarray]] = {}
        for (svc, _pod), le_map in bucket_data.items():
            if not le_map:
                continue
            bounds = np.array(sorted(le_map.keys()))
            # Convert cumulative to incremental
            cum_counts = np.array([le_map[b] for b in bounds])
            inc_counts = np.diff(np.concatenate([[0.0], cum_counts]))
            # All supported sources (Istio ms, OTel HTTP ms, OTel gRPC ms old semconv)
            # use millisecond bucket boundaries. Always divide by 1000 to get seconds.
            bounds_s = bounds / 1000.0
            samples = reconstruct_from_histogram(bounds_s, inc_counts)
            svc_samples.setdefault(svc, []).append(samples)

        for svc, row in svc_idx.items():
            sample_lists = svc_samples.get(svc, [])
            M_t[row, M_LATENCY_P99] = aggregate_p99_union(sample_lists)

    def _fill_error_rate(
        self,
        M_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        raw: dict[str, list[dict[str, Any]]],
    ) -> None:
        """HTTP error rate = errors / total, volume-weighted over pods."""
        pod_total: dict[str, dict[str, float]] = {}
        pod_errors: dict[str, dict[str, float]] = {}

        for item in raw.get("http_requests_total", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            pod, val = self._pod_value(item)
            pod_total.setdefault(svc, {})[pod] = val

        for item in raw.get("http_requests_errors", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            pod, val = self._pod_value(item)
            pod_errors.setdefault(svc, {})[pod] = val

        for svc, row in svc_idx.items():
            total_map = pod_total.get(svc, {})
            err_map = pod_errors.get(svc, {})
            if not total_map:
                continue
            pods = list(total_map.keys())
            volumes = np.array([total_map[p] for p in pods], dtype=np.float32)
            rates = np.array(
                [err_map.get(p, 0.0) / max(total_map[p], 1e-9) for p in pods],
                dtype=np.float32,
            )
            M_t[row, M_ERROR_RATE] = aggregate_volume_weighted(rates, volumes)

    def _fill_net_sat(
        self,
        M_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        raw: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Network transmit bytes/s — max over pods."""
        svc_vals: dict[str, list[float]] = {}
        for item in raw.get("net_transmit_bytes", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            _, val = self._pod_value(item)
            svc_vals.setdefault(svc, []).append(val)

        for svc, row in svc_idx.items():
            vals = svc_vals.get(svc, [])
            if vals:
                M_t[row, M_NET_SAT] = aggregate_max(np.array(vals, dtype=np.float32))

    def _fill_disk_io(
        self,
        M_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        raw: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Disk IOPS — max over pods."""
        svc_vals: dict[str, list[float]] = {}
        for item in raw.get("disk_iops", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            _, val = self._pod_value(item)
            svc_vals.setdefault(svc, []).append(val)

        for svc, row in svc_idx.items():
            vals = svc_vals.get(svc, [])
            if vals:
                M_t[row, M_DISK_IO] = aggregate_max(np.array(vals, dtype=np.float32))

    def _fill_queue_depth(
        self,
        M_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        raw: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Queue depth (pending requests) — max over pods."""
        svc_vals: dict[str, list[float]] = {}
        for item in raw.get("queue_depth", []):
            svc = self._service_label(item["metric"], self._aliases)
            if svc not in svc_idx:
                continue
            _, val = self._pod_value(item)
            svc_vals.setdefault(svc, []).append(val)

        for svc, row in svc_idx.items():
            vals = svc_vals.get(svc, [])
            if vals:
                M_t[row, M_QUEUE_DEPTH] = aggregate_max(np.array(vals, dtype=np.float32))
