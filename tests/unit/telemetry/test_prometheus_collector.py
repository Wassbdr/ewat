"""Unit tests for PrometheusCollector using a minimal HTTP stub."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from telemetry.collectors.prometheus_collector import PrometheusCollector
from telemetry.feature_names import M_CPU_UTIL, METRICS_DIM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prom_result(metric: dict, value: float) -> dict:
    """Build a Prometheus instant-query result item."""
    return {"metric": metric, "value": [0, str(value)]}


def _success_response(results: list) -> dict:
    return {"status": "success", "data": {"resultType": "vector", "result": results}}


def _make_fake_get(responses: dict[str, dict]):
    """Return a callable that routes by query substring, for mocking session.get."""

    def fake_get(url, params=None, **kwargs):
        query = (params or {}).get("query", "")
        for key, resp_data in responses.items():
            if key in query:
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json = MagicMock(return_value=resp_data)
                return mock_resp
        # default: empty success
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=_success_response([]))
        return mock_resp

    return fake_get


def _make_collector(responses: dict[str, dict]) -> PrometheusCollector:
    """Build a PrometheusCollector whose HTTP session is stubbed."""
    collector = PrometheusCollector(endpoint="http://fake-prom:9090", namespace="ewat")
    collector._session = MagicMock()
    collector._session.get = MagicMock(side_effect=_make_fake_get(responses))
    return collector


# ---------------------------------------------------------------------------
# CPU utilisation tests (Bug 1.1 regression)
# ---------------------------------------------------------------------------


class TestCpuUtilisation:
    def test_cpu_util_with_valid_limit(self):
        """Pods with known CPU limits produce correct utilisation fractions."""
        responses = {
            "container_cpu_usage_seconds_total": _success_response([
                _prom_result({"pod": "pod-a", "service": "svc-a", "namespace": "ewat"}, 0.3),
            ]),
            "kube_pod_container_resource_limits": _success_response([
                _prom_result(
                    {
                        "pod": "pod-a",
                        "service": "svc-a",
                        "namespace": "ewat",
                        "resource": "cpu",
                    },
                    1.0,
                ),
            ]),
        }
        collector = _make_collector(responses)
        M_t, services = collector.collect(timestamp=1_000_000.0)

        assert "svc-a" in services
        row = services.index("svc-a")
        assert M_t[row, M_CPU_UTIL] == pytest.approx(0.3, abs=1e-5)

    def test_cpu_util_without_limit_is_nan(self):
        """Pods without CPU limits must NOT inflate utilisation — result is NaN.

        Regression for Bug 1.1: the old code used max(val, 1e-9) as divisor,
        which yielded utilisation ~ 10^9 for pods without limits.
        """
        responses = {
            # Usage present but no matching limit entry
            "container_cpu_usage_seconds_total": _success_response([
                _prom_result({"pod": "pod-no-limit", "service": "svc-b", "namespace": "ewat"}, 0.5),
            ]),
            # cpu_limit query returns empty (pod has no configured limit)
            "kube_pod_container_resource_limits": _success_response([]),
        }
        collector = _make_collector(responses)
        M_t, services = collector.collect(timestamp=1_000_000.0)

        assert "svc-b" in services
        row = services.index("svc-b")
        assert np.isnan(M_t[row, M_CPU_UTIL]), (
            f"Expected NaN for pod without CPU limit, got {M_t[row, M_CPU_UTIL]}"
        )

    def test_cpu_util_zero_limit_is_nan(self):
        """A limit value of 0 must be ignored (not stored), producing NaN."""
        responses = {
            "container_cpu_usage_seconds_total": _success_response([
                _prom_result({"pod": "p1", "service": "svc-c", "namespace": "ewat"}, 0.2),
            ]),
            "kube_pod_container_resource_limits": _success_response([
                # limit=0 is invalid and must not be stored
                _prom_result(
                    {
                        "pod": "p1",
                        "service": "svc-c",
                        "namespace": "ewat",
                        "resource": "cpu",
                    },
                    0.0,
                ),
            ]),
        }
        collector = _make_collector(responses)
        M_t, services = collector.collect(timestamp=1_000_000.0)

        row = services.index("svc-c")
        assert np.isnan(M_t[row, M_CPU_UTIL])

    def test_cpu_util_max_over_pods(self):
        """CPU utilisation is the max over pods, not the mean."""
        responses = {
            "container_cpu_usage_seconds_total": _success_response([
                _prom_result({"pod": "p1", "service": "svc-d", "namespace": "ewat"}, 0.1),
                _prom_result({"pod": "p2", "service": "svc-d", "namespace": "ewat"}, 0.9),
            ]),
            "kube_pod_container_resource_limits": _success_response([
                _prom_result(
                    {
                        "pod": "p1",
                        "service": "svc-d",
                        "namespace": "ewat",
                        "resource": "cpu",
                    },
                    1.0,
                ),
                _prom_result(
                    {
                        "pod": "p2",
                        "service": "svc-d",
                        "namespace": "ewat",
                        "resource": "cpu",
                    },
                    1.0,
                ),
            ]),
        }
        collector = _make_collector(responses)
        M_t, services = collector.collect(timestamp=1_000_000.0)

        row = services.index("svc-d")
        assert M_t[row, M_CPU_UTIL] == pytest.approx(0.9, abs=1e-5)

    def test_ram_util_without_limit_is_nan(self):
        """RAM utilization must be NaN when no memory limit is configured."""
        collector = PrometheusCollector(endpoint="http://fake-prom:9090", namespace="ewat")

        M_t = np.full((1, METRICS_DIM), np.nan, dtype=np.float32)
        raw = {
            "ram_usage": [
                _prom_result(
                    {"pod": "pod-no-limit", "service": "svc-ram", "namespace": "ewat"},
                    256.0,
                ),
            ],
            "ram_limit": [],
        }

        collector._fill_ram(M_t, {"svc-ram": 0}, raw)
        assert np.isnan(M_t[0, 1])


# ---------------------------------------------------------------------------
# Queue depth — gauge vs counter (Bug 1.2 regression)
# ---------------------------------------------------------------------------


class TestQueueDepthQuery:
    def test_queue_depth_query_uses_gauge(self):
        """The queue_depth PromQL query must use the gauge metric, not _total counter.

        Regression for Bug 1.2: envoy_cluster_upstream_rq_pending_total is a
        monotonically increasing counter and must NOT be used for queue depth.
        """
        from telemetry.collectors.prometheus_collector import _QUERIES

        query = _QUERIES["queue_depth"]
        assert "envoy_cluster_upstream_rq_pending_total" not in query, (
            "queue_depth must not use the _total counter — use the gauge instead"
        )
        assert "envoy_cluster_upstream_rq_pending" in query


class TestServiceFallbackFromPod:
    def test_service_label_inferred_from_deployment_pod(self):
        metric = {"namespace": "ewat", "pod": "frontend-proxy-7747cb74bb-kwlv5"}
        assert PrometheusCollector._service_label(metric) == "frontend-proxy"

    def test_collect_discovers_service_without_service_label(self):
        responses = {
            "container_cpu_usage_seconds_total": _success_response([
                _prom_result({"pod": "checkout-f87b6c457-4fbgw", "namespace": "ewat"}, 0.2),
            ]),
            "kube_pod_container_resource_limits": _success_response([
                _prom_result(
                    {"pod": "checkout-f87b6c457-4fbgw", "namespace": "ewat", "resource": "cpu"},
                    1.0,
                ),
            ]),
        }
        collector = _make_collector(responses)
        M_t, services = collector.collect(timestamp=1_000_000.0)

        assert "checkout" in services
        row = services.index("checkout")
        assert M_t[row, M_CPU_UTIL] == pytest.approx(0.2, abs=1e-5)


# ---------------------------------------------------------------------------
# JSON parse error handling (Bug 1.5 regression)
# ---------------------------------------------------------------------------


class TestJsonParseError:
    def test_json_error_returns_empty_not_crash(self):
        """Prometheus returning an HTML error page must not raise JSONDecodeError.

        Regression for Bug 1.5: unprotected resp.json() crashes on HTML responses.
        """
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=ValueError("No JSON object could be decoded"))

        collector = PrometheusCollector(endpoint="http://fake-prom:9090")
        collector._session = MagicMock()
        collector._session.get = MagicMock(return_value=mock_resp)

        # Must not raise; should return NaN matrix instead
        M_t, services = collector.collect(timestamp=1_000_000.0)

        # All features should be NaN (no data parsed)
        assert M_t.size == 0 or np.all(np.isnan(M_t))

    def test_output_shape_always_valid(self):
        """Even when all queries fail, the output shape is (0, METRICS_DIM) or valid."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=ValueError("bad json"))

        collector = PrometheusCollector(
            endpoint="http://fake-prom:9090",
            services=["svc-x"],
        )
        collector._session = MagicMock()
        collector._session.get = MagicMock(return_value=mock_resp)

        M_t, services = collector.collect(timestamp=1_000_000.0)

        assert M_t.ndim == 2
        assert M_t.shape[1] == METRICS_DIM


class TestLatencyFallbackUnits:
    def test_fallback_latency_bucket_converts_ms_to_s(self):
        """OTel fallback uses otel_http_server_duration_milliseconds — buckets in ms.

        Both the primary Istio query and the OTel fallback use milliseconds.
        _fill_latency must divide by 1000 unconditionally.
        Regression: old code assumed OTel fallback buckets were in seconds.
        """
        collector = PrometheusCollector(endpoint="http://fake-prom:9090", namespace="ewat")
        # Simulate fallback query result: le values are in milliseconds
        # le=100ms, le=200ms → P99 should be ~0.1–0.2 s after /1000 conversion
        M_t = np.full((1, METRICS_DIM), np.nan, dtype=np.float32)
        raw = {
            "http_request_duration_bucket": [
                _prom_result(
                    {"k8s_pod_name": "p1", "service_name": "svc-lat",
                     "k8s_namespace_name": "ewat", "le": "100"},
                    10.0,
                ),
                _prom_result(
                    {"k8s_pod_name": "p1", "service_name": "svc-lat",
                     "k8s_namespace_name": "ewat", "le": "200"},
                    20.0,
                ),
            ]
        }

        collector._fill_latency(M_t, {"svc-lat": 0}, raw)

        # le=100ms → 0.1s after /1000; P99 ~ 0.1–0.2s
        assert 0.05 < M_t[0, 2] < 0.5, f"P99 should be ~0.1–0.2s, got {M_t[0, 2]}"

    def test_otel_fallback_uses_service_name_label(self):
        """_service_label must extract service from 'service_name' OTel attribute."""
        from telemetry.collectors.prometheus_collector import PrometheusCollector
        metric = {"service_name": "cart", "k8s_pod_name": "cart-66785b767c-jmxdv",
                  "k8s_namespace_name": "ewat"}
        result = PrometheusCollector._service_label(metric)
        assert result == "cart"

    def test_otel_fallback_uses_k8s_pod_name_label(self):
        """_pod_value must extract pod name from 'k8s_pod_name' OTel attribute."""
        collector = PrometheusCollector(endpoint="http://fake-prom:9090")
        item = {
            "metric": {"k8s_pod_name": "cart-66785b767c-jmxdv", "k8s_namespace_name": "ewat"},
            "value": [1234567890.0, "42.0"],
        }
        pod, val = collector._pod_value(item)
        assert pod == "cart-66785b767c-jmxdv"
        assert val == pytest.approx(42.0)

    def test_grpc_latency_bucket_merged_into_p99(self):
        """gRPC histogram buckets (grpc_request_duration_bucket) contribute to P99.

        A service with only gRPC traffic (no HTTP) must have non-NaN latency.
        Buckets use milliseconds → _fill_latency must divide by 1000.
        """
        collector = PrometheusCollector(endpoint="http://fake-prom:9090", namespace="ewat")
        M_t = np.full((1, METRICS_DIM), np.nan, dtype=np.float32)
        raw = {
            "http_request_duration_bucket": [],  # no HTTP traffic
            "grpc_request_duration_bucket": [
                _prom_result(
                    {"k8s_pod_name": "p1", "service_name": "svc-grpc",
                     "k8s_namespace_name": "ewat", "le": "50"},
                    5.0,
                ),
                _prom_result(
                    {"k8s_pod_name": "p1", "service_name": "svc-grpc",
                     "k8s_namespace_name": "ewat", "le": "100"},
                    10.0,
                ),
            ],
        }

        collector._fill_latency(M_t, {"svc-grpc": 0}, raw)

        # le=50ms → 0.05s; le=100ms → 0.1s; P99 should be within that range
        assert not np.isnan(M_t[0, 2]), "gRPC service must have non-NaN latency"
        assert 0.01 < M_t[0, 2] < 0.2, f"P99 should be ~0.05–0.1s, got {M_t[0, 2]}"

    def test_grpc_and_http_latency_merged_per_service(self):
        """When a service has both HTTP and gRPC traffic, both buckets are merged."""
        collector = PrometheusCollector(endpoint="http://fake-prom:9090", namespace="ewat")
        M_t = np.full((1, METRICS_DIM), np.nan, dtype=np.float32)
        raw = {
            "http_request_duration_bucket": [
                _prom_result(
                    {"k8s_pod_name": "p1", "service_name": "svc-mixed",
                     "k8s_namespace_name": "ewat", "le": "100"},
                    5.0,
                ),
            ],
            "grpc_request_duration_bucket": [
                _prom_result(
                    {"k8s_pod_name": "p1", "service_name": "svc-mixed",
                     "k8s_namespace_name": "ewat", "le": "500"},
                    15.0,
                ),
            ],
        }

        collector._fill_latency(M_t, {"svc-mixed": 0}, raw)

        # Both sources contribute; result should be non-NaN
        assert not np.isnan(M_t[0, 2]), "Mixed HTTP+gRPC service must have non-NaN latency"
