"""Canonical Prometheus range-query definitions used by Phase 1 (recording)
and Phase 2 (feature building).

This module is the single source of truth for the PromQL templates that
feed M(t). It is deliberately dependency-free (pure strings) so that
both the online recorder and the offline feature builder can share it
without pulling in ``requests`` or ``numpy``.

Each entry describes:
- ``name``: logical feature key (e.g. ``cpu_usage``).
- ``primary``: the preferred PromQL template (Istio/kubelet labels).
- ``fallback``: an alternative template targeting OTel SDK labels when the
  primary series is empty.

All templates accept two positional substitutions:
- ``{namespace}`` — Kubernetes namespace filter.
- ``{window}`` — PromQL rate/irate window, e.g. ``"2m"``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromQuerySpec:
    """A single logical Prometheus query with a primary + fallback template.

    Both templates return the same *logical* feature. The fallback is tried
    only when the primary produces an empty result set.
    """

    name: str
    primary: str
    fallback: str | None = None


QUERIES: list[PromQuerySpec] = [
    PromQuerySpec(
        name="cpu_usage",
        primary=(
            "sum by (pod, service, namespace) ("
            "  rate(container_cpu_usage_seconds_total"
            "{{namespace='{namespace}', container!=''}}[{window}])"
            ")"
        ),
    ),
    PromQuerySpec(
        name="cpu_limit",
        primary=(
            "sum by (pod, service, namespace) ("
            "  kube_pod_container_resource_limits"
            "{{namespace='{namespace}', resource='cpu', container!=''}}"
            ")"
        ),
    ),
    PromQuerySpec(
        name="ram_usage",
        primary=(
            "sum by (pod, service, namespace) ("
            "  container_memory_working_set_bytes"
            "{{namespace='{namespace}', container!=''}}"
            ")"
        ),
    ),
    PromQuerySpec(
        name="ram_limit",
        primary=(
            "sum by (pod, service, namespace) ("
            "  kube_pod_container_resource_limits"
            "{{namespace='{namespace}', resource='memory', container!=''}}"
            ")"
        ),
    ),
    PromQuerySpec(
        name="http_request_duration_bucket",
        primary=(
            "sum by (pod, service, namespace, le) ("
            "  rate(istio_request_duration_milliseconds_bucket"
            "{{namespace='{namespace}', reporter='destination'}}[{window}])"
            ")"
        ),
        fallback=(
            "sum by (k8s_pod_name, service_name, k8s_namespace_name, le) ("
            "  rate(otel_http_server_duration_milliseconds_bucket"
            "{{k8s_namespace_name='{namespace}'}}[{window}])"
            ")"
        ),
    ),
    PromQuerySpec(
        name="http_requests_total",
        primary=(
            "sum by (pod, service, namespace) ("
            "  rate(istio_requests_total"
            "{{namespace='{namespace}', reporter='destination'}}[{window}])"
            ")"
        ),
        fallback=(
            "sum by (k8s_pod_name, service_name, k8s_namespace_name) ("
            "  rate(otel_http_server_duration_milliseconds_count"
            "{{k8s_namespace_name='{namespace}'}}[{window}])"
            ")"
        ),
    ),
    PromQuerySpec(
        name="http_requests_errors",
        primary=(
            "sum by (pod, service, namespace) ("
            "  rate(istio_requests_total"
            "{{namespace='{namespace}', reporter='destination',"
            "   response_code=~'[45][0-9][0-9]'}}[{window}])"
            ")"
        ),
        fallback=(
            "sum by (k8s_pod_name, service_name, k8s_namespace_name) ("
            "  rate(otel_http_server_duration_milliseconds_count"
            "{{k8s_namespace_name='{namespace}',"
            "   http_status_code=~'[45][0-9][0-9]'}}[{window}])"
            ")"
        ),
    ),
    PromQuerySpec(
        name="grpc_request_duration_bucket",
        primary=(
            "sum by (k8s_pod_name, service_name, k8s_namespace_name, le) ("
            "  rate(otel_rpc_server_duration_milliseconds_bucket"
            "{{k8s_namespace_name='{namespace}'}}[{window}])"
            ")"
        ),
        fallback=(
            "sum by (k8s_pod_name, service_name, namespace, le) ("
            "  rate(rpc_server_duration_milliseconds_bucket"
            "{{namespace='{namespace}'}}[{window}])"
            ")"
        ),
    ),
    PromQuerySpec(
        name="net_transmit_bytes",
        primary=(
            "sum by (pod, service, namespace) ("
            "  rate(container_network_transmit_bytes_total"
            "{{namespace='{namespace}'}}[{window}])"
            ")"
        ),
    ),
    PromQuerySpec(
        name="disk_iops",
        primary=(
            "sum by (pod, service, namespace) ("
            "  rate(container_fs_reads_total{{namespace='{namespace}'}}[{window}])"
            "  + rate(container_fs_writes_total{{namespace='{namespace}'}}[{window}])"
            ")"
        ),
    ),
    PromQuerySpec(
        name="queue_depth",
        primary=(
            "sum by (pod, service, namespace) ("
            "  envoy_cluster_upstream_rq_pending{{namespace='{namespace}'}}"
            ")"
        ),
        fallback=(
            "sum by (pod, namespace) ("
            "  rate(kube_pod_container_status_restarts_total"
            "{{namespace='{namespace}'}}[{window}])"
            ")"
        ),
    ),
]


def render(spec: PromQuerySpec, namespace: str, window: str) -> tuple[str, str | None]:
    """Return the (primary, fallback) PromQL strings for one query spec."""
    primary = spec.primary.format(namespace=namespace, window=window)
    fallback = (
        spec.fallback.format(namespace=namespace, window=window) if spec.fallback else None
    )
    return primary, fallback
