"""Unit tests for trace and log collectors using in-memory stubs."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from telemetry.collectors.log_collector import (
    LogCollector,
    LogQueryBackend,
    LogRecord,
    classify_level,
)
from telemetry.collectors.trace_collector import (
    JaegerBackend,
    Span,
    SpanQueryBackend,
    TraceCollector,
    _compute_trace_structures,
)
from telemetry.feature_names import LOGS_DIM, TRACES_DIM

# ---------------------------------------------------------------------------
# In-memory backends
# ---------------------------------------------------------------------------


class InMemorySpanBackend(SpanQueryBackend):
    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    def fetch_spans(self, start_unix_s: float, end_unix_s: float) -> list[Span]:
        return self._spans


class InMemoryLogBackend(LogQueryBackend):
    def __init__(self, records: list[LogRecord]) -> None:
        self._records = records

    def fetch_logs(self, start_unix_s: float, end_unix_s: float) -> list[LogRecord]:
        return self._records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_span(
    trace_id: str,
    span_id: str,
    parent: str,
    svc: str,
    duration_s: float = 0.01,
    status: str = "OK",
    is_retry: bool = False,
    start_time_s: float | None = None,
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        service_name=svc,
        duration_s=duration_s,
        status_code=status,
        is_retry=is_retry,
        start_time_s=start_time_s,
    )


# ---------------------------------------------------------------------------
# Trace structure tests
# ---------------------------------------------------------------------------


class TestComputeTraceStructures:
    def test_single_span_depth_one(self):
        spans = [make_span("t1", "s1", "", "svc-a")]
        structs = _compute_trace_structures(spans)
        assert structs["t1"]["max_depth"] == 1

    def test_chain_depth(self):
        # t1 → s1 → s2 → s3 (chain of 3)
        spans = [
            make_span("t1", "s1", "", "svc-a"),
            make_span("t1", "s2", "s1", "svc-a"),
            make_span("t1", "s3", "s2", "svc-a"),
        ]
        structs = _compute_trace_structures(spans)
        assert structs["t1"]["max_depth"] == 3

    def test_fan_out(self):
        # s1 has 2 children → avg_fan_out = (2 + 0 + 0) / 3
        spans = [
            make_span("t1", "s1", "", "svc-a"),
            make_span("t1", "s2", "s1", "svc-b"),
            make_span("t1", "s3", "s1", "svc-c"),
        ]
        structs = _compute_trace_structures(spans)
        avg_fo = structs["t1"]["avg_fan_out"]
        assert avg_fo == pytest.approx(2 / 3, abs=1e-5)


# ---------------------------------------------------------------------------
# TraceCollector tests
# ---------------------------------------------------------------------------


class TestTraceCollector:
    def _make_spans(self) -> list[Span]:
        return [
            make_span("t1", "s1", "", "svc-a", duration_s=0.1),
            make_span("t1", "s2", "s1", "svc-a", duration_s=0.2, status="ERROR"),
            make_span("t2", "s3", "", "svc-b", duration_s=0.05, is_retry=True),
            make_span("t2", "s4", "s3", "svc-b", duration_s=0.03),
        ]

    def test_output_shape(self):
        collector = TraceCollector(backend=InMemorySpanBackend(self._make_spans()))
        T_t, services = collector.collect()
        n = len(services)
        assert T_t.shape == (n, TRACES_DIM)

    def test_services_discovered(self):
        collector = TraceCollector(backend=InMemorySpanBackend(self._make_spans()))
        T_t, services = collector.collect()
        assert "svc-a" in services
        assert "svc-b" in services

    def test_abnormal_rate_svc_a(self):
        # svc-a has 1 ERROR out of 2 spans
        svc_idx = {"svc-a": 0, "svc-b": 1}
        collector = TraceCollector(backend=InMemorySpanBackend(self._make_spans()))
        T_t, services = collector.collect(service_index=svc_idx)
        row = svc_idx["svc-a"]
        assert T_t[row, 1] == pytest.approx(0.5)  # T_ABNORMAL_RATE

    def test_retry_rate_svc_b(self):
        svc_idx = {"svc-a": 0, "svc-b": 1}
        collector = TraceCollector(backend=InMemorySpanBackend(self._make_spans()))
        T_t, _ = collector.collect(service_index=svc_idx)
        row = svc_idx["svc-b"]
        assert T_t[row, 4] == pytest.approx(0.5)  # T_RETRY_RATE (1 retry / 2 spans)

    def test_empty_spans_all_nan(self):
        collector = TraceCollector(
            backend=InMemorySpanBackend([]), services=["svc-x"]
        )
        T_t, services = collector.collect()
        assert np.all(np.isnan(T_t))

    def test_service_index_respected(self):
        svc_idx = {"svc-a": 0, "svc-b": 1}
        collector = TraceCollector(backend=InMemorySpanBackend(self._make_spans()))
        T_t, services = collector.collect(service_index=svc_idx)
        assert T_t.shape[0] == 2

    def test_service_allowlist_propagated_to_backend(self):
        class _AllowlistBackend(InMemorySpanBackend):
            def __init__(self, spans: list[Span]) -> None:
                super().__init__(spans)
                self._service_allowlist: set[str] | None = None

        backend = _AllowlistBackend(self._make_spans())
        collector = TraceCollector(backend=backend)

        svc_idx = {"svc-a": 0, "svc-b": 1}
        collector.collect(service_index=svc_idx)

        assert backend._service_allowlist == {"svc-a", "svc-b"}


# ---------------------------------------------------------------------------
# classify_level tests
# ---------------------------------------------------------------------------


class TestClassifyLevel:
    def test_prefilled_error(self):
        rec = LogRecord("svc", "pod", "...", level="ERROR")
        assert classify_level(rec) == "ERROR"

    def test_prefilled_warn(self):
        rec = LogRecord("svc", "pod", "...", level="WARNING")
        assert classify_level(rec) == "WARN"

    def test_body_fallback_error(self):
        rec = LogRecord("svc", "pod", "2024-01-01 ERROR connection refused")
        assert classify_level(rec) == "ERROR"

    def test_body_fallback_warn(self):
        rec = LogRecord("svc", "pod", "WARN memory high")
        assert classify_level(rec) == "WARN"

    def test_info_default(self):
        rec = LogRecord("svc", "pod", "request processed successfully")
        assert classify_level(rec) == "INFO"


# ---------------------------------------------------------------------------
# LogCollector tests
# ---------------------------------------------------------------------------


class TestLogCollector:
    def _make_records(self) -> list[LogRecord]:
        return [
            LogRecord("svc-a", "pod-1", "INFO service started"),
            LogRecord("svc-a", "pod-1", "ERROR database timeout"),
            LogRecord("svc-a", "pod-1", "WARN high memory"),
            LogRecord("svc-b", "pod-2", "INFO request ok"),
            LogRecord("svc-b", "pod-2", "INFO request ok"),
        ]

    def test_output_shape(self):
        collector = LogCollector(backend=InMemoryLogBackend(self._make_records()))
        L_t, services = collector.collect()
        assert L_t.shape == (len(services), LOGS_DIM)

    def test_error_rate_svc_a(self):
        svc_idx = {"svc-a": 0, "svc-b": 1}
        collector = LogCollector(
            backend=InMemoryLogBackend(self._make_records()),
        )
        L_t, _ = collector.collect(service_index=svc_idx)
        # svc-a: 1 ERROR out of 3 lines
        assert L_t[0, 0] == pytest.approx(1 / 3, abs=1e-5)

    def test_warn_rate_svc_a(self):
        svc_idx = {"svc-a": 0, "svc-b": 1}
        collector = LogCollector(backend=InMemoryLogBackend(self._make_records()))
        L_t, _ = collector.collect(service_index=svc_idx)
        # svc-a: 1 WARN out of 3 lines
        assert L_t[0, 1] == pytest.approx(1 / 3, abs=1e-5)

    def test_lexical_entropy_positive(self):
        svc_idx = {"svc-a": 0, "svc-b": 1}
        collector = LogCollector(backend=InMemoryLogBackend(self._make_records()))
        L_t, _ = collector.collect(service_index=svc_idx)
        # Lexical entropy should be > 0 for diverse logs
        assert L_t[0, 3] > 0.0

    def test_semantic_anomaly_nan_without_centroid(self):
        svc_idx = {"svc-a": 0}
        collector = LogCollector(backend=InMemoryLogBackend(self._make_records()))
        L_t, _ = collector.collect(service_index=svc_idx)
        assert np.isnan(L_t[0, 2])

    def test_empty_records_all_nan(self):
        collector = LogCollector(
            backend=InMemoryLogBackend([]), services=["svc-z"]
        )
        L_t, services = collector.collect()
        assert np.all(np.isnan(L_t))

    def test_semantic_disabled_keeps_nan(self):
        svc_idx = {"svc-a": 0}

        class _DummyScorer:
            centroid = np.ones(3, dtype=np.float32)

            def score(self, _lines):
                return 0.42

        collector = LogCollector(
            backend=InMemoryLogBackend(self._make_records()),
            semantic_scorers={"svc-a": _DummyScorer()},
            semantic_enabled=False,
        )
        L_t, _ = collector.collect(service_index=svc_idx)
        assert np.isnan(L_t[0, 2])

    def test_collect_with_records_returns_raw_records(self):
        records = self._make_records()
        collector = LogCollector(backend=InMemoryLogBackend(records))
        _L_t, _services, raw_records = collector.collect_with_records()
        assert len(raw_records) == len(records)


# ---------------------------------------------------------------------------
# BFS depth correctness (Bug 1.3 regression)
# ---------------------------------------------------------------------------


class TestBfsDepth:
    def test_wide_tree_depth_correct(self):
        """BFS must correctly compute depth=2 for a root with two direct children.

        With DFS (pop()) this would be correct too, but the key invariant is
        that the traversal visits all nodes and reports the true tree height.
        """
        spans = [
            make_span("t1", "root", "", "svc-a"),
            make_span("t1", "c1", "root", "svc-b"),
            make_span("t1", "c2", "root", "svc-c"),
        ]
        structs = _compute_trace_structures(spans)
        assert structs["t1"]["max_depth"] == 2

    def test_deep_left_path_not_missed(self):
        """Depth 4 via leftmost chain must be found even when right branch is shallow.

        Tree:  root → a → b → c  (depth 4)
                    └→ d          (depth 2)
        Both BFS and DFS must report depth 4; ensures no early exit.
        """
        spans = [
            make_span("t1", "root", "", "svc-a"),
            make_span("t1", "a", "root", "svc-a"),
            make_span("t1", "b", "a", "svc-a"),
            make_span("t1", "c", "b", "svc-a"),   # depth 4
            make_span("t1", "d", "root", "svc-a"),  # depth 2
        ]
        structs = _compute_trace_structures(spans)
        assert structs["t1"]["max_depth"] == 4

    def test_start_time_s_flows_through_span(self):
        """Span.start_time_s must be stored and accessible for build_windowed."""
        ts = 1_700_000_000.0
        span = make_span("t1", "s1", "", "svc-a", start_time_s=ts)
        assert span.start_time_s == ts

    def test_span_without_start_time_returns_none(self):
        """Spans without startTime must have start_time_s=None, not 0."""
        span = make_span("t1", "s1", "", "svc-a")
        assert span.start_time_s is None


# ---------------------------------------------------------------------------
# JSON error resilience — Jaeger backend (Bug 1.5 regression)
# ---------------------------------------------------------------------------


class TestJaegerBackendJsonError:
    def _make_backend(self) -> JaegerBackend:
        backend = JaegerBackend.__new__(JaegerBackend)
        backend._endpoint = "http://fake-jaeger:16686"
        backend._namespace = "ewat"
        backend._timeout = 5.0
        backend._limit = 100
        backend._service_allowlist = None
        import requests
        backend._session = requests.Session()
        return backend

    def test_services_json_error_returns_empty(self):
        """JaegerBackend._get_services must return [] on JSON decode error.

        Regression for Bug 1.5: unprotected resp.json() crashed on HTML pages.
        """
        backend = self._make_backend()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=ValueError("bad json"))
        backend._session.get = MagicMock(return_value=mock_resp)

        result = backend._get_services()
        assert result == []

    def test_traces_json_error_skips_service(self):
        """JaegerBackend.fetch_spans must skip a service when JSON parse fails.

        A HTML 502 from the Jaeger query service must not propagate as an
        exception — the service is skipped and an empty list returned.
        """
        backend = self._make_backend()

        services_resp = MagicMock()
        services_resp.raise_for_status = MagicMock()
        services_resp.json = MagicMock(return_value={"data": ["svc-x"]})

        traces_resp = MagicMock()
        traces_resp.raise_for_status = MagicMock()
        traces_resp.json = MagicMock(side_effect=ValueError("bad json"))

        backend._session.get = MagicMock(side_effect=[services_resp, traces_resp])

        spans = backend.fetch_spans(0.0, 100.0)
        assert spans == []


# ---------------------------------------------------------------------------
# JSON error resilience — Loki backend (Bug 1.5 regression)
# ---------------------------------------------------------------------------


class TestLokiBackendJsonError:
    def test_fetch_logs_json_error_returns_empty(self):
        """LokiBackend.fetch_logs must return [] on JSON decode error.

        Regression for Bug 1.5.
        """
        from telemetry.collectors.log_collector import LokiBackend

        backend = LokiBackend.__new__(LokiBackend)
        backend._endpoint = "http://fake-loki:3100"
        backend._namespace = "ewat"
        backend._timeout = 5.0
        backend._limit = 1000
        import requests
        backend._session = requests.Session()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=ValueError("bad json"))
        backend._session.get = MagicMock(return_value=mock_resp)

        records = backend.fetch_logs(0.0, 100.0)
        assert records == []
