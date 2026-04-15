"""Unit tests for telemetry.signal_builder.

Uses stub collectors that return controlled data so no live cluster is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from telemetry.feature_names import (
    LOGS_SLICE,
    METRICS_SLICE,
    SIGNAL_DIM,
    TRACES_SLICE,
    FEATURE_NAMES,
)
from telemetry.signal_builder import SignalBuilder, SignalSnapshot


# ---------------------------------------------------------------------------
# Stub collectors
# ---------------------------------------------------------------------------


class _StubPrometheus:
    """Returns a fixed M(t) for two services."""

    def collect(self, timestamp=None, service_index=None):
        services = ["svc-a", "svc-b"]
        svc_idx = service_index or {s: i for i, s in enumerate(services)}
        n = len(svc_idx)
        M = np.zeros((n, 7), dtype=np.float32)
        M[0, :] = [0.5, 0.6, 0.01, 0.02, 1e6, 100.0, 5.0]
        M[1, :] = [0.8, 0.9, 0.05, 0.10, 2e6, 200.0, 10.0]
        return M, sorted(svc_idx, key=lambda s: svc_idx[s])


class _StubTraces:
    def collect(self, timestamp=None, service_index=None):
        services = ["svc-a", "svc-b"]
        svc_idx = service_index or {s: i for i, s in enumerate(services)}
        n = len(svc_idx)
        T = np.full((n, 6), 0.1, dtype=np.float32)
        return T, sorted(svc_idx, key=lambda s: svc_idx[s])


class _StubLogs:
    def collect(self, timestamp=None, service_index=None):
        services = ["svc-a", "svc-b"]
        svc_idx = service_index or {s: i for i, s in enumerate(services)}
        n = len(svc_idx)
        L = np.full((n, 4), 0.2, dtype=np.float32)
        return L, sorted(svc_idx, key=lambda s: svc_idx[s])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSignalSnapshot:
    def test_shape(self):
        S = np.ones((5, SIGNAL_DIM), dtype=np.float32)
        snap = SignalSnapshot(S=S, services=["s"] * 5, timestamp=0.0)
        assert snap.S.shape == (5, SIGNAL_DIM)
        assert snap.M.shape == (5, 7)
        assert snap.T.shape == (5, 6)
        assert snap.L.shape == (5, 4)

    def test_n_nan_counts_correctly(self):
        S = np.ones((3, SIGNAL_DIM), dtype=np.float32)
        S[0, 0] = float("nan")
        S[1, 5] = float("nan")
        snap = SignalSnapshot(S=S, services=["a", "b", "c"], timestamp=0.0)
        assert snap.n_nan == 2


class TestSignalBuilder:
    def _builder_with_stubs(self):
        b = SignalBuilder(
            prometheus=_StubPrometheus(),
            traces=_StubTraces(),
            logs=_StubLogs(),
            services=["svc-a", "svc-b"],
        )
        return b

    def test_build_returns_snapshot(self):
        snap = self._builder_with_stubs().build()
        assert isinstance(snap, SignalSnapshot)

    def test_signal_shape(self):
        snap = self._builder_with_stubs().build()
        assert snap.S.shape == (2, SIGNAL_DIM)

    def test_services_ordered(self):
        snap = self._builder_with_stubs().build()
        assert snap.services == ["svc-a", "svc-b"]

    def test_metrics_filled(self):
        snap = self._builder_with_stubs().build()
        assert not np.any(np.isnan(snap.M))

    def test_traces_filled(self):
        snap = self._builder_with_stubs().build()
        assert not np.any(np.isnan(snap.T))

    def test_logs_filled(self):
        snap = self._builder_with_stubs().build()
        assert not np.any(np.isnan(snap.L))

    def test_no_collectors_all_nan(self):
        builder = SignalBuilder(services=["svc-x"])
        snap = builder.build()
        assert snap.S.shape == (1, SIGNAL_DIM)
        assert np.all(np.isnan(snap.S))

    def test_only_prometheus(self):
        builder = SignalBuilder(prometheus=_StubPrometheus(), services=["svc-a", "svc-b"])
        snap = builder.build()
        assert not np.any(np.isnan(snap.M))
        assert np.all(np.isnan(snap.T))
        assert np.all(np.isnan(snap.L))

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == SIGNAL_DIM

    def test_slice_coverage(self):
        # Slices must cover all 17 columns without overlap
        all_cols = list(range(METRICS_SLICE.start, METRICS_SLICE.stop))
        all_cols += list(range(TRACES_SLICE.start, TRACES_SLICE.stop))
        all_cols += list(range(LOGS_SLICE.start, LOGS_SLICE.stop))
        assert sorted(all_cols) == list(range(SIGNAL_DIM))


class TestSignalBuilderCollectorFailure:
    """The builder must not crash when a collector raises an exception."""

    def test_prometheus_failure_graceful(self):
        class _BrokenPrometheus:
            def collect(self, **kwargs):
                raise RuntimeError("network unreachable")

        builder = SignalBuilder(
            prometheus=_BrokenPrometheus(),
            services=["svc-a"],
        )
        snap = builder.build()
        # M should remain NaN, snapshot should still be returned
        assert snap.S.shape == (1, SIGNAL_DIM)
        assert np.all(np.isnan(snap.M))
