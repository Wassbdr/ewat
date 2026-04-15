"""Signal builder — assembles S(t) ∈ ℝ^{N×17} = [M(t) | T(t) | L(t)].

Coordinates the three collectors, aligns their service indices, and
concatenates the feature matrices into the unified signal tensor consumed
by the EWAT pipeline (drift detector, encoder, etc.).

Usage
-----
>>> from omegaconf import OmegaConf
>>> cfg = OmegaConf.load("configs/default.yaml")
>>> builder = SignalBuilder.from_config(cfg)
>>> S_t, services, ts = builder.build()
>>> assert S_t.shape == (len(services), 17)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from telemetry.feature_names import LOGS_SLICE, METRICS_SLICE, SIGNAL_DIM, TRACES_SLICE
from telemetry.collectors.log_collector import LogCollector
from telemetry.collectors.prometheus_collector import PrometheusCollector
from telemetry.collectors.trace_collector import TraceCollector

logger = logging.getLogger(__name__)


@dataclass
class SignalSnapshot:
    """A single time-stamped observation of S(t).

    Attributes
    ----------
    S:
        Float32 array of shape (N, 17). NaN entries indicate missing data.
    services:
        List of N service names aligned with rows of S.
    timestamp:
        Unix timestamp (seconds) at which the snapshot was collected.
    n_nan:
        Total number of NaN cells in S (diagnostic).
    """

    S: npt.NDArray[np.float32]
    services: list[str]
    timestamp: float
    n_nan: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_nan = int(np.isnan(self.S).sum())

    @property
    def M(self) -> npt.NDArray[np.float32]:
        """Metrics sub-matrix M(t) ∈ ℝ^{N×7}."""
        return self.S[:, METRICS_SLICE]

    @property
    def T(self) -> npt.NDArray[np.float32]:
        """Trace sub-matrix T(t) ∈ ℝ^{N×6}."""
        return self.S[:, TRACES_SLICE]

    @property
    def L(self) -> npt.NDArray[np.float32]:
        """Log sub-matrix L(t) ∈ ℝ^{N×4}."""
        return self.S[:, LOGS_SLICE]


class SignalBuilder:
    """Coordinates the three collectors and assembles S(t).

    Parameters
    ----------
    prometheus:
        Fitted :class:`PrometheusCollector` instance (or ``None`` to skip
        M(t) — all 7 metrics columns will be NaN).
    traces:
        Fitted :class:`TraceCollector` instance (or ``None`` to skip T(t)).
    logs:
        Fitted :class:`LogCollector` instance (or ``None`` to skip L(t)).
    services:
        Canonical ordered list of N services. When provided the matrix
        dimension is fixed regardless of what each collector discovers. When
        ``None`` the union of all discovered services is used.
    """

    def __init__(
        self,
        prometheus: PrometheusCollector | None = None,
        traces: TraceCollector | None = None,
        logs: LogCollector | None = None,
        services: list[str] | None = None,
    ) -> None:
        self._prometheus = prometheus
        self._traces = traces
        self._logs = logs
        self._services = services

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Any) -> "SignalBuilder":
        """Construct a SignalBuilder from a Hydra/OmegaConf config object.

        Parameters
        ----------
        cfg:
            Root config node loaded from ``configs/default.yaml``.
            Expected keys under ``telemetry``:
                - prometheus.endpoint
                - jaeger.endpoint       (for traces via Jaeger HTTP API)
                - loki.endpoint         (for logs via Loki HTTP API)
            Expected keys under ``cluster``:
                - namespace

        Returns
        -------
        SignalBuilder
        """
        ns: str = cfg.cluster.namespace

        prom_endpoint: str = cfg.telemetry.prometheus.endpoint

        # Prometheus collector (always created; some queries may return no data
        # if Istio/Envoy metrics are not present)
        prometheus = PrometheusCollector(
            endpoint=prom_endpoint,
            namespace=ns,
        )

        # Trace collector — requires a Jaeger endpoint
        traces: TraceCollector | None = None
        jaeger_cfg = cfg.telemetry.get("jaeger", {})
        jaeger_url: str = jaeger_cfg.get("endpoint", "") if jaeger_cfg else ""
        if jaeger_url:
            from telemetry.collectors.trace_collector import JaegerBackend

            backend = JaegerBackend(endpoint=jaeger_url, namespace=ns)
            traces = TraceCollector(backend=backend)
        else:
            logger.warning(
                "No jaeger.endpoint configured; T(t) will be all NaN. "
                "Set telemetry.jaeger.endpoint in configs/default.yaml."
            )

        # Log collector — requires a Loki endpoint
        logs: LogCollector | None = None
        loki_cfg = cfg.telemetry.get("loki", {})
        loki_url: str = loki_cfg.get("endpoint", "") if loki_cfg else ""
        if loki_url:
            from telemetry.collectors.log_collector import LokiBackend

            log_backend = LokiBackend(endpoint=loki_url, namespace=ns)
            logs = LogCollector(backend=log_backend)
        else:
            logger.warning(
                "No loki.endpoint configured; L(t) will be partial (lexical "
                "entropy and level rates unavailable). "
                "Set telemetry.loki.endpoint in configs/default.yaml."
            )

        return cls(prometheus=prometheus, traces=traces, logs=logs)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(self, timestamp: float | None = None) -> SignalSnapshot:
        """Collect all modalities and return S(t).

        Parameters
        ----------
        timestamp:
            Target Unix timestamp (seconds). Defaults to now.

        Returns
        -------
        SignalSnapshot
            Contains S ∈ ℝ^{N×17}, services list, and diagnostics.
        """
        ts = timestamp or time.time()

        # Step 1: discover services (use Prometheus as the primary discoverer)
        services, svc_idx = self._get_service_index(ts)
        n = len(services)

        S_t = np.full((n, SIGNAL_DIM), float("nan"), dtype=np.float32)

        # Step 2: fill M(t)
        if self._prometheus is not None:
            try:
                M_t, _ = self._prometheus.collect(timestamp=ts, service_index=svc_idx)
                S_t[:, METRICS_SLICE] = M_t
            except Exception:
                logger.exception("PrometheusCollector.collect() failed")

        # Step 3: fill T(t)
        if self._traces is not None:
            try:
                T_t, _ = self._traces.collect(timestamp=ts, service_index=svc_idx)
                S_t[:, TRACES_SLICE] = T_t
            except Exception:
                logger.exception("TraceCollector.collect() failed")

        # Step 4: fill L(t)
        if self._logs is not None:
            try:
                L_t, _ = self._logs.collect(timestamp=ts, service_index=svc_idx)
                S_t[:, LOGS_SLICE] = L_t
            except Exception:
                logger.exception("LogCollector.collect() failed")

        snapshot = SignalSnapshot(S=S_t, services=services, timestamp=ts)

        if snapshot.n_nan > 0:
            nan_frac = snapshot.n_nan / S_t.size
            logger.debug(
                "SignalBuilder: S(t) has %d NaN cells (%.1f%%) at t=%.0f",
                snapshot.n_nan,
                100.0 * nan_frac,
                ts,
            )

        return snapshot

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_service_index(self, ts: float) -> tuple[list[str], dict[str, int]]:
        """Discover the canonical service list.

        Priority: explicit list → Prometheus discovery → empty fallback.
        """
        if self._services is not None:
            services = sorted(self._services)
            return services, {s: i for i, s in enumerate(services)}

        if self._prometheus is not None:
            try:
                _M, services = self._prometheus.collect(timestamp=ts)
                svc_idx = {s: i for i, s in enumerate(services)}
                return services, svc_idx
            except Exception:
                logger.exception("Service discovery via Prometheus failed")

        return [], {}
