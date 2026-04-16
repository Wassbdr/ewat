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

from telemetry.collectors.log_collector import LogCollector
from telemetry.collectors.prometheus_collector import PrometheusCollector
from telemetry.collectors.trace_collector import TraceCollector
from telemetry.feature_names import LOGS_SLICE, METRICS_SLICE, SIGNAL_DIM, TRACES_SLICE

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
    log_records:
        Raw log lines used to compute L(t) for this snapshot window.
    """

    S: npt.NDArray[np.float32]
    services: list[str]
    timestamp: float
    log_records: list[Any] = field(default_factory=list)
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
    def from_config(
        cls,
        cfg: Any,
        *,
        semantic_enabled: bool = True,
        traces_enabled: bool = True,
        services: list[str] | None = None,
    ) -> SignalBuilder:
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
        semantic_enabled:
            Enable SentenceBERT-based semantic log features.
        traces_enabled:
            Enable Jaeger trace collection. When ``False``, T(t) remains NaN.

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
            services=services,
        )

        traces: TraceCollector | None = None
        if traces_enabled:
            # Trace collector — requires a Jaeger endpoint
            jaeger_cfg = cfg.telemetry.get("jaeger", {})
            jaeger_url: str = jaeger_cfg.get("endpoint", "") if jaeger_cfg else ""
            if jaeger_url:
                from telemetry.collectors.trace_collector import JaegerBackend

                jaeger_collection_cfg = jaeger_cfg.get("collection", {}) if jaeger_cfg else {}
                backend = JaegerBackend(
                    endpoint=jaeger_url,
                    namespace=ns,
                    timeout=float(jaeger_collection_cfg.get("request_timeout_s", 15.0)),
                    limit=int(jaeger_collection_cfg.get("limit_per_service", 20)),
                    fetch_total_timeout_s=float(
                        jaeger_collection_cfg.get("fetch_total_timeout_s", 10.0)
                    ),
                    max_parallel=int(jaeger_collection_cfg.get("max_parallel", 8)),
                )
                traces = TraceCollector(
                    backend=backend,
                    window_s=float(jaeger_collection_cfg.get("trace_window_s", 120.0)),
                    services=services,
                    cache_ttl_s=float(jaeger_collection_cfg.get("span_cache_ttl_s", 30.0)),
                )
            else:
                logger.warning(
                    "No jaeger.endpoint configured; T(t) will be all NaN. "
                    "Set telemetry.jaeger.endpoint in configs/default.yaml."
                )
        else:
            logger.info("Trace collection disabled; T(t) will be all NaN.")

        # Log collector — requires a Loki endpoint
        logs: LogCollector | None = None
        loki_cfg = cfg.telemetry.get("loki", {})
        loki_url: str = loki_cfg.get("endpoint", "") if loki_cfg else ""
        if loki_url:
            from telemetry.collectors.log_collector import LokiBackend

            log_backend = LokiBackend(endpoint=loki_url, namespace=ns)
            logs = LogCollector(
                backend=log_backend,
                semantic_enabled=semantic_enabled,
                services=services,
            )
        else:
            logger.warning(
                "No loki.endpoint configured; L(t) will be partial (lexical "
                "entropy and level rates unavailable). "
                "Set telemetry.loki.endpoint in configs/default.yaml."
            )

        return cls(prometheus=prometheus, traces=traces, logs=logs, services=services)

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

        # Step 1 + 2: single Prometheus call for both service discovery and M(t).
        # Previously _get_service_index() called collect() for discovery and then
        # build() called it again to fill M(t) — two round-trips that doubled the
        # fallback warnings on every 15-second tick.
        M_t_raw: npt.NDArray[np.float32] | None = None
        raw_services: list[str] = []

        if self._prometheus is not None:
            try:
                if self._services is not None:
                    canonical_idx = {s: i for i, s in enumerate(self._services)}
                    M_t_raw, raw_services = self._prometheus.collect(
                        timestamp=ts,
                        service_index=canonical_idx,
                    )
                else:
                    M_t_raw, raw_services = self._prometheus.collect(timestamp=ts)
            except Exception:
                logger.exception("PrometheusCollector.collect() failed")

        services, svc_idx = self._build_service_index(raw_services)
        n = len(services)
        S_t = np.full((n, SIGNAL_DIM), float("nan"), dtype=np.float32)

        # Align the already-fetched M(t) rows to the canonical service order.
        if M_t_raw is not None and n > 0:
            disc_idx = {s: i for i, s in enumerate(raw_services)}
            for svc, row in svc_idx.items():
                src = disc_idx.get(svc)
                if src is not None:
                    S_t[row, METRICS_SLICE] = M_t_raw[src]

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
                L_t, _, log_records = self._logs.collect_with_records(
                    timestamp=ts,
                    service_index=svc_idx,
                )
                S_t[:, LOGS_SLICE] = L_t
            except Exception:
                logger.exception("LogCollector.collect() failed")
                log_records = []
        else:
            log_records = []

        snapshot = SignalSnapshot(S=S_t, services=services, timestamp=ts, log_records=log_records)

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

    def _build_service_index(
        self, discovered: list[str]
    ) -> tuple[list[str], dict[str, int]]:
        """Merge explicitly configured services with Prometheus-discovered ones.

        Priority: explicit list (if set) unioned with discovered; else discovered only.
        """
        if self._services is not None:
            services = list(self._services)
        else:
            services = sorted(set(discovered))
        return services, {s: i for i, s in enumerate(services)}
