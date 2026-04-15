"""Log collector — extracts L(t) ∈ ℝ^{N×4} from OTel log records.

Queries a log backend (Loki HTTP or an in-memory stub) for raw log lines
in the window [t-W, t], then computes per-service log features.

Features (columns 13–16 in S(t), columns 0–3 in L(t)):
    0  log_error_rate     Fraction of ERROR-level log lines
    1  log_warn_rate      Fraction of WARN-level log lines
    2  semantic_anomaly   Mean cosine distance to the normal centroid (SentenceBERT)
    3  lexical_entropy    Shannon entropy of token distribution (bits)

Aggregation (pod → service):
    log_error_rate    → volume_weighted
    log_warn_rate     → volume_weighted
    semantic_anomaly  → median (one scorer per service is the normal usage)
    lexical_entropy   → median

Data sources:
    Primary:  Loki HTTP query API (LogQL)
    Fallback: OTLP log records streamed directly to the collector
              (not yet wired; add LokiBackend or OTLPLogBackend as needed)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from telemetry.feature_names import LOGS_DIM
from telemetry.features.aggregation import aggregate_volume_weighted
from telemetry.features.lexical import lexical_entropy
from telemetry.features.semantic import SemanticAnomalyScorer

# Local column indices within L_t (shape N×4) — NOT global S(t) indices
_L_ERROR_RATE = 0
_L_WARN_RATE = 1
_L_SEMANTIC_ANOMALY = 2
_L_LEXICAL_ENTROPY = 3

logger = logging.getLogger(__name__)

# Level classification patterns
_LEVEL_ERROR = re.compile(
    r"\b(?:ERROR|CRITICAL|FATAL|SEVERE|EMERGENCY)\b", re.IGNORECASE
)
_LEVEL_WARN = re.compile(r"\b(?:WARN(?:ING)?)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LogRecord:
    """Minimal log record.

    Parameters
    ----------
    service_name:
        Kubernetes service emitting this log line.
    pod_name:
        Pod that produced the line (optional; used for volume-weighting).
    body:
        Raw log line text.
    level:
        Pre-parsed severity level string, e.g. ``"ERROR"`` or ``"INFO"``.
        When empty the level is inferred from ``body`` via regex.
    """

    service_name: str
    pod_name: str
    body: str
    level: str = ""


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class LogQueryBackend(ABC):
    """Abstract backend for fetching log records in a time window."""

    @abstractmethod
    def fetch_logs(self, start_unix_s: float, end_unix_s: float) -> list[LogRecord]:
        """Return log records whose timestamp falls in [start_unix_s, end_unix_s].

        Parameters
        ----------
        start_unix_s:
            Window start (Unix epoch, seconds).
        end_unix_s:
            Window end (Unix epoch, seconds).

        Returns
        -------
        list[LogRecord]
        """
        ...


# ---------------------------------------------------------------------------
# Loki HTTP backend
# ---------------------------------------------------------------------------


class LokiBackend(LogQueryBackend):
    """Grafana Loki HTTP query range backend.

    Parameters
    ----------
    endpoint:
        Loki query-frontend base URL, e.g. ``http://loki.monitoring.svc:3100``.
    namespace:
        Kubernetes namespace. Used in the LogQL label selector.
    timeout:
        HTTP request timeout in seconds.
    limit:
        Maximum log lines per query.
    """

    def __init__(
        self,
        endpoint: str,
        namespace: str = "ewat",
        timeout: float = 10.0,
        limit: int = 5000,
    ) -> None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self._endpoint = endpoint.rstrip("/")
        self._namespace = namespace
        self._timeout = min(timeout, 5.0)  # hard cap: fail fast
        self._limit = limit

        _retry = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        _adapter = HTTPAdapter(max_retries=_retry)
        self._session = requests.Session()
        self._session.mount("http://", _adapter)
        self._session.mount("https://", _adapter)

    def fetch_logs(self, start_unix_s: float, end_unix_s: float) -> list[LogRecord]:
        """Execute a LogQL range query for the namespace."""
        import requests

        # OTel logs pushed via gateway use the k8s_namespace_name label
        # (added by the k8sattributes processor), not the bare "namespace" label.
        query = f'{{k8s_namespace_name="{self._namespace}"}}'
        params = {
            "query": query,
            "start": int(start_unix_s * 1e9),  # Loki uses nanoseconds
            "end": int(end_unix_s * 1e9),
            "limit": self._limit,
            "direction": "forward",
        }
        try:
            resp = self._session.get(
                f"{self._endpoint}/loki/api/v1/query_range",
                params=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Loki fetch_logs error: %s", exc)
            return []

        try:
            payload = resp.json()
        except ValueError as exc:
            logger.error("Loki JSON parse error: %s", exc)
            return []
        return self._parse_response(payload)

    def _parse_response(self, data: dict) -> list[LogRecord]:
        records: list[LogRecord] = []
        for stream in data.get("data", {}).get("result", []):
            labels = stream.get("stream", {})
            svc = (
                labels.get("service_name")           # OTel SDK resource attribute (via gateway)
                or labels.get("service")              # Istio / Kubernetes convention
                or labels.get("app")
                or labels.get("app_kubernetes_io_name")
                or ""
            )
            pod = (
                labels.get("k8s_pod_name")            # OTel k8sattributes
                or labels.get("pod")                  # Prometheus / bare k8s
                or ""
            )
            for _ts, line in stream.get("values", []):
                records.append(LogRecord(service_name=svc, pod_name=pod, body=line))
        return records


# ---------------------------------------------------------------------------
# Level classification helper
# ---------------------------------------------------------------------------


def classify_level(record: LogRecord) -> str:
    """Return ``'ERROR'``, ``'WARN'``, or ``'INFO'`` for a log record.

    Uses the pre-parsed ``level`` field when available, otherwise falls back
    to regex on the body text.
    """
    if record.level:
        lvl = record.level.upper()
        if _LEVEL_ERROR.match(lvl):
            return "ERROR"
        if _LEVEL_WARN.match(lvl):
            return "WARN"
        return "INFO"

    body = record.body
    if _LEVEL_ERROR.search(body):
        return "ERROR"
    if _LEVEL_WARN.search(body):
        return "WARN"
    return "INFO"


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------


class LogCollector:
    """Fetch L(t) ∈ ℝ^{N×4} from a log query backend.

    Parameters
    ----------
    backend:
        Implementation of :class:`LogQueryBackend`.
    window_s:
        Look-back window in seconds.
    semantic_scorers:
        Dict mapping service_name → :class:`SemanticAnomalyScorer`.
        When a service has no scorer the semantic anomaly feature is NaN.
        Call :meth:`fit_semantic_centroid` after a reference window to populate.
    services:
        Optional explicit list of service names.
    """

    def __init__(
        self,
        backend: LogQueryBackend,
        window_s: float = 120.0,
        semantic_scorers: dict[str, SemanticAnomalyScorer] | None = None,
        services: list[str] | None = None,
        semantic_enabled: bool = True,
    ) -> None:
        self._backend = backend
        self._window_s = window_s
        self._semantic_scorers: dict[str, SemanticAnomalyScorer] = semantic_scorers or {}
        self._services = services
        self._semantic_enabled = semantic_enabled

    # ------------------------------------------------------------------
    # Centroid fitting
    # ------------------------------------------------------------------

    def fit_semantic_centroid(
        self,
        reference_records: list[LogRecord],
    ) -> LogCollector:
        """Compute per-service SentenceBERT centroids from reference log records.

        Intended to be called once on a representative "normal" window
        (e.g. first 5 minutes after a stable deployment).

        Parameters
        ----------
        reference_records:
            Log records from the normal reference window.

        Returns
        -------
        self
        """
        svc_lines: dict[str, list[str]] = {}
        for rec in reference_records:
            if rec.service_name:
                svc_lines.setdefault(rec.service_name, []).append(rec.body)

        for svc, lines in svc_lines.items():
            scorer = self._semantic_scorers.setdefault(svc, SemanticAnomalyScorer())
            scorer.fit(lines)
            logger.info(
                "LogCollector: fitted centroid for service '%s' (%d lines)",
                svc,
                len(lines),
            )

        return self

    # ------------------------------------------------------------------
    # Main collect
    # ------------------------------------------------------------------

    def collect(
        self,
        timestamp: float | None = None,
        service_index: dict[str, int] | None = None,
    ) -> tuple[npt.NDArray[np.float32], list[str]]:
        L_t, services, _ = self.collect_with_records(
            timestamp=timestamp,
            service_index=service_index,
        )
        return L_t, services

    def collect_with_records(
        self,
        timestamp: float | None = None,
        service_index: dict[str, int] | None = None,
    ) -> tuple[npt.NDArray[np.float32], list[str], list[LogRecord]]:
        """Return L(t) for the window ending at ``timestamp``.

        Parameters
        ----------
        timestamp:
            Unix timestamp (seconds). Defaults to now.
        service_index:
            Pre-defined mapping service → row index. Must be consistent with
            the other collectors so matrices align when concatenated.

        Returns
        -------
        L_t:
            Float32 array of shape (N, 4). NaN where no logs observed.
        services:
            List of N service names.
        """
        import time

        ts = timestamp or time.time()
        records = self._backend.fetch_logs(ts - self._window_s, ts)

        services, svc_idx = self._resolve_services(records, service_index)
        n = len(services)
        L_t = np.full((n, LOGS_DIM), float("nan"), dtype=np.float32)

        if not records:
            return L_t, services, []

        self._fill_features(L_t, svc_idx, records)
        return L_t, services, records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_services(
        self,
        records: list[LogRecord],
        service_index: dict[str, int] | None,
    ) -> tuple[list[str], dict[str, int]]:
        if service_index is not None:
            services = sorted(service_index, key=lambda s: service_index[s])
            return services, service_index

        discovered: set[str] = {r.service_name for r in records if r.service_name}
        if self._services is not None:
            discovered |= set(self._services)
        services = sorted(discovered)
        return services, {s: i for i, s in enumerate(services)}

    def _fill_features(
        self,
        L_t: npt.NDArray[np.float32],
        svc_idx: dict[str, int],
        records: list[LogRecord],
    ) -> None:
        # Group records by (service, pod)
        SvcPod = tuple[str, str]
        pod_records: dict[SvcPod, list[LogRecord]] = {}
        for rec in records:
            if rec.service_name in svc_idx:
                key = (rec.service_name, rec.pod_name)
                pod_records.setdefault(key, []).append(rec)

        # Group by service (for entropy + semantic scoring)
        svc_all_records: dict[str, list[LogRecord]] = {}
        for (svc, _pod), recs in pod_records.items():
            svc_all_records.setdefault(svc, []).extend(recs)

        for svc, row in svc_idx.items():
            all_recs = svc_all_records.get(svc, [])
            if not all_recs:
                continue

            # Collect per-pod counts for volume-weighted aggregation
            pod_svc_keys = [k for k in pod_records if k[0] == svc]
            pod_n_total: list[float] = []
            pod_error_rate: list[float] = []
            pod_warn_rate: list[float] = []

            for key in pod_svc_keys:
                recs = pod_records[key]
                n_total = len(recs)
                n_error = sum(1 for r in recs if classify_level(r) == "ERROR")
                n_warn = sum(1 for r in recs if classify_level(r) == "WARN")
                pod_n_total.append(float(n_total))
                pod_error_rate.append(n_error / max(n_total, 1))
                pod_warn_rate.append(n_warn / max(n_total, 1))

            volumes = np.array(pod_n_total, dtype=np.float32)
            L_t[row, _L_ERROR_RATE] = aggregate_volume_weighted(
                np.array(pod_error_rate, dtype=np.float32), volumes
            )
            L_t[row, _L_WARN_RATE] = aggregate_volume_weighted(
                np.array(pod_warn_rate, dtype=np.float32), volumes
            )

            # Lexical entropy on all lines for this service
            lines = [r.body for r in all_recs]
            L_t[row, _L_LEXICAL_ENTROPY] = lexical_entropy(lines)

            # Semantic anomaly — requires fitted centroid
            scorer = self._semantic_scorers.get(svc)
            if (
                self._semantic_enabled
                and scorer is not None
                and scorer.centroid is not None
            ):
                L_t[row, _L_SEMANTIC_ANOMALY] = scorer.score(lines)
            # else: stays NaN until centroid is fitted
