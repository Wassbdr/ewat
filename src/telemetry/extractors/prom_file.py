"""Offline Prometheus extractor — computes M(t) from a range-query dump.

We re-use the existing :class:`PrometheusCollector` for all the pod→service
aggregation logic (CPU util, histogram-based P99, volume-weighted error
rate, etc.) by subclassing it and overriding ``_query_all`` so that
instant-query results at timestamp ``ts`` are reconstructed by
**nearest-timestamp lookup** into the parsed range dump.

Using nearest neighbour (rather than linear interpolation) matches the
semantics of a real Prometheus instant query at ``ts`` in a world where
metric evaluation is already a step function between scrapes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from telemetry.collectors.prometheus_collector import PrometheusCollector

logger = logging.getLogger(__name__)


class FilePrometheusCollector(PrometheusCollector):
    """Drop-in replacement of :class:`PrometheusCollector` that reads from a dump.

    Parameters
    ----------
    range_results:
        Dict ``{query_name -> prometheus_range_response}`` as produced by
        :class:`telemetry.recorder.TelemetryRecorder` and serialised to
        ``prometheus_range.json.gz`` in Phase 1.
    fallback_used:
        Dict ``{query_name -> bool}`` indicating whether the fallback PromQL
        template was the source of the data. Stored for provenance; the
        feature-filling logic is insensitive to primary/fallback because
        ``PrometheusCollector._service_label`` handles both label sets.
    namespace, services, aliases:
        Same meaning as in :class:`PrometheusCollector`.
    """

    def __init__(
        self,
        range_results: dict[str, Any],
        fallback_used: dict[str, bool] | None = None,
        namespace: str = "ewat",
        services: list[str] | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            endpoint="file://offline",
            namespace=namespace,
            services=services,
            aliases=aliases,
        )
        self._parsed: dict[str, list[dict[str, Any]]] = {}
        self._time_grid: dict[str, np.ndarray] = {}
        self._values_matrix: dict[str, list[np.ndarray]] = {}
        self._fallback_used = dict(fallback_used or {})
        self._parse_range_results(range_results)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_range_results(self, range_results: dict[str, Any]) -> None:
        """Convert each range-query response into lookup-friendly arrays.

        For every query we keep:
        - ``_parsed[name]`` — list of ``{"metric": {...}, "values_ts": np.ndarray,
          "values": np.ndarray}`` so we can do nearest-timestamp lookup fast.
        """
        for name, response in range_results.items():
            data = (response or {}).get("data", {}) or {}
            rows = data.get("result", []) or []
            parsed: list[dict[str, Any]] = []
            for row in rows:
                metric = row.get("metric", {}) or {}
                values = row.get("values", []) or []
                if not values:
                    continue
                ts_array = np.asarray([float(v[0]) for v in values], dtype=np.float64)
                val_array = np.asarray([_safe_float(v[1]) for v in values], dtype=np.float64)
                parsed.append({
                    "metric": dict(metric),
                    "ts": ts_array,
                    "values": val_array,
                })
            self._parsed[name] = parsed

    # ------------------------------------------------------------------
    # Overridden hook
    # ------------------------------------------------------------------

    def _query_all(self, ts: float) -> dict[str, list[dict[str, Any]]]:
        """Build instant-query result sets at ``ts`` from the parsed dump."""
        results: dict[str, list[dict[str, Any]]] = {}
        self._query_used_fallback = {}
        for name, parsed_rows in self._parsed.items():
            items: list[dict[str, Any]] = []
            for row in parsed_rows:
                val = _nearest_value(row["ts"], row["values"], ts)
                if val is None or np.isnan(val):
                    continue
                items.append({"metric": row["metric"], "value": [ts, str(val)]})
            results[name] = items
            self._query_used_fallback[name] = self._fallback_used.get(name, False)
        return results

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def available_queries(self) -> list[str]:
        return sorted(self._parsed.keys())

    def coverage_window(self) -> tuple[float | None, float | None]:
        """Return ``(min_ts, max_ts)`` across all parsed series (or Nones)."""
        tmin: float | None = None
        tmax: float | None = None
        for parsed_rows in self._parsed.values():
            for row in parsed_rows:
                ts = row["ts"]
                if ts.size == 0:
                    continue
                tmin = float(ts[0]) if tmin is None else min(tmin, float(ts[0]))
                tmax = float(ts[-1]) if tmax is None else max(tmax, float(ts[-1]))
        return tmin, tmax


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if f != f:  # NaN guard
        return float("nan")
    return f


def _nearest_value(ts_array: np.ndarray, val_array: np.ndarray, target: float) -> float | None:
    """Return the sample whose timestamp is closest to ``target``.

    Returns ``None`` when ``ts_array`` is empty. The caller filters NaNs.
    """
    if ts_array.size == 0:
        return None
    idx = int(np.abs(ts_array - target).argmin())
    return float(val_array[idx])
