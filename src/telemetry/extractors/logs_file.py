"""Offline log extractor — parses a Loki dump into :class:`LogRecord` objects.

Reuses :class:`telemetry.collectors.log_collector.LogCollector` via a file-backed
:class:`LogQueryBackend` so the 4 log features (error rate, warn rate,
lexical entropy, semantic anomaly) stay computed the same way as in the
old online pipeline.

The Loki dump is expected to be a list of ``{"labels": {...}, "values": [[ts, line], ...]}``
entries, already namespaced by ``k8s_namespace_name`` at fetch time. The
loader preserves the timestamp on each record so the collector can serve
arbitrary sub-windows.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

from telemetry.collectors.log_collector import LogQueryBackend, LogRecord


@dataclass
class _TimedRecord:
    ts_s: float
    record: LogRecord


class InMemoryLogBackend(LogQueryBackend):
    """Serve :class:`LogRecord` objects from a pre-parsed dump by timestamp."""

    def __init__(self, timed_records: list[_TimedRecord]) -> None:
        timed_records.sort(key=lambda tr: tr.ts_s)
        self._records = [tr.record for tr in timed_records]
        self._keys = [tr.ts_s for tr in timed_records]

    def fetch_logs(self, start_unix_s: float, end_unix_s: float) -> list[LogRecord]:
        if not self._records:
            return []
        lo = bisect.bisect_left(self._keys, start_unix_s)
        hi = bisect.bisect_right(self._keys, end_unix_s)
        return list(self._records[lo:hi])


def parse_loki_dump(dump: dict[str, Any]) -> list[_TimedRecord]:
    """Flatten a Loki dump into timestamped :class:`LogRecord` objects.

    ``values`` entries come as ``[ts_ns_str, line]``; ts is converted to
    seconds. The service / pod labels follow the same resolution order as
    :meth:`LokiBackend._parse_response`.
    """
    out: list[_TimedRecord] = []
    for stream in dump.get("streams", []) or []:
        labels = stream.get("labels", {}) or {}
        values = stream.get("values", []) or []
        svc = (
            labels.get("service_name")
            or labels.get("service")
            or labels.get("app")
            or labels.get("app_kubernetes_io_name")
            or ""
        )
        pod = labels.get("k8s_pod_name") or labels.get("pod") or ""
        for entry in values:
            if len(entry) < 2:
                continue
            ts_raw, line = entry[0], entry[1]
            try:
                ts_ns = int(ts_raw)
            except (TypeError, ValueError):
                continue
            ts_s = ts_ns / 1e9
            out.append(_TimedRecord(ts_s=ts_s, record=LogRecord(
                service_name=svc, pod_name=pod, body=str(line)
            )))
    return out


def apply_aliases(records: list[_TimedRecord], aliases: dict[str, str]) -> None:
    """Rewrite ``record.service_name`` in place using the aliases map."""
    if not aliases:
        return
    for tr in records:
        if tr.record.service_name in aliases:
            tr.record.service_name = aliases[tr.record.service_name]
