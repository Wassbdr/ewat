"""Feature index constants for the 17-dimensional signal S(t) ∈ ℝ^{N×17}.

S(t) = [M(t) | T(t) | L(t)]

M(t) ∈ ℝ^{N×7}  — Prometheus metrics
T(t) ∈ ℝ^{N×6}  — OTel trace features
L(t) ∈ ℝ^{N×4}  — OTel log features
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# M(t) — Metrics features (indices 0–6)
# ---------------------------------------------------------------------------
M_CPU_UTIL = 0       # CPU utilisation (fraction of limit)
M_RAM_UTIL = 1       # RAM utilisation (fraction of limit)
M_LATENCY_P99 = 2    # HTTP request latency P99 (seconds)
M_ERROR_RATE = 3     # HTTP error rate (4xx + 5xx / total)
M_NET_SAT = 4        # Network saturation (bytes/s, normalised)
M_DISK_IO = 5        # Disk I/O (IOPS/s)
M_QUEUE_DEPTH = 6    # Queue depth / pending requests

# ---------------------------------------------------------------------------
# T(t) — Trace features (indices 7–12)
# ---------------------------------------------------------------------------
T_SPAN_DUR_P99 = 7       # P99 span duration (seconds) — P99 on union of raw durations
T_ABNORMAL_RATE = 8      # Fraction of error/abnormal spans
T_TRACE_DEPTH = 9        # Median max depth of trace trees
T_FAN_OUT = 10           # Median fan-out (children per span)
T_RETRY_RATE = 11        # Fraction of retry spans
T_LATENCY_CV = 12        # Latency coefficient of variation (std/mean)

# ---------------------------------------------------------------------------
# L(t) — Log features (indices 13–16)
# ---------------------------------------------------------------------------
L_ERROR_RATE = 13        # Fraction of ERROR-level log lines
L_WARN_RATE = 14         # Fraction of WARN-level log lines
L_SEMANTIC_ANOMALY = 15  # Mean cosine distance to normal centroid (SentenceBERT)
L_LEXICAL_ENTROPY = 16   # Lexical entropy of token distribution

# ---------------------------------------------------------------------------
# Convenient groupings
# ---------------------------------------------------------------------------
METRICS_SLICE = slice(0, 7)
TRACES_SLICE = slice(7, 13)
LOGS_SLICE = slice(13, 17)

SIGNAL_DIM = 17
METRICS_DIM = 7
TRACES_DIM = 6
LOGS_DIM = 4

# Human-readable names in signal order
FEATURE_NAMES: list[str] = [
    "cpu_util",
    "ram_util",
    "latency_p99",
    "error_rate_http",
    "net_sat",
    "disk_io",
    "queue_depth",
    "span_dur_p99",
    "abnormal_span_rate",
    "trace_depth",
    "fan_out",
    "retry_rate",
    "latency_cv",
    "log_error_rate",
    "log_warn_rate",
    "semantic_anomaly",
    "lexical_entropy",
]

assert len(FEATURE_NAMES) == SIGNAL_DIM

# Aggregation rule for each feature (used by collectors when reducing pods→service)
# "max"              → saturation metrics
# "volume_weighted"  → rate metrics
# "p99_union"        → latency metrics (percentile on union of all durations)
# "median"           → structural / distributional metrics
AGGREGATION_RULE: dict[str, str] = {
    "cpu_util": "max",
    "ram_util": "max",
    "latency_p99": "p99_union",
    "error_rate_http": "volume_weighted",
    "net_sat": "max",
    "disk_io": "max",
    "queue_depth": "max",
    "span_dur_p99": "p99_union",
    "abnormal_span_rate": "volume_weighted",
    "trace_depth": "median",
    "fan_out": "median",
    "retry_rate": "volume_weighted",
    "latency_cv": "median",
    "log_error_rate": "volume_weighted",
    "log_warn_rate": "volume_weighted",
    "semantic_anomaly": "median",
    "lexical_entropy": "median",
}
