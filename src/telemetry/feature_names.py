"""Feature schema registry for the telemetry signal S(t).

Two schemas coexist:

- ``"v4"`` (a.k.a. v3/v4 datasets, Online Boutique): S(t) ∈ ℝ^{N×17},
  M(t) ∈ ℝ^{N×7} | T(t) ∈ ℝ^{N×6} | L(t) ∈ ℝ^{N×4}.
- ``"v5.1"`` (Train Ticket): S(t) ∈ ℝ^{N×18},
  M(t) ∈ ℝ^{N×10} | T(t) ∈ ℝ^{N×4} | L(t) ∈ ℝ^{N×4}.

The module-level constants (``FEATURE_NAMES``, ``SIGNAL_DIM``, the index
constants and slices) describe the **v4 schema** and are kept for backward
compatibility. New code should resolve a schema explicitly via
:func:`get_schema` / :func:`signal_dim`, or infer it from an array with
:func:`schema_for_dim`.
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

# ---------------------------------------------------------------------------
# Schema registry (D1, audit 2026-06)
# ---------------------------------------------------------------------------
# Single source of truth for every signal schema in the project. The v5.1
# names were previously duplicated in v5/collect/build_features_v5.py and
# scripts/validate_v5.py; both now import from here.

SCHEMA_V4 = "v4"
SCHEMA_V5_1 = "v5.1"

FEATURE_NAMES_V4: list[str] = FEATURE_NAMES

# v5.1 (Train Ticket, 2026-06): vs v4 — span_dur_p99 dropped (≡ latency_p99,
# ρ=1.0 when latency is trace-sourced), retry_rate dropped (structurally dead
# on TT), queue_depth dropped (no Istio/Envoy), +mem_limit_ratio (replaces
# oom_events: container_oom_events_total reads 0 on observit-cluster1),
# +3 JVM features (Spring Boot signal), log_warn_rate → restart_count.
FEATURE_NAMES_V5_1: list[str] = [
    # M(t) infra + JVM (0-9)
    "cpu_util", "ram_util", "latency_p99", "error_rate_http", "net_sat",
    "disk_io", "mem_limit_ratio", "jvm_heap_ratio", "jvm_gc_util", "jvm_threads_blocked",
    # T(t) traces (10-13)
    "abnormal_span_rate", "trace_depth", "fan_out", "latency_cv",
    # L(t) logs (14-17)
    "log_error_rate", "restart_count", "semantic_anomaly", "lexical_entropy",
]

SCHEMAS: dict[str, list[str]] = {
    SCHEMA_V4: FEATURE_NAMES_V4,
    SCHEMA_V5_1: FEATURE_NAMES_V5_1,
}

MODALITY_SLICES: dict[str, dict[str, slice]] = {
    SCHEMA_V4: {"M": slice(0, 7), "T": slice(7, 13), "L": slice(13, 17)},
    SCHEMA_V5_1: {"M": slice(0, 10), "T": slice(10, 14), "L": slice(14, 18)},
}

assert all(len(SCHEMAS[v]) == MODALITY_SLICES[v]["L"].stop for v in SCHEMAS)


def get_schema(version: str) -> list[str]:
    """Return the ordered feature names for a schema version.

    Parameters
    ----------
    version: One of ``"v4"`` or ``"v5.1"``.

    Raises
    ------
    KeyError: If ``version`` is not a registered schema.
    """
    if version not in SCHEMAS:
        raise KeyError(
            f"unknown feature schema {version!r}; registered: {sorted(SCHEMAS)}"
        )
    return list(SCHEMAS[version])


def signal_dim(version: str) -> int:
    """Return the signal dimensionality (number of features) for a schema."""
    return len(get_schema(version))


def schema_for_dim(dim: int) -> str:
    """Infer the schema version from a signal's last-axis dimension.

    Useful for consumers that only see the array (e.g. validators on legacy
    episodes whose metadata predates ``signal_feature_names``).

    Raises
    ------
    KeyError: If no registered schema has ``dim`` features.
    """
    for version, names in SCHEMAS.items():
        if len(names) == dim:
            return version
    raise KeyError(
        f"no registered feature schema with {dim} features; "
        f"registered dims: { {v: len(n) for v, n in SCHEMAS.items()} }"
    )
