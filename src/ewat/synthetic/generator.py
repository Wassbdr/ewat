"""Seeded generator of synthetic microservice episodes.

An episode is a signal ``S(t) ∈ ℝ^{T×6×17}`` (the v4 feature schema of
:mod:`telemetry.feature_names`, 30 s per step) over the six canonical
Online Boutique services, plus a dynamic call-graph adjacency
``A(t) ∈ ℝ^{T×6×6×3}`` (volume, latency, error channels).

Each episode has up to three phases:

1. ``normal``  — stationary load with AR(1) noise;
2. ``drift``   — harmless traffic ramp (load × 1.4–2.2): CPU, network, queue
   and disk rise on every service, latency barely moves, no errors;
3. ``fault``   — one of :data:`FAULT_TYPES` injected on one random service
   (never the load generator), on top of the drifted load.

``drift_traffic_ramp`` episodes stop after phase 2: they contain no fault and
must not raise an alert. Scenario names follow ``configs/collection.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from telemetry import feature_names as fn

SERVICES: list[str] = [
    "frontend",
    "recommendation",
    "cart",
    "ad",
    "product-catalog",
    "load-generator",
]
# caller → callee
CALL_GRAPH: list[tuple[str, str]] = [
    ("load-generator", "frontend"),
    ("frontend", "recommendation"),
    ("frontend", "cart"),
    ("frontend", "ad"),
    ("frontend", "product-catalog"),
    ("recommendation", "product-catalog"),
]

BENIGN_DRIFT = "drift_traffic_ramp"
FAULT_TYPES: list[str] = [
    "fail_slow_latency",
    "intermittent_error",
    "cpu_starvation",
    "resource_leak",
]
SCENARIOS: list[str] = [BENIGN_DRIFT, *FAULT_TYPES]

STEP_SECONDS = 30.0
N_FEAT = fn.SIGNAL_DIM

_IDX = {s: i for i, s in enumerate(SERVICES)}
_LOADGEN = _IDX["load-generator"]
_FRONTEND = _IDX["frontend"]


@dataclass
class Episode:
    """One synthetic episode and its ground truth."""

    signal: np.ndarray  # (T, N, 17) float32, raw units
    adjacency: np.ndarray  # (T, N, N, 3) float32
    phase: np.ndarray  # (T,) str: normal | drift | fault
    scenario: str
    seed: int
    drift_start: int
    fault_start: int | None
    target: str | None
    params: dict = field(default_factory=dict)

    @property
    def fault_type(self) -> str | None:
        return None if self.scenario == BENIGN_DRIFT else self.scenario

    @property
    def T(self) -> int:  # noqa: N802
        return int(self.signal.shape[0])


def _baseline(rng: np.random.Generator) -> np.ndarray:
    """Per-service steady-state value of every feature, shape (N, 17)."""
    n = len(SERVICES)
    b = np.empty((n, N_FEAT))
    b[:, fn.M_CPU_UTIL] = rng.uniform(0.15, 0.35, n)
    b[:, fn.M_RAM_UTIL] = rng.uniform(0.30, 0.50, n)
    b[:, fn.M_LATENCY_P99] = rng.uniform(0.05, 0.20, n)
    b[:, fn.M_ERROR_RATE] = rng.uniform(0.001, 0.004, n)
    b[:, fn.M_NET_SAT] = rng.uniform(0.15, 0.35, n)
    b[:, fn.M_DISK_IO] = rng.uniform(40.0, 120.0, n)
    b[:, fn.M_QUEUE_DEPTH] = rng.uniform(2.0, 6.0, n)
    b[:, fn.T_SPAN_DUR_P99] = b[:, fn.M_LATENCY_P99] * rng.uniform(0.8, 0.95, n)
    b[:, fn.T_ABNORMAL_RATE] = rng.uniform(0.001, 0.004, n)
    b[:, fn.T_TRACE_DEPTH] = rng.uniform(3.0, 5.0, n)
    b[:, fn.T_FAN_OUT] = rng.uniform(1.5, 3.0, n)
    b[:, fn.T_RETRY_RATE] = rng.uniform(0.002, 0.006, n)
    b[:, fn.T_LATENCY_CV] = rng.uniform(0.25, 0.45, n)
    b[:, fn.L_ERROR_RATE] = rng.uniform(0.002, 0.006, n)
    b[:, fn.L_WARN_RATE] = rng.uniform(0.01, 0.03, n)
    b[:, fn.L_SEMANTIC_ANOMALY] = rng.uniform(0.05, 0.15, n)
    b[:, fn.L_LEXICAL_ENTROPY] = rng.uniform(4.0, 5.0, n)
    return b


def _ar1_noise(rng: np.random.Generator, shape: tuple[int, ...], phi: float = 0.6) -> np.ndarray:
    """Unit-variance AR(1) noise along axis 0."""
    eps = rng.standard_normal(shape)
    out = np.empty(shape)
    out[0] = eps[0]
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, shape[0]):
        out[t] = phi * out[t - 1] + scale * eps[t]
    return out


def _ramp(T: int, start: int, length: int) -> np.ndarray:
    """0 before ``start``, linear 0→1 over ``length`` steps, then 1."""
    t = np.arange(T, dtype=np.float64)
    return np.clip((t - start + 1) / max(length, 1), 0.0, 1.0)


def _adjacency(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Dynamic call-graph adjacency from the generated metrics."""
    T, n, _ = x.shape
    adj = np.zeros((T, n, n, 3))
    for caller, callee in CALL_GRAPH:
        i, j = _IDX[caller], _IDX[callee]
        vol = x[:, j, fn.M_NET_SAT] / base[j, fn.M_NET_SAT]
        lat = x[:, j, fn.M_LATENCY_P99] / base[j, fn.M_LATENCY_P99]
        err = 1.0 + 50.0 * x[:, j, fn.M_ERROR_RATE]
        for c, v in enumerate((vol, lat, err)):
            v = np.clip(v, 0.0, 10.0)
            adj[:, i, j, c] = v
            adj[:, j, i, c] = v
    return adj


def generate_episode(
    scenario: str,
    seed: int,
    *,
    n_normal: int | None = None,
    n_drift: int | None = None,
    n_fault: int = 30,
    noise: float = 0.05,
) -> Episode:
    """Generate one episode.

    Parameters
    ----------
    scenario:
        One of :data:`SCENARIOS`.
    seed:
        Seed of the episode; the same ``(scenario, seed)`` always yields the
        same arrays.
    n_normal, n_drift:
        Phase lengths in steps. Drawn from ``[35, 45]`` and ``[30, 40]`` when
        ``None``.
    n_fault:
        Length of the fault phase (also appended, fault-free, to
        ``drift_traffic_ramp`` episodes so every episode has the same shape).
    noise:
        Relative standard deviation of the AR(1) measurement noise.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario)])
    n_normal = int(rng.integers(35, 46)) if n_normal is None else n_normal
    n_drift = int(rng.integers(30, 41)) if n_drift is None else n_drift
    T = n_normal + n_drift + n_fault
    n = len(SERVICES)
    drift_start = n_normal
    fault_start = n_normal + n_drift

    base = _baseline(rng)
    x = np.broadcast_to(base, (T, n, N_FEAT)).copy()

    # ---- Phase 2: harmless traffic ramp (all services) -------------------
    load_gain = float(rng.uniform(1.4, 2.2))
    ramp_len = int(rng.integers(4, 9))
    load = 1.0 + (load_gain - 1.0) * _ramp(T, drift_start, ramp_len)  # (T,)
    L = load[:, None]
    x[..., fn.M_CPU_UTIL] *= L**0.8
    x[..., fn.M_NET_SAT] *= L
    x[..., fn.M_QUEUE_DEPTH] *= L
    x[..., fn.M_DISK_IO] *= L**0.5
    x[..., fn.M_RAM_UTIL] += 0.04 * (L - 1.0)
    x[..., fn.M_LATENCY_P99] *= 1.0 + 0.05 * (L - 1.0)
    x[..., fn.T_SPAN_DUR_P99] *= 1.0 + 0.05 * (L - 1.0)

    # ---- Phase 3: fault on one service ------------------------------------
    params: dict = {"load_gain": load_gain, "ramp_len": ramp_len}
    target: str | None = None
    has_fault = scenario != BENIGN_DRIFT
    if has_fault:
        candidates = [i for i in range(n) if i != _LOADGEN]
        k = int(rng.choice(candidates))
        target = SERVICES[k]
        onset = _ramp(T, fault_start, 2)
        up = _FRONTEND if k != _FRONTEND else None  # caller that sees propagation

        if scenario == "fail_slow_latency":
            m = float(rng.uniform(2.5, 5.0))
            f = 1.0 + (m - 1.0) * onset
            for j in (fn.M_LATENCY_P99, fn.T_SPAN_DUR_P99):
                x[:, k, j] *= f
            x[:, k, fn.T_LATENCY_CV] += 0.35 * onset
            x[:, k, fn.M_QUEUE_DEPTH] *= 1.0 + 0.5 * onset
            if up is not None:
                x[:, up, fn.M_LATENCY_P99] *= 1.0 + 0.3 * (m - 1.0) * onset
                x[:, up, fn.T_SPAN_DUR_P99] *= 1.0 + 0.3 * (m - 1.0) * onset
            params["latency_factor"] = m
        elif scenario == "intermittent_error":
            e = float(rng.uniform(0.06, 0.20))
            x[:, k, fn.M_ERROR_RATE] += e * onset
            x[:, k, fn.T_ABNORMAL_RATE] += e * onset
            x[:, k, fn.L_ERROR_RATE] += 0.7 * e * onset
            x[:, k, fn.T_RETRY_RATE] += 0.4 * e * onset
            x[:, k, fn.L_SEMANTIC_ANOMALY] += 0.15 * onset
            if up is not None:
                x[:, up, fn.M_ERROR_RATE] += 0.3 * e * onset
                x[:, up, fn.T_RETRY_RATE] += 0.2 * e * onset
            params["error_rate"] = e
        elif scenario == "cpu_starvation":
            c = float(rng.uniform(0.85, 0.97))
            x[:, k, fn.M_CPU_UTIL] += (c - x[:, k, fn.M_CPU_UTIL]) * onset
            lat = float(rng.uniform(1.4, 2.0))
            x[:, k, fn.M_LATENCY_P99] *= 1.0 + (lat - 1.0) * onset
            x[:, k, fn.T_SPAN_DUR_P99] *= 1.0 + (lat - 1.0) * onset
            x[:, k, fn.M_QUEUE_DEPTH] *= 1.0 + float(rng.uniform(1.0, 2.0)) * onset
            x[:, k, fn.T_LATENCY_CV] += 0.15 * onset
            params["cpu_level"] = c
        elif scenario == "resource_leak":
            slope = float(rng.uniform(0.012, 0.022))
            steps = np.clip(np.arange(T) - fault_start + 1, 0, None).astype(np.float64)
            x[:, k, fn.M_RAM_UTIL] = np.minimum(x[:, k, fn.M_RAM_UTIL] + slope * steps, 0.98)
            pressure = np.clip(x[:, k, fn.M_RAM_UTIL] - 0.75, 0.0, None) / 0.23
            x[:, k, fn.L_WARN_RATE] += 0.05 * pressure
            x[:, k, fn.M_CPU_UTIL] *= 1.0 + 0.4 * pressure
            x[:, k, fn.M_LATENCY_P99] *= 1.0 + 0.6 * pressure
            params["leak_per_step"] = slope

    # ---- Measurement noise (multiplicative AR(1)) -------------------------
    x *= 1.0 + noise * _ar1_noise(rng, x.shape)
    x = np.clip(x, 0.0, None)
    for j in (fn.M_CPU_UTIL, fn.M_RAM_UTIL):
        x[..., j] = np.clip(x[..., j], 0.0, 1.0)

    phase = np.array(["normal"] * n_normal + ["drift"] * n_drift + ["fault"] * n_fault)
    if not has_fault:
        phase[fault_start:] = "drift"

    return Episode(
        signal=x.astype(np.float32),
        adjacency=_adjacency(x, base).astype(np.float32),
        phase=phase,
        scenario=scenario,
        seed=seed,
        drift_start=drift_start,
        fault_start=fault_start if has_fault else None,
        target=target,
        params=params,
    )


def call_graph_adjacency(signal: np.ndarray, n_ref: int = 10) -> np.ndarray:
    """Adjacency for a metrics window that came without one.

    Uses the synthetic call graph and the median of the first ``n_ref`` steps
    as baseline. Only meaningful for the six canonical services.
    """
    x = np.nan_to_num(np.asarray(signal, dtype=np.float64))
    base = np.maximum(np.median(x[:n_ref], axis=0), 1e-6)
    return _adjacency(x, base).astype(np.float32)
