"""Adapt RCAEval RE2-OB dataset → EWAT episode format for zero-shot evaluation.

Real RCAEval RE2-OB layout (inspected from actual files):
    RE2-OB/
        {fault_service}_{fault_type}/        e.g. checkoutservice_cpu/
            {instance}/                      e.g. 1/, 2/, 3/
                metrics.csv       wide format: time,{svc}_{metric},... (1s resolution)
                tracets_lat.csv   pre-aggregated latency: time,{svc}_{op},... (15s)
                tracets_err.csv   pre-aggregated error rate: same columns (15s)
                logs.csv          time,timestamp,container_name,message,level,...
                inject_time.txt   Unix timestamp (integer seconds)
                traces.csv        raw span data (used for depth/fan-out)

EWAT episode output:
    data/features/rcaeval/{episode_id}/
        signal.npz            signal: (T, N=6, 17) float32
        adjacency.npz         adjacency: (T, N, N, 3) float32
        labels.parquet
        services.json
        metadata.json

Workflow
--------
# Inspect first case:
    python -m scripts.adapt_rcaeval --inspect --data-dir data/raw/rcaeval/RE2-OB

# Convert all cases:
    python -m scripts.adapt_rcaeval --data-dir data/raw/rcaeval/RE2-OB \\
        --output data/features/rcaeval

# Assemble into split:
    python -m scripts.assemble_dataset \\
        --features-root data/features/rcaeval \\
        --output data/datasets/ewat_rcaeval --stratified

Notes
-----
- Only the 6 EWAT services (ad, cart, frontend, load-generator, product-catalog,
  recommendation) are kept from the 11 Online Boutique services. Root causes in
  unmapped services (checkout, currency, payment, …) are recorded but produce
  all-NaN rows in the signal for that service.
- disk_io (M5) is available only for adservice, emailservice, recommendationservice,
  redis in RCAEval — other services get NaN.
- queue_length (M6), retry_rate (T4), semantic_anomaly (L2) → NaN.
- Metrics are 1-second counters/gauges resampled to 30-second windows.
- Istio latency columns (istio-latency-99) are used for latency_p99 directly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── Service mapping ────────────────────────────────────────────────────────────

EWAT_SERVICES: list[str] = [
    "ad", "cart", "frontend", "load-generator", "product-catalog", "recommendation"
]
N_SERVICES = len(EWAT_SERVICES)
SVC_IDX = {s: i for i, s in enumerate(EWAT_SERVICES)}

# RCAEval service name → EWAT canonical name
RCAEVAL_TO_EWAT: dict[str, str] = {
    "adservice":             "ad",
    "cartservice":           "cart",
    "frontendservice":       "frontend",
    "frontend":              "frontend",
    "loadgenerator":         "load-generator",
    "productcatalogservice": "product-catalog",
    "recommendationservice": "recommendation",
}

# Fault type → EWAT scenario name (best effort)
FAULT_TO_SCENARIO: dict[str, str] = {
    "cpu":    "cpu_starvation",
    "mem":    "memory_pressure",
    "delay":  "fail_slow_latency",
    "loss":   "network_loss",
    "disk":   "disk_io_fault",
    "socket": "fail_slow_cpu",
}

FAULT_TO_CATEGORY: dict[str, str] = {
    "cpu":    "contention",
    "mem":    "contention",
    "delay":  "slowdown",
    "loss":   "network",
    "disk":   "io",
    "socket": "contention",
}

STEP_S = 30.0  # seconds per EWAT timestep

# ── Column parsing helpers ─────────────────────────────────────────────────────

# Ordered longest-first to avoid prefix collisions (e.g. "frontend" vs "frontendservice")
_KNOWN_RCAEVAL_SVCS = sorted(RCAEVAL_TO_EWAT.keys(), key=len, reverse=True)


def _parse_metric_col(col: str) -> tuple[str | None, str | None]:
    """Split 'adservice_container-cpu-usage-seconds-total' → ('adservice', 'container-cpu-...')"""
    for svc in _KNOWN_RCAEVAL_SVCS:
        prefix = svc + "_"
        if col.startswith(prefix):
            return svc, col[len(prefix):]
    return None, None


def _tracets_svc(col: str) -> str | None:
    """Extract service from tracets column like 'adservice_Recv' or 'frontendservice_frontend'."""
    for svc in _KNOWN_RCAEVAL_SVCS:
        if col.startswith(svc + "_"):
            return svc
    return None


# ── Metric extraction: M(t) ───────────────────────────────────────────────────
#
# Feature index → metric suffix → aggregation
#   "rate"  : cumulative counter → (last - first) / window_s (clipped ≥ 0)
#   "gauge" : instantaneous → mean over window
#   "ratio" : numerator_suffix / denominator_suffix → mean rate ratio

_M_SPEC: list[tuple[int, str, str, str | None]] = [
    # (feat_idx, metric_suffix, agg, denominator_suffix or None)
    (0, "container-cpu-usage-seconds-total",   "rate",  None),  # cpu_utilization
    (1, "container-memory-usage-bytes",         "gauge", "container-spec-memory-limit-bytes"),  # ram_util (normalised)
    (2, "istio-latency-99",                     "gauge", None),  # latency_p99 (ms)
    (3, "istio-error-total",                   "rate",  "istio-request-total"),  # error_rate
    (4, "container-network-receive-bytes-total","rate",  None),  # net_sat (receive only; add tx below)
    (5, "container-blkio-device-usage-total",  "rate",  None),  # disk_io
    # feat 6 (queue_length): NaN — no matching metric
]
_NET_TX_SUFFIX = "container-network-transmit-bytes-total"


def _resample_series(ts: np.ndarray, vals: np.ndarray, grid: np.ndarray,
                     agg: str = "mean") -> np.ndarray:
    """Resample a 1-s series to the 30-s grid. Returns (T,) float array."""
    out = np.full(len(grid), np.nan)
    for t, t0 in enumerate(grid):
        mask = (ts >= t0) & (ts < t0 + STEP_S)
        if not mask.any():
            continue
        w_vals = vals[mask]
        if agg == "mean":
            out[t] = float(np.nanmean(w_vals))
        elif agg == "last_minus_first":
            out[t] = float(w_vals[-1] - w_vals[0])
    return out


def _build_M(metrics: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Build M(t) ∈ ℝ^{T×N×7}."""
    T = len(grid)
    M = np.full((T, N_SERVICES, 7), np.nan, dtype=np.float64)
    ts = metrics["time"].values.astype(float)

    for feat_idx, suffix, agg, denom_suffix in _M_SPEC:
        for rcaeval_svc, ewat_svc in RCAEVAL_TO_EWAT.items():
            si = SVC_IDX[ewat_svc]
            col = f"{rcaeval_svc}_{suffix}"
            if col not in metrics.columns:
                continue
            vals = metrics[col].values.astype(float)

            if agg == "rate":
                # Compute per-window rate: Δcounter / STEP_S
                for t, t0 in enumerate(grid):
                    mask = (ts >= t0) & (ts < t0 + STEP_S)
                    if mask.sum() < 2:
                        continue
                    delta = float(vals[mask][-1]) - float(vals[mask][0])
                    rate = max(delta, 0.0) / STEP_S

                    if denom_suffix:
                        d_col = f"{rcaeval_svc}_{denom_suffix}"
                        if d_col in metrics.columns:
                            d_vals = metrics[d_col].values.astype(float)
                            d_delta = max(float(d_vals[mask][-1]) - float(d_vals[mask][0]), 0.0)
                            rate = rate / d_delta if d_delta > 1e-9 else np.nan
                    # accumulate (multiple services map to same EWAT slot)
                    if np.isnan(M[t, si, feat_idx]):
                        M[t, si, feat_idx] = rate
                    else:
                        M[t, si, feat_idx] += rate

            elif agg == "gauge":
                if denom_suffix:
                    d_col = f"{rcaeval_svc}_{denom_suffix}"
                    denom_vals = metrics[d_col].values.astype(float) if d_col in metrics.columns else None
                for t, t0 in enumerate(grid):
                    mask = (ts >= t0) & (ts < t0 + STEP_S)
                    if not mask.any():
                        continue
                    v = float(np.nanmean(vals[mask]))
                    if denom_suffix and denom_vals is not None:
                        d = float(np.nanmean(denom_vals[mask]))
                        v = v / d if d > 1.0 else np.nan
                    M[t, si, feat_idx] = v

    # Add network transmit to receive (feat 4 = total net_sat)
    for rcaeval_svc, ewat_svc in RCAEVAL_TO_EWAT.items():
        si = SVC_IDX[ewat_svc]
        tx_col = f"{rcaeval_svc}_{_NET_TX_SUFFIX}"
        if tx_col not in metrics.columns:
            continue
        tx_vals = metrics[tx_col].values.astype(float)
        for t, t0 in enumerate(grid):
            mask = (ts >= t0) & (ts < t0 + STEP_S)
            if mask.sum() < 2:
                continue
            tx_rate = max(tx_vals[mask][-1] - tx_vals[mask][0], 0.0) / STEP_S
            if np.isnan(M[t, si, 4]):
                M[t, si, 4] = tx_rate
            else:
                M[t, si, 4] += tx_rate

    return M.astype(np.float32)


# ── Trace feature extraction: T(t) ───────────────────────────────────────────

def _build_T(tracets_lat: pd.DataFrame | None, tracets_err: pd.DataFrame | None,
             traces_raw: pd.DataFrame | None, grid: np.ndarray) -> np.ndarray:
    """Build T(t) ∈ ℝ^{T×N×6}.

    Cols: span_dur_median, abnormal_span_rate, trace_depth, fan_out,
          retry_rate (NaN), latency_cv
    Uses pre-aggregated tracets_lat / tracets_err for speed.
    Falls back to raw traces.csv for depth/fan-out when available.
    """
    T = len(grid)
    Tc = np.full((T, N_SERVICES, 6), np.nan, dtype=np.float64)

    # ── From tracets_lat (P50 latency per service-operation, 15s bins) ─────────
    if tracets_lat is not None and "time" in tracets_lat.columns:
        lat_ts = tracets_lat["time"].values.astype(float)
        for col in tracets_lat.columns:
            if col == "time":
                continue
            rcaeval_svc = _tracets_svc(col)
            if rcaeval_svc is None:
                continue
            ewat_svc = RCAEVAL_TO_EWAT.get(rcaeval_svc)
            if ewat_svc is None:
                continue
            si = SVC_IDX[ewat_svc]
            vals = tracets_lat[col].values.astype(float)
            for t, t0 in enumerate(grid):
                mask = (lat_ts >= t0) & (lat_ts < t0 + STEP_S)
                if not mask.any():
                    continue
                w = vals[mask]
                w = w[~np.isnan(w)]
                if not len(w):
                    continue
                med = float(np.median(w))
                if np.isnan(Tc[t, si, 0]):
                    Tc[t, si, 0] = med
                else:
                    Tc[t, si, 0] = (Tc[t, si, 0] + med) / 2.0
                # latency_cv from spread across operations
                if len(w) > 1:
                    mn = float(np.mean(w))
                    if mn > 0:
                        cv = float(np.std(w)) / mn
                        if np.isnan(Tc[t, si, 5]):
                            Tc[t, si, 5] = cv
                        else:
                            Tc[t, si, 5] = (Tc[t, si, 5] + cv) / 2.0

    # ── From tracets_err (error rate per service-operation, 15s bins) ──────────
    if tracets_err is not None and "time" in tracets_err.columns:
        err_ts = tracets_err["time"].values.astype(float)
        for col in tracets_err.columns:
            if col == "time":
                continue
            rcaeval_svc = _tracets_svc(col)
            if rcaeval_svc is None:
                continue
            ewat_svc = RCAEVAL_TO_EWAT.get(rcaeval_svc)
            if ewat_svc is None:
                continue
            si = SVC_IDX[ewat_svc]
            vals = tracets_err[col].values.astype(float)
            for t, t0 in enumerate(grid):
                mask = (err_ts >= t0) & (err_ts < t0 + STEP_S)
                if not mask.any():
                    continue
                w = vals[mask]
                w = w[~np.isnan(w)]
                if not len(w):
                    continue
                err = float(np.mean(w))
                if np.isnan(Tc[t, si, 1]):
                    Tc[t, si, 1] = err
                else:
                    Tc[t, si, 1] = (Tc[t, si, 1] + err) / 2.0

    # ── From raw traces: depth and fan-out ─────────────────────────────────────
    if traces_raw is not None and not traces_raw.empty:
        cols = [c.lower() for c in traces_raw.columns]
        traces_raw = traces_raw.copy()
        traces_raw.columns = cols

        ts_col = next((c for c in cols if c in ("time", "timestamp", "starttime",
                                                  "starttimemillis")), None)
        svc_col = next((c for c in cols if c in ("servicename", "service_name", "service")), None)
        tid_col = next((c for c in cols if "traceid" in c or c == "trace_id"), None)
        pid_col = next((c for c in cols if "parentspan" in c or c == "parent_span_id"), None)

        if ts_col and svc_col:
            traces_raw["_ts"] = pd.to_numeric(traces_raw[ts_col], errors="coerce")
            # Some traces use HH:MM timestamps — those will become NaN → skip
            traces_raw = traces_raw.dropna(subset=["_ts"])
            # If timestamps look like milliseconds (> 1e12), convert to seconds
            if traces_raw["_ts"].median() > 1e12:
                traces_raw["_ts"] = traces_raw["_ts"] / 1000.0

            traces_raw["_ewat_svc"] = traces_raw[svc_col].map(RCAEVAL_TO_EWAT)

            for t, t0 in enumerate(grid):
                w = traces_raw[(traces_raw["_ts"] >= t0) & (traces_raw["_ts"] < t0 + STEP_S)]
                if w.empty:
                    continue
                if tid_col:
                    for tid, tg in w.groupby(tid_col):
                        svcs_in_trace = tg["_ewat_svc"].dropna().unique()
                        depth = len(tg)
                        fan_out = len(svcs_in_trace)
                        # Attribute to root service (no parent)
                        if pid_col:
                            roots = tg[tg[pid_col].isna() | (tg[pid_col].astype(str) == "")]
                        else:
                            roots = tg.head(1)
                        for _, root_row in roots.iterrows():
                            rsvc = RCAEVAL_TO_EWAT.get(str(root_row.get(svc_col, "")))
                            if rsvc and rsvc in SVC_IDX:
                                ri = SVC_IDX[rsvc]
                                prev_d = Tc[t, ri, 2]
                                prev_f = Tc[t, ri, 3]
                                Tc[t, ri, 2] = depth if np.isnan(prev_d) else (prev_d + depth) / 2
                                Tc[t, ri, 3] = fan_out if np.isnan(prev_f) else (prev_f + fan_out) / 2

    return Tc.astype(np.float32)


# ── Log feature extraction: L(t) ─────────────────────────────────────────────

def _lexical_entropy(messages: list[str]) -> float:
    tokens = " ".join(messages).lower().split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def _build_L(logs: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Build L(t) ∈ ℝ^{T×N×4}.

    Cols: log_error_rate, log_warn_rate, semantic_anomaly (NaN), lexical_entropy
    """
    T = len(grid)
    L = np.full((T, N_SERVICES, 4), np.nan, dtype=np.float64)

    if logs.empty:
        return L.astype(np.float32)

    logs = logs.copy()
    logs.columns = [c.lower().strip() for c in logs.columns]

    # Use 'timestamp' (nanosecond epoch) if available, else 'time' (HH:MM string → useless)
    ts_col = "timestamp" if "timestamp" in logs.columns else None
    svc_col = next((c for c in logs.columns if c in ("container_name", "service",
                                                       "service_name")), None)
    msg_col = next((c for c in logs.columns if c in ("message", "msg", "body")), None)
    lvl_col = next((c for c in logs.columns if c in ("level", "severity", "loglevel")), None)

    if ts_col is None:
        return L.astype(np.float32)

    logs["_ts"] = pd.to_numeric(logs[ts_col], errors="coerce")
    # nanosecond → second
    if logs["_ts"].dropna().median() > 1e15:
        logs["_ts"] = logs["_ts"] / 1e9

    if svc_col:
        logs["_ewat_svc"] = logs[svc_col].map(RCAEVAL_TO_EWAT)

    for t, t0 in enumerate(grid):
        w = logs[(logs["_ts"] >= t0) & (logs["_ts"] < t0 + STEP_S)]
        if w.empty:
            continue

        for ewat_svc, si in SVC_IDX.items():
            if svc_col:
                sw = w[w["_ewat_svc"] == ewat_svc]
            else:
                sw = w
            if sw.empty:
                continue

            n = len(sw)
            if lvl_col:
                lvls = sw[lvl_col].astype(str).str.lower()
                L[t, si, 0] = float((lvls == "error").sum()) / n
                L[t, si, 1] = float(lvls.isin({"warn", "warning"}).sum()) / n
            elif msg_col:
                msgs = sw[msg_col].astype(str)
                L[t, si, 0] = float(msgs.str.contains(r"\berror\b", case=False).sum()) / n
                L[t, si, 1] = float(msgs.str.contains(r"\bwarn\b", case=False).sum()) / n

            # semantic_anomaly → NaN
            L[t, si, 2] = np.nan

            if msg_col:
                L[t, si, 3] = _lexical_entropy(sw[msg_col].dropna().astype(str).tolist())

    return L.astype(np.float32)


# ── Adjacency from metrics (Istio) ────────────────────────────────────────────

def _build_adjacency_from_metrics(metrics: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Build A(t) ∈ ℝ^{T×N×N×3} from Istio per-service metrics.

    Since RCAEval does not provide per-edge Istio metrics (only per-service),
    we fall back to: if service B calls service A and both have significant
    traffic in the window, connect them with volume=request_rate,
    latency=istio-latency-99, error=istio-error-rate.

    For a proper call graph we would need per-edge span data.
    Here we use a fixed topology derived from Online Boutique known architecture.
    """
    T = len(grid)
    A = np.zeros((T, N_SERVICES, N_SERVICES, 3), dtype=np.float32)
    ts = metrics["time"].values.astype(float)

    # Online Boutique known call edges (caller → callee), restricted to EWAT services
    # Derived from the official service graph
    KNOWN_EDGES = [
        ("frontend", "ad"),
        ("frontend", "cart"),
        ("frontend", "product-catalog"),
        ("frontend", "recommendation"),
        ("load-generator", "frontend"),
        ("recommendation", "product-catalog"),
        ("cart", "product-catalog"),
    ]

    for t, t0 in enumerate(grid):
        mask = (ts >= t0) & (ts < t0 + STEP_S)
        if not mask.any():
            continue

        # Per-service request rate and latency from Istio
        svc_req: dict[str, float] = {}
        svc_lat: dict[str, float] = {}
        svc_err: dict[str, float] = {}
        for rcaeval_svc, ewat_svc in RCAEVAL_TO_EWAT.items():
            req_col = f"{rcaeval_svc}_istio-request-total"
            lat_col = f"{rcaeval_svc}_istio-latency-99"
            err_col = f"{rcaeval_svc}_istio-error-total"

            if req_col in metrics.columns:
                v = metrics[req_col].values[mask]
                delta = max(float(v[-1]) - float(v[0]), 0.0) / STEP_S
                svc_req[ewat_svc] = delta

            if lat_col in metrics.columns:
                svc_lat[ewat_svc] = float(np.nanmean(metrics[lat_col].values[mask]))

            if err_col in metrics.columns and req_col in metrics.columns:
                e_v = metrics[err_col].values[mask]
                r_v = metrics[req_col].values[mask]
                e_d = max(float(e_v[-1]) - float(e_v[0]), 0.0)
                r_d = max(float(r_v[-1]) - float(r_v[0]), 0.0)
                svc_err[ewat_svc] = e_d / r_d if r_d > 1e-9 else 0.0

        for caller, callee in KNOWN_EDGES:
            if caller not in SVC_IDX or callee not in SVC_IDX:
                continue
            ci, cj = SVC_IDX[caller], SVC_IDX[callee]
            A[t, ci, cj, 0] = svc_req.get(callee, 0.0)   # volume at callee
            A[t, ci, cj, 1] = svc_lat.get(callee, 0.0)   # latency at callee
            A[t, ci, cj, 2] = svc_err.get(callee, 0.0)   # error rate at callee

    return A


# ── Case loading ──────────────────────────────────────────────────────────────

def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  [warn] cannot read {path.name}: {e}")
        return pd.DataFrame()


def _load_instance(instance_dir: Path) -> dict:
    fault_dir = instance_dir.parent
    name_parts = fault_dir.name.rsplit("_", 1)
    fault_service = name_parts[0]
    fault_type = name_parts[1] if len(name_parts) > 1 else "unknown"

    inject_time = np.nan
    it = instance_dir / "inject_time.txt"
    if it.exists():
        try:
            inject_time = float(it.read_text().strip())
        except ValueError:
            pass

    return {
        "episode_id": f"rcaeval_{fault_dir.name}_{instance_dir.name}",
        "fault_service": fault_service,
        "fault_type": fault_type,
        "scenario": FAULT_TO_SCENARIO.get(fault_type, fault_type),
        "inject_time": inject_time,
        "root_cause_ewat": RCAEVAL_TO_EWAT.get(fault_service, fault_service),
        "metrics":     _read_csv_safe(instance_dir / "metrics.csv"),
        "tracets_lat": _read_csv_safe(instance_dir / "tracets_lat.csv"),
        "tracets_err": _read_csv_safe(instance_dir / "tracets_err.csv"),
        "logs":        _read_csv_safe(instance_dir / "logs.csv"),
        "traces":      _read_csv_safe(instance_dir / "traces.csv"),
    }


def _find_instances(re2_ob_dir: Path) -> list[Path]:
    instances = []
    for fault_dir in sorted(re2_ob_dir.iterdir()):
        if not fault_dir.is_dir():
            continue
        for inst in sorted(fault_dir.iterdir()):
            if inst.is_dir() and inst.name.isdigit():
                instances.append(inst)
    return instances


# ── Episode writer ────────────────────────────────────────────────────────────

def _write_episode(case: dict, output_root: Path) -> bool:
    ep_id = case["episode_id"]
    out_dir = output_root / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = case["metrics"]
    if metrics.empty or "time" not in metrics.columns:
        print(f"  [skip] {ep_id}: no metrics")
        return False

    # Build 30-second grid from metrics timestamps
    t_min = float(metrics["time"].min())
    t_max = float(metrics["time"].max())
    if t_max - t_min < STEP_S * 5:
        print(f"  [skip] {ep_id}: episode too short ({t_max - t_min:.0f}s)")
        return False
    grid = np.arange(t_min, t_max, STEP_S)
    T = len(grid)

    M = _build_M(metrics, grid)
    Tc = _build_T(case["tracets_lat"], case["tracets_err"], case["traces"], grid)
    L = _build_L(case["logs"], grid)

    # Fill structurally-absent features with 0 (not NaN):
    # M[:,6] = queue_length — not available in RCAEval Prometheus scrape
    # T[:,4] = retry_rate   — not available without raw span retry annotation
    # L[:,2] = semantic_anomaly — requires SentenceBERT (offline model)
    M[:, :, 6] = np.where(np.isnan(M[:, :, 6]), 0.0, M[:, :, 6])
    Tc[:, :, 4] = np.where(np.isnan(Tc[:, :, 4]), 0.0, Tc[:, :, 4])
    L[:, :, 2] = np.where(np.isnan(L[:, :, 2]), 0.0, L[:, :, 2])

    signal = np.concatenate([M, Tc, L], axis=2).astype(np.float32)
    A = _build_adjacency_from_metrics(metrics, grid)

    assert signal.shape == (T, N_SERVICES, 17), f"shape mismatch: {signal.shape}"
    assert A.shape == (T, N_SERVICES, N_SERVICES, 3)

    np.savez_compressed(out_dir / "signal.npz", signal=signal)
    np.savez_compressed(out_dir / "adjacency.npz", adjacency=A)
    (out_dir / "services.json").write_text(json.dumps(EWAT_SERVICES))

    # Labels
    inject_t = case["inject_time"]
    rc_svc = case["root_cause_ewat"]
    category = "anomaly"
    rows = []
    for t, ts in enumerate(grid):
        regime = "normal"
        is_inj = False
        if not math.isnan(inject_t) and ts >= inject_t:
            regime = "injection"
            is_inj = True
        rows.append({
            "timestamp": float(ts),
            "regime": regime,
            "category": category,
            "scenario": case["scenario"],
            "target_services": json.dumps([rc_svc] if rc_svc else []),
            "chaos_resource": case["fault_type"],
            "episode_id": ep_id,
            "drift_flag": False,
            "target_service": rc_svc,
            "is_injection": is_inj,
        })
    pd.DataFrame(rows).to_parquet(out_dir / "labels.parquet", index=False)

    nan_frac = float(np.isnan(signal).mean())
    t_min_f = float(grid[0])
    t_max_f = float(grid[-1]) + STEP_S
    inject_f = float(inject_t) if not math.isnan(inject_t) else t_min_f
    (out_dir / "metadata.json").write_text(json.dumps({
        "episode_id": ep_id,
        "scenario": {
            "name": case["scenario"],
            "category": FAULT_TO_CATEGORY.get(case["fault_type"], "anomaly"),
        },
        "boundaries": {
            "baseline_start": t_min_f,
            "baseline_end": inject_f,
            "injection_start": inject_f,
            "injection_end": t_max_f,
            "recovery_start": t_max_f,
            "recovery_end": t_max_f,
        },
        "canonical_services": EWAT_SERVICES,
        "fault_service": case["fault_service"],
        "fault_type": case["fault_type"],
        "root_cause_ewat": rc_svc,
        "inject_time": inject_f,
        "grid_step_s": STEP_S,
        "T": T, "N": N_SERVICES, "D": 17,
        "quality_snapshot": {"signal_nan_ratio": round(nan_frac, 4)},
        "source": "rcaeval_re2_ob",
    }, indent=2))

    print(f"  {ep_id}: T={T}  NaN={nan_frac:.1%}")
    return True


# ── Inspect mode ─────────────────────────────────────────────────────────────

def _inspect(re2_ob_dir: Path) -> None:
    instances = _find_instances(re2_ob_dir)
    print(f"Found {len(instances)} instances across {len(set(p.parent for p in instances))} fault types.\n")

    inst = instances[0]
    print(f"Sample: {inst.parent.name}/{inst.name}\n")

    metrics = _read_csv_safe(inst / "metrics.csv")
    if not metrics.empty:
        cols = [c for c in metrics.columns if c != "time"]
        svcs_found = set()
        for c in cols:
            svc, _ = _parse_metric_col(c)
            if svc:
                svcs_found.add(svc)
        print(f"metrics.csv: {len(metrics)} rows × {len(cols)+1} cols")
        print(f"  time range: {metrics['time'].min():.0f} – {metrics['time'].max():.0f}")
        print(f"  services detected: {sorted(svcs_found)}")
        print(f"  sample cols: {cols[:8]}")

    lat = _read_csv_safe(inst / "tracets_lat.csv")
    if not lat.empty:
        print(f"\ntracets_lat.csv: {len(lat)} rows × {len(lat.columns)} cols")
        print(f"  sample cols: {list(lat.columns[:6])}")

    err = _read_csv_safe(inst / "tracets_err.csv")
    if not err.empty:
        print(f"tracets_err.csv: {len(err)} rows × {len(err.columns)} cols")

    logs = _read_csv_safe(inst / "logs.csv")
    if not logs.empty:
        print(f"\nlogs.csv: {len(logs)} rows")
        print(f"  columns: {list(logs.columns)}")
        if "level" in logs.columns:
            print(f"  level counts: {logs['level'].value_counts().to_dict()}")

    it = inst / "inject_time.txt"
    print(f"\ninject_time: {it.read_text().strip() if it.exists() else 'MISSING'}")
    print(f"root_cause (from dir): {inst.parent.name}")

    ewat_mapped = {s: RCAEVAL_TO_EWAT.get(s, "(unmapped)") for s in sorted(svcs_found)}
    print(f"\nService mapping:\n" + "\n".join(f"  {k} → {v}" for k, v in ewat_mapped.items()))


# ── CLI ───────────────────────────────────────────────────────────────────────

ZENODO_RE2_OB_URL = "https://zenodo.org/records/14590730/files/RE2-OB.zip"


def _download(dest_dir: Path) -> Path:
    import subprocess, zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "RE2-OB.zip"

    if zip_path.exists() and zip_path.stat().st_size > 1e6:
        print(f"Archive already present: {zip_path}")
    else:
        print(f"Downloading RCAEval RE2-OB from Zenodo …")
        result = subprocess.run(
            ["wget", "-c", "-O", str(zip_path), ZENODO_RE2_OB_URL], check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"wget failed. Download manually: {ZENODO_RE2_OB_URL}")

    print(f"Extracting {zip_path} …")
    with __import__("zipfile").ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    subdirs = [p for p in dest_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    return subdirs[0] if len(subdirs) == 1 else dest_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adapt RCAEval RE2-OB → EWAT episode format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Inspect (already downloaded):
  python -m scripts.adapt_rcaeval --inspect --data-dir data/raw/rcaeval/RE2-OB

# Convert all 90 instances:
  python -m scripts.adapt_rcaeval --data-dir data/raw/rcaeval/RE2-OB \\
      --output data/features/rcaeval

# Convert only CPU fault cases:
  python -m scripts.adapt_rcaeval --data-dir data/raw/rcaeval/RE2-OB \\
      --faults cpu --output data/features/rcaeval

# Assemble dataset after conversion:
  python -m scripts.assemble_dataset \\
    --features-root data/features/rcaeval \\
    --output data/datasets/ewat_rcaeval --stratified
""",
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="RE2-OB directory (or download destination with --download)")
    p.add_argument("--download", action="store_true",
                   help="Download RE2-OB.zip from Zenodo into --data-dir first")
    p.add_argument("--output", type=Path, default=Path("data/features/rcaeval"))
    p.add_argument("--inspect", action="store_true",
                   help="Inspect first instance and exit")
    p.add_argument("--faults", nargs="*", default=None,
                   help="Filter to fault types (e.g. cpu mem delay)")
    p.add_argument("--services", nargs="*", default=None,
                   help="Filter to fault services (e.g. frontend cartservice)")
    p.add_argument("--max-cases", type=int, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    data_dir = args.data_dir
    if args.download:
        data_dir = _download(data_dir)
        print(f"Data ready at: {data_dir}\n")

    if not data_dir.exists():
        raise FileNotFoundError(
            f"--data-dir not found: {data_dir}\n"
            "  Use --download or point to the extracted RE2-OB folder."
        )

    if args.inspect:
        _inspect(data_dir)
        return

    instances = _find_instances(data_dir)
    if args.faults:
        instances = [p for p in instances
                     if any(p.parent.name.endswith("_" + f) for f in args.faults)]
    if args.services:
        instances = [p for p in instances
                     if any(p.parent.name.startswith(s) for s in args.services)]
    if args.max_cases:
        instances = instances[:args.max_cases]

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(instances)} instances → {args.output}\n")
    ok, skip = 0, 0
    for inst in instances:
        print(f"[{ok + skip + 1}/{len(instances)}] {inst.parent.name}/{inst.name}")
        case = _load_instance(inst)
        if _write_episode(case, args.output):
            ok += 1
        else:
            skip += 1

    print(f"\nDone: {ok} converted, {skip} skipped.")
    print(
        f"\nNext:\n"
        f"  python -m scripts.assemble_dataset \\\n"
        f"    --features-root {args.output} \\\n"
        f"    --output data/datasets/ewat_rcaeval --stratified"
    )


if __name__ == "__main__":
    main()
