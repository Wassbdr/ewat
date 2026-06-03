"""EWAT v5 — Phase 2 : dumps Train Ticket → contrat per-épisode v4-conforme.

Construit S(t) ∈ ℝ^{T×N×17}, le masque, et G(t) ∈ ℝ^{T×N×N×3} en RÉUTILISANT
les modules matures du repo (graphe + indices de trace), de sorte que
`scripts/assemble_dataset.py` et `scripts/validate_dataset.py` fonctionnent
sans modification.

Sourcing TT v5.1, 18 features (pas d'Istio/OTel HTTP → latence/erreur via traces) :
  M[0] cpu_util, M[1] ram_util            ← cAdvisor (dump Prometheus)
  M[2] latency_p99                        ← SpanLatencyIndex (traces)
  M[3] error_rate_http                    ← SpanErrorRateIndex (traces)
  M[4] net_sat, M[5] disk_io              ← cAdvisor
  M[6] mem_limit_ratio                    ← cAdvisor working_set/limite (saturation)
  M[7] jvm_heap_ratio, M[8] jvm_gc_util,
  M[9] jvm_threads_blocked                ← jmx_prometheus_javaagent (annotations)
  T[10] abnormal_span_rate, T[11] trace_depth,
  T[12] fan_out, T[13] latency_cv         ← TraceCollector (modules existants)
  L[14] log_error_rate, L[17] lexical_entropy  ← Loki
  L[15] restart_count                     ← kube-state-metrics (dump Prometheus)
  L[16] semantic_anomaly                  ← SentenceBERT (collect/semantic.py)
G(t) ← compute_graph_for_window (volume, latence médiane, taux d'erreur).

Sortie : signal.npz, signal_mask.npz, adjacency.npz, labels.parquet,
services.json, metadata.json, graph_stats.csv, feature_provenance.json.

Usage (PYTHONPATH inclut src/) :
    python -m collect.build_features_v5 --dump <ep_dir> --out <ep_dir> \
        --episode-id <id> --scenario cpu_stress --category contention --step 30
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# src/ doit être sur le PYTHONPATH (le repo l'ajoute via conftest/scripts ;
# ici on l'insère défensivement).
_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from telemetry.collectors.trace_collector import TraceCollector  # noqa: E402
from telemetry.extractors.traces_file import (  # noqa: E402
    InMemorySpanBackend,
    SpanErrorRateIndex,
    SpanLatencyIndex,
    apply_aliases,
    compute_graph_for_window,
    parse_jaeger_dump,
)

try:
    from graph.diagnostics import compute_stats as _graph_stats
except Exception:  # diagnostics optionnel
    _graph_stats = None

# Schéma v5.1 (18 features). Évolution vs v5.0 : suppression de span_dur_p99
# (≡ latency_p99, ρ=1.0) et retry_rate (structurellement mort sur TT) ; ajout de
# 3 features JVM (le signal manquant pour un système Spring Boot + bugs F).
# M[6] : mem_limit_ratio remplace oom_events (container_oom_events_total lit 0
# partout sur observit-cluster1 — vérifié 2026-06-02 ; saturation mémoire utile).
FEATURE_NAMES = [
    # M(t) infra + JVM (0-9)
    "cpu_util", "ram_util", "latency_p99", "error_rate_http", "net_sat",
    "disk_io", "mem_limit_ratio", "jvm_heap_ratio", "jvm_gc_util", "jvm_threads_blocked",
    # T(t) traces (10-13)
    "abnormal_span_rate", "trace_depth", "fan_out", "latency_cv",
    # L(t) logs (14-17)
    "log_error_rate", "restart_count", "semantic_anomaly", "lexical_entropy",
]
N_FEATURES = 18
SCHEMA_VERSION = "v5.1"
# Tranches modales (pour quality_snapshot et l'analyse)
M_SLICE, T_SLICE, L_SLICE = slice(0, 10), slice(10, 14), slice(14, 18)

_LEVEL_RE = re.compile(r'"s":"E"|\bERROR\b|\bSEVERE\b|\bFATAL\b')


def _grid(t0: float, t1: float, step: int) -> np.ndarray:
    n = max(1, int(round((t1 - t0) / step)))
    return t0 + step * np.arange(n + 1)


def _bin(ts: float, t0: float, step: int, T: int) -> int:
    return min(T - 1, max(0, int((ts - t0) / step)))


# ───────────────────────── M(t) cAdvisor ─────────────────────────
def _metrics_cadvisor(prom: dict, services: list[str], idx: dict, t_grid, step):
    """cpu/ram (max), net/disk (somme rx+tx, reads+writes), oom (somme),
    restart (max), depuis le dump Prometheus du probe."""
    T = len(t_grid) - 1
    t0 = t_grid[0]
    out = {f: np.full((len(services), T), np.nan, np.float32)
           for f in ["cpu_util", "ram_util", "net_sat", "disk_io", "mem_limit",
                     "restart_count", "jvm_heap_used", "jvm_heap_max",
                     "jvm_gc_util", "jvm_threads_blocked"]}

    def accum(keys, feat, agg):
        buckets = defaultdict(list)
        for k in keys:
            for s in prom.get(k, []) if isinstance(prom.get(k), list) else []:
                pod = s.get("metric", {}).get("pod", "")
                svc = re.sub(r"-[a-z0-9]+-[a-z0-9]+$", "", pod)
                if svc not in idx:
                    continue
                for ts, val in s.get("values", []):
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    buckets[(idx[svc], _bin(float(ts), t0, step, T))].append(v)
        for (si, b), vals in buckets.items():
            out[feat][si, b] = max(vals) if agg == "max" else sum(vals)

    accum(["cpu"], "cpu_util", "max")
    accum(["ram"], "ram_util", "max")
    accum(["net_rx", "net_tx"], "net_sat", "sum")
    accum(["fs_reads", "fs_writes"], "disk_io", "sum")
    accum(["mem_limit"], "mem_limit", "max")
    accum(["restarts"], "restart_count", "max")
    # JVM (jmx_prometheus_javaagent) — saturation → max
    accum(["jvm_heap_used"], "jvm_heap_used", "max")
    accum(["jvm_heap_max"], "jvm_heap_max", "max")
    accum(["jvm_gc_sum"], "jvm_gc_util", "max")
    accum(["jvm_threads_blocked"], "jvm_threads_blocked", "max")
    # jvm_heap_ratio = used / max (par cellule), borné [0,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = out["jvm_heap_used"] / out["jvm_heap_max"]
    out["jvm_heap_ratio"] = np.clip(ratio, 0.0, 1.0).astype(np.float32)
    # mem_limit_ratio = working_set / limite conteneur ∈ [0,1] (saturation mémoire).
    # Remplace oom_events (nul sur ce cluster). Capte memory_stress/pressure (numérateur
    # monte) et F3 (dénominateur abaissé 500→250Mi → ratio → 1.0).
    with np.errstate(divide="ignore", invalid="ignore"):
        mem_ratio = out["ram_util"] / out["mem_limit"]
    out["mem_limit_ratio"] = np.clip(mem_ratio, 0.0, 1.0).astype(np.float32)
    return out


# ───────────────────────── L(t) Loki ─────────────────────────
def _lex_entropy(lines):
    counts, total = defaultdict(int), 0
    for ln in lines:
        for tok in re.findall(r"[A-Za-z]+", ln):
            counts[tok.lower()] += 1; total += 1
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _logs(loki: dict, services: list[str], idx: dict, t_grid, step):
    T = len(t_grid) - 1
    t0 = t_grid[0]
    by = defaultdict(list)
    for st in loki.get("streams", []):
        svc = st.get("stream", {}).get("app", "")
        if svc not in idx:
            continue
        for ts_ns, line in st.get("values", []):
            by[(idx[svc], _bin(int(ts_ns) / 1e9, t0, step, T))].append(line)
    out = {f: np.full((len(services), T), np.nan, np.float32)
           for f in ["log_error_rate", "semantic_anomaly", "lexical_entropy"]}
    for (si, b), lines in by.items():
        n = len(lines)
        out["log_error_rate"][si, b] = sum(bool(_LEVEL_RE.search(x)) for x in lines) / n
        out["lexical_entropy"][si, b] = _lex_entropy(lines)
    return out, by  # by = buckets (svc_idx, bin) -> lines, pour l'anomalie sémantique


def impute(S, mask, names):
    """NaN → activité 0 (T/L et compteurs M) ; gauges cpu/ram = forward+back-fill
    temporel par service (un scrape manqué ≠ ressource nulle), reste → 0.

    S est de forme (T, N, F) : axe 0 = temps, axe 1 = service.
    """
    S = S.copy()
    # gauges = saturation (forward-fill si scrape manqué). jvm_threads_blocked est
    # un compteur transitoire (0 = pas de contention) → NaN→0, pas de forward-fill.
    gauges = {"cpu_util", "ram_util", "jvm_heap_ratio", "mem_limit_ratio"}
    for fi, fn in enumerate(names):
        plane = S[:, :, fi]  # (T, N)
        if fn in gauges:
            for si in range(plane.shape[1]):          # par service (colonne)
                col = plane[:, si]
                if np.isnan(col).all():
                    col[:] = 0.0
                    continue
                last = np.nan
                for t in range(len(col)):             # forward-fill
                    if np.isnan(col[t]):
                        col[t] = last
                    else:
                        last = col[t]
                first = int(np.argmax(~np.isnan(col))) if (~np.isnan(col)).any() else 0
                col[:first] = col[first]              # back-fill le début
                plane[:, si] = col
        plane[np.isnan(plane)] = 0.0
        S[:, :, fi] = plane
    return S


def build(dump: Path, services: list[str], step: int, aliases: dict | None = None,
          with_semantic: bool = True):
    aliases = aliases or {}
    prom = json.load(gzip.open(dump / "prometheus.json.gz", "rt"))
    jae = json.load(gzip.open(dump / "jaeger.json.gz", "rt"))
    loki = json.load(gzip.open(dump / "loki.json.gz", "rt"))

    idx = {s: i for i, s in enumerate(services)}
    N = len(services)

    # grille temporelle depuis Prometheus
    all_ts = [float(v[0]) for s in prom.get("cpu", []) if isinstance(s, dict)
              for v in s.get("values", [])]
    if not all_ts:
        raise SystemExit("dump Prometheus vide (pas de série cpu)")
    t0, t1 = min(all_ts), max(all_ts)
    t_grid = _grid(t0, t1, step)
    T = len(t_grid) - 1

    # spans + indices de trace (modules existants)
    spans = parse_jaeger_dump(jae)
    apply_aliases(spans, aliases)
    lat_idx = SpanLatencyIndex(jae, services, aliases=aliases)
    err_idx = SpanErrorRateIndex(jae, services, aliases=aliases, grpc_callee_map={})
    trace_collector = TraceCollector(InMemorySpanBackend(spans), window_s=step,
                                     services=services, aliases={})

    # tenseurs
    S = np.full((T, N, N_FEATURES), np.nan, np.float32)
    adjacency = np.zeros((T, N, N, 3), np.float32)

    M = _metrics_cadvisor(prom, services, idx, t_grid, step)
    L, log_buckets = _logs(loki, services, idx, t_grid, step)

    # M cAdvisor + JVM (transpose service×T → T×service)
    for fname in ["cpu_util", "ram_util", "net_sat", "disk_io", "mem_limit_ratio",
                  "restart_count", "jvm_heap_ratio", "jvm_gc_util", "jvm_threads_blocked"]:
        S[:, :, FEATURE_NAMES.index(fname)] = M[fname].T
    for fname in ["log_error_rate", "lexical_entropy"]:
        S[:, :, FEATURE_NAMES.index(fname)] = L[fname].T

    # anomalie sémantique (SentenceBERT) — centroïde normal = premier quart des
    # bins (toujours en phase baseline/pre). Désactivable via with_semantic=False.
    fi_sem = FEATURE_NAMES.index("semantic_anomaly")
    if with_semantic:
        try:
            from collect.semantic import compute_semantic
            baseline_bins = set(range(max(1, T // 4)))
            sem = compute_semantic(log_buckets, services, T, baseline_bins)  # (N,T)
            S[:, :, fi_sem] = sem.T
        except Exception as e:  # pragma: no cover
            print(f"[semantic] désactivé ({e})", flush=True)

    # par fenêtre : latence/erreur (traces), T(t), G(t)
    svc_index = {s: i for i, s in enumerate(services)}
    fi_lat, fi_err = FEATURE_NAMES.index("latency_p99"), FEATURE_NAMES.index("error_rate_http")
    # TraceCollector renvoie 6 colonnes : [span_dur_p99, abnormal_span_rate,
    # trace_depth, fan_out, retry_rate, latency_cv]. v5.1 garde [1,2,3,5]
    # (drop span_dur_p99 ≡ latence, drop retry_rate mort) → indices 10-13.
    TRACE_COLS = [1, 2, 3, 5]
    for b in range(T):
        ws, we = float(t_grid[b]), float(t_grid[b + 1])
        p99 = lat_idx.p99_for_window(ws, we)
        er = err_idx.error_rate_for_window(ws, we)
        for s, i in idx.items():
            v = p99.get(s, float("nan"))
            if v == v:
                S[b, i, fi_lat] = v
            e = er.get(s, float("nan"))
            if e == e:
                S[b, i, fi_err] = e
        # T(t) via collecteur existant (fenêtre = [we-step, we])
        T_t, _ = trace_collector.collect(timestamp=we, service_index=svc_index)
        S[b, :, T_SLICE] = T_t[:, TRACE_COLS]
        # G(t)
        adjacency[b] = compute_graph_for_window(spans, services, ws, we)

    mask = ~np.isnan(S)  # True = présent
    S_imp = impute(S, mask, FEATURE_NAMES)
    return {
        "signal": S_imp, "signal_raw": S, "missing_mask": ~mask,
        "adjacency": adjacency, "services": services, "t_grid": t_grid[:-1],
        "feature_names": FEATURE_NAMES,
    }


def write_episode(res: dict, out: Path, episode_id: str, scenario: str, category: str,
                  target_services: list[str], chaos_resource: str, boundaries: dict,
                  regime: np.ndarray, intensity: np.ndarray, fault_type: str,
                  bug_id: str | None, held_out: bool, step: int):
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    S = res["signal"]
    T, N, _ = S.shape
    np.savez_compressed(out / "signal.npz", signal=S)
    np.savez_compressed(out / "signal_raw.npz", signal_raw=res["signal_raw"])
    np.savez_compressed(out / "signal_mask.npz", missing_mask=res["missing_mask"])
    np.savez_compressed(out / "adjacency.npz", adjacency=res["adjacency"])
    json.dump(res["services"], open(out / "services.json", "w"), indent=1)

    drift = category in ("drift", "overlap")
    rows = []
    for t in range(T):
        rows.append({
            "timestamp": float(res["t_grid"][t]),
            "regime": str(regime[t]),
            "category": category,
            "scenario": scenario,
            "target_services": json.dumps(target_services),
            "target_service": target_services[0] if target_services else "",
            "chaos_resource": chaos_resource,
            "episode_id": episode_id,
            "drift_flag": bool(drift),
            "is_injection": regime[t] == "injection",
            "intensity_t": float(intensity[t]),
            "fault_type": fault_type,
            "bug_id": bug_id or "",
            "held_out_flag": bool(held_out),
        })
    pd.DataFrame(rows).to_parquet(out / "labels.parquet")

    raw = res["signal_raw"]
    qsnap = {
        "signal_nan_ratio": float(np.isnan(raw).mean()),
        "metrics_nan_ratio": float(np.isnan(raw[:, :, M_SLICE]).mean()),
        "traces_nan_ratio": float(np.isnan(raw[:, :, T_SLICE]).mean()),
        "logs_nan_ratio": float(np.isnan(raw[:, :, L_SLICE]).mean()),
    }
    meta = {
        "episode_id": episode_id,
        "scenario": {"name": scenario, "category": category, "file": chaos_resource,
                     "targets": target_services, "description": "", "duration_nominal_s": 0.0},
        "boundaries": boundaries,
        "canonical_services": res["services"],
        "feature_set": "v5",
        "grid_step_s": float(step),
        "dataset_schema_version": SCHEMA_VERSION,
        "signal_feature_names": FEATURE_NAMES,
        "quality_snapshot": qsnap,
        "artifacts": {
            "signal": {"path": "signal.npz", "key": "signal", "shape": [T, N, N_FEATURES], "dtype": "float32"},
            "signal_mask": {"path": "signal_mask.npz", "key": "missing_mask", "shape": [T, N, N_FEATURES], "dtype": "bool"},
            "adjacency": {"path": "adjacency.npz", "key": "adjacency", "shape": [T, N, N, 3], "dtype": "float32"},
            "labels": {"path": "labels.parquet", "n_rows": T},
            "services": {"path": "services.json", "n_services": N},
        },
    }
    json.dump(meta, open(out / "metadata.json", "w"), indent=2)

    # graph_stats.csv
    if _graph_stats is not None:
        try:
            import csv
            stats = []
            for b in range(T):
                adj = res["adjacency"][b]
                n_edges = int((adj[:, :, 0] > 0).sum())
                stats.append({"timestep": b, "n_edges": n_edges,
                              "density": n_edges / max(1, N * (N - 1))})
            with open(out / "graph_stats.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["timestep", "n_edges", "density"])
                w.writeheader(); w.writerows(stats)
        except Exception:
            pass
    json.dump({"builder": "build_features_v5", "schema": SCHEMA_VERSION},
              open(out / "feature_provenance.json", "w"), indent=2)
    return qsnap


def _load_services() -> list[str]:
    return json.load(open(Path(__file__).parent / "tt_services.json"))


def build_episode(ep_dir: Path, services: list[str] | None = None,
                  step: int | None = None, with_semantic: bool = True) -> dict:
    """Phase 2 OFFLINE : un dossier d'épisode (dumps + episode_meta.json) →
    contrat v4 complet (signal/mask/adjacency/labels/metadata...).

    Lit `episode_meta.json` (écrit par run_episode) pour les boundaries + meta,
    calcule régime + intensity_t par step, écrit tout le contrat. Idempotent :
    saute si signal.npz existe déjà et pas de --force (géré par le batch).
    """
    services = services or _load_services()
    meta = json.load(open(ep_dir / "episode_meta.json"))
    step = step or int(meta.get("step", 30))

    res = build(ep_dir, services, step, with_semantic=with_semantic)
    T = res["signal"].shape[0]
    rel = res["t_grid"] - res["t_grid"][0]
    br = meta["boundaries_rel"]
    inj0, inj1, ramp_s = br["injection_start"], br["injection_end"], meta.get("ramp_s", 0)
    regime = np.array(["normal"] * T, dtype=object)
    intensity = np.zeros(T)
    for i, g in enumerate(rel):
        if inj0 <= g < inj1:
            regime[i] = "injection"
            into = g - inj0
            intensity[i] = min(1.0, into / ramp_s) if ramp_s > 0 else 1.0
        elif g >= inj1:
            regime[i] = "recovery"
    t0 = res["t_grid"][0]
    bnd = {"baseline_start": t0, "injection_start": inj0 + t0,
           "injection_end": inj1 + t0, "recovery_end": br["recovery_end"] + t0}
    q = write_episode(
        res, ep_dir, meta["episode_id"], meta["scenario"], meta.get("category", "unknown"),
        meta.get("targets", []), meta.get("chaos_resource", ""), bnd, regime, intensity,
        "bug" if meta.get("is_bug") else "chaos", meta.get("bug_id"),
        bool(meta.get("held_out")), step)
    return {"episode_id": meta["episode_id"], "shape": list(res["signal"].shape),
            "raw_nan": q["signal_nan_ratio"],
            "g_edges": int((res["adjacency"][:, :, :, 0] > 0).sum()),
            "regime_counts": {r: int((regime == r).sum()) for r in ["normal", "injection", "recovery"]}}


# Worker module-level (picklable pour ProcessPoolExecutor). (ep_str, force, sem)
def _build_one_worker(spec: tuple) -> str | None:
    ep_str, force, sem = spec
    ep = Path(ep_str)
    if not (ep / "episode_meta.json").exists():
        return None
    if (ep / "signal.npz").exists() and not force:
        return f"skip {ep.name} (déjà buildé)"
    try:
        r = build_episode(ep, _load_services(), with_semantic=sem)
        return f"OK {r['episode_id']} {r['shape']} edges={r['g_edges']} raw_nan={r['raw_nan']:.0%}"
    except Exception as e:
        return f"FAIL {ep.name}: {e}"


def main() -> None:
    p = argparse.ArgumentParser(description="EWAT v5 feature builder (Phase 2 offline)")
    p.add_argument("--episode", help="un dossier d'épisode (dumps + episode_meta.json)")
    p.add_argument("--raw-root", help="racine : build tous les épisodes (batch)")
    p.add_argument("--workers", type=int, default=4, help="processus parallèles (batch)")
    p.add_argument("--force", action="store_true", help="rebuild même si signal.npz existe")
    p.add_argument("--no-semantic", action="store_true")
    args = p.parse_args()
    sem = not args.no_semantic

    if args.episode:
        print(_build_one_worker((args.episode, args.force, sem)))
    elif args.raw_root:
        eps = [str(p) for p in sorted(Path(args.raw_root).iterdir()) if p.is_dir()]
        print(f"build batch : {len(eps)} dossiers, {args.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        specs = [(e, args.force, sem) for e in eps]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for msg in ex.map(_build_one_worker, specs):
                if msg:
                    print(msg, flush=True)
    else:
        p.error("--episode ou --raw-root requis")


if __name__ == "__main__":
    main()
