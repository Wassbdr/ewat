"""EWAT v5 — Phase 2 : dumps bruts Train Ticket → tenseur S(t) ∈ ℝ^{N×T×17}.

Lit les dumps gzip produits par collect.probe (prometheus/jaeger/loki) et
construit le signal S(t) = [M(t) | T(t) | L(t)] sur une grille temporelle de
pas `step` secondes, agrégé par service TT.

Jeu de features « Lean enrichi » (cf. docs/dataset_v5_plan.md §0.5) :
  M(t) 7 : cpu, ram, disk_io, net, blkio, oom_events, restart_count
  T(t) 7 : span_dur_p99, latency_cv, error_rate, trace_depth, fan_out,
           abnormal_span_rate, retry_rate
  L(t) 3 : log_error_rate, log_semantic_anomaly(*), lexical_entropy
(*) log_semantic_anomaly : placeholder 0.0 ici (SentenceBERT branché en aval,
    coûteux ; on garde la colonne pour le schéma).

Sortie : <out>/features.npz  (S: (N,T,17) float32, services: (N,), feature_names,
         t_grid: (T,)).

Usage :
    python -m collect.build_features_tt --dump /tmp/ep_xxx --out /tmp/ep_xxx --step 30
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    # M(t)
    "cpu", "ram", "disk_io", "net", "blkio", "oom_events", "restart_count",
    # T(t)
    "span_dur_p99", "latency_cv", "error_rate", "trace_depth", "fan_out",
    "abnormal_span_rate", "retry_rate",
    # L(t)
    "log_error_rate", "log_semantic_anomaly", "lexical_entropy",
]
N_FEATURES = len(FEATURE_NAMES)

_POD_SUFFIX = re.compile(r"-[a-z0-9]+-[a-z0-9]+$")  # <svc>-<rs>-<pod>
_LEVEL_RE = re.compile(r'"s":"E"|\bERROR\b|\bSEVERE\b', re.IGNORECASE)
_WARN_RE = re.compile(r'"s":"W"|\bWARN\b', re.IGNORECASE)


def _svc_of_pod(pod: str) -> str:
    return _POD_SUFFIX.sub("", pod)


def _load(dump: Path, name: str):
    with gzip.open(dump / f"{name}.json.gz", "rt") as f:
        return json.load(f)


def _grid(t0: float, t1: float, step: int) -> np.ndarray:
    n = max(1, int(round((t1 - t0) / step)))
    return t0 + step * np.arange(n + 1)


def _bin_index(ts: float, t0: float, step: int, T: int) -> int:
    return min(T - 1, max(0, int((ts - t0) / step)))


# ───────────────────────── M(t) ─────────────────────────
def build_metrics(prom: dict, services: list[str], t_grid: np.ndarray, step: int) -> dict:
    """Agrège les séries cAdvisor par service sur la grille. Saturation→max, IO→somme."""
    T = len(t_grid) - 1
    t0 = t_grid[0]
    idx = {s: i for i, s in enumerate(services)}
    # init
    M = {feat: np.full((len(services), T), np.nan) for feat in
         ["cpu", "ram", "net", "disk_io", "blkio", "oom_events", "restart_count"]}

    def accum(metric_keys: list[str], feat: str, agg: str):
        """Agrège une ou plusieurs familles de métriques vers `feat`.

        Plusieurs clés (ex. net_rx+net_tx) sont d'abord additionnées par
        (série, point) conceptuellement : on collecte toutes leurs valeurs dans
        le même bucket puis on applique l'agrégation intra-service (max/sum).
        """
        buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
        for metric_key in metric_keys:
            series = prom.get(metric_key)
            if not isinstance(series, list):
                continue
            for s in series:
                pod = s.get("metric", {}).get("pod", "")
                svc = _svc_of_pod(pod)
                if svc not in idx:
                    continue
                for ts, val in s.get("values", []):
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    b = _bin_index(float(ts), t0, step, T)
                    buckets[(idx[svc], b)].append(v)
        for (si, b), vals in buckets.items():
            if agg == "max":
                M[feat][si, b] = max(vals)
            elif agg == "sum":
                M[feat][si, b] = sum(vals)
            elif agg == "mean":
                M[feat][si, b] = sum(vals) / len(vals)

    accum(["cpu"], "cpu", "max")            # saturation → max
    accum(["ram"], "ram", "max")            # saturation → max
    accum(["net_rx", "net_tx"], "net", "sum")        # rx + tx
    accum(["fs_reads", "fs_writes"], "disk_io", "sum")  # reads + writes
    accum(["blkio"], "blkio", "sum")
    accum(["oom"], "oom_events", "sum")
    accum(["restarts"], "restart_count", "max")
    return M


# ───────────────────────── T(t) ─────────────────────────
def build_traces(jae: dict, services: list[str], t_grid: np.ndarray, step: int) -> dict:
    T = len(t_grid) - 1
    t0 = t_grid[0]
    idx = {s: i for i, s in enumerate(services)}
    # collecte par (service, bin)
    durs: dict[tuple[int, int], list[float]] = defaultdict(list)
    errs: dict[tuple[int, int], list[int]] = defaultdict(list)
    children: dict[tuple[int, int], list[int]] = defaultdict(list)
    depths: dict[tuple[int, int], list[int]] = defaultdict(list)
    ops: dict[tuple[int, int], list[str]] = defaultdict(list)

    for svc, trs in jae.get("traces", {}).items():
        if not isinstance(trs, list):
            continue
        for tr in trs:
            procs = tr.get("processes", {})
            spans = tr.get("spans", [])
            # index spanID -> children count + depth
            child_count: dict[str, int] = defaultdict(int)
            parent: dict[str, str] = {}
            for sp in spans:
                for ref in sp.get("references", []):
                    if ref.get("refType") == "CHILD_OF":
                        parent[sp["spanID"]] = ref.get("spanID")
                        child_count[ref.get("spanID")] += 1
            for sp in spans:
                pid = sp.get("processID")
                sname = procs.get(pid, {}).get("serviceName", "")
                if sname not in idx:
                    continue
                b = _bin_index(sp.get("startTime", 0) / 1e6, t0, step, T)
                key = (idx[sname], b)
                durs[key].append(sp.get("duration", 0) / 1000.0)  # µs→ms
                # error from http.status_code tag
                code = 0
                for tag in sp.get("tags", []):
                    if tag.get("key") == "http.status_code":
                        try:
                            code = int(tag.get("value", 0))
                        except (TypeError, ValueError):
                            code = 0
                errs[key].append(1 if code >= 400 else 0)
                children[key].append(child_count.get(sp["spanID"], 0))
                # depth: remonter la chaîne parent
                d, cur, guard = 0, sp["spanID"], 0
                while cur in parent and guard < 50:
                    cur = parent[cur]; d += 1; guard += 1
                depths[key].append(d)
                ops[key].append(sp.get("operationName", ""))

    Tt = {f: np.full((len(services), T), np.nan) for f in
          ["span_dur_p99", "latency_cv", "error_rate", "trace_depth", "fan_out",
           "abnormal_span_rate", "retry_rate"]}
    for key, dl in durs.items():
        si, b = key
        arr = np.array(dl)
        Tt["span_dur_p99"][si, b] = np.percentile(arr, 99)
        Tt["latency_cv"][si, b] = (arr.std() / arr.mean()) if arr.mean() > 0 else 0.0
        Tt["error_rate"][si, b] = np.mean(errs[key]) if errs[key] else 0.0
        Tt["abnormal_span_rate"][si, b] = np.mean(errs[key]) if errs[key] else 0.0
        Tt["trace_depth"][si, b] = np.median(depths[key]) if depths[key] else 0.0
        Tt["fan_out"][si, b] = np.median(children[key]) if children[key] else 0.0
        # retry_rate : opérations répétées dans le bin
        opl = ops[key]
        Tt["retry_rate"][si, b] = (1 - len(set(opl)) / len(opl)) if opl else 0.0
    return Tt


# ───────────────────────── L(t) ─────────────────────────
def _lexical_entropy(lines: list[str]) -> float:
    if not lines:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for ln in lines:
        for tok in re.findall(r"[A-Za-z]+", ln):
            counts[tok.lower()] += 1; total += 1
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def build_logs(loki: dict, services: list[str], t_grid: np.ndarray, step: int) -> dict:
    T = len(t_grid) - 1
    t0 = t_grid[0]
    idx = {s: i for i, s in enumerate(services)}
    by: dict[tuple[int, int], list[str]] = defaultdict(list)
    for st in loki.get("streams", []):
        # le label Loki `app` est déjà le nom de service propre (ex.
        # ts-order-other-service) — ne PAS lui appliquer _svc_of_pod.
        svc = st.get("stream", {}).get("app", "")
        if svc not in idx:
            continue
        for ts_ns, line in st.get("values", []):
            b = _bin_index(int(ts_ns) / 1e9, t0, step, T)
            by[(idx[svc], b)].append(line)
    L = {f: np.full((len(services), T), np.nan) for f in
         ["log_error_rate", "log_semantic_anomaly", "lexical_entropy"]}
    for (si, b), lines in by.items():
        n = len(lines)
        L["log_error_rate"][si, b] = sum(bool(_LEVEL_RE.search(x)) for x in lines) / n if n else 0.0
        L["lexical_entropy"][si, b] = _lexical_entropy(lines)
        L["log_semantic_anomaly"][si, b] = 0.0  # SentenceBERT en aval
    return L


def impute(S: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Impute le NaN avec une sémantique d'absence d'activité.

    - M(t) saturation (cpu, ram) : forward-fill puis back-fill par service
      (un scrape manqué ≠ ressource nulle) ; reste → 0.
    - M(t) IO/compteurs (disk_io, net, blkio, oom, restart) : NaN → 0 (pas d'IO/évt).
    - T(t) et L(t) : NaN → 0 (aucun span / aucun log = aucune activité).

    Un service présent mais non sollicité dans un bin a donc une activité 0,
    pas une valeur manquante — convention standard sur Train Ticket
    (Eadro, RCAEval RE2-TT).
    """
    S = S.copy()
    gauges = {"cpu", "ram"}
    for fi, fname in enumerate(feature_names):
        plane = S[:, :, fi]
        if fname in gauges:
            # forward-fill puis back-fill le long du temps, par service
            for si in range(plane.shape[0]):
                row = plane[si]
                if np.isnan(row).all():
                    row[:] = 0.0
                    continue
                last = np.nan
                for t in range(len(row)):
                    if np.isnan(row[t]):
                        row[t] = last
                    else:
                        last = row[t]
                # back-fill le début
                first_valid = np.argmax(~np.isnan(row)) if (~np.isnan(row)).any() else 0
                row[:first_valid] = row[first_valid]
            plane[np.isnan(plane)] = 0.0
        else:
            plane[np.isnan(plane)] = 0.0
        S[:, :, fi] = plane
    return S


def build(dump: Path, step: int, services: list[str] | None = None) -> dict:
    prom = _load(dump, "prometheus")
    jae = _load(dump, "jaeger")
    loki = _load(dump, "loki")

    # liste de services : services applicatifs TT uniquement (noeuds du graphe
    # d'appel). On exclut les bases (-mongo/-mysql) qui n'émettent ni trace ni
    # log applicatif → sinon NaN T(t)/L(t) artificiellement gonflé.
    if services is None:
        svset = set()
        for s in prom.get("cpu", []):
            pod = s.get("metric", {}).get("pod", "")
            svc = _svc_of_pod(pod)
            if svc.startswith("ts-") and not svc.endswith(("-mongo", "-mysql")):
                svset.add(svc)
        services = sorted(svset)

    # grille temporelle depuis les bornes Prometheus
    all_ts = [float(v[0]) for s in prom.get("cpu", []) if isinstance(s, dict)
              for v in s.get("values", [])]
    t0, t1 = (min(all_ts), max(all_ts)) if all_ts else (0, step)
    t_grid = _grid(t0, t1, step)
    T = len(t_grid) - 1

    M = build_metrics(prom, services, t_grid, step)
    Tt = build_traces(jae, services, t_grid, step)
    L = build_logs(loki, services, t_grid, step)

    N = len(services)
    S = np.full((N, T, N_FEATURES), np.nan, dtype=np.float32)
    allf = {**M, **Tt, **L}
    for fi, fname in enumerate(FEATURE_NAMES):
        if fname in allf:
            S[:, :, fi] = allf[fname]
    # raw_nan : taux de NaN avant imputation (traçabilité qualité)
    raw_nan = float(np.isnan(S).mean())
    S_imp = impute(S, list(FEATURE_NAMES))
    return {"S": S_imp, "S_raw": S, "services": np.array(services),
            "feature_names": np.array(FEATURE_NAMES), "t_grid": t_grid[:-1],
            "raw_nan_frac": np.array([raw_nan])}


def main() -> None:
    p = argparse.ArgumentParser(description="EWAT v5 TT feature builder")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=30)
    args = p.parse_args()
    res = build(Path(args.dump), args.step)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", **res)
    S = res["S"]
    S_raw = res["S_raw"]
    print(f"S(t) shape = {S.shape}  (N={S.shape[0]} services, T={S.shape[1]} steps, F={S.shape[2]})")
    print(f"NaN après imputation = {100 * np.isnan(S).mean():.1f}%  |  NaN brut (avant) = {100 * np.isnan(S_raw).mean():.1f}%")
    print("NaN brut par feature (avant imputation) :")
    S = S_raw
    for fi, fname in enumerate(FEATURE_NAMES):
        pct = 100 * np.isnan(S[:, :, fi]).mean()
        print(f"  {fname:<22} {pct:5.1f}%")
    print(f"écrit : {out/'features.npz'}")


if __name__ == "__main__":
    main()
