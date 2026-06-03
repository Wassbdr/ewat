"""EWAT v5 — sonde de collecte télémétrie Train Ticket (Phase 1, dumps bruts).

Prouve et exécute le pull des 3 sources S(t) pour le namespace `tt` sur une
fenêtre [start, end], via les endpoints confirmés le 2026-06-01 :
  - M(t) Prometheus : monitoring/monitoring-kube-prometheus-prometheus:9090 (cAdvisor)
  - T(t) Jaeger     : tt/jaeger-query:16686
  - L(t) Loki       : monitoring-metrics/loki:3100 (promtail cluster-wide)

Suppose des port-forwards déjà ouverts (cf. constantes PORTS) ; sinon les ouvre.
Écrit les dumps bruts gzip dans <out>/{prometheus,jaeger,loki}.json.gz —
mêmes noms que record_episode pour réutiliser la Phase 2 en aval.

Usage :
    python -m collect.probe --out /tmp/ep_test --window 120 --step 30
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

NAMESPACE = "tt"  # défaut mono-runner

# Contexte kubectl épinglé sur les port-forwards (immunise contre une bascule de
# contexte : un pf vers le mauvais cluster échouerait silencieusement). Cf. inject.py.
_KCTX = os.environ.get("V5_KUBE_CONTEXT", "observit-cluster1")


def pf_config(ns: str = NAMESPACE, offset: int = 0) -> dict:
    """Config des 3 port-forwards pour un runner.

    Prometheus & Loki sont partagés (`monitoring-metrics`) ; Jaeger est
    par-namespace (`<ns>/jaeger-query`). Les ports locaux sont décalés de
    `offset` pour que PLUSIEURS runners coexistent sur le même poste sans
    collision (tt: offset 0 → 19090/16686/13100 ; tt-b: offset 10 → 19100/...).
    """
    return {
        "prometheus": {"svc": "svc/prometheus-server", "ns": "monitoring-metrics",
                       "local": 19090 + offset, "remote": 80},
        "jaeger": {"svc": "svc/jaeger-query", "ns": ns,
                   "local": 16686 + offset, "remote": 16686},
        "loki": {"svc": "svc/loki", "ns": "monitoring-metrics",
                 "local": 13100 + offset, "remote": 3100},
    }


def _prom_queries(ns: str = NAMESPACE) -> dict:
    """Requêtes M(t) (cAdvisor + kube-state + JVM) filtrées sur `ns`."""
    return {
        "cpu": f'rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!=""}}[2m])',
        "ram": f'container_memory_working_set_bytes{{namespace="{ns}",container!=""}}',
        "net_rx": f'rate(container_network_receive_bytes_total{{namespace="{ns}"}}[2m])',
        "net_tx": f'rate(container_network_transmit_bytes_total{{namespace="{ns}"}}[2m])',
        "fs_reads": f'rate(container_fs_reads_bytes_total{{namespace="{ns}",container!=""}}[2m])',
        "fs_writes": f'rate(container_fs_writes_bytes_total{{namespace="{ns}",container!=""}}[2m])',
        "blkio": f'rate(container_blkio_device_usage_total{{namespace="{ns}"}}[2m])',
        # saturation mémoire = working_set / limite conteneur (cf. build_features_v5).
        # Remplace container_oom_events_total, qui lit 0 partout sur observit-cluster1
        # (cAdvisor/kernel ne surface pas ce compteur — vérifié 2026-06-02).
        "mem_limit": f'container_spec_memory_limit_bytes{{namespace="{ns}",container!=""}}',
        "restarts": f'kube_pod_container_status_restarts_total{{namespace="{ns}"}}',
        # JVM (jmx_prometheus_javaagent scrapé par pod-annotation discovery)
        "jvm_heap_used": f'jvm_memory_bytes_used{{namespace="{ns}",area="heap"}}',
        "jvm_heap_max": f'jvm_memory_bytes_max{{namespace="{ns}",area="heap"}}',
        "jvm_gc_sum": f'rate(jvm_gc_collection_seconds_sum{{namespace="{ns}"}}[2m])',
        # threads BLOQUÉS (contention) — voir build_features_v5 (ρ ram_util faible).
        "jvm_threads_blocked": f'jvm_threads_state{{namespace="{ns}",state="BLOCKED"}}',
    }


def _get(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


_READY_URL = {
    "prometheus": "http://127.0.0.1:{local}/-/ready",
    "jaeger": "http://127.0.0.1:{local}/api/services",
    "loki": "http://127.0.0.1:{local}/ready",
}


def _pf_one(cfg: dict) -> subprocess.Popen:
    return subprocess.Popen(
        ["kubectl", "--context", _KCTX, "-n", cfg["ns"], "port-forward", cfg["svc"],
         f'{cfg["local"]}:{cfg["remote"]}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _pf_ready(name: str, cfg: dict, timeout: float = 10) -> bool:
    url = _READY_URL[name].format(local=cfg["local"])
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _ensure_pf(pf: dict | None = None, retries: int = 6) -> list[subprocess.Popen]:
    """Ouvre les 3 port-forwards (config `pf`, défaut = pf_config() mono-runner)
    et VÉRIFIE leur connectivité avant de rendre la main (cause racine d'un
    blocage précédent : un pf mort + requêtes séquentielles bloquaient >1h).
    Relance un pf qui ne répond pas.

    Tolérances élargies (2026-06-03) pour la VM : Prometheus est PARTAGÉ par les 3
    runners (un seul `monitoring-metrics/prometheus-server`) et le réseau VM→cluster
    est plus lent → quand les 3 ouvrent leur pf Prometheus en même temps, /-/ready
    pouvait dépasser 4 s × 3 essais. timeout 10 s, 6 essais, sleeps plus longs."""
    pf = pf or pf_config()
    procs: dict[str, subprocess.Popen] = {}
    for name, cfg in pf.items():
        procs[name] = _pf_one(cfg)
    time.sleep(8)
    for name, cfg in pf.items():
        for _ in range(retries):
            if _pf_ready(name, cfg):
                break
            try:
                procs[name].terminate()
            except Exception:
                pass
            procs[name] = _pf_one(cfg)
            time.sleep(7)
        else:
            raise RuntimeError(f"port-forward {name} ({cfg['svc']}) ne répond pas après {retries} essais")
    return list(procs.values())


def pull_prometheus(start: float, end: float, step: int, ns: str = NAMESPACE,
                    pf: dict | None = None) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    pf = pf or pf_config(ns)
    base = f"http://127.0.0.1:{pf['prometheus']['local']}/api/v1/query_range"
    queries = list(_prom_queries(ns).items())

    def _fetch(item):
        name, q = item
        params = urllib.parse.urlencode({"query": q, "start": start, "end": end, "step": step})
        try:
            return name, _get(f"{base}?{params}").get("data", {}).get("result", [])
        except Exception as e:
            return name, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=8) as ex:
        return {name: series for name, series in ex.map(_fetch, queries)}


def pull_jaeger(start: float, end: float, chunk_s: int = 60, limit: int = 1500,
                ns: str = NAMESPACE, pf: dict | None = None) -> dict:
    """Pull Jaeger par tranches temporelles pour éviter le plafond `limit`
    par requête (200 traces/service saturait dès 5 min → biais à l'échelle).

    On découpe [start, end] en fenêtres de `chunk_s` s, on requête chaque
    service par chunk (limit élevé/chunk), puis on **fusionne en une liste
    plate de traces dédupliquées par traceID** — exactement le schéma attendu
    par `src/telemetry/extractors/traces_file.py::parse_jaeger_dump`
    (`dump["traces"]` = list[trace]).
    """
    from concurrent.futures import ThreadPoolExecutor

    pf = pf or pf_config(ns)
    base = f"http://127.0.0.1:{pf['jaeger']['local']}/api"
    svcs = _get(f"{base}/services").get("data", []) or []
    ts_svcs = [s for s in svcs if s.startswith("ts-")]

    # tâches (service, chunk) ; parallélisées pour tenir à l'échelle 30 min.
    tasks = []
    for s in ts_svcs:
        t = start
        while t < end:
            t_end = min(t + chunk_s, end)
            tasks.append((s, t, t_end))
            t = t_end

    def _fetch(task):
        s, t, t_end = task
        params = urllib.parse.urlencode({
            "service": s, "start": int(t * 1e6), "end": int(t_end * 1e6), "limit": limit})
        try:
            return _get(f"{base}/traces?{params}").get("data", []) or []
        except Exception:
            return []

    merged: dict[str, dict] = {}  # traceID -> trace object (dédup)
    # 16 workers : Jaeger /api/traces est lent par appel (~2-3s même après
    # drainage) ; le coût est dominé par le NOMBRE d'appels (services × chunks),
    # donc on parallélise agressivement.
    with ThreadPoolExecutor(max_workers=16) as ex:
        for traces_chunk in ex.map(_fetch, tasks):
            for tr in traces_chunk:
                tid = tr.get("traceID", "")
                if tid and tid not in merged:
                    merged[tid] = tr
    traces = list(merged.values())
    return {"services": ts_svcs, "n_traces_total": len(traces), "traces": traces}


def pull_loki(start: float, end: float, chunk_s: int = 30, ns: str = NAMESPACE,
              pf: dict | None = None) -> dict:
    """Pull Loki par tranches temporelles pour éviter le plafond de 5000 lignes
    qui tasserait tous les logs dans le bin le plus récent.

    On découpe [start, end] en fenêtres de `chunk_s` secondes, on requête
    chacune (limit 5000/chunk, direction=forward), puis on fusionne les streams
    par jeu de labels.
    """
    from concurrent.futures import ThreadPoolExecutor

    pf = pf or pf_config(ns)
    base = f"http://127.0.0.1:{pf['loki']['local']}/loki/api/v1/query_range"
    q = f'{{namespace="{ns}"}}'
    # chunks temporels parallélisés (sous charge, Loki = le goulot : 60 chunks ×
    # milliers de lignes ; le séquentiel dominait les ~8 min de collecte).
    chunks = []
    t = start
    while t < end:
        chunks.append((t, min(t + chunk_s, end)))
        t = chunks[-1][1]

    def _fetch(c):
        a, b = c
        params = urllib.parse.urlencode({
            "query": q, "start": int(a * 1e9), "end": int(b * 1e9),
            "limit": 5000, "direction": "forward"})
        try:
            return _get(f"{base}?{params}").get("data", {}).get("result", [])
        except Exception:
            return []

    merged: dict[str, dict] = {}
    total = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for result in ex.map(_fetch, chunks):
            for s in result:
                key = json.dumps(s.get("stream", {}), sort_keys=True)
                if key not in merged:
                    merged[key] = {"stream": s["stream"], "values": []}
                merged[key]["values"].extend(s.get("values", []))
                total += len(s.get("values", []))
    streams = list(merged.values())
    return {"n_streams": len(streams), "n_lines": total, "streams": streams}


def main() -> None:
    p = argparse.ArgumentParser(description="EWAT v5 TT telemetry probe")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=float, default=120, help="fenêtre passée en secondes")
    p.add_argument("--step", type=int, default=30)
    p.add_argument("--no-pf", action="store_true", help="port-forwards déjà ouverts")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = time.time()
    start = end - args.window

    procs = [] if args.no_pf else _ensure_pf()
    try:
        prom = pull_prometheus(start, end, args.step)
        jae = pull_jaeger(start, end)
        loki = pull_loki(start, end)
    finally:
        for pr in procs:
            pr.terminate()

    for name, data in [("prometheus", prom), ("jaeger", jae), ("loki", loki)]:
        with gzip.open(out / f"{name}.json.gz", "wt") as f:
            json.dump(data, f)

    # résumé
    prom_series = {k: (len(v) if isinstance(v, list) else "ERR") for k, v in prom.items()}
    print("=== M(t) Prometheus séries par métrique ===")
    for k, v in prom_series.items():
        print(f"  {k:<10} {v}")
    print(f"=== T(t) Jaeger : {len(jae['services'])} services, {jae['n_traces_total']} traces ===")
    print(f"=== L(t) Loki : {loki.get('n_streams','ERR')} streams, {loki.get('n_lines','ERR')} lignes ===")
    print(f"dumps écrits dans {out}/")


if __name__ == "__main__":
    main()
