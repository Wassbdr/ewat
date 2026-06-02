"""EWAT v5 — moniteur de santé Train Ticket pendant la collecte.

Surveille en continu le namespace TT et émet une ligne par changement d'état
notable (pod non prêt, crashloop, eviction, restart en hausse). Conçu pour être
lancé en arrière-plan (ou via l'outil Monitor) pendant une campagne longue : le
pilote a montré qu'auth pouvait accumuler des centaines de restarts sous pression.

Émet sur stdout (une ligne = un événement) :
    OK ready=63/64
    DEGRADED ready=58/64 notready=ts-auth-service,ts-order-service
    CRASHLOOP ts-auth-service restarts=12
    RECOVERED ready=64/64

Usage :
    python -m collect.health_monitor --namespace tt --interval 30
"""

from __future__ import annotations

import argparse
import subprocess
import time


def _snapshot(namespace: str) -> dict:
    r = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace, "--no-headers"],
        capture_output=True, text=True)
    rows = [l.split() for l in r.stdout.splitlines() if l.strip()]
    ready, notready, crashloop = 0, [], []
    for cols in rows:
        if len(cols) < 4:
            continue
        name, rd, status, restarts = cols[0], cols[1], cols[2], cols[3]
        a, _, b = rd.partition("/")
        is_ready = a == b and a != "0"
        if is_ready:
            ready += 1
        else:
            notready.append(name)
        if "CrashLoop" in status:
            crashloop.append((name, restarts))
    return {"total": len(rows), "ready": ready, "notready": notready, "crashloop": crashloop}


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 TT health monitor")
    ap.add_argument("--namespace", default="tt")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--ready-floor", type=float, default=0.90)
    args = ap.parse_args()

    prev_state = None
    while True:
        s = _snapshot(args.namespace)
        frac = s["ready"] / max(1, s["total"])
        state = "OK" if frac >= args.ready_floor else "DEGRADED"
        if s["crashloop"]:
            cl = ",".join(f"{n}({r})" for n, r in s["crashloop"])
            print(f"CRASHLOOP {cl}", flush=True)
        if state != prev_state:
            if state == "OK":
                print(f"{'RECOVERED' if prev_state else 'OK'} ready={s['ready']}/{s['total']}", flush=True)
            else:
                nr = ",".join(s["notready"][:8])
                print(f"DEGRADED ready={s['ready']}/{s['total']} notready={nr}", flush=True)
            prev_state = state
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
