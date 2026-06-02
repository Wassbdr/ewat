"""EWAT v5 — reset d'état Train Ticket entre épisodes.

Empêche la dérive de baseline sur une longue campagne (accumulation d'orders en
base, entrées Nacos/registry stale, état JVM). Deux modes :

- ``light`` (défaut) : ne fait qu'attendre un cool-down (laisse le système se
  stabiliser sans rien redémarrer). Rapide.
- ``deep`` : rolling-restart des services stateful + leurs MongoDB pour repartir
  d'un état propre. Plus lent (~minutes), à lancer périodiquement (tous les K
  épisodes) plutôt qu'à chaque épisode.

Usage :
    python -m collect.reset_tt_state --mode light --cooldown 30
    python -m collect.reset_tt_state --mode deep --namespace tt
"""

from __future__ import annotations

import argparse
import subprocess
import time

# Services porteurs d'état accumulé (orders, réservations, paiements).
STATEFUL = [
    "ts-order-service", "ts-order-other-service",
    "ts-preserve-service", "ts-preserve-other-service",
    "ts-inside-payment-service", "ts-payment-service",
    "ts-cancel-service", "ts-rebook-service",
]
STATEFUL_DBS = [
    "ts-order-mongo", "ts-order-other-mongo", "ts-payment-mongo",
    "ts-inside-payment-mongo",
]


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def reset_light(cooldown: int) -> None:
    print(f"[reset] light : cooldown {cooldown}s", flush=True)
    time.sleep(cooldown)


def reset_deep(namespace: str, cooldown: int) -> None:
    print(f"[reset] deep : rolling-restart stateful ({len(STATEFUL)} svc + "
          f"{len(STATEFUL_DBS)} db) dans {namespace}", flush=True)
    for d in STATEFUL_DBS + STATEFUL:
        _run(["kubectl", "rollout", "restart", "deploy", "-n", namespace, d])
    for d in STATEFUL_DBS + STATEFUL:
        subprocess.run(["kubectl", "rollout", "status", "deploy", "-n", namespace, d,
                        "--timeout=300s"], capture_output=True, text=True)
    print(f"[reset] deep terminé, cooldown {cooldown}s", flush=True)
    time.sleep(cooldown)


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 TT state reset")
    ap.add_argument("--mode", choices=["light", "deep"], default="light")
    ap.add_argument("--namespace", default="tt")
    ap.add_argument("--cooldown", type=int, default=30)
    args = ap.parse_args()
    if args.mode == "deep":
        reset_deep(args.namespace, args.cooldown)
    else:
        reset_light(args.cooldown)


if __name__ == "__main__":
    main()
