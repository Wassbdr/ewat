"""EWAT v5 — reset d'état Train Ticket entre épisodes.

Empêche la dérive de baseline sur une longue campagne (accumulation d'orders en
base, entrées Nacos/registry stale, état JVM). Deux modes :

- ``light`` (défaut) : ne fait qu'attendre un cool-down (laisse le système se
  stabiliser sans rien redémarrer). Rapide.
- ``deep`` : rolling-restart des services stateful + leurs MongoDB pour repartir
  d'un état propre. Plus lent (~minutes), à lancer périodiquement (tous les K
  épisodes) plutôt qu'à chaque épisode.
- ``reseed`` : recharge les données de seed (comptes, trajets, routes…) en
  redémarrant les services applicatifs SANS toucher aux mongos. À lancer quand
  l'appli répond mais ne sert plus de parcours métier (login KO, 0 trajet) —
  symptôme d'une base éphémère vidée par un redémarrage. Cf. ``reset_reseed``.

Usage :
    python -m collect.reset_tt_state --mode light --cooldown 30
    python -m collect.reset_tt_state --mode deep --namespace tt
    python -m collect.reset_tt_state --mode reseed --namespace tt-c
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

# Contexte kubectl épinglé (cf. inject.py/probe.py) : ce reset fait du rolling-restart
# (write) — un contexte qui bascule restarterait des pods du mauvais cluster.
_KC = ["kubectl", "--context", os.environ.get("V5_KUBE_CONTEXT", "observit-cluster1")]

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
# Jaeger all-in-one stocke les spans EN MÉMOIRE : il gonfle sur toute la campagne
# → les requêtes de collecte passent de ~100 ms à 60 s+ (vu 07-20 : jaeger à 3 h
# d'uptime = 62 s/requête, contre 104 ms fraîchement redémarré). Aggravé par le fix
# chaos-daemon (les vraies fautes génèrent bien plus de spans). On le vide au deep
# reset (entre épisodes → sûr) pour garder les pulls rapides. Bonus : isolation
# per-épisode propre (plus de spans d'épisodes antérieurs dans le store).
# VALIDÉ bout-en-bout 07-21 : restart tt-b jaeger (2.6Go, 11s/requête) → charge
# 150s → 31 services re-tracés, requête 1.2s, mem 73Mi. L'ingestion recouvre
# (les services se reconnectent), et le reset étant ENTRE épisodes, le baseline
# laisse le temps au store de se remplir avant l'injection.
TELEMETRY = ["jaeger"]


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


# Nombre de deployments redémarrés simultanément. Depuis le passage des limites CPU
# à 500m, redémarrer les 43 services d'un coup laisse 43 JVM en phase de boot
# réclamer jusqu'à 21 cores sur des nœuds qui en ont 7 : le 2026-07-30, 57p5t est
# monté à 105 % de CPU et le reseed n'avançait plus (23/69 pods prêts, login à 000).
# Par lots, le pic reste borné et chaque lot démarre vite.
BATCH = int(os.environ.get("V5_ROLLOUT_BATCH", "8"))


def _rollout(namespace: str, names: list[str], timeout: str = "300s") -> None:
    """Rolling-restart PAR LOTS, en attendant la readiness de chaque lot.

    On ne rend la main qu'une fois tous les lots prêts — c'est ce qui permet de
    séquencer bases → services, tout en évitant la ruée de démarrages simultanés
    qui sature le CPU des nœuds (cf. BATCH).
    """
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        for d in chunk:
            _run([*_KC, "rollout", "restart", "deploy", "-n", namespace, d])
        for d in chunk:
            subprocess.run([*_KC, "rollout", "status", "deploy", "-n", namespace, d,
                            f"--timeout={timeout}"], capture_output=True, text=True)


def _app_deployments(namespace: str) -> list[str]:
    """Deployments du namespace SAUF les bases (`*-mongo`).

    Le reseed doit redémarrer les *services* (qui rechargent leurs données au
    boot) sans toucher aux *bases*, qui doivent rester debout pour recevoir ce
    seed.
    """
    out = _run([*_KC, "get", "deploy", "-n", namespace, "-o", "name"])
    names = [ln.split("/")[-1].strip() for ln in out.splitlines() if ln.strip()]
    return [n for n in names if "-mongo" not in n]


def reset_light(cooldown: int) -> None:
    print(f"[reset] light : cooldown {cooldown}s", flush=True)
    time.sleep(cooldown)


def reset_reseed(namespace: str, cooldown: int) -> None:
    """Recharge les données de seed de TOUS les services applicatifs.

    Les mongos TT sont ÉPHÉMÈRES (aucun volume, 0 PVC dans les 3 namespaces) et
    les données de seed ne sont chargées qu'au DÉMARRAGE du service propriétaire.
    Tout redémarrage d'un mongo (éviction, incident nœud, deep reset) vide donc sa
    base DÉFINITIVEMENT jusqu'à ce que le service redémarre — sans aucun signal :
    les pods restent `Running 1/1`. Constaté le 2026-07-26 : comptes
    `fdse_microservice` disparus de tt-b/tt-c (login KO) et trajets vides sur tt-c
    (`ts-travel-mongo` : 2 documents contre 7 sur tt/tt-b).

    On redémarre les services SANS toucher aux mongos : chaque service se re-seede
    contre une base vivante.
    """
    svcs = _app_deployments(namespace)
    print(f"[reset] reseed : rolling-restart de {len(svcs)} services applicatifs "
          f"(mongos conservés) dans {namespace}", flush=True)
    _rollout(namespace, svcs)
    print(f"[reset] reseed terminé, cooldown {cooldown}s", flush=True)
    time.sleep(cooldown)


def reset_deep(namespace: str, cooldown: int) -> None:
    """Repart d'un état propre : vide l'état accumulé, puis laisse les services re-seeder.

    ORDRE CRITIQUE (corrigé le 2026-07-26) : bases D'ABORD (restart + attente
    Ready), services ENSUITE. L'ancienne version restartait tout en un seul lot
    (`STATEFUL_DBS + STATEFUL + TELEMETRY`) → un service pouvait se seeder avant
    que son mongo soit prêt, et le seed était perdu (bases éphémères, cf.
    `reset_reseed`) — le deep reset s'auto-infligeait la panne qu'il devait éviter.
    """
    print(f"[reset] deep : rolling-restart stateful ({len(STATEFUL)} svc + "
          f"{len(STATEFUL_DBS)} db) + jaeger dans {namespace}", flush=True)
    _rollout(namespace, STATEFUL_DBS)  # 1) bases vidées ET prêtes
    _rollout(namespace, STATEFUL)      # 2) services : se re-seedent contre des bases vivantes
    _rollout(namespace, TELEMETRY)     # 3) télémétrie (vide le store Jaeger en mémoire)
    print(f"[reset] deep terminé, cooldown {cooldown}s", flush=True)
    time.sleep(cooldown)


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 TT state reset")
    ap.add_argument("--mode", choices=["light", "deep", "reseed"], default="light")
    ap.add_argument("--namespace", default="tt")
    ap.add_argument("--cooldown", type=int, default=30)
    args = ap.parse_args()
    if args.mode == "deep":
        reset_deep(args.namespace, args.cooldown)
    elif args.mode == "reseed":
        reset_reseed(args.namespace, args.cooldown)
    else:
        reset_light(args.cooldown)


if __name__ == "__main__":
    main()
