"""EWAT v5 — générateur de charge Train Ticket.

Pilote un mix pondéré de scénarios métier (réservation, annulation, paiement,
consign, rebook…) contre une instance Train Ticket, avec N utilisateurs
concurrents. Sert de charge nominale stable pendant les phases baseline /
pre-injection / recovery d'un épisode, et de charge ciblée pendant l'injection.

Usage :
    python -m loadgen.runner --address http://<CLUSTER_NODE_IP>:32677 \
        --users 10 --duration 600 --rps-log 30

    # mode ciblé (charge orientée vers un bug F)
    python -m loadgen.runner --address ... --scenario query_and_cancel --users 20
"""

from __future__ import annotations

import argparse
import logging
import random
import threading
import time
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from loadgen import scenarios  # noqa: E402
from loadgen.queries import Query  # noqa: E402


# Opérations « légères » directes (méthodes Query) pour réveiller les services
# que les flux métier ne touchent pas : food, route, assurance, contacts, admin.
def op_food(q: Query):
    q.query_food()

def op_route(q: Query):
    q.query_route()

def op_cheapest(q: Query):
    q.query_cheapest(date="2026-12-01")

def op_min_station(q: Query):
    q.query_min_station(date="2026-12-01")

def op_quickest(q: Query):
    q.query_quickest(date="2026-12-01")

def op_assurances(q: Query):
    q.query_assurances()

def op_contacts(q: Query):
    q.query_contacts()

def op_admin_config(q: Query):
    q.admin_login()
    q.query_admin_basic_config()

def op_admin_price(q: Query):
    q.admin_login()
    q.query_admin_basic_price()

def op_admin_travel(q: Query):
    q.admin_login()
    q.query_admin_travel()


# Registre des opérations légères (résolu dans le worker si absent de scenarios).
_EXTRA_OPS = {
    "op_food": op_food, "op_route": op_route, "op_cheapest": op_cheapest,
    "op_min_station": op_min_station, "op_quickest": op_quickest,
    "op_assurances": op_assurances, "op_contacts": op_contacts,
    "op_admin_config": op_admin_config, "op_admin_price": op_admin_price,
    "op_admin_travel": op_admin_travel,
}

# Mix nominal élargi : flux métier profonds (preserve/pay/cancel/collect/
# consign/rebook/execute) + opérations légères pour couvrir tout le graphe TT.
NOMINAL_MIX: dict[str, int] = {
    # flux métier multi-services (chaînes profondes)
    "query_and_preserve": 22,
    "query_and_pay": 16,   # → payment, inside-payment, voucher
    "query_and_cancel": 10,
    "query_and_collect": 7,
    "query_and_consign": 7,
    "query_and_rebook": 5,
    "query_and_execute": 5,
    # opérations légères (couverture large)
    "op_food": 5, "op_route": 4, "op_cheapest": 3, "op_min_station": 2,
    "op_quickest": 2, "op_assurances": 3, "op_contacts": 2,
    "op_admin_config": 3, "op_admin_price": 2, "op_admin_travel": 2,
}


@dataclass
class Stats:
    ok: int = 0
    err: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def hit(self, success: bool) -> None:
        with self.lock:
            if success:
                self.ok += 1
            else:
                self.err += 1


def _weighted_choice(mix: dict[str, int]) -> str:
    names = list(mix)
    weights = list(mix.values())
    return random.choices(names, weights=weights, k=1)[0]


def _worker(address: str, stop: threading.Event, stats: Stats, scenario: str | None) -> None:
    """Un utilisateur virtuel : login puis boucle de scénarios jusqu'au stop."""
    q = Query(address)
    try:
        q.login()
    except Exception:
        stats.hit(False)
        return

    last_relogin = time.time()
    while not stop.is_set():
        # re-login périodique (~90 s) pour générer des spans ts-auth-service
        if time.time() - last_relogin > 90:
            try:
                q.login()
            except Exception:
                pass
            last_relogin = time.time()

        name = scenario or _weighted_choice(NOMINAL_MIX)
        fn = getattr(scenarios, name, None) or _EXTRA_OPS.get(name)
        if fn is None:
            stats.hit(False)
            time.sleep(1)
            continue
        try:
            fn(q)
            stats.hit(True)
        except Exception:
            stats.hit(False)
        # petite pause pour ne pas saturer le cluster (contrainte CPU partagé)
        time.sleep(random.uniform(0.3, 1.2))


def run(
    address: str,
    users: int,
    duration: float,
    scenario: str | None = None,
    rps_log: float = 30.0,
) -> Stats:
    """Lance `users` workers pendant `duration` secondes.

    Parameters
    ----------
    address : str
        URL de base Train Ticket (NodePort UI dashboard).
    users : int
        Nombre d'utilisateurs virtuels concurrents.
    duration : float
        Durée totale en secondes.
    scenario : str | None
        Si fourni, tous les workers exécutent ce seul scénario (mode ciblé).
        Sinon, mix pondéré NOMINAL_MIX (mode nominal).
    rps_log : float
        Période (s) d'affichage du throughput cumulé.
    """
    stop = threading.Event()
    stats = Stats()
    threads = [
        threading.Thread(target=_worker, args=(address, stop, stats, scenario), daemon=True)
        for _ in range(users)
    ]
    for t in threads:
        t.start()

    start = time.time()
    last_log = start
    last_ok = 0
    while time.time() - start < duration:
        time.sleep(1)
        now = time.time()
        if now - last_log >= rps_log:
            with stats.lock:
                ok, err = stats.ok, stats.err
            rps = (ok - last_ok) / (now - last_log)
            print(
                f"[{int(now - start):>4}s] ok={ok} err={err} "
                f"rps~{rps:.1f} users={users} mode={scenario or 'nominal'}",
                flush=True,
            )
            last_log, last_ok = now, ok

    stop.set()
    for t in threads:
        t.join(timeout=5)
    with stats.lock:
        print(f"DONE ok={stats.ok} err={stats.err} duration={duration}s users={users}", flush=True)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="EWAT v5 Train Ticket load generator")
    p.add_argument("--address", required=True, help="Base URL TT, ex http://<CLUSTER_NODE_IP>:32677")
    p.add_argument("--users", type=int, default=10, help="utilisateurs virtuels concurrents")
    p.add_argument("--duration", type=float, default=600, help="durée en secondes")
    p.add_argument("--scenario", default=None, help="scénario unique (mode ciblé)")
    p.add_argument("--rps-log", type=float, default=30, help="période de log throughput (s)")
    args = p.parse_args()
    run(args.address, args.users, args.duration, args.scenario, args.rps_log)


if __name__ == "__main__":
    main()
