"""EWAT v5 — smoke test métier Train Ticket (préalable à un épisode).

Vérifie qu'un parcours métier passe VRAIMENT, avant de brûler ~33 min de collecte.

Le gate télémétrie (`run_campaign._backends_scraping`) contrôle que les *backends*
collectent ; le gate brut (`run_episode`) contrôle que des données *existent*.
Aucun des deux ne contrôle que l'*application* sert encore des parcours. Or TT peut
être intégralement `Running 1/1` et ne plus rien servir — constaté le 2026-07-26 :

  - comptes `fdse_microservice` disparus de tt-b/tt-c → login KO (bases éphémères
    vidées par un redémarrage, jamais re-seedées : cf. reset_tt_state.reset_reseed) ;
  - `ts-travel-mongo` à 2 documents sur tt-c → 0 trajet → `_preserve_one` abandonne ;
  - `ts-ticketinfo-service` en CrashLoop sur tt → recherche de trajets en HTTP 500.

Un épisode collecté dans cet état n'a AUCUN parcours métier : les services
transactionnels profonds (payment, cancel, rebook, execute) ne sont jamais
sollicités et T(t) est massivement imputé — exactement le défaut qu'on cherche à
corriger. Ce test échoue en quelques secondes au lieu de le découvrir à l'audit.

On valide la chaîne complète login → trajets → contacts → preserve, qui est le
préalable exact de `loadgen.scenarios.full_journey`.

Usage :
    python -m collect.smoke --address http://<NODE_IP>:32677
    # exit 0 = parcours OK ; exit 1 = KO, dernière ligne « SMOKE FAIL step=<étape> »
"""

from __future__ import annotations

import argparse
import sys
import time

# Étapes, dans l'ordre où elles conditionnent le parcours métier. Le nom est repris
# tel quel par run_campaign._repair_app pour cibler les services à redémarrer.
STEP_LOGIN = "login"
STEP_TRIPS = "trips"
STEP_PRESERVE = "preserve"

# Les appels loadgen ne posent AUCUN timeout → un TT dégradé bloquerait la campagne.
# 25 s : mesuré à chaud, trips ≈ 5-7 s et preserve ≈ 13-20 s (JVM throttlées à 200m CPU).
HTTP_TIMEOUT = 25.0
# Un TT fraîchement redémarré est FROID (JIT, pools, registre) : mesuré à 11 s pour un
# login et > 25 s pour une recherche de trajets, contre < 1 s et 5 s une fois chaud. Or
# la campagne fait un deep reset tous les N épisodes — sans ce réessai, le smoke test
# échouerait juste après, déclencherait une réparation, donc de NOUVEAUX redémarrages :
# une spirale qui empirerait exactement ce qu'elle croit réparer. La 1re tentative sert
# de réchauffement, la 2e juge.
ATTEMPTS = 2
RETRY_PAUSE_S = 20.0


def _bound_session(q) -> None:
    """Impose un timeout à toutes les requêtes de la session `Query`.

    `loadgen.queries.Query` appelle `session.post(...)` sans `timeout` : sur un TT
    dégradé (vu à 60 s/requête sous chaos réseau), le smoke test bloquerait la
    campagne. On enveloppe `session.request` en conservant headers et cookies.
    """
    original = q.session.request

    def _with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", HTTP_TIMEOUT)
        return original(*args, **kwargs)

    q.session.request = _with_timeout


def _fail(step: str, detail: str = "") -> int:
    print(f"  ✗ {step}{(' — ' + detail) if detail else ''}", flush=True)
    print(f"SMOKE FAIL step={step}", flush=True)
    return 1


def _smoke_once(address: str, verbose: bool = True) -> int:
    """Un passage du parcours de validation. Retourne un code de sortie (0 = OK)."""
    from loadgen import scenarios
    from loadgen.queries import Query

    q = Query(address)
    _bound_session(q)

    # 1) login — les comptes de seed existent-ils encore ?
    # `Query.login()` suppose « HTTP 200 = succès », or TT renvoie 200 avec
    # {"status":0,"msg":"Incorrect username or password."} quand le compte manque
    # → il lève AttributeError sur `data.get(...)`. On traite les deux cas.
    t0 = time.time()
    try:
        ok = q.login()
    except Exception as e:  # noqa: BLE001 — tout échec de login est un échec de gate
        return _fail(STEP_LOGIN, f"{type(e).__name__}: {e}")
    if not ok or not q.token:
        return _fail(STEP_LOGIN, "compte absent ou identifiants refusés")
    if verbose:
        print(f"  ✓ {STEP_LOGIN} ({time.time() - t0:.1f}s)", flush=True)

    # 2) trajets — les données de seed voyage/route sont-elles présentes ?
    # Les deux routes servent le mix nominal (high-speed 60 % / normal 40 %,
    # cf. loadgen.scenarios.highspeed_weights) : si l'une est vide, une bonne part
    # des parcours abandonne silencieusement. On exige donc les deux.
    date = scenarios._PRESERVE_DATE
    routes = [
        ("high_speed", lambda: q.query_high_speed_ticket(
            place_pair=("Shang Hai", "Su Zhou"), time=date)),
        ("normal", lambda: q.query_normal_ticket(
            place_pair=("Shang Hai", "Nan Jing"), time=date)),
    ]
    for label, fetch in routes:
        t0 = time.time()
        try:
            trips = fetch()
        except Exception as e:  # noqa: BLE001
            return _fail(STEP_TRIPS, f"{label}: {type(e).__name__}: {e}")
        # Les collecteurs loadgen renvoient None si la requête a ÉCHOUÉ (non-200 ou
        # data absente) et une liste vide s'il n'y a simplement aucun trajet. Les
        # deux cassent le parcours mais n'appellent pas la même réparation : None =
        # un service du chemin critique est KO (vu sur tt : ts-ticketinfo en
        # CrashLoop → HTTP 500) ; [] = données de seed perdues (vu sur tt-c).
        if trips is None:
            return _fail(STEP_TRIPS, f"{label}: requête en échec "
                                     f"(service du chemin critique KO ?)")
        if not trips:
            return _fail(STEP_TRIPS, f"{label}: 0 trajet (seed voyage/route perdu ?)")
        if verbose:
            print(f"  ✓ {STEP_TRIPS}/{label}: {len(trips)} trajet(s) "
                  f"({time.time() - t0:.1f}s)", flush=True)

    # 3) preserve — la chaîne transactionnelle (contacts → seat → order) répond-elle ?
    # C'est le point de bascule de `full_journey` : sans ordre créé, pay/collect/
    # cancel/rebook/consign ne s'exécutent jamais et les services profonds restent
    # non tracés.
    t0 = time.time()
    try:
        scenarios._ensure_contact(q)
        order = scenarios._preserve_one(q)
    except Exception as e:  # noqa: BLE001
        return _fail(STEP_PRESERVE, f"{type(e).__name__}: {e}")
    if not order:
        return _fail(STEP_PRESERVE, "aucun ordre créé")
    if verbose:
        print(f"  ✓ {STEP_PRESERVE}: order={order[0]} trip={order[1]} "
              f"({time.time() - t0:.1f}s)", flush=True)

    print("SMOKE OK", flush=True)
    return 0


def run_smoke(address: str, verbose: bool = True, attempts: int = ATTEMPTS,
              pause_s: float = RETRY_PAUSE_S) -> int:
    """Parcours de validation, tolérant au démarrage à froid.

    N'échoue qu'après `attempts` passages : le premier réchauffe (JIT, pools,
    registre), les suivants jugent. Sans cela, tout redémarrage récent de TT ferait
    diagnostiquer à tort une panne applicative.
    """
    rc = 1
    for i in range(attempts):
        rc = _smoke_once(address, verbose=verbose)
        if rc == 0:
            return 0
        if i < attempts - 1:
            print(f"  … échec au 1er passage — nouvel essai dans {pause_s:.0f}s "
                  f"(démarrage à froid ?) [{i + 2}/{attempts}]", flush=True)
            time.sleep(pause_s)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 — smoke test métier Train Ticket")
    ap.add_argument("--address", required=True, help="http://<node_ip>:<nodeport> du namespace")
    ap.add_argument("--attempts", type=int, default=ATTEMPTS,
                    help="passages avant de conclure à l'échec (le 1er réchauffe TT)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    sys.exit(run_smoke(args.address, verbose=not args.quiet, attempts=args.attempts))


if __name__ == "__main__":
    main()
