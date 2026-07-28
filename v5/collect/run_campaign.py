"""EWAT v5 — driver de collecte massive Train Ticket.

Boucle sur le catalogue (scénarios chaos + bugs F) × répétitions, en produisant
un épisode conforme par itération. Robuste pour une campagne de plusieurs jours :

- **checkpoint/reprise idempotente** : un épisode déjà validé est sauté.
- **gate qualité par épisode** : appelle `scripts/validate_v5.py` ; si échec,
  marque `.quality_failed` et retente jusqu'à `--max-retries`.
- **reset d'état** périodique (tous les `--reset-every` épisodes : deep ; sinon light).
- **held-out** : les 3 chaos held-out + bugs F sont marqués `held_out_flag` (→ test only).
- **moniteur santé** : vérifie TT avant chaque épisode ; pause si dégradé.

Usage :
    python -m collect.run_campaign --reps 30 --out-root data/raw_v5 \
        --address http://<CLUSTER_NODE_IP>:32677
    # reprise : relancer la même commande, les épisodes validés sont sautés.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from collect import reset_tt_state, run_episode

V5 = Path(__file__).resolve().parents[1]
REPO = V5.parent

HELD_OUT_CHAOS = {"held_io_latency", "held_net_bandwidth", "held_kernel_fault"}

# Contexte kubectl épinglé (cf. inject.py / probe.py / run_episode.py).
KCTX = os.environ.get("V5_KUBE_CONTEXT", "observit-cluster1")
KCTX_ARGS = ["--context", KCTX]


def _assert_context(namespace: str) -> None:
    """Préflight bloquant : le contexte épinglé doit exister ET voir le namespace
    cible. Évite de lancer une campagne de plusieurs jours contre le mauvais
    cluster (bascule de contexte vue en 2026-06-03 — inject échouait en silence)."""
    r = subprocess.run(["kubectl", *KCTX_ARGS, "get", "ns", namespace, "--no-headers"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[campaign] PRÉFLIGHT ÉCHEC : contexte '{KCTX}' ne voit pas le namespace "
                 f"'{namespace}' (rc={r.returncode}). {r.stderr.strip()}\n"
                 f"  → vérifier `kubectl config get-contexts` ou définir V5_KUBE_CONTEXT.")
    print(f"[campaign] préflight OK : contexte={KCTX} namespace={namespace} visible", flush=True)


def _catalog() -> dict:
    return yaml.safe_load(open(V5 / "chaos" / "catalog.yaml"))


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tt_healthy(namespace: str) -> bool:
    r = subprocess.run(
        ["kubectl", *KCTX_ARGS, "get", "pods", "-n", namespace, "--no-headers"],
        capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if not lines:
        return False
    ready = sum(1 for l in lines if "1/1" in l.split()[1:2] or l.split()[1].startswith("1/1"))
    total = len(lines)
    return ready / total >= 0.90  # ≥90% pods prêts


def _backends_scraping(namespace: str, min_cpu_series: int = 40) -> bool:
    """Pré-check télémétrie AVANT un épisode : Prometheus scrape-t-il vraiment ce
    namespace, et Loki répond-il ? Cause racine de la perte du 6-7 juin (incident
    jnk2v) : Prometheus restait `/-/ready` mais ne scrapait plus les pods tt évincés
    → 33 min collectées pour `prom=0`. On vérifie la PRÉSENCE de séries cAdvisor
    fraîches (pas juste la readiness), et la readiness Loki. Fail-open sur erreur
    réseau (un blip ne doit pas stopper une campagne de plusieurs jours).

    Le seuil est ABSOLU, pas « ≥ 1 série » : lors de la panne cAdvisor du 17-20
    juillet (5 nœuds en HTTP 503, payload trop gros pour le timeout de scrape),
    le namespace ne remontait qu'1 à 13 séries sur 64 — assez pour passer un test
    « > 0 », et ~450 épisodes ont été collectés SANS métriques M(t) (cpu, ram,
    net, disk, mem_limit tous NaN) avant qu'on s'en aperçoive. TT a 64 services :
    en dessous de `min_cpu_series`, la collecte est inexploitable, on met en pause.
    """
    import urllib.parse
    import urllib.request

    node = os.environ.get("V5_NODE_IP", "172.16.203.12")
    prom = f"http://{node}:{os.environ.get('V5_PROM_NODEPORT', '32700')}"
    loki = f"http://{node}:{os.environ.get('V5_LOKI_NODEPORT', '32701')}"
    # 1) Prometheus couvre-t-il VRAIMENT le namespace ? (≥ min_cpu_series séries)
    q = f'count(container_cpu_usage_seconds_total{{namespace="{namespace}",container!=""}})'
    try:
        url = f"{prom}/api/v1/query?{urllib.parse.urlencode({'query': q})}"
        with urllib.request.urlopen(url, timeout=8) as r:
            res = json.load(r).get("data", {}).get("result", [])
        n_series = float(res[0]["value"][1]) if res else 0.0
        if n_series < min_cpu_series:
            print(f"[campaign] couverture cAdvisor insuffisante sur {namespace} : "
                  f"{n_series:.0f} séries cpu < {min_cpu_series} — pause "
                  f"(épisodes sans métriques M(t) sinon)", flush=True)
            return False
    except Exception as e:
        print(f"[campaign] check Prometheus {namespace} échec ({e}) — fail-open", flush=True)
        return True
    # 2) Loki répond-il ? (loki-0 Pending pendant l'incident → logs=0)
    try:
        urllib.request.urlopen(f"{loki}/ready", timeout=8)
    except Exception:
        print(f"[campaign] Loki ne répond pas ({loki}) — pause", flush=True)
        return False
    return True


def _nodes_ram_ok(ceiling: float = 90.0) -> bool:
    """Garde-fou RAM — contrainte *binding* à 3 runners (CPU large, RAM tendue :
    1 runner ≈ 20 GB JVM+mongos). Retourne False si un nœud dépasse `ceiling` %
    de mémoire → pause (une éviction redémarre les mongos, dont les bases sont
    éphémères : le seed est alors perdu et TT ne sert plus de parcours, cf.
    reset_tt_state.reset_reseed). Fail-open si `kubectl top` échoue (un blip
    metrics-server ne doit pas stopper une campagne de plusieurs jours).

    On surveille TOUS les nœuds hors control-plane. L'ancienne version filtrait
    sur `"workers" in nom` alors que Train Ticket tourne en réalité sur les nœuds
    `collectors` : le garde-fou était aveugle sur les seuls nœuds qui comptent
    (constaté le 2026-07-27, collecteurs à 85-88 % sans qu'aucune pause ne se
    déclenche).
    """
    r = subprocess.run(["kubectl", *KCTX_ARGS, "top", "nodes", "--no-headers"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return True  # fail-open
    hot = []
    for line in r.stdout.splitlines():
        cols = line.split()
        # NAME  CPU(cores)  CPU%  MEM(bytes)  MEM%
        if len(cols) < 5 or "masters" in cols[0]:
            continue
        try:
            mem_pct = float(cols[4].rstrip("%"))
        except ValueError:
            continue
        if mem_pct > ceiling:
            hot.append(f"{cols[0].split('-')[-1]}={mem_pct:.0f}%")
    if hot:
        print(f"[campaign] RAM nœud > {ceiling:.0f}% ({','.join(hot)}) — pause "
              f"(éviction = mongos redémarrés = seed perdu)", flush=True)
        return False
    return True


# Services propriétaires du seed, par étape du smoke test en échec. Les mongos TT
# sont éphémères et le seed n'est rechargé qu'au démarrage du service propriétaire
# (cf. reset_tt_state.reset_reseed) → redémarrer LE bon service restaure les données
# sans passer par un reseed complet du namespace (~43 services, plusieurs minutes).
# Interrupteur d'urgence : V5_APP_GATE=0 désactive entièrement le gate applicatif.
# Indispensable — sans lui, un gate qui se trompe bloque toute la campagne et il
# faut modifier le code pour repartir.
APP_GATE = os.environ.get("V5_APP_GATE", "1") not in ("0", "false", "no")
WARMUP_S = int(os.environ.get("V5_WARMUP_S", "90"))
# Plafonds par cycle : au-delà, on abandonne l'épisode au lieu de continuer à
# redémarrer des services (cf. la boucle du 2026-07-28). Rechargés à chaque succès.
MAX_TARGETED_REPAIRS = int(os.environ.get("V5_MAX_REPAIRS", "2"))
MAX_RESEEDS = int(os.environ.get("V5_MAX_RESEEDS", "1"))
_repair_budget = {"targeted": 0, "reseed": 0}

REPAIR_SERVICES = {
    "login": ["ts-auth-service", "ts-user-service"],
    "trips": ["ts-ticketinfo-service", "ts-travel-service", "ts-travel2-service",
              "ts-route-service", "ts-basic-service", "ts-station-service",
              "ts-train-service", "ts-price-service"],
    # ts-security-service : indispensable ici. Sans son seed `ts.security_config`,
    # il renvoie une NullPointerException sur « Get Security Config Info » et
    # `preserve` échoue en HTTP 500 — diagnostiqué le 2026-07-27 sur tt, où la
    # collection avait disparu alors qu'elle contenait 2 documents sur tt-b.
    "preserve": ["ts-preserve-service", "ts-contacts-service", "ts-seat-service",
                 "ts-order-service", "ts-config-service", "ts-security-service",
                 "ts-assurance-service"],
}


def _app_healthy(namespace: str, address: str, timeout: int = 420) -> tuple[bool, str]:
    """L'APPLICATION sert-elle encore un parcours métier ? (cf. collect.smoke)

    `_backends_scraping` vérifie que les backends collectent ; le gate brut de
    `run_episode` vérifie que des données existent. Aucun ne vérifie que TT sert
    encore des parcours — or TT peut être `Running 1/1` partout et ne plus rien
    servir (bases éphémères vidées, service du chemin critique en CrashLoop :
    constaté le 2026-07-26 sur les 3 namespaces). Un épisode collecté ainsi n'a
    aucun flux transactionnel et son T(t) est massivement imputé.

    Lancé en sous-processus pour un timeout DUR : les appels loadgen ne posent pas
    de timeout et un TT dégradé bloquerait la campagne. Le budget couvre les
    réessais anti-démarrage-à-froid de `collect.smoke` (un deep reset précède
    régulièrement cet appel). Retourne (ok, étape_en_échec). Fail-open si le test
    lui-même est inexécutable.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-m", "collect.smoke", "--address", address],
            capture_output=True, text=True, cwd=str(V5), timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(V5)})
    except subprocess.TimeoutExpired:
        print(f"[campaign] smoke test {namespace} : timeout {timeout}s — TT bloqué",
              flush=True)
        return False, "timeout"
    except Exception as e:
        print(f"[campaign] smoke test {namespace} inexécutable ({e}) — fail-open",
              flush=True)
        return True, ""
    if r.returncode == 0:
        return True, ""
    step = ""
    for line in r.stdout.splitlines():
        if line.startswith("SMOKE FAIL step="):
            step = line.split("=", 1)[1].strip()
    detail = next((ln.strip() for ln in r.stdout.splitlines()
                   if ln.strip().startswith("✗")), "")
    print(f"[campaign] smoke test {namespace} ÉCHEC (étape={step or '?'}) {detail}",
          flush=True)
    return False, step


def _warmup(address: str, seconds: int = 90) -> None:
    """Charge de réchauffement après un redémarrage de services.

    Un service qui vient de redémarrer est `Ready` mais FROID : JIT non chauffé,
    pools de connexions vides, registre à re-résoudre. Mesuré à 11 s pour un login
    et > 25 s pour une recherche de trajets, contre < 1 s à chaud. Le juger tout de
    suite relance une réparation, donc de nouveaux redémarrages : c'est la boucle
    observée le 2026-07-28 (9 h sans un seul épisode abouti).
    """
    print(f"[campaign] réchauffement {seconds}s avant de rejuger ...", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "loadgen.runner", "--address", address,
             "--users", "4", "--duration", str(seconds), "--rps-log", str(seconds)],
            capture_output=True, text=True, cwd=str(V5), timeout=seconds + 120,
            env={**os.environ, "PYTHONPATH": str(V5)})
    except Exception as e:
        print(f"[campaign] réchauffement interrompu ({e})", flush=True)


def _repair_app(namespace: str, step: str) -> bool:
    """Réparation ciblée du seed : redémarre les services de l'étape en échec.

    Attendre ne sert à rien sur une base éphémère vidée — elle ne se re-remplit
    jamais toute seule. On agit au lieu de mettre en pause.
    """
    svcs = REPAIR_SERVICES.get(step)
    if not svcs:
        return False
    print(f"[campaign] réparation {namespace} (étape={step}) : restart "
          f"{', '.join(svcs)}", flush=True)
    reset_tt_state._rollout(namespace, svcs)
    return True


def _stall_flag(out_root: Path, namespace: str, reason: str | None) -> None:
    """Marqueur de stagnation lisible de l'extérieur (`ls`, dashboard, alerting).

    La panne RKE2 du 2026-07-22 a coûté ~10 h : la campagne se mettait
    correctement en pause, mais rien ne le signalait autrement qu'une ligne de log
    noyée parmi des milliers. C'est le seul type de panne hors de notre périmètre
    (plan de contrôle du cluster) — donc au minimum, la rendre visible tout de
    suite. Un fichier par namespace : les 3 runners partagent le même out-root.
    """
    flag = out_root / f"_STALLED_{namespace}"
    if reason is None:
        flag.unlink(missing_ok=True)
        return
    flag.write_text(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {reason}\n")


def _validate(ep_dir: Path) -> bool:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validate_v5.py"), "--episode", str(ep_dir)],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": __import__("os").environ.get("PATH", "")})
    print(r.stdout.strip().splitlines()[-1] if r.stdout else "(no output)", flush=True)
    return r.returncode == 0


def collect_episode(scenario: str, rep: int, out_root: Path, address: str,
                    users: int, is_bug: bool, held_out: bool, max_retries: int,
                    namespace: str, pf_offset: int = 0, ram_ceiling: float = 90.0,
                    target: str | None = None, peak: str = "high",
                    min_cpu_series: int = 40) -> bool:
    for attempt in range(max_retries + 1):
        ep_id = f"episode_{scenario}_{rep:03d}_{_ts()}"
        ep_dir = out_root / ep_id
        # santé TT (readiness pods) + RAM nœuds (anti-saturation à 3 runners) avant épisode
        waited = 0
        while (not _tt_healthy(namespace) or not _nodes_ram_ok(ram_ceiling)
               or not _backends_scraping(namespace, min_cpu_series)) and waited < 600:
            print(f"[campaign] TT dégradé / RAM haute / backend KO, pause 30s ...", flush=True)
            time.sleep(30); waited += 30
        # BLOCAGE DUR sur la télémétrie : dégradation pods/RAM = transitoire (on
        # tente quand même après l'attente), mais si Prometheus ne couvre toujours
        # pas le namespace, l'épisode sortirait SANS métriques M(t). On refuse de
        # brûler 33 min pour des données inexploitables (panne cAdvisor 17-20/07 :
        # ~450 épisodes vides parce que l'attente expirait puis collectait quand même).
        if not _backends_scraping(namespace, min_cpu_series):
            print(f"[campaign] ABANDON {ep_id} : télémétrie insuffisante après "
                  f"{waited}s d'attente — épisode NON collecté", flush=True)
            return False
        # GATE APPLICATIF (~20 s) : les backends collectent, mais TT sert-il encore
        # des parcours ? Sans ce contrôle, un namespace au seed perdu produit 33 min
        # sans aucun flux transactionnel — les services profonds (payment, cancel,
        # rebook, execute) restent non tracés et l'épisode est inexploitable, sans
        # qu'aucun autre gate ne s'en aperçoive. On répare puis on réessaie ;
        # l'escalade va du ciblé (quelques services) au reseed complet du namespace.
        app_ok, step = (True, "") if not APP_GATE else _app_healthy(namespace, address)
        if not app_ok:
            # Budget de réparation. Sans plafond, chaque épisode relance un cycle
            # restart → test à froid → échec → restart, qui détruit plus qu'il ne
            # répare : le 2026-07-28, 9 h sans un seul épisode abouti, les deux
            # runners en boucle. On répare peu, on RÉCHAUFFE avant de rejuger, et
            # on abandonne l'épisode plutôt que de s'acharner. Le budget se
            # recharge dès qu'un épisode aboutit.
            if (_repair_budget["targeted"] < MAX_TARGETED_REPAIRS
                    and _repair_app(namespace, step)):
                _repair_budget["targeted"] += 1
                _warmup(address, WARMUP_S)
                app_ok, step = _app_healthy(namespace, address)
            if not app_ok and _repair_budget["reseed"] < MAX_RESEEDS:
                reset_tt_state.reset_reseed(namespace, cooldown=60)
                _repair_budget["reseed"] += 1
                _warmup(address, WARMUP_S)
                app_ok, step = _app_healthy(namespace, address)
        if not app_ok:
            print(f"[campaign] ABANDON {ep_id} : parcours métier KO (étape={step}) "
                  f"— budget réparation {_repair_budget} — épisode NON collecté",
                  flush=True)
            return False
        try:
            # COLLECTE uniquement (pas de build) — Record→Build→Assemble.
            res = run_episode.run_episode(scenario, ep_dir, address, users,
                                          run_episode.STEP_S, is_bug, held_out,
                                          namespace, pf_offset, target, peak)
        except Exception as e:
            print(f"[campaign] {scenario} rep{rep} attempt{attempt} EXC: {e}", flush=True)
            (ep_dir).mkdir(parents=True, exist_ok=True)
            (ep_dir / ".raw_failed").write_text(f"exception: {e}")
            continue
        # gate BRUT (la collecte a-t-elle capté assez de données ?)
        if res.get("raw_ok"):
            # Un épisode abouti prouve que TT est sain : on recharge le budget de
            # réparation, pour qu'une panne future puisse encore être réparée.
            _repair_budget["targeted"] = _repair_budget["reseed"] = 0
            print(f"[campaign] OK {ep_id} traces={res['n_traces']} logs={res['n_log_lines']} "
                  f"prom={res['n_prom_series']} svc={res.get('n_traced_services')} "
                  f"fault={res.get('fault_score', float('nan')):.2f} "
                  f"collect={res['collect_s']}s", flush=True)
            return True
        print(f"[campaign] FAIL raw-gate {ep_id} "
              f"({'; '.join(res.get('gate_reasons') or [])}) attempt {attempt}", flush=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 collection campaign driver")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out-root", type=Path, default=REPO / "data" / "raw_v5")
    ap.add_argument("--address", default="http://<CLUSTER_NODE_IP>:32677")
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--namespace", default="tt")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--reset-every", type=int, default=10, help="deep reset tous les N épisodes")
    ap.add_argument("--only", default="", help="liste de scénarios (CSV) pour restreindre")
    ap.add_argument("--rep-start", type=int, default=0, help="rep de début (split multi-runner)")
    ap.add_argument("--rep-end", type=int, default=None, help="rep de fin exclue (défaut = reps)")
    ap.add_argument("--pf-offset", type=int, default=0, help="décalage ports locaux (multi-runner ; ex. tt=0, tt-b=10)")
    ap.add_argument("--held-out-cap", type=int, default=28, help="reps max pour les scénarios held-out (test-only)")
    ap.add_argument("--min-cpu-series", type=int, default=40,
                    help="séries cpu cAdvisor minimum pour lancer un épisode "
                         "(TT=64 services ; sous ce seuil les métriques M(t) manquent)")
    ap.add_argument("--ram-ceiling", type=float, default=90.0,
                    help="pause si un nœud worker dépasse ce %% de RAM (garde-fou 3 runners)")
    ap.add_argument("--stall-alert-s", type=int, default=3600,
                    help="alerte + marqueur _STALLED_<ns> si aucun épisode collecté "
                         "depuis ce délai (défaut 1 h)")
    args = ap.parse_args()
    rep_end = args.rep_end if args.rep_end is not None else args.reps

    _assert_context(args.namespace)  # préflight : bon cluster avant tout
    args.out_root.mkdir(parents=True, exist_ok=True)
    cat = _catalog()
    scenarios = [s["name"] for s in cat["scenarios"]]
    bugs = [b["id"] for b in cat["bugs"] if b.get("status") == "ready"]  # F1 d'abord
    # rotation de cible + catégorie par scénario (design d'échantillonnage v5.2)
    pools = {s["name"]: s.get("target_pool") for s in cat["scenarios"]}
    cats = {s["name"]: s.get("category", "") for s in cat["scenarios"]}
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s in keep]
        bugs = [b for b in bugs if b in keep]

    # plan d'épisodes : (name, is_bug, held_out)
    plan: list[tuple[str, bool, bool]] = []
    for s in scenarios:
        plan.append((s, False, s in HELD_OUT_CHAOS))
    for b in bugs:
        plan.append((b, True, True))  # bugs = held-out (test only)

    # un épisode "collecté OK" = episode_meta.json présent ET pas de .raw_failed
    def _collected_ok(e: Path) -> bool:
        return (e / "episode_meta.json").exists() and not (e / ".raw_failed").exists()

    done = sum(1 for p in args.out_root.iterdir() if _collected_ok(p)) \
        if args.out_root.exists() else 0
    print(f"[campaign] {len(plan)} (scénario,type) × {args.reps} reps ; déjà {done} épisodes collectés", flush=True)

    print(f"[campaign] ns={args.namespace} reps[{args.rep_start}:{rep_end}] pf_offset={args.pf_offset} "
          f"held-out cap={args.held_out_cap}", flush=True)
    episode_n = 0
    last_ok = time.time()  # référence de l'alerte de stagnation
    for rep in range(args.rep_start, rep_end):
        for (name, is_bug, held_out) in plan:
            # held-out plafonnés (test-only) : pas besoin de 30 reps
            if held_out and rep >= args.held_out_cap:
                continue
            episode_n += 1
            # reprise idempotente : sauter si déjà collecté
            existing = list(args.out_root.glob(f"episode_{name}_{rep:03d}_*"))
            if any(_collected_ok(e) for e in existing):
                continue
            # reset périodique
            mode = "deep" if (episode_n % args.reset_every == 0) else "light"
            subprocess.run([sys.executable, "-m", "collect.reset_tt_state",
                            "--mode", mode, "--namespace", args.namespace,
                            "--cooldown", "30"], cwd=str(V5))
            # Design d'échantillonnage v5.2 — tout DÉTERMINISTE par rep (reprise
            # idempotente : même rep → même config, quel que soit le redémarrage) :
            #  - rotation de cible : round-robin sur le pool (7-8 reps / type×cible)
            #  - pic d'intensité : 60% high / 30% med / 10% low (pannes subtiles)
            #  - charge : 12 / 10 / 14 users (niveaux de trafic)
            pool = pools.get(name)
            target = pool[rep % len(pool)] if (pool and not is_bug) else None
            peak = "high"
            if not is_bug and cats.get(name) not in ("drift", "normal", "overlap"):
                peak = "high" if rep % 10 < 6 else ("med" if rep % 10 < 9 else "low")
            users = [args.users, max(8, args.users - 2), args.users + 2][rep % 3]
            got = collect_episode(name, rep, args.out_root, args.address, users,
                                  is_bug, held_out, args.max_retries, args.namespace,
                                  args.pf_offset, args.ram_ceiling, target, peak,
                                  args.min_cpu_series)
            # Alerte de stagnation : une campagne qui pause proprement pendant des
            # heures reste une campagne qui n'avance pas. On le signale au lieu de
            # le laisser découvrir le lendemain.
            if got:
                last_ok = time.time()
                _stall_flag(args.out_root, args.namespace, None)
            else:
                idle = time.time() - last_ok
                if idle >= args.stall_alert_s:
                    msg = (f"aucun épisode collecté depuis {idle / 3600:.1f} h "
                           f"(ns={args.namespace})")
                    print(f"[campaign] ⚠ ALERTE STAGNATION : {msg}", flush=True)
                    _stall_flag(args.out_root, args.namespace, msg)

    print("[campaign] terminé.", flush=True)


if __name__ == "__main__":
    main()
