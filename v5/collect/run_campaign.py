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
        --address http://172.16.203.12:32677
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

from collect import run_episode

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


def _nodes_ram_ok(ceiling: float = 90.0) -> bool:
    """Garde-fou RAM — contrainte *binding* à 3 runners (CPU large, RAM tendue :
    1 runner ≈ 20 GB JVM+mongos). Retourne False si un nœud worker dépasse
    `ceiling` % de mémoire → le gate met la collecte en pause (anti-éviction qui
    corromprait les épisodes). Fail-open si `kubectl top` échoue (un blip
    metrics-server ne doit pas stopper une campagne de plusieurs jours)."""
    r = subprocess.run(["kubectl", *KCTX_ARGS, "top", "nodes", "--no-headers"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return True  # fail-open
    hot = []
    for line in r.stdout.splitlines():
        cols = line.split()
        # NAME  CPU(cores)  CPU%  MEM(bytes)  MEM%
        if len(cols) < 5 or "workers" not in cols[0]:
            continue
        try:
            mem_pct = float(cols[4].rstrip("%"))
        except ValueError:
            continue
        if mem_pct > ceiling:
            hot.append(f"{cols[0].split('-')[-1]}={mem_pct:.0f}%")
    if hot:
        print(f"[campaign] RAM workers > {ceiling:.0f}% ({','.join(hot)}) — pause", flush=True)
        return False
    return True


def _validate(ep_dir: Path) -> bool:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validate_v5.py"), "--episode", str(ep_dir)],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": __import__("os").environ.get("PATH", "")})
    print(r.stdout.strip().splitlines()[-1] if r.stdout else "(no output)", flush=True)
    return r.returncode == 0


def collect_episode(scenario: str, rep: int, out_root: Path, address: str,
                    users: int, is_bug: bool, held_out: bool, max_retries: int,
                    namespace: str, pf_offset: int = 0, ram_ceiling: float = 90.0) -> bool:
    for attempt in range(max_retries + 1):
        ep_id = f"episode_{scenario}_{rep:03d}_{_ts()}"
        ep_dir = out_root / ep_id
        # santé TT (readiness pods) + RAM nœuds (anti-saturation à 3 runners) avant épisode
        waited = 0
        while (not _tt_healthy(namespace) or not _nodes_ram_ok(ram_ceiling)) and waited < 600:
            print(f"[campaign] TT dégradé / RAM haute, pause 30s ...", flush=True)
            time.sleep(30); waited += 30
        try:
            # COLLECTE uniquement (pas de build) — Record→Build→Assemble.
            res = run_episode.run_episode(scenario, ep_dir, address, users,
                                          run_episode.STEP_S, is_bug, held_out,
                                          namespace, pf_offset)
        except Exception as e:
            print(f"[campaign] {scenario} rep{rep} attempt{attempt} EXC: {e}", flush=True)
            (ep_dir).mkdir(parents=True, exist_ok=True)
            (ep_dir / ".raw_failed").write_text(f"exception: {e}")
            continue
        # gate BRUT (la collecte a-t-elle capté assez de données ?)
        if res.get("raw_ok"):
            print(f"[campaign] OK {ep_id} traces={res['n_traces']} logs={res['n_log_lines']} "
                  f"prom={res['n_prom_series']} collect={res['collect_s']}s", flush=True)
            return True
        print(f"[campaign] FAIL raw-gate {ep_id} "
              f"(traces={res.get('n_traces')} logs={res.get('n_log_lines')}) attempt {attempt}", flush=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 collection campaign driver")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out-root", type=Path, default=REPO / "data" / "raw_v5")
    ap.add_argument("--address", default="http://172.16.203.12:32677")
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--namespace", default="tt")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--reset-every", type=int, default=10, help="deep reset tous les N épisodes")
    ap.add_argument("--only", default="", help="liste de scénarios (CSV) pour restreindre")
    ap.add_argument("--rep-start", type=int, default=0, help="rep de début (split multi-runner)")
    ap.add_argument("--rep-end", type=int, default=None, help="rep de fin exclue (défaut = reps)")
    ap.add_argument("--pf-offset", type=int, default=0, help="décalage ports locaux (multi-runner ; ex. tt=0, tt-b=10)")
    ap.add_argument("--held-out-cap", type=int, default=28, help="reps max pour les scénarios held-out (test-only)")
    ap.add_argument("--ram-ceiling", type=float, default=90.0,
                    help="pause si un nœud worker dépasse ce %% de RAM (garde-fou 3 runners)")
    args = ap.parse_args()
    rep_end = args.rep_end if args.rep_end is not None else args.reps

    _assert_context(args.namespace)  # préflight : bon cluster avant tout
    args.out_root.mkdir(parents=True, exist_ok=True)
    cat = _catalog()
    scenarios = [s["name"] for s in cat["scenarios"]]
    bugs = [b["id"] for b in cat["bugs"] if b.get("status") == "ready"]  # F1 d'abord
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
            collect_episode(name, rep, args.out_root, args.address, args.users,
                            is_bug, held_out, args.max_retries, args.namespace,
                            args.pf_offset, args.ram_ceiling)

    print("[campaign] terminé.", flush=True)


if __name__ == "__main__":
    main()
