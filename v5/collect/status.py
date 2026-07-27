"""EWAT v5 — état de la campagne de collecte (live, pas de log périmé).

Le suivi historique affichait la DERNIÈRE LIGNE de log de chaque runner. Or un
épisode dure ~33 min en produisant très peu de sorties : un runner qui a pausé
puis repris garde le message de pause affiché pendant de longues minutes et
paraît bloqué alors qu'il travaille. Le 2026-07-27, ça a fait diagnostiquer à
tort un runner en panne — il collectait normalement.

Ici, la source de vérité est l'ÉTAT DU CLUSTER, pas le log :
  - un chaos actif (avec son âge) = un épisode est réellement en cours ;
  - la couverture cAdvisor dit si le gate télémétrie laisse passer ;
  - le marqueur `_STALLED_<ns>` n'apparaît qu'après 1 h sans épisode abouti,
    donc jamais sur une pause passagère.
La dernière ligne de log reste affichée, mais HORODATÉE : sa péremption se voit.

Usage :
    python -m collect.status --out-root ../data/raw_v5_2
    watch -n60 'python -m collect.status --out-root ../data/raw_v5_2'
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

NAMESPACES = ("tt", "tt-b", "tt-c")
CHAOS_KINDS = ("stresschaos", "networkchaos", "podchaos", "dnschaos",
               "timechaos", "iochaos")
KCTX = os.environ.get("V5_KUBE_CONTEXT", "observit-cluster1")
KC = ["kubectl", "--context", KCTX]
NODE_IP = os.environ.get("V5_NODE_IP", "172.16.203.12")
PROM_NP = os.environ.get("V5_PROM_NODEPORT", "32700")
MIN_CPU_SERIES = 40  # même seuil que run_campaign._backends_scraping


def _sh(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}min"
    return f"{seconds / 3600:.1f}h"


def _chaos_state(ns: str) -> tuple[int, int, str]:
    """(chaos en cours, chaos ORPHELINS, détail).

    Un CR Chaos Mesh ne disparaît pas à l'expiration de sa `duration` : il reste
    jusqu'à suppression explicite. Sa simple présence ne prouve donc PAS qu'un
    épisode tourne — vu le 2026-07-27 sur tt-c : un `container-kill` de 62 s
    encore `AllInjected=True, AllRecovered=False` 13 min plus tard, qui étranglait
    le namespace pendant que son runner paraissait juste « en pause ».

    On distingue donc via le statut : non recovered ET plus vieux que sa duration
    = orphelin (anomalie à nettoyer), pas activité normale.
    """
    import json as _json
    running = orphan = 0
    detail = []
    for kind in CHAOS_KINDS:
        out = _sh([*KC, "get", kind, "-n", ns, "-o", "json"])
        if not out.strip():
            continue
        try:
            items = _json.loads(out).get("items", [])
        except Exception:
            continue
        for it in items:
            name = it["metadata"]["name"]
            created = it["metadata"].get("creationTimestamp", "")
            spec_dur = str(it["spec"].get("duration", "")).rstrip("s")
            raw_conds = it.get("status", {}).get("conditions", [])
            conds = {c.get("type"): c.get("status") for c in raw_conds}
            recovered = conds.get("AllRecovered") == "True"
            age_s = 0.0
            try:
                t0 = dt.datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=dt.UTC)
                age_s = (dt.datetime.now(dt.UTC) - t0).total_seconds()
            except Exception:
                pass
            try:
                dur_s = float(spec_dur)
            except ValueError:
                dur_s = 0.0
            # marge : le contrôleur peut mettre quelques dizaines de s à réconcilier
            if not recovered and dur_s and age_s > dur_s + 120:
                orphan += 1
                detail.append(f"{name} ORPHELIN {_age(age_s)}>{spec_dur}s")
            else:
                running += 1
                detail.append(f"{name} {_age(age_s)}")
    return running, orphan, ", ".join(detail)


def _ready(ns: str) -> tuple[int, int]:
    out = _sh([*KC, "get", "pods", "-n", ns, "--no-headers"])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return sum(1 for ln in lines if "1/1" in ln.split()[1:2]), len(lines)


def _cadvisor_series(ns: str) -> int:
    q = f'count(container_cpu_usage_seconds_total{{namespace="{ns}",container!=""}})'
    url = (f"http://{NODE_IP}:{PROM_NP}/api/v1/query?"
           + urllib.parse.urlencode({"query": q}))
    try:
        import json
        with urllib.request.urlopen(url, timeout=8) as r:
            res = json.load(r).get("data", {}).get("result", [])
        return int(float(res[0]["value"][1])) if res else 0
    except Exception:
        return -1  # inconnu


def _last_log(out_root: Path, ns: str) -> tuple[str, float]:
    """(dernière ligne utile, âge du fichier en secondes)."""
    log = out_root / f"_campaign_{ns}.log"
    if not log.exists():
        return "(pas de log)", -1.0
    try:
        tail = _sh(["tail", "-n", "40", str(log)])
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        return (lines[-1][:70] if lines else "(vide)"), time.time() - log.stat().st_mtime
    except Exception:
        return "(illisible)", -1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 — état live de la collecte")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--target", type=int, default=890, help="objectif d'épisodes")
    args = ap.parse_args()
    root: Path = args.out_root

    eps = sorted(root.glob("episode_*"))
    built = sum(1 for e in eps if (e / "signal.npz").exists())
    failed = sum(1 for e in eps if (e / ".raw_failed").exists())
    pct = 100 * len(eps) / max(args.target, 1)
    now = dt.datetime.now().strftime("%H:%M:%S")

    print(f"══════ EWAT v5.2 — {now} ══════")
    print(f"épisodes: {len(eps)} / ~{args.target} ({pct:.0f}%)   "
          f"échecs: {failed}   buildés: {built}")

    # dernier épisode abouti : le vrai indicateur d'avancement
    metas = [e for e in eps if (e / "episode_meta.json").exists()]
    if metas:
        newest = max(metas, key=lambda p: (p / "episode_meta.json").stat().st_mtime)
        idle = time.time() - (newest / "episode_meta.json").stat().st_mtime
        flag = "  ⚠ STAGNATION" if idle > 3600 else ""
        print(f"dernier épisode abouti: il y a {_age(idle)}{flag}")

    print("\n── runners (état LIVE du cluster) ──")
    for ns in NAMESPACES:
        running, orphan, detail = _chaos_state(ns)
        ready, total = _ready(ns)
        series = _cadvisor_series(ns)
        gate = "OK" if series >= MIN_CPU_SERIES else (
            "?" if series < 0 else f"PAUSE ({series}<{MIN_CPU_SERIES})")
        if orphan:
            state = f"⚠ {orphan} CHAOS ORPHELIN — à nettoyer"
        elif running:
            state = "épisode en cours"
        else:
            state = "entre deux épisodes"
        stalled = (root / f"_STALLED_{ns}").exists()
        line, log_age = _last_log(root, ns)
        print(f"  {ns:<5} {state:<36} pods={ready}/{total}  cAdvisor={gate}"
              f"{'  ⚠ STALLED' if stalled else ''}")
        if detail:
            print(f"        chaos: {detail}")
        print(f"        log (il y a {_age(log_age)}) : {line}")

    procs = _sh(["pgrep", "-fc", "collect.run_campaign"]).strip() or "0"
    print(f"\n── procs: {procs} runner(s) (attendu 3) ──")


if __name__ == "__main__":
    main()
