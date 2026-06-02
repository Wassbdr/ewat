"""EWAT v5 — orchestrateur d'épisode Train Ticket (anatomie 30 min, T=60).

Enchaîne baseline → pre-injection → ramp-up → injection → recovery avec charge
continue + injection chaos (ou swap bug F), collecte les 3 sources sur toute la
fenêtre, et construit le contrat per-épisode v4-conforme via build_features_v5.

Anatomie par défaut (step 30 s ⇒ 30 min) :
    baseline 12 · pre 14 · ramp 6 · injection 20 · recovery 8   = 60 steps.
Le ramp-up monte l'intensité low→med→high sur la phase ramp (angle précursion),
puis high stable en injection.

Usage (PYTHONPATH inclut src/) :
    python -m collect.run_episode --scenario cpu_stress --category contention \
        --out data/raw_v5/episode_cpu_stress_000_<tsZ>
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from collect import probe

import os

STEP_S = 30

# Anatomie en steps (× STEP_S secondes). 30 min par défaut ; override possible
# via V5_PHASES="b,pre,ramp,inj,rec" (steps) pour les tests rapides.
PHASES = {"baseline": 12, "pre": 14, "ramp": 6, "injection": 20, "recovery": 8}
if os.environ.get("V5_PHASES"):
    _vals = [int(x) for x in os.environ["V5_PHASES"].split(",")]
    PHASES = dict(zip(["baseline", "pre", "ramp", "injection", "recovery"], _vals))
RAMP_INTENSITIES = ["low", "med", "high"]  # répartis sur la phase ramp


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _v5_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _category_of(scenario: str, catalog: dict) -> tuple[str, list[str], str]:
    """Retourne (category, target_services, kind) depuis le catalogue chaos."""
    for s in catalog.get("scenarios", []):
        if s["name"] == scenario:
            tgt = s.get("target") or (s["parts"][0]["target"] if s.get("parts") else "")
            return s.get("category", "unknown"), [tgt] if tgt else [], s.get("kind", "")
    for b in catalog.get("bugs", []):
        if b["id"] == scenario:
            return "bug", [b.get("service", "")], "bug"
    return "unknown", [], ""


def run_episode(scenario: str, out: Path, address: str, users: int, step: int,
                is_bug: bool, held_out: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    v5 = _v5_dir()
    import yaml
    catalog = yaml.safe_load(open(v5 / "chaos" / "catalog.yaml"))
    category, targets, _kind = _category_of(scenario, catalog)

    dur = {k: PHASES[k] * step for k in PHASES}
    total = sum(dur.values())

    # charge continue
    load = subprocess.Popen(
        [sys.executable, "-m", "loadgen.runner", "--address", address,
         "--users", str(users), "--duration", str(total + 30), "--rps-log", "300"],
        cwd=str(v5), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    boundaries: dict[str, float] = {}
    t_start = time.time()

    def mark(name):
        boundaries[name] = time.time() - t_start

    try:
        mark("baseline_start")
        print(f"[{scenario}] baseline {dur['baseline']}s + pre {dur['pre']}s ...", flush=True)
        time.sleep(dur["baseline"] + dur["pre"])
        mark("injection_start")  # ramp + injection comptent comme régime injection

        if is_bug:
            # Un bug (swap image / patch mem-limit) déclenche un reboot du pod ;
            # sous pression CPU il met plusieurs minutes à redémarrer. On attend
            # que le pod fautif soit prêt AVANT de compter la fenêtre active,
            # sinon on ne capte que le reboot et pas la signature de la panne.
            bug_svc = targets[0] if targets else None
            print(f"[{scenario}] inject bug ({scenario}) sur {bug_svc} ...", flush=True)
            _run([sys.executable, "-m", "chaos.inject", "apply-bug", scenario], cwd=str(v5))
            if bug_svc:
                print(f"[{scenario}] attente redémarrage pod fautif ...", flush=True)
                _run(["kubectl", "rollout", "status", "deploy", "-n", "tt", bug_svc,
                      "--timeout=600s"])
            # fenêtre active du bug (charge tourne, la panne se manifeste)
            time.sleep(dur["ramp"] + dur["injection"])
            print(f"[{scenario}] restauration bug ...", flush=True)
            _run([sys.executable, "-m", "chaos.inject", "delete-bug", scenario], cwd=str(v5))
        else:
            # ramp-up : intensité croissante
            ramp_each = dur["ramp"] / len(RAMP_INTENSITIES)
            for inten in RAMP_INTENSITIES:
                print(f"[{scenario}] ramp intensité={inten} ...", flush=True)
                _run([sys.executable, "-m", "chaos.inject", "apply", scenario,
                      "--intensity", inten, "--duration", f"{int(ramp_each)+2}s"], cwd=str(v5))
                time.sleep(ramp_each)
            # injection stable high
            print(f"[{scenario}] injection high {dur['injection']}s ...", flush=True)
            _run([sys.executable, "-m", "chaos.inject", "apply", scenario,
                  "--intensity", "high", "--duration", f"{dur['injection']}s"], cwd=str(v5))
            time.sleep(dur["injection"])
            _run([sys.executable, "-m", "chaos.inject", "delete", scenario], cwd=str(v5))

        mark("injection_end")
        print(f"[{scenario}] recovery {dur['recovery']}s ...", flush=True)
        time.sleep(dur["recovery"])
        mark("recovery_end")
    finally:
        load.terminate()
    t_end = time.time()

    # collecte
    print(f"[{scenario}] collecte fenêtre {total}s ...", flush=True)
    procs = probe._ensure_pf()
    try:
        prom = probe.pull_prometheus(t_start, t_end, step)
        jae = probe.pull_jaeger(t_start, t_end, chunk_s=60, limit=1500)
        loki = probe.pull_loki(t_start, t_end, chunk_s=step)
    finally:
        for p in procs:
            p.terminate()
    for name, data in [("prometheus", prom), ("jaeger", jae), ("loki", loki)]:
        with gzip.open(out / f"{name}.json.gz", "wt") as f:
            json.dump(data, f)

    # === SÉPARATION collecte/build (Record → Build → Assemble) ===
    # On NE build PAS ici. On écrit episode_meta.json avec tout ce dont la Phase 2
    # offline (build_features_v5 --raw-root) a besoin pour reconstruire le contrat
    # + les labels (boundaries relatives + ramp). Les dumps bruts sont sacrés.
    episode_id = out.name
    meta = {
        "episode_id": episode_id, "scenario": scenario, "category": category,
        "targets": targets, "chaos_resource": (f"v5-{scenario}" if not is_bug else f"bug-{scenario}"),
        "is_bug": is_bug, "bug_id": (scenario if is_bug else None),
        "held_out": held_out, "step": step, "ramp_s": dur["ramp"], "t_start": t_start,
        "boundaries_rel": {  # secondes relatives au début de la fenêtre de collecte
            "baseline_start": boundaries.get("baseline_start", 0.0),
            "injection_start": boundaries["injection_start"],
            "injection_end": boundaries["injection_end"],
            "recovery_end": boundaries["recovery_end"],
        },
    }
    json.dump(meta, open(out / "episode_meta.json", "w"), indent=2)

    # contrôle qualité BRUT (léger, pas de build) — gate de collecte
    n_traces = jae.get("n_traces_total", 0)
    n_logs = loki.get("n_lines", 0)
    n_prom = len(prom.get("cpu", [])) if isinstance(prom.get("cpu"), list) else 0
    ok = n_traces > 0 and n_logs > 0 and n_prom > 0
    if not ok:
        (out / ".raw_failed").write_text(f"traces={n_traces} logs={n_logs} prom={n_prom}")
    return {
        "episode_id": episode_id, "raw_ok": ok,
        "n_traces": n_traces, "n_log_lines": n_logs, "n_prom_series": n_prom,
        "collect_s": round(time.time() - t_end, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="EWAT v5 episode orchestrator (T=60)")
    p.add_argument("--scenario", required=True)
    p.add_argument("--category", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--address", default="http://172.16.203.12:32677")
    p.add_argument("--users", type=int, default=12)
    p.add_argument("--step", type=int, default=STEP_S)
    p.add_argument("--bug", action="store_true")
    p.add_argument("--held-out", action="store_true")
    args = p.parse_args()
    res = run_episode(args.scenario, Path(args.out), args.address, args.users,
                      args.step, args.bug, args.held_out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
