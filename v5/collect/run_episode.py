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

# Contexte kubectl épinglé sur toutes les commandes (cf. inject.py / probe.py) :
# immunise la collecte contre une bascule de contexte (vue en session 2026-06-03).
KCTX = os.environ.get("V5_KUBE_CONTEXT", "observit-cluster1")
KCTX_ARGS = ["--context", KCTX]

# Anatomie en steps (× STEP_S secondes). 30 min par défaut ; override possible
# via V5_PHASES="b,pre,ramp,inj,rec" (steps) pour les tests rapides.
PHASES = {"baseline": 12, "pre": 14, "ramp": 6, "injection": 20, "recovery": 8}
if os.environ.get("V5_PHASES"):
    _vals = [int(x) for x in os.environ["V5_PHASES"].split(",")]
    PHASES = dict(zip(["baseline", "pre", "ramp", "injection", "recovery"], _vals))
RAMP_INTENSITIES = ["low", "med", "high"]  # répartis sur la phase ramp


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _run_logged(cmd: list[str], tag: str, **kw) -> subprocess.CompletedProcess:
    """Comme _run mais réaffiche stdout/stderr (les injections bug étaient
    fire-and-forget : un échec de restauration passait silencieusement)."""
    r = _run(cmd, **kw)
    out, err = (r.stdout or "").strip(), (r.stderr or "").strip()
    if out:
        print(f"[{tag}] {out}", flush=True)
    if err:
        print(f"[{tag}] STDERR {err}", flush=True)
    return r


def _restore_bug(scenario: str, bug_svc: str | None, namespace: str, v5: Path,
                 nsargs: list[str], faulty_image: str | None, retries: int = 2) -> bool:
    """Restaure l'état sain APRÈS un bug, de façon vérifiée (corrige une race où
    le delete-bug de l'épisode ne reprenait pas : déploiement laissé sur l'image
    fautive → contamination des épisodes suivants). delete-bug → attente rollout →
    vérif image → retry. Bloque jusqu'à restauration confirmée (ou échec loggé)."""
    for attempt in range(retries + 1):
        _run_logged([sys.executable, "-m", "chaos.inject", "delete-bug", scenario, *nsargs],
                    f"{scenario}/restore", cwd=str(v5))
        if not bug_svc:
            return True
        _run(["kubectl", *KCTX_ARGS, "rollout", "status", "deploy", "-n", namespace, bug_svc,
              "--timeout=300s"])
        if not faulty_image:  # bug non-image (ex. mem_limit) : delete-bug + rollout suffit
            return True
        cur = _run(["kubectl", *KCTX_ARGS, "get", "deploy", "-n", namespace, bug_svc, "-o",
                    "jsonpath={.spec.template.spec.containers[0].image}"]).stdout.strip()
        if cur != faulty_image:
            print(f"[{scenario}/restore] OK image saine = {cur}", flush=True)
            return True
        print(f"[{scenario}/restore] image encore fautive ({cur}) — retry {attempt + 1}/{retries}",
              flush=True)
    print(f"[{scenario}/restore] ÉCHEC: image fautive persistante après {retries + 1} essais", flush=True)
    return False


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


# ───────────────────── Seuils du gate brut (surchargeables par env) ─────────────────────
# L'ancien gate était `traces>0 and logs>0 and prom>0` : un épisode à 12 traces, sans
# aucun service profond tracé et sans faute manifestée, passait. C'est ce qui a laissé
# entrer ~450 épisodes vides (17-20/07) et rendu nécessaire un audit *a posteriori*.
# Les seuils visent le CATASTROPHIQUE (collecte cassée), pas la perfection : un épisode
# sain trace 19-28 services sur 41, donc le plancher est bas à dessein — la qualité fine
# se joue côté loadgen/build, pas en rejetant des épisodes.
MIN_TRACES = int(os.environ.get("V5_MIN_TRACES", "500"))
MIN_TRACED_SERVICES = int(os.environ.get("V5_MIN_TRACED_SERVICES", "15"))
FAULT_MIN_RATIO = float(os.environ.get("V5_FAULT_MIN_RATIO", "1.15"))
# Scénarios sans faute attendue : un score plat y est NORMAL, on ne teste pas.
# `overlap` (θ_drift∩anomaly) porte bien une composante chaos → testé.
NO_FAULT_CATEGORIES = ("normal", "drift")


def _traced_services(traces: list) -> set[str]:
    """Services TT réellement porteurs de spans sur la fenêtre collectée.

    `jaeger["services"]` liste ce que Jaeger CONNAÎT (son historique), pas ce qui a
    été tracé pendant l'épisode : seul le comptage sur les spans mesure la vraie
    couverture de T(t).
    """
    svcs: set[str] = set()
    for tr in traces:
        for proc in (tr.get("processes") or {}).values():
            name = proc.get("serviceName")
            if name and name.startswith("ts-"):
                svcs.add(name)
    return svcs


def _p99(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * 0.99), len(ordered) - 1)]


def _span_stats(traces: list, lo: float, hi: float) -> dict[str, dict]:
    """Par service : {durées ms, n_spans, n_err} pour les spans démarrés dans [lo, hi[.

    L'agrégation est PAR SERVICE, pas globale : le chaos frappe un service parmi 41
    (`mode: one`), donc son effet se dilue complètement dans un p99 calculé sur
    l'union des spans. C'est ce qui faisait rejeter des épisodes parfaitement
    valides avec un score collé à 1.00 (constaté le 2026-07-27).
    """
    out: dict[str, dict] = {}
    for tr in traces:
        procs = tr.get("processes") or {}
        for sp in tr.get("spans") or []:
            start = sp.get("startTime")
            if start is None or not (lo <= start / 1e6 < hi):
                continue
            svc = (procs.get(sp.get("processID")) or {}).get("serviceName")
            if not svc:
                continue
            rec = out.setdefault(svc, {"dur": [], "n": 0, "err": 0})
            rec["dur"].append(sp.get("duration", 0) / 1000.0)
            rec["n"] += 1
            for tag in sp.get("tags") or []:
                if tag.get("key") == "error" and tag.get("value") in (True, "true"):
                    rec["err"] += 1
                    break
    return out


def _fault_visible(traces: list, t_start: float, boundaries: dict) -> tuple[bool, float]:
    """La faute injectée a-t-elle laissé une signature mesurable dans les traces ?

    Même principe que `scripts/fault_presence.py` : rapport injection/baseline sur la
    meilleure de deux grandeurs qui bougent pour *tous* les types de faute (latence,
    erreurs), et surtout **max à travers les services** — le chaos ne frappe qu'un
    service, c'est donc le pire touché qui porte le signal. Calculé sur les spans
    BRUTS car au moment du gate `signal.npz` n'existe pas encore (Record → Build →
    Assemble).

    On compare le DERNIER TIERS de l'injection (intensité au pic, le ramp monte
    progressivement) à la baseline. Cible le mode d'échec réel — « le chaos n'a jamais
    été appliqué » donne un rapport ≈ 1.0 (cas chaos-daemon absent, 17-20/07) — sans
    rejeter les fautes subtiles (peak=low), d'où un seuil bas. Retourne (visible,
    score) ; score NaN et visible True si trop peu de spans pour conclure : on ne
    bloque jamais sur une incertitude.
    """
    base_lo = t_start + boundaries.get("baseline_start", 0.0)
    inj_lo = t_start + boundaries.get("injection_start", 0.0)
    inj_hi = t_start + boundaries.get("injection_end", 0.0)
    peak_lo = inj_lo + (inj_hi - inj_lo) * 2.0 / 3.0
    base = _span_stats(traces, base_lo, inj_lo)
    peak = _span_stats(traces, peak_lo, inj_hi)
    best = float("nan")
    for svc, p in peak.items():
        b = base.get(svc)
        if not b or b["n"] < 20 or p["n"] < 10:
            continue  # trop peu de spans pour ce service : on ne conclut pas
        ratio_lat = _p99(p["dur"]) / max(_p99(b["dur"]), 1e-6)
        base_err, peak_err = b["err"] / b["n"], p["err"] / p["n"]
        if base_err > 1e-6:
            ratio_err = peak_err / base_err
        else:
            ratio_err = 5.0 if peak_err > 1e-6 else 1.0
        score = max(ratio_lat, ratio_err)
        if best != best or score > best:  # best est NaN au premier passage
            best = score
    if best != best:
        return True, float("nan")
    return best >= FAULT_MIN_RATIO, best


def run_episode(scenario: str, out: Path, address: str, users: int, step: int,
                is_bug: bool, held_out: bool, namespace: str = "tt",
                pf_offset: int = 0, target: str | None = None,
                peak_intensity: str = "high") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    v5 = _v5_dir()
    import yaml
    catalog = yaml.safe_load(open(v5 / "chaos" / "catalog.yaml"))
    category, targets, _kind = _category_of(scenario, catalog)
    if target:
        targets = [target]  # rotation : la cible choisie devient la vérité des labels
    nsargs = ["--namespace", namespace]  # passé à chaos.inject

    # Jitter d'onset : baseline/pre tirés PAR ÉPISODE (seed = nom d'épisode →
    # déterministe à la reprise). Sans jitter, tous les épisodes injectent au même
    # step → un modèle peut apprendre la POSITION au lieu du signal (fuite
    # positionnelle, cf. stress test A1 v3). V5_PHASES (mode test) désactive.
    phases = dict(PHASES)
    if not os.environ.get("V5_PHASES"):
        import random as _random
        _rng = _random.Random(f"jitter:{out.name}")
        phases["baseline"] = _rng.randint(8, 16)
        phases["pre"] = _rng.randint(10, 18)
    dur = {k: phases[k] * step for k in phases}
    total = sum(dur.values())

    # Charge = mix nominal (NOMINAL_MIX) pour TOUS les épisodes, y compris bugs.
    # Le champ catalog `load:` (charge ciblée mono-scénario) a été testé pour les
    # bugs (2026-06-03) et ABANDONNÉ : query_and_cancel seul → couverture trace
    # 8/41 (< plancher 18, validate FAIL), ne trace même pas voucher, et ne fait
    # PAS émerger F1 (bug de logique async, invisible en télémétrie infra/trace
    # quelle que soit la charge). Le mix nominal donne 29/41 tracés et passe le gate.
    load = subprocess.Popen(
        [sys.executable, "-m", "loadgen.runner", "--address", address,
         "--users", str(users), "--duration", str(total + 30), "--rps-log", "300"],
        cwd=str(v5), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    boundaries: dict[str, float] = {}
    t_start = time.time()

    def mark(name):
        boundaries[name] = time.time() - t_start

    # Nettoyage garanti : un épisode INTERROMPU (exception, kill) ne doit jamais
    # laisser de traînée (manifests chaos orphelins / image fautive) — vu 2× en
    # campagne : les reliquats contaminent les épisodes suivants du namespace.
    bug_svc = targets[0] if (is_bug and targets) else None
    faulty_image = next((b.get("image") for b in catalog.get("bugs", [])
                         if b["id"] == scenario), None) if is_bug else None
    cleaned = False
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
            print(f"[{scenario}] inject bug ({scenario}) sur {bug_svc} ...", flush=True)
            _run_logged([sys.executable, "-m", "chaos.inject", "apply-bug", scenario, *nsargs],
                        f"{scenario}/apply-bug", cwd=str(v5))
            if bug_svc:
                print(f"[{scenario}] attente redémarrage pod fautif ...", flush=True)
                _run(["kubectl", *KCTX_ARGS, "rollout", "status", "deploy", "-n", namespace,
                      bug_svc, "--timeout=600s"])
            # fenêtre active du bug (charge tourne, la panne se manifeste)
            time.sleep(dur["ramp"] + dur["injection"])
            print(f"[{scenario}] restauration bug (vérifiée) ...", flush=True)
            _restore_bug(scenario, bug_svc, namespace, v5, nsargs, faulty_image)
        elif category == "normal":
            # Run sain complet : AUCUNE injection. La fenêtre « injection » du
            # timeline est vide ; featurize (is_normal) garde regime=normal partout.
            print(f"[{scenario}] normal : aucune injection, fenêtre "
                  f"{dur['ramp'] + dur['injection']}s ...", flush=True)
            time.sleep(dur["ramp"] + dur["injection"])
        elif category in ("drift", "overlap"):
            # Drift bénin OU overlap (θ_drift∩anomaly) : UNE seule action au début
            # de la fenêtre (pas de ramp d'intensité, sinon apply répété →
            # rollouts multiples + écrasement du replica baseline sauvegardé →
            # restauration cassée). Pour overlap, inject applique drift natif +
            # chaos (intensité high) simultanément sur le même service.
            win = dur["ramp"] + dur["injection"]
            print(f"[{scenario}] {category} (natif) sur {targets} ...", flush=True)
            _run([sys.executable, "-m", "chaos.inject", "apply", scenario,
                  "--intensity", "high", "--duration", f"{win}s", *nsargs], cwd=str(v5))
            time.sleep(win)
            _run([sys.executable, "-m", "chaos.inject", "delete", scenario, *nsargs], cwd=str(v5))
        else:
            # ramp-up : intensité croissante, TRONQUÉE au pic demandé — les épisodes
            # plafonnés à low/med échantillonnent des pannes subtiles (spectre de
            # difficulté early-warning), au lieu de toujours finir à high.
            tgt_args = ["--target", target] if target else []
            ramp_levels = RAMP_INTENSITIES[:RAMP_INTENSITIES.index(peak_intensity) + 1]
            ramp_each = dur["ramp"] / len(ramp_levels)
            for inten in ramp_levels:
                print(f"[{scenario}] ramp intensité={inten} ...", flush=True)
                _run([sys.executable, "-m", "chaos.inject", "apply", scenario,
                      "--intensity", inten, "--duration", f"{int(ramp_each)+2}s",
                      *tgt_args, *nsargs], cwd=str(v5))
                time.sleep(ramp_each)
            # injection stable au pic
            print(f"[{scenario}] injection {peak_intensity} {dur['injection']}s ...", flush=True)
            _run([sys.executable, "-m", "chaos.inject", "apply", scenario,
                  "--intensity", peak_intensity, "--duration", f"{dur['injection']}s",
                  *tgt_args, *nsargs], cwd=str(v5))
            time.sleep(dur["injection"])
            _run([sys.executable, "-m", "chaos.inject", "delete", scenario, *nsargs], cwd=str(v5))

        cleaned = True  # chaque branche a fait son delete/restore
        mark("injection_end")
        print(f"[{scenario}] recovery {dur['recovery']}s ...", flush=True)
        time.sleep(dur["recovery"])
        mark("recovery_end")
    finally:
        load.terminate()
        if not cleaned:
            # épisode interrompu AVANT son nettoyage : purge best-effort
            print(f"[{scenario}] interrompu — nettoyage chaos/bug de secours", flush=True)
            try:
                if is_bug:
                    _restore_bug(scenario, bug_svc, namespace, v5, nsargs, faulty_image)
                elif category != "normal":
                    _run([sys.executable, "-m", "chaos.inject", "delete", scenario, *nsargs],
                         cwd=str(v5))
            except Exception as e:  # le nettoyage ne doit pas masquer l'erreur d'origine
                print(f"[{scenario}] nettoyage de secours échoué: {e}", flush=True)
    t_end = time.time()

    # collecte (port-forwards namespacés + offset pour coexistence multi-runner).
    # Délai de drainage : Jaeger all-in-one est très lent à INTERROGER tant qu'il
    # ingère le flux de spans de la charge ; on laisse 20 s après l'arrêt de la
    # charge pour qu'il draine avant de requêter (sinon /api/traces explose).
    print(f"[{scenario}] drainage 20s puis collecte fenêtre {total}s (ns={namespace}, NodePort) ...", flush=True)
    time.sleep(20)
    # Collecte en DIRECT via NodePort (plus de port-forward : cf. probe.nodeport_bases).
    # 3 pulls concurrents. Jaeger : chunks larges (300 s) → ÷5 le nombre d'appels.
    from concurrent.futures import ThreadPoolExecutor

    def _collect_once():
        timings = {}

        def _timed(name, fn, *a, **k):
            _t = time.time(); r = fn(*a, **k); timings[name] = round(time.time() - _t, 1); return r

        with ThreadPoolExecutor(max_workers=3) as ex:
            f_prom = ex.submit(_timed, "prom", probe.pull_prometheus, t_start, t_end, step, namespace)
            f_jae = ex.submit(_timed, "jaeger", probe.pull_jaeger, t_start, t_end, 300, 1500, namespace)
            f_loki = ex.submit(_timed, "loki", probe.pull_loki, t_start, t_end, step, namespace)
            return f_prom.result(), f_jae.result(), f_loki.result(), timings

    # Retry du COLLECT (pas des 33 min de phases) : sur un cluster instable (nœuds
    # taintés/drainés, pods qui churent), un blip de quelques s pendant le pull
    # (timed out / connection reset) faisait perdre tout l'épisode. On ré-essaie le
    # pull jusqu'à 3× avec pause → on absorbe les blips au lieu de jeter 33 min.
    prom, jae, loki = {}, {}, {}
    for attempt in range(3):
        try:
            prom, jae, loki, timings = _collect_once()
            print(f"[{scenario}] pull timings: {timings}", flush=True)
            break
        except Exception as e:
            print(f"[{scenario}] collecte échec essai {attempt + 1}/3 ({e}) — retry 25s", flush=True)
            time.sleep(25)
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
        "held_out": held_out, "namespace": namespace, "step": step, "ramp_s": dur["ramp"], "t_start": t_start,
        "phases": phases,  # phases effectives (jitter d'onset par épisode)
        "peak_intensity": peak_intensity,
        # valeur numérique du pic pour intensity_t (featurize plafonne dessus)
        "peak_value": {"low": 1 / 3, "med": 2 / 3, "high": 1.0}[peak_intensity],
        "boundaries_rel": {  # secondes relatives au début de la fenêtre de collecte
            "baseline_start": boundaries.get("baseline_start", 0.0),
            "injection_start": boundaries["injection_start"],
            "injection_end": boundaries["injection_end"],
            "recovery_end": boundaries["recovery_end"],
        },
    }
    json.dump(meta, open(out / "episode_meta.json", "w"), indent=2)

    # contrôle qualité BRUT (léger, pas de build) — gate de collecte.
    # On ne demande plus « des données existent-elles ? » mais « sont-elles
    # exploitables ? » : volume, couverture des services tracés, et manifestation
    # effective de la faute (cf. seuils en tête de module).
    n_traces = jae.get("n_traces_total", 0)
    n_logs = loki.get("n_lines", 0)
    n_prom = len(prom.get("cpu", [])) if isinstance(prom.get("cpu"), list) else 0
    traces_list = jae.get("traces") or []
    n_svc = len(_traced_services(traces_list))

    reasons = []
    if n_traces < MIN_TRACES:
        reasons.append(f"traces={n_traces}<{MIN_TRACES}")
    if n_logs <= 0:
        reasons.append("logs=0")
    if n_prom <= 0:
        reasons.append("prom=0")
    if n_svc < MIN_TRACED_SERVICES:
        reasons.append(f"services_tracés={n_svc}<{MIN_TRACED_SERVICES}")
    fault_score = float("nan")
    if category not in NO_FAULT_CATEGORIES:
        visible, fault_score = _fault_visible(traces_list, t_start, boundaries)
        if not visible:
            reasons.append(f"faute invisible (score={fault_score:.2f}<{FAULT_MIN_RATIO})")

    ok = not reasons
    if not ok:
        # Format compatible avec le triage existant (`traces=… logs=… prom=…`),
        # enrichi du motif exact pour ne plus avoir à deviner à l'audit.
        (out / ".raw_failed").write_text(
            f"traces={n_traces} logs={n_logs} prom={n_prom} services={n_svc} "
            f"fault_score={fault_score:.2f} | " + "; ".join(reasons))
    return {
        "episode_id": episode_id, "raw_ok": ok,
        "n_traces": n_traces, "n_log_lines": n_logs, "n_prom_series": n_prom,
        "n_traced_services": n_svc, "fault_score": fault_score,
        "gate_reasons": reasons,
        "collect_s": round(time.time() - t_end, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="EWAT v5 episode orchestrator (T=60)")
    p.add_argument("--scenario", required=True)
    p.add_argument("--category", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--address", default="http://<CLUSTER_NODE_IP>:32677")
    p.add_argument("--users", type=int, default=12)
    p.add_argument("--step", type=int, default=STEP_S)
    p.add_argument("--bug", action="store_true")
    p.add_argument("--held-out", action="store_true")
    p.add_argument("--namespace", default="tt")
    p.add_argument("--pf-offset", type=int, default=0, help="décalage ports locaux (multi-runner)")
    p.add_argument("--target", default=None, help="cible (rotation ; doit appartenir au target_pool)")
    p.add_argument("--peak-intensity", default="high", choices=["low", "med", "high"],
                   help="pic d'intensité de l'injection (pannes subtiles = low/med)")
    args = p.parse_args()
    res = run_episode(args.scenario, Path(args.out), args.address, args.users,
                      args.step, args.bug, args.held_out, args.namespace, args.pf_offset,
                      args.target, args.peak_intensity)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
