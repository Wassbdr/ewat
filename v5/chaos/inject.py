"""EWAT v5 — injecteur de chaos Train Ticket depuis catalog.yaml.

Rend un scénario du catalogue (avec intensité low/med/high) en manifeste(s)
Chaos Mesh et l'applique / le supprime via kubectl. Gère les scénarios simples,
composites (plusieurs ressources), et les bugs réels (swap d'image).

Usage :
    python -m chaos.inject list
    python -m chaos.inject apply cpu_stress --intensity high
    python -m chaos.inject delete cpu_stress
    python -m chaos.inject apply-bug F1
    python -m chaos.inject delete-bug F1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CATALOG = Path(__file__).parent / "catalog.yaml"

# Contexte kubectl ÉPINGLÉ sur toutes les commandes (chemin d'échec silencieux :
# si le contexte courant bascule, inject taperait le mauvais cluster sans erreur
# → chaos appliqué ailleurs / pas appliqué → épisodes corrompus en douce).
# Configurable (V5_KUBE_CONTEXT) pour la VM ; défaut = cluster de prod EWAT.
_KCTX = os.environ.get("V5_KUBE_CONTEXT", "observit-cluster1")
_KC = ["kubectl", "--context", _KCTX]


def _load() -> dict:
    with open(CATALOG) as f:
        return yaml.safe_load(f)


def _subst(obj, repl: dict):
    """Substitue récursivement les placeholders {{KEY}} dans les valeurs str."""
    if isinstance(obj, dict):
        return {k: _subst(v, repl) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, repl) for v in obj]
    if isinstance(obj, str):
        for k, v in repl.items():
            obj = obj.replace("{{" + k + "}}", str(v))
        return obj
    return obj


def _repl_for(cat: dict, intensity: str) -> dict:
    lv = cat["intensity_levels"][intensity]
    return {
        "CPU_LOAD": lv["cpu_load"],
        "MEM_SIZE": lv["mem_size"],
        "LATENCY": lv["latency"],
        "LOSS": lv["loss"],
        "WORKERS": lv["workers"],
    }


# Champs Chaos Mesh qui exigent un entier (le reste reste string).
_INT_KEYS = {"load", "workers", "percent", "limit", "buffer"}


def _cast_ints(obj):
    """Caste en int les valeurs des clés numériques exigées par Chaos Mesh."""
    if isinstance(obj, dict):
        return {
            k: (int(v) if k in _INT_KEYS and isinstance(v, str) and v.lstrip("-").isdigit()
                else _cast_ints(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_cast_ints(v) for v in obj]
    return obj


def _selector(ns: str, target: str) -> dict:
    return {"namespaces": [ns], "labelSelectors": {"app": target}}


def _rfc1123(name: str) -> str:
    """Nom de ressource K8s valide (RFC 1123 : pas d'underscore)."""
    return name.replace("_", "-")


def _render(scn: dict, ns: str, repl: dict, name_suffix: str = "") -> list[dict]:
    """Rend un scénario en une liste de manifestes Chaos Mesh."""
    manifests = []
    base = _rfc1123(scn["name"])
    if scn.get("kind") == "composite":
        for i, part in enumerate(scn["parts"]):
            spec = _subst(copy.deepcopy(part["spec"]), repl)
            spec["selector"] = _selector(ns, part["target"])
            spec.setdefault("duration", "{{DURATION}}")
            manifests.append({
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": part["kind"],
                "metadata": {"name": f"v5-{base}-{i}{name_suffix}", "namespace": ns},
                "spec": spec,
            })
    else:
        spec = _subst(copy.deepcopy(scn["spec"]), repl)
        spec["selector"] = _selector(ns, scn["target"])
        spec.setdefault("duration", "{{DURATION}}")
        manifests.append({
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": scn["kind"],
            "metadata": {"name": f"v5-{base}{name_suffix}", "namespace": ns},
            "spec": spec,
        })
    return manifests


def _kubectl(action: str, manifests: list[dict]) -> None:
    doc = "\n---\n".join(yaml.safe_dump(m) for m in manifests)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(doc)
        path = f.name
    cmd = [*_KC, action, "-f", path]
    if action == "delete":
        cmd.append("--wait=false")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())


def _get_scn(cat: dict, name: str) -> dict:
    for s in cat["scenarios"]:
        if s["name"] == name:
            return s
    raise SystemExit(f"scénario inconnu: {name}")


def cmd_list(cat: dict) -> None:
    print(f"{'SCÉNARIO':<28} {'CATÉGORIE':<12} {'KIND':<14} CIBLE")
    for s in cat["scenarios"]:
        tgt = s.get("target", "+".join(p["target"] for p in s.get("parts", [])))
        print(f"{s['name']:<28} {s['category']:<12} {s['kind']:<14} {tgt}")
    print("\nBUGS (swap image, test-only):")
    for b in cat["bugs"]:
        print(f"  {b['id']:<5} {b['status']:<9} {b['service']:<28} {b['image'] or '(à builder)'}")


# ─────────────────────── Drift bénin (pas de Chaos Mesh) ───────────────────
# θ_drift = dérive bénigne (rolling deploy, autoscaling) : PAS une anomalie.
# Réalisé par des opérations kubectl natives (restart/scale), pas des manifestes
# Chaos Mesh. L'épisode reste labellé regime=normal + drift_flag (cf.
# build_features_v5 : category=="drift" → régime non-anomalie, fenêtre drift_flag).
def _drift_state_path(name: str, ns: str) -> str:
    # namespacé : 2 runners (tt / tt-b) sur le même scénario ne doivent pas
    # s'écraser mutuellement le replica baseline sauvegardé.
    return f"/tmp/ewat_drift_{ns}_{_rfc1123(name)}.json"


def _apply_drift(cat: dict, scn: dict) -> None:
    ns, svc, kind = cat["namespace"], scn["target"], scn["kind"]
    if kind == "rollout":
        print(f"[drift] rollout restart {svc}")
        subprocess.run([*_KC, "rollout", "restart", "-n", ns, f"deploy/{svc}"], check=False)
    elif kind == "scale":
        cur = _kget(ns, svc, "{.spec.replicas}") or "1"
        json.dump({"service": svc, "replicas": cur},
                  open(_drift_state_path(scn["name"], ns), "w"))
        target = str(scn.get("replicas", 3))
        print(f"[drift] scale {svc}: {cur} -> {target}")
        subprocess.run([*_KC, "scale", "-n", ns, f"deploy/{svc}", f"--replicas={target}"],
                       check=False)
    else:
        raise SystemExit(f"drift kind inconnu: {kind}")


def _delete_drift(cat: dict, scn: dict) -> None:
    ns, svc, kind = cat["namespace"], scn["target"], scn["kind"]
    if kind == "rollout":
        return  # le restart se termine seul — rien à annuler
    if kind == "scale":
        st = {}
        try:
            st = json.load(open(_drift_state_path(scn["name"], ns)))
        except Exception:
            pass
        back = st.get("replicas", "1")
        print(f"[drift] scale back {svc} -> {back}")
        subprocess.run([*_KC, "scale", "-n", ns, f"deploy/{svc}", f"--replicas={back}"],
                       check=False)


_DRIFT_KINDS = {"rollout", "scale"}


def _overlap_chaos_pseudo(scn: dict) -> dict:
    """Sous-scénario chaos d'un overlap, au format attendu par _render."""
    c = scn["chaos"]
    return {"name": scn["name"], "kind": c["kind"], "target": c["target"], "spec": c["spec"]}


def cmd_apply(cat: dict, name: str, intensity: str, duration: str,
              target: str | None = None) -> None:
    scn = _get_scn(cat, name)
    if scn.get("kind") == "none":
        print(f"[normal] {name}: no-op (aucune injection)")
        return
    if scn.get("kind") in _DRIFT_KINDS:
        return _apply_drift(cat, scn)
    if scn.get("kind") == "overlap":
        # θ_drift∩anomaly : drift natif + chaos simultanés sur le même service
        _apply_drift(cat, {**scn["drift"], "name": scn["name"]})
        scn = _overlap_chaos_pseudo(scn)
    # Rotation de cible : --target doit appartenir au pool déclaré du scénario
    # (décorrèle type↔service ; validation stricte = pas de chaos sur une cible
    # arbitraire par typo). Sans --target, cible historique du catalogue.
    if target is not None:
        allowed = set(scn.get("target_pool", [])) | {scn.get("target")}
        if target not in allowed:
            raise SystemExit(
                f"{name}: cible {target!r} hors pool {sorted(allowed - {None})}")
        scn = {**scn, "target": target}
    repl = _repl_for(cat, intensity)
    repl["DURATION"] = duration
    repl["TARGET"] = scn.get("target", "")  # ex. containerNames: ["{{TARGET}}"]
    manifests = _render(scn, cat["namespace"], repl)
    # injecter la duration substituée + caster les champs entiers
    manifests = _cast_ints(_subst(manifests, {"DURATION": duration}))
    _kubectl("apply", manifests)


def cmd_delete(cat: dict, name: str) -> None:
    scn = _get_scn(cat, name)
    if scn.get("kind") == "none":
        print(f"[normal] {name}: no-op")
        return
    if scn.get("kind") in _DRIFT_KINDS:
        return _delete_drift(cat, scn)
    drift_part = None
    if scn.get("kind") == "overlap":
        drift_part = {**scn["drift"], "name": scn["name"]}
        scn = _overlap_chaos_pseudo(scn)
    # delete matche par nom de ressource (v5-<scenario>) — indépendant de la cible
    manifests = _render(scn, cat["namespace"], {"DURATION": "1s", "TARGET": scn.get("target", "")})
    _kubectl("delete", manifests)
    if drift_part is not None:
        _delete_drift(cat, drift_part)


def _get_bug(cat: dict, bug_id: str) -> dict:
    for b in cat["bugs"]:
        if b["id"] == bug_id:
            return b
    raise SystemExit(f"bug inconnu: {bug_id}")


def _state_path(bug_id: str, ns: str) -> str:
    # namespacé : 2 runners (tt / tt-b) ne doivent pas se croiser les états sains
    return f"/tmp/ewat_bug_{ns}_{bug_id}.json"


def _kget(ns: str, svc: str, jsonpath: str) -> str:
    r = subprocess.run([*_KC, "get", "deploy", "-n", ns, svc, "-o",
                        "jsonpath=" + jsonpath], capture_output=True, text=True)
    return r.stdout.strip()


def cmd_apply_bug(cat: dict, bug_id: str) -> None:
    """Injecte un bug F. Deux modes :
    - image    : swap de l'image saine vers l'image fautive (F1).
    - mem_limit : abaisse la limite mémoire conteneur sous l'empreinte JVM
                  → OOMKill (F3, mécanisme JVM/Docker authentique, sans rebuild).
    L'état sain est sauvegardé dans /tmp pour restauration par delete-bug.
    """
    b = _get_bug(cat, bug_id)
    ns = cat["namespace"]
    mode = b.get("mode", "image")
    svc = b["service"]

    if mode == "image":
        if not b.get("image"):
            raise SystemExit(f"{bug_id}: image indisponible (status={b.get('status')}).")
        healthy = _kget(ns, svc, "{.spec.template.spec.containers[0].image}")
        json.dump({"mode": "image", "service": svc, "healthy": healthy},
                  open(_state_path(bug_id, ns), "w"))
        print(f"{bug_id} [image] {svc}: {healthy} -> {b['image']}")
        subprocess.run([*_KC, "set", "image", "-n", ns,
                        f"deploy/{svc}", f"{svc}={b['image']}"], check=False)
    elif mode == "mem_limit":
        healthy = _kget(ns, svc, "{.spec.template.spec.containers[0].resources.limits.memory}") or "500Mi"
        json.dump({"mode": "mem_limit", "service": svc, "healthy": healthy},
                  open(_state_path(bug_id, ns), "w"))
        faulty = b["mem_limit"]
        print(f"{bug_id} [mem_limit] {svc}: {healthy} -> {faulty}")
        patch = {"spec": {"template": {"spec": {"containers": [
            {"name": svc, "resources": {"limits": {"memory": faulty}}}]}}}}
        subprocess.run([*_KC, "patch", "deploy", "-n", ns, svc, "--type=strategic",
                        "-p", json.dumps(patch)], check=False)
    elif mode == "env":
        # Bug reproduit par CONFIGURATION (pas de rebuild), comme F3 : on patche des
        # variables d'env JVM/Spring (ex. -Dserver.tomcat.max-threads=5 → épuisement
        # du pool de threads sous charge parallèle). On capture la valeur saine de
        # CHAQUE var pour restaurer à l'identique (une var préexistante ne doit pas
        # être perdue → sinon contamination des épisodes suivants, cf. _restore_bug).
        envs = b["env"]  # {VAR: value}
        prev = {k: _kget(ns, svc,
                         "{.spec.template.spec.containers[0].env[?(@.name==\"" + k + "\")].value}")
                for k in envs}
        json.dump({"mode": "env", "service": svc, "prev": prev},
                  open(_state_path(bug_id, ns), "w"))
        # env_append : CONCATÈNE à la valeur saine au lieu de l'écraser. Indispensable
        # pour JAVA_TOOL_OPTIONS, qui porte déjà `-javaagent:/jmxagent/jmx.jar=...` sur
        # tous les services TT : l'écraser supprimerait l'agent JMX → plus aucune
        # métrique jvm_* sur le service fautif, donc plus de symptôme observable.
        append = bool(b.get("env_append", False))
        pairs = [f"{k}={(prev[k] + ' ' + v).strip() if (append and prev[k]) else v}"
                 for k, v in envs.items()]
        print(f"{bug_id} [env] {svc}: set {pairs} (prev={prev})")
        subprocess.run([*_KC, "set", "env", "-n", ns, f"deploy/{svc}", *pairs], check=False)
    else:
        raise SystemExit(f"{bug_id}: mode inconnu {mode}")


def cmd_delete_bug(cat: dict, bug_id: str, healthy_override: str | None = None) -> None:
    """Restaure l'état sain (lu depuis /tmp ou override)."""
    b = _get_bug(cat, bug_id)
    ns = cat["namespace"]
    svc = b["service"]
    st = {}
    try:
        st = json.load(open(_state_path(bug_id, ns)))
    except Exception:
        pass
    mode = st.get("mode", b.get("mode", "image"))
    healthy = healthy_override or st.get("healthy")
    if mode == "image":
        if not healthy:
            raise SystemExit(f"{bug_id}: image saine inconnue (pas d'état sauvegardé).")
        subprocess.run([*_KC, "set", "image", "-n", ns,
                        f"deploy/{svc}", f"{svc}={healthy}"], check=False)
    elif mode == "mem_limit":
        healthy = healthy or "500Mi"
        patch = {"spec": {"template": {"spec": {"containers": [
            {"name": svc, "resources": {"limits": {"memory": healthy}}}]}}}}
        subprocess.run([*_KC, "patch", "deploy", "-n", ns, svc, "--type=strategic",
                        "-p", json.dumps(patch)], check=False)
    elif mode == "env":
        # restaure chaque var : valeur saine préexistante → set back ; absente → unset
        prev = st.get("prev") or {k: "" for k in b.get("env", {})}
        args = [f"{k}={v}" if v else f"{k}-" for k, v in prev.items()]
        subprocess.run([*_KC, "set", "env", "-n", ns, f"deploy/{svc}", *args], check=False)
        print(f"{bug_id}: env restauré -> {prev}")
        return
    print(f"{bug_id}: restauré -> {healthy}")


def main() -> None:
    cat = _load()
    p = argparse.ArgumentParser(description="EWAT v5 chaos injector")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("apply"); a.add_argument("name"); a.add_argument("--intensity", default="high"); a.add_argument("--duration", default="600s"); a.add_argument("--namespace", default=None); a.add_argument("--target", default=None, help="cible (doit appartenir au target_pool du scénario)")
    d = sub.add_parser("delete"); d.add_argument("name"); d.add_argument("--namespace", default=None)
    ab = sub.add_parser("apply-bug"); ab.add_argument("bug_id"); ab.add_argument("--namespace", default=None)
    db = sub.add_parser("delete-bug"); db.add_argument("bug_id"); db.add_argument("healthy_image", nargs="?", default=None); db.add_argument("--namespace", default=None)
    args = p.parse_args()

    # override du namespace cible (multi-runner) — toutes les fonctions lisent cat["namespace"]
    if getattr(args, "namespace", None):
        cat["namespace"] = args.namespace

    if args.cmd == "list":
        cmd_list(cat)
    elif args.cmd == "apply":
        cmd_apply(cat, args.name, args.intensity, args.duration, args.target)
    elif args.cmd == "delete":
        cmd_delete(cat, args.name)
    elif args.cmd == "apply-bug":
        cmd_apply_bug(cat, args.bug_id)
    elif args.cmd == "delete-bug":
        cmd_delete_bug(cat, args.bug_id, args.healthy_image)


if __name__ == "__main__":
    main()
