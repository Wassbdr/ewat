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
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CATALOG = Path(__file__).parent / "catalog.yaml"


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
        "MEM_PCT": lv["mem_pct"],
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
    cmd = ["kubectl", action, "-f", path]
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


def cmd_apply(cat: dict, name: str, intensity: str, duration: str) -> None:
    scn = _get_scn(cat, name)
    repl = _repl_for(cat, intensity)
    repl["DURATION"] = duration
    manifests = _render(scn, cat["namespace"], repl)
    # injecter la duration substituée + caster les champs entiers
    manifests = _cast_ints(_subst(manifests, {"DURATION": duration}))
    _kubectl("apply", manifests)


def cmd_delete(cat: dict, name: str) -> None:
    scn = _get_scn(cat, name)
    manifests = _render(scn, cat["namespace"], {"DURATION": "1s"})
    _kubectl("delete", manifests)


def _get_bug(cat: dict, bug_id: str) -> dict:
    for b in cat["bugs"]:
        if b["id"] == bug_id:
            return b
    raise SystemExit(f"bug inconnu: {bug_id}")


def _state_path(bug_id: str) -> str:
    return f"/tmp/ewat_bug_{bug_id}.json"


def _kget(ns: str, svc: str, jsonpath: str) -> str:
    r = subprocess.run(["kubectl", "get", "deploy", "-n", ns, svc, "-o",
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
                  open(_state_path(bug_id), "w"))
        print(f"{bug_id} [image] {svc}: {healthy} -> {b['image']}")
        subprocess.run(["kubectl", "set", "image", "-n", ns,
                        f"deploy/{svc}", f"{svc}={b['image']}"], check=False)
    elif mode == "mem_limit":
        healthy = _kget(ns, svc, "{.spec.template.spec.containers[0].resources.limits.memory}") or "500Mi"
        json.dump({"mode": "mem_limit", "service": svc, "healthy": healthy},
                  open(_state_path(bug_id), "w"))
        faulty = b["mem_limit"]
        print(f"{bug_id} [mem_limit] {svc}: {healthy} -> {faulty}")
        patch = {"spec": {"template": {"spec": {"containers": [
            {"name": svc, "resources": {"limits": {"memory": faulty}}}]}}}}
        subprocess.run(["kubectl", "patch", "deploy", "-n", ns, svc, "--type=strategic",
                        "-p", json.dumps(patch)], check=False)
    else:
        raise SystemExit(f"{bug_id}: mode inconnu {mode}")


def cmd_delete_bug(cat: dict, bug_id: str, healthy_override: str | None = None) -> None:
    """Restaure l'état sain (lu depuis /tmp ou override)."""
    b = _get_bug(cat, bug_id)
    ns = cat["namespace"]
    svc = b["service"]
    st = {}
    try:
        st = json.load(open(_state_path(bug_id)))
    except Exception:
        pass
    mode = st.get("mode", b.get("mode", "image"))
    healthy = healthy_override or st.get("healthy")
    if mode == "image":
        if not healthy:
            raise SystemExit(f"{bug_id}: image saine inconnue (pas d'état sauvegardé).")
        subprocess.run(["kubectl", "set", "image", "-n", ns,
                        f"deploy/{svc}", f"{svc}={healthy}"], check=False)
    elif mode == "mem_limit":
        healthy = healthy or "500Mi"
        patch = {"spec": {"template": {"spec": {"containers": [
            {"name": svc, "resources": {"limits": {"memory": healthy}}}]}}}}
        subprocess.run(["kubectl", "patch", "deploy", "-n", ns, svc, "--type=strategic",
                        "-p", json.dumps(patch)], check=False)
    print(f"{bug_id}: restauré -> {healthy}")


def main() -> None:
    cat = _load()
    p = argparse.ArgumentParser(description="EWAT v5 chaos injector")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("apply"); a.add_argument("name"); a.add_argument("--intensity", default="high"); a.add_argument("--duration", default="600s")
    d = sub.add_parser("delete"); d.add_argument("name")
    ab = sub.add_parser("apply-bug"); ab.add_argument("bug_id")
    db = sub.add_parser("delete-bug"); db.add_argument("bug_id"); db.add_argument("healthy_image", nargs="?", default=None)
    args = p.parse_args()

    if args.cmd == "list":
        cmd_list(cat)
    elif args.cmd == "apply":
        cmd_apply(cat, args.name, args.intensity, args.duration)
    elif args.cmd == "delete":
        cmd_delete(cat, args.name)
    elif args.cmd == "apply-bug":
        cmd_apply_bug(cat, args.bug_id)
    elif args.cmd == "delete-bug":
        cmd_delete_bug(cat, args.bug_id, args.healthy_image)


if __name__ == "__main__":
    main()
