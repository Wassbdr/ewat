"""EWAT — Audit de fuite d'infrastructure pour la publication d'un dataset.

Scanne tous les fichiers d'un dataset (ou d'un dossier release) destinés à être
publiés et échoue (exit 1) si un motif de fuite d'infrastructure est trouvé :
IP privées, noms de nœuds/cluster, DNS internes, chemins home absolus, endpoints
de télémétrie, identité de l'auteur.

Read-only : ne modifie jamais les fichiers scannés. Écrit optionnellement un
rapport JSON (``--report``).

La surface de fuite réelle a été identifiée dans le contrat d'artefacts :
- ``metadata.json`` peut contenir ``config`` / ``base_config`` (endpoints, node_ip,
  cluster, mlflow home path) sur les datasets v4 ; le build v5
  (``v5/collect/build_features_v5.py``) émet une métadonnée *lean* sans ces blocs.
- ``dataset.json`` (écrit par ``assemble_dataset``) contient ``features_root`` =
  chemin absolu ``/home/...``.
- ``labels.parquet`` / ``services.json`` ne doivent porter que des noms de
  services publics (``ts-*`` pour Train Ticket, Online Boutique pour v4).

Ce script est le *gate* de publication : ``build_release_v5.py`` refuse de packager
si l'audit échoue.

Usage
-----
    python scripts/audit_leak_v5.py --dataset data/datasets/ewat_v5
    python scripts/audit_leak_v5.py --release release/ewat_v5 --report leak_audit.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

# --- Motifs de fuite -------------------------------------------------------

# IPv4 privées (RFC 1918) : 10.x, 172.16-31.x, 192.168.x. On évite de flagger
# les octets isolés en exigeant au moins deux groupes numériques.
_LEAK_PATTERNS: dict[str, re.Pattern] = {
    "private_ipv4": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3})\b"
    ),
    "internal_dns": re.compile(
        r"\b(?:rancher|devolab|observit|cattle|svc\.cluster\.local)[\w.-]*", re.I
    ),
    "dot_lan": re.compile(r"\b[\w-]+\.lan\b", re.I),
    "node_name": re.compile(r"\b[\w-]*workers-[a-z0-9]+-[a-z0-9]+\b", re.I),
    "home_path": re.compile(r"/home/[\w./-]+"),
    "author_identity": re.compile(r"\b(?:wassim|badraoui|devoteam)\b", re.I),
    "telemetry_endpoint": re.compile(
        r"\b(?:prometheus|jaeger|loki|otel[\w-]*collector|mlflow)[\w.-]*:\d{2,5}", re.I
    ),
    "cluster_api": re.compile(r"https?://[\w.-]*(?:rancher|devolab|k8s/clusters)[\w./-]*", re.I),
}

# Blocs de métadonnées entiers qui ne devraient jamais figurer dans une release.
# Leur simple présence est un échec (indépendamment du contenu).
_FORBIDDEN_META_KEYS = {"config", "base_config"}

# Clés de dataset.json qui portent des chemins absolus internes.
_FORBIDDEN_DATASET_KEYS = {"features_root"}

# Fichiers dont le contenu est du texte à scanner (JSON/CSV/txt/md/cff/yaml).
_TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md", ".cff", ".yaml", ".yml"}

# Fichiers documentaires où l'identité de l'auteur (nom, affiliation) est
# légitime et publique — la citation EN A BESOIN. Les motifs d'INFRA (IP, nœuds,
# DNS, endpoints, chemins home) y restent interdits.
_DOC_FILES = {"README.md", "DATASHEET.md", "CITATION.cff", "LICENSE"}
_DOC_ALLOWED_PATTERNS = {"author_identity"}


def _iter_strings(obj, path: str = ""):
    """Parcours récursif d'un objet JSON, yield (json_path, valeur_str)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _scan_text(text: str, source: str) -> list[dict]:
    findings = []
    for name, pat in _LEAK_PATTERNS.items():
        for m in pat.finditer(text):
            findings.append({"file": source, "pattern": name, "match": m.group(0)[:120]})
    return findings


def _scan_json_structure(data, source: str, forbidden_keys: set[str]) -> list[dict]:
    """Flag la présence de clés interdites (blocs config/base_config, features_root)."""
    findings = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in forbidden_keys:
                    findings.append(
                        {"file": source, "pattern": "forbidden_key", "match": k}
                    )
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return findings


def _read_parquet_as_text(fp: Path) -> str:
    import pandas as pd

    df = pd.read_parquet(fp)
    # On ne scanne que les colonnes texte (les tenseurs/floats ne fuient pas).
    parts = []
    for col in df.columns:
        if df[col].dtype == object:
            parts.append("\n".join(df[col].astype(str).unique().tolist()))
    return "\n".join(parts)


def _load_text(fp: Path) -> str | None:
    if fp.suffix == ".gz":
        try:
            return gzip.open(fp, "rt", errors="replace").read()
        except OSError:
            return None
    if fp.suffix == ".parquet":
        return _read_parquet_as_text(fp)
    if fp.suffix in _TEXT_SUFFIXES:
        return fp.read_text(errors="replace")
    return None


def _walk_files(root: Path):
    """Yield tous les fichiers sous *root*, en suivant les répertoires symlinkés.

    Les datasets assemblés symlinkent ``episodes/<id>`` vers ``data/features/...`` ;
    la release copie les fichiers réels, donc l'audit doit scanner la cible réelle.
    """
    for dirpath, _dirs, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            yield Path(dirpath) / name


def audit(root: Path, allowlist: set[str]) -> dict:
    """Audite un dossier. Retourne un rapport {clean, findings, scanned}."""
    findings: list[dict] = []
    scanned: list[str] = []

    for fp in sorted(_walk_files(root)):
        if not fp.is_file():
            continue
        rel = os.path.relpath(fp, root)

        # Tenseurs : vérifier seulement que signal.npz ne contient pas signal_raw.
        if fp.suffix == ".npz":
            scanned.append(rel)
            if fp.name == "signal.npz":
                import numpy as np

                with np.load(fp) as z:
                    if "signal_raw" in z.files:
                        findings.append(
                            {"file": rel, "pattern": "raw_signal_in_release",
                             "match": "signal.npz contient signal_raw"}
                        )
            continue

        text = _load_text(fp)
        if text is None:
            continue
        scanned.append(rel)

        # Scan texte transverse.
        is_doc = fp.name in _DOC_FILES
        for f in _scan_text(text, rel):
            # Tolérer les tokens de l'allowlist (ex. noms de services publics).
            if f["match"].lower() in allowlist:
                continue
            # Identité d'auteur légitime dans les fichiers documentaires.
            if is_doc and f["pattern"] in _DOC_ALLOWED_PATTERNS:
                continue
            findings.append(f)

        # Scan structurel des JSON (clés interdites).
        if fp.suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            if data is not None:
                if fp.name == "metadata.json":
                    findings += _scan_json_structure(data, rel, _FORBIDDEN_META_KEYS)
                if fp.name == "dataset.json":
                    findings += _scan_json_structure(data, rel, _FORBIDDEN_DATASET_KEYS)

    return {
        "root": str(root),
        "clean": len(findings) == 0,
        "n_files_scanned": len(scanned),
        "n_findings": len(findings),
        "findings": findings,
        "patterns_tested": sorted(_LEAK_PATTERNS) + ["forbidden_key", "raw_signal_in_release"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit de fuite d'infra avant publication.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path, help="Racine d'un dataset assemblé.")
    src.add_argument("--release", type=Path, help="Racine d'un dossier release.")
    ap.add_argument(
        "--allow", nargs="*", default=[],
        help="Tokens supplémentaires à tolérer (ex. noms de services publics).",
    )
    ap.add_argument("--report", type=Path, help="Écrire le rapport JSON ici.")
    args = ap.parse_args(argv)

    root = args.dataset or args.release
    if not root.exists():
        print(f"[FAIL] chemin introuvable : {root}", file=sys.stderr)
        return 2

    allowlist = {t.lower() for t in args.allow}
    report = audit(root, allowlist)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))

    if report["clean"]:
        print(f"[OK] {report['n_files_scanned']} fichiers scannés, aucune fuite.")
        return 0

    print(f"[FAIL] {report['n_findings']} fuite(s) sur {report['n_files_scanned']} fichiers :",
          file=sys.stderr)
    for f in report["findings"][:40]:
        print(f"  - {f['file']}: [{f['pattern']}] {f['match']}", file=sys.stderr)
    if report["n_findings"] > 40:
        print(f"  ... (+{report['n_findings'] - 40} autres)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
