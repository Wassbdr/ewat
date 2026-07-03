"""EWAT — Packaging d'une release publique du dataset (artefacts sanitisés).

Construit un dossier ``release/<name>/`` auto-portant à partir d'un dataset
assemblé (``data/datasets/<name>``), en **sanitisant** au passage la métadonnée
qui fuite l'infrastructure interne.

Le contrat d'artefacts scientifiques est conservé intégralement (tenseurs, labels,
services, adjacency) ; seuls les champs d'infra sont retirés/réécrits :

- ``metadata.json`` : suppression des blocs ``config`` et ``base_config``
  (node_ip, cluster, endpoints télémétrie, mlflow home path).
- ``feature_provenance.json`` : suppression de ``source_episode_dir`` (chemin home).
- ``dataset.json`` : ``features_root`` (chemin absolu interne) retiré.

Le script REFUSE de packager si l'audit de fuite (``audit_leak_v5.py``) ne passe
pas sur la sortie — la sanitization est donc *vérifiée*, pas supposée.

Produit dans ``release/<name>/`` :
    data/<episode_id>/{signal,signal_mask,adjacency}.npz, labels.parquet,
                      services.json, metadata.json, feature_provenance.json,
                      graph_stats.csv
    dataset.json, index.parquet, split.json, services.json, summary.csv
    schema.json           # noms des features + slices M/T/L (depuis le registre)
    SHA256SUMS            # checksums de tous les fichiers de données
    leak_audit.json       # rapport d'audit (doit être clean)

Usage
-----
    python scripts/build_release_v5.py --dataset data/datasets/ewat_v5 \
        --out release/ewat_v5 --allow ts-order ts-travel ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_leak_v5 import audit  # noqa: E402

# Champs de métadonnée à retirer (fuite d'infra), quel que soit le schéma.
_META_DROP = ("config", "base_config")
_PROVENANCE_DROP = ("source_episode_dir",)
_DATASET_DROP = ("features_root",)

# Fichiers copiés tels quels (aucune fuite : tenseurs, labels publics, graphes).
_COPY_VERBATIM = {
    "signal.npz", "signal_mask.npz", "adjacency.npz",
    "labels.parquet", "services.json", "graph_stats.csv",
}
# signal_raw.npz est explicitement EXCLU de la release (données brutes).
_EXCLUDE = {"signal_raw.npz"}


def _sanitize_json(fp: Path, drop_keys: tuple[str, ...]) -> dict:
    data = json.loads(fp.read_text())
    for k in drop_keys:
        data.pop(k, None)
    return data


def _copy_episode(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fp in sorted(src_dir.iterdir()):
        if fp.name in _EXCLUDE:
            continue
        if fp.name == "metadata.json":
            clean = _sanitize_json(fp, _META_DROP)
            (dst_dir / fp.name).write_text(json.dumps(clean, indent=2))
        elif fp.name == "feature_provenance.json":
            clean = _sanitize_json(fp, _PROVENANCE_DROP)
            (dst_dir / fp.name).write_text(json.dumps(clean, indent=2))
        elif fp.name in _COPY_VERBATIM:
            shutil.copy2(fp, dst_dir / fp.name)
        # tout autre fichier inattendu est ignoré (allowlist stricte)


def _write_schema(out: Path) -> None:
    from src.telemetry import feature_names as fn

    schema = {
        "schema_version": fn.SCHEMA_V5_1,
        "signal_dim": fn.signal_dim(fn.SCHEMA_V5_1),
        "feature_names": fn.get_schema(fn.SCHEMA_V5_1),
        "modality_slices": {
            m: [sl.start, sl.stop] for m, sl in fn.MODALITY_SLICES[fn.SCHEMA_V5_1].items()
        },
        "signal_shape": ["T", "N_services", "signal_dim"],
        "adjacency_shape": ["T", "N_services", "N_services", 3],
        "adjacency_edge_dims": ["volume", "latency_med", "error_rate"],
        "label_columns": {
            "regime": "normal|injection|drift_anomaly|recovery",
            "category": "scenario category",
            "scenario": "chaos scenario name",
            "drift_flag": "bool — benign drift active",
            "is_injection": "bool — chaos injection active",
            "intensity_t": "float [0,1] — ramped fault intensity",
            "fault_type": "chaos|bug",
            "bug_id": "F1|F3|'' — real-bug identifier",
            "held_out_flag": "bool — novelty split (test-only)",
        },
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2))


def _write_checksums(out: Path) -> None:
    lines = []
    for dirpath, _dirs, files in os.walk(out):
        for name in sorted(files):
            if name == "SHA256SUMS":
                continue
            fp = Path(dirpath) / name
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            rel = os.path.relpath(fp, out)
            lines.append(f"{h}  {rel}")
    (out / "SHA256SUMS").write_text("\n".join(sorted(lines)) + "\n")


def build_release(dataset: Path, out: Path, allow: list[str]) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    data_out = out / "data"
    data_out.mkdir()

    # 1) Racine du dataset (sanitisée).
    (out / "dataset.json").write_text(
        json.dumps(_sanitize_json(dataset / "dataset.json", _DATASET_DROP), indent=2)
    )
    for name in ("split.json", "services.json"):
        shutil.copy2(dataset / name, out / name)
    for name in ("index.parquet", "summary.csv"):
        if (dataset / name).exists():
            shutil.copy2(dataset / name, out / name)

    # 2) Épisodes (résolution des symlinks episodes/<id> → features/...).
    episodes_dir = dataset / "episodes"
    n_ep = 0
    for ep in sorted(episodes_dir.iterdir()):
        real = ep.resolve()
        if not real.is_dir():
            continue
        _copy_episode(real, data_out / ep.name)
        n_ep += 1

    # 3) Fichiers documentaires + loader (copiés depuis scripts/release_assets/).
    assets = Path(__file__).resolve().parent / "release_assets"
    loader = Path(__file__).resolve().parent / "release_load_ewat.py"
    shutil.copy2(loader, out / "load_ewat.py")
    for name in ("README.md", "DATASHEET.md", "LICENSE", "CITATION.cff"):
        src = assets / name
        if src.exists():
            shutil.copy2(src, out / name)

    # 4) Schéma + checksums (après tous les fichiers, checksums inclut la doc).
    _write_schema(out)
    _write_checksums(out)

    # 4) Audit de fuite (gate) sur la sortie réelle.
    allowlist = {t.lower() for t in allow}
    report = audit(out, allowlist)
    (out / "leak_audit.json").write_text(json.dumps(report, indent=2))
    report["n_episodes"] = n_ep
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Packager une release publique du dataset.")
    ap.add_argument("--dataset", type=Path, required=True, help="Dataset assemblé source.")
    ap.add_argument("--out", type=Path, required=True, help="Dossier release de sortie.")
    ap.add_argument("--allow", nargs="*", default=[], help="Tokens tolérés à l'audit.")
    args = ap.parse_args(argv)

    if not (args.dataset / "dataset.json").exists():
        print(f"[FAIL] dataset introuvable : {args.dataset}", file=sys.stderr)
        return 2

    report = build_release(args.dataset, args.out, args.allow)

    if not report["clean"]:
        print(f"[FAIL] release NON publiable — {report['n_findings']} fuite(s) résiduelle(s) :",
              file=sys.stderr)
        for f in report["findings"][:20]:
            print(f"  - {f['file']}: [{f['pattern']}] {f['match']}", file=sys.stderr)
        return 1

    print(f"[OK] release construite : {args.out}")
    print(f"     {report['n_episodes']} épisodes, {report['n_files_scanned']} fichiers, audit CLEAN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
