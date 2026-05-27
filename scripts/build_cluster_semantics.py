"""Build semantic naming table for EWAT clusters from manifest + fiches.

Reads cluster_manifest.json and permutation_importance fiches (cluster_*.json),
then writes cluster_semantics.json and cluster_semantics.md for the thesis report.

Usage
-----
    python -m scripts.build_cluster_semantics \\
        --typing-dir experiments/typing \\
        --output experiments/typing
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

# Human-readable labels keyed by dominant Chaos Mesh scenario
_SCENARIO_LABELS: dict[str, tuple[str, str, str]] = {
    "memory_pressure": ("Pression mémoire", "Memory pressure", "anomaly"),
    "fail_slow_cpu": ("CPU lent", "Slow CPU", "anomaly"),
    "cpu_starvation": ("Contention CPU", "CPU starvation", "anomaly"),
    "intermittent_error": ("Erreurs intermittentes", "Intermittent errors", "anomaly"),
    "fail_slow_latency": ("Latence lente", "Slow latency", "anomaly"),
    "noisy_neighbor": ("Voisin bruyant", "Noisy neighbor", "anomaly"),
    "resource_leak": ("Fuite de ressources", "Resource leak", "anomaly"),
    "crash": ("Crash pod", "Pod crash", "anomaly"),
    "oom": ("OOM", "Out-of-memory", "anomaly"),
    "network_loss": ("Perte réseau", "Network loss", "anomaly"),
    "drift_traffic_ramp": ("Rampe de trafic (drift)", "Traffic ramp (drift)", "drift"),
    "drift_rolling_deploy": ("Déploiement progressif (drift)", "Rolling deploy (drift)", "drift"),
    "drift_config_change": ("Changement config (drift)", "Config change (drift)", "drift"),
    "drift_scale_up": ("Autoscaling (drift)", "Scale-up (drift)", "drift"),
    "faulty_deploy_overlap": (
        "Déploiement défectueux (drift ∩ anomalie)",
        "Faulty deployment (drift ∩ anomaly)",
        "drift_and_anomaly",
    ),
}


def _load_manifest(typing_dir: Path) -> dict[str, dict]:
    path = typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run experiments/typing/train.py first.")
    return json.loads(path.read_text())


def _cluster_stats(manifest: dict[str, dict], n_clusters: int) -> dict[int, dict]:
    by_cluster: dict[int, list[str]] = {c: [] for c in range(n_clusters)}
    for meta in manifest.values():
        c = int(meta["cluster"])
        by_cluster[c].append(meta.get("scenario", "unknown"))

    stats: dict[int, dict] = {}
    for c, scenarios in by_cluster.items():
        if not scenarios:
            continue
        counts = Counter(scenarios)
        dom, dom_n = counts.most_common(1)[0]
        stats[c] = {
            "n_episodes": len(scenarios),
            "dominant_scenario": dom,
            "purity": round(dom_n / len(scenarios), 4),
            "top_scenarios": dict(counts.most_common(5)),
        }
    return stats


def _top_features_from_fiche(fiches_dir: Path, cluster_id: int, k: int = 3) -> list[str]:
    fiche_path = fiches_dir / f"cluster_{cluster_id}.json"
    if not fiche_path.exists():
        return []
    fiche = json.loads(fiche_path.read_text())
    if fiche.get("top5_features"):
        return list(fiche["top5_features"][:k])
    imp = fiche.get("feature_importance") or {}
    ranked = sorted(imp.items(), key=lambda x: -abs(x[1]))
    return [name for name, _ in ranked[:k] if abs(_[1]) > 1e-9]


def _suggest_name(dominant: str, top_feats: list[str]) -> tuple[str, str]:
    if dominant in _SCENARIO_LABELS:
        fr, en, _ = _SCENARIO_LABELS[dominant]
        if top_feats:
            fr = f"{fr} ({', '.join(top_feats[:2])})"
            en = f"{en} ({', '.join(top_feats[:2])})"
        return fr, en
    return dominant.replace("_", " ").title(), dominant


def build_semantics(typing_dir: Path) -> dict:
    manifest = _load_manifest(typing_dir)
    n_clusters = max(int(m["cluster"]) for m in manifest.values()) + 1
    stats = _cluster_stats(manifest, n_clusters)
    fiches_dir = typing_dir / "fiches"

    clusters: dict[str, dict] = {}
    for c in range(n_clusters):
        if c not in stats:
            continue
        st = stats[c]
        dom = st["dominant_scenario"]
        top_feats = _top_features_from_fiche(fiches_dir, c)
        name_fr, name_en = _suggest_name(dom, top_feats)
        regime = _SCENARIO_LABELS.get(dom, ("", "", "anomaly"))[2]

        clusters[str(c)] = {
            "name_fr": name_fr,
            "name_en": name_en,
            "dominant_scenario": dom,
            "purity": st["purity"],
            "n_episodes": st["n_episodes"],
            "top_scenarios": st["top_scenarios"],
            "top_features": top_feats,
            "regime": regime,
        }

    return {
        "generated": str(date.today()),
        "source": "cluster_manifest.json + fiches/cluster_*.json (permutation_importance)",
        "n_clusters": n_clusters,
        "clusters": clusters,
    }


def _write_markdown(data: dict, out_path: Path) -> None:
    lines = [
        "# Nommage sémantique des clusters EWAT\n",
        f"_Généré le {data['generated']}_\n",
        "| Cluster | Nom (FR) | Scénario dominant | Pureté | N ép. | Top-3 features |",
        "|---------|----------|-------------------|--------|-------|----------------|",
    ]
    for cid in sorted(data["clusters"], key=int):
        c = data["clusters"][cid]
        feats = ", ".join(c.get("top_features") or []) or "—"
        lines.append(
            f"| C{cid} | {c['name_fr']} | `{c['dominant_scenario']}` | "
            f"{c['purity']:.3f} | {c['n_episodes']} | {feats} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cluster semantic naming table")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output or args.typing_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_semantics(args.typing_dir)
    json_path = out_dir / "cluster_semantics.json"
    md_path = out_dir / "cluster_semantics.md"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    _write_markdown(data, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
