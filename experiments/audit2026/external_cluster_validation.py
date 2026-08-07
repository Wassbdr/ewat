"""Validation EXTERNE du typage Phase H-bis (audit 2026-06, critique L9).

La silhouette (H1) est une métrique interne : elle dit que les embeddings
forment de beaux clusters, pas que ces clusters correspondent à de vrais
types de pannes. Référence externe v3 : NMI = 0.518, pureté = 0.503.

Ce script calcule, pour chaque graine de phase_h2 et chaque split :
NMI, ARI et pureté entre les labels de clusters (train = fit_predict,
val/test = nearest-centroid) et les scénarios Chaos Mesh (vérité terrain
indépendante). C'est LE chiffre présentable du typage.

Usage
-----
    python -m experiments.audit2026.external_cluster_validation \\
        [--phase-dir experiments/multiseed/phase_h2] \\
        [--dataset data/datasets/ewat_v4_strat]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.audit2026.common import write_results


def _purity(clusters: np.ndarray, scenarios: np.ndarray) -> float:
    """Moyenne pondérée de la fraction majoritaire par cluster."""
    total = len(clusters)
    s = 0.0
    for c in np.unique(clusters):
        mask = clusters == c
        _, counts = np.unique(scenarios[mask], return_counts=True)
        s += counts.max()
    return float(s / total)


def main() -> None:
    p = argparse.ArgumentParser(description="Validation externe clusters H-bis")
    p.add_argument("--phase-dir", type=Path,
                   default=Path("experiments/multiseed/phase_h2"))
    p.add_argument("--dataset", type=Path,
                   default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/audit2026/external_clusters"))
    args = p.parse_args()

    split = json.loads((args.dataset / "split.json").read_text())
    index = pd.read_parquet(args.dataset / "index.parquet")
    scen_of = dict(zip(index["episode_id"], index["scenario"]))

    per_seed: dict[str, dict] = {}
    for seed_dir in sorted(args.phase_dir.glob("seed_*")):
        art = seed_dir / "typing" / "cluster_artifacts"
        if not (art / "labels_test.npy").exists():
            continue
        row: dict[str, dict] = {}
        for sp in ("train", "val", "test"):
            labels = np.load(art / f"labels_{sp}.npy")
            ep_ids = split[sp]
            if len(labels) != len(ep_ids):
                print(f"  ⚠ {seed_dir.name}/{sp}: {len(labels)} labels vs "
                      f"{len(ep_ids)} épisodes — sauté")
                continue
            scenarios = np.array([scen_of[e] for e in ep_ids])
            row[sp] = {
                "nmi": float(normalized_mutual_info_score(scenarios, labels)),
                "ari": float(adjusted_rand_score(scenarios, labels)),
                "purity": _purity(labels, scenarios),
                "n_clusters_used": int(len(np.unique(labels))),
            }
        per_seed[seed_dir.name] = row
        t = row.get("test", {})
        print(f"{seed_dir.name}: test NMI={t.get('nmi', float('nan')):.3f} "
              f"ARI={t.get('ari', float('nan')):.3f} "
              f"purity={t.get('purity', float('nan')):.3f} "
              f"(K utilisés={t.get('n_clusters_used')})")

    def agg(metric: str, sp: str) -> tuple[float, float]:
        vals = [r[sp][metric] for r in per_seed.values() if sp in r]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {sp: {m: {"mean": agg(m, sp)[0], "std": agg(m, sp)[1]}
                    for m in ("nmi", "ari", "purity")}
               for sp in ("train", "val", "test")}

    nmi_m, nmi_s = agg("nmi", "test")
    ari_m, ari_s = agg("ari", "test")
    pur_m, pur_s = agg("purity", "test")
    lines = [
        "# Validation externe des clusters — Phase H-bis vs scénarios Chaos Mesh",
        "",
        "_La silhouette (H1, interne) ne valide pas le SENS des clusters._",
        "_Référence v3 (graine 42, K=10 auto) : NMI = 0.518, pureté = 0.503._",
        "",
        "| Split | NMI | ARI | Pureté |",
        "|---|---|---|---|",
    ]
    for sp in ("train", "val", "test"):
        s = summary[sp]
        lines.append(
            f"| {sp} | {s['nmi']['mean']:.3f} ± {s['nmi']['std']:.3f} "
            f"| {s['ari']['mean']:.3f} ± {s['ari']['std']:.3f} "
            f"| {s['purity']['mean']:.3f} ± {s['purity']['std']:.3f} |"
        )
    lines += [
        "",
        f"**Test (10 graines)** : NMI {nmi_m:.3f} ± {nmi_s:.3f}, "
        f"ARI {ari_m:.3f} ± {ari_s:.3f}, pureté {pur_m:.3f} ± {pur_s:.3f}.",
        "",
        "Lecture : si NMI/pureté restent ~0.5 malgré la silhouette 0.843, le",
        "gain des fixes est géométrique, pas sémantique — les clusters sont",
        "plus nets mais pas plus alignés sur les types de pannes réels. À",
        "reporter tel quel ; le typage présentable reste B2 (supervisé sur",
        "cible indépendante).",
    ]
    write_results(args.output, {"per_seed": per_seed, "summary": summary}, lines)


if __name__ == "__main__":
    main()
