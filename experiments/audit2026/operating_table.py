"""Table opérationnelle B2 calibrée — typage avec abstention (audit 2026-06).

Remplace l'ancienne table seuil/FA/lead (invalide : probas non calibrées,
FA 100 %, lead = identification de scénario). Question opérationnelle
honnête : « quand B2 voit la fenêtre pré-injection d'un épisode, à quelle
confiance peut-il router l'incident vers le bon scénario, et que vaut cette
confiance après calibration ? »

Pour une grille de seuils t : couverture (% épisodes test où max proba
calibrée ≥ t) et précision top-1 parmi les épisodes couverts. Calibration
isotonique fittée sur le split val (probas OvR poolées, protocole E2).

Usage
-----
    python -m experiments.audit2026.operating_table \\
        --dataset data/datasets/ewat_v4_strat --features-root data/features/v4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from experiments.audit2026.common import (
    build_b2_xy,
    fit_predict_proba,
    load_manifest,
    write_results,
)
from utils.seeding import seed_everything

THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    p = argparse.ArgumentParser(description="Table opérationnelle B2 calibrée")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/audit2026/operating_table"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)

    manifest, classes = load_manifest(args.dataset)
    n_classes = len(classes)
    X_tr, y_tr, _ = build_b2_xy(manifest, args.features_root, "train", classes, k=args.k)
    X_va, y_va, _ = build_b2_xy(manifest, args.features_root, "val", classes, k=args.k)
    X_te, y_te, _ = build_b2_xy(manifest, args.features_root, "test", classes, k=args.k)

    _, p_va = fit_predict_proba(X_tr, y_tr, X_va, n_classes)
    _, p_te = fit_predict_proba(X_tr, y_tr, X_te, n_classes)

    # Calibration isotonique sur les probas OvR poolées du val (protocole E2)
    y_pool = np.concatenate([(y_va == c).astype(int) for c in range(n_classes)])
    p_pool = np.concatenate([p_va[:, c] for c in range(n_classes)])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_pool, y_pool)
    p_te_cal = iso.transform(p_te.ravel()).reshape(p_te.shape)

    rows = []
    for variant, probs in (("brut", p_te), ("calibré", p_te_cal)):
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        for t in THRESHOLDS:
            covered = conf >= t
            n_cov = int(covered.sum())
            acc = float((pred[covered] == y_te[covered]).mean()) if n_cov else float("nan")
            rows.append({"variant": variant, "threshold": t,
                         "coverage": n_cov / len(y_te), "n_covered": n_cov,
                         "top1_accuracy": acc})

    lines = [
        "# Table opérationnelle — typage B2 calibré avec abstention (audit 2026-06)",
        "",
        "Cible indépendante (15 scénarios Chaos Mesh), fenêtre pré-injection",
        f"k={args.k}, calibration isotonique fittée sur val. Remplace l'ancienne",
        "table seuil/FA/lead (probas non calibrées, comportement identificateur",
        "de scénario — cf. L9.2).",
        "",
        "| Seuil de confiance | Couverture (brut) | Top-1 (brut) | Couverture (calibré) | Top-1 (calibré) |",
        "|---|---|---|---|---|",
    ]
    by_t: dict[float, dict[str, dict]] = {}
    for r in rows:
        by_t.setdefault(r["threshold"], {})[r["variant"]] = r
    for t, d in sorted(by_t.items()):
        b, c = d["brut"], d["calibré"]
        lines.append(
            f"| ≥ {t:.1f} | {b['coverage']:.0%} ({b['n_covered']}) "
            f"| {b['top1_accuracy']:.3f} | {c['coverage']:.0%} ({c['n_covered']}) "
            f"| {c['top1_accuracy']:.3f} |"
        )
    lines += [
        "",
        "Lecture : la colonne « calibré » est la seule interprétable en",
        "probabilité (E2). Le couple (couverture, top-1) à chaque seuil donne",
        "le point de fonctionnement du tri d'incidents : abstention au-dessous",
        "du seuil, routage vers le type prédit au-dessus.",
    ]
    write_results(args.output, {"rows": rows, "k": args.k}, lines)
    for t, d in sorted(by_t.items()):
        c = d["calibré"]
        print(f"t={t:.1f}  calibré: couverture={c['coverage']:.0%}  "
              f"top1={c['top1_accuracy']:.3f}")


if __name__ == "__main__":
    main()
