"""E6 (audit 2026-06) — Borne supérieure (oracle) du dataset v4_strat.

Sans borne haute, « B2 = 0.920 » n'est pas interprétable : le gap vers 1.0
est-il du bruit irréductible du dataset ou de la marge de progression ?

Deux bornes, par construction de plus en plus optimistes :

- **train+val** : LR fitté sur train∪val (tout ce qu'un modèle honnête peut
  voir), évalué sur test — la meilleure version « légale » de B2.
- **oracle (fit-on-all)** : LR fitté sur train∪val∪test, évalué sur test.
  CIRCULAIRE PAR CONSTRUCTION — c'est le plafond de ce que cette classe de
  modèle peut extraire de ces features sur ces épisodes. Tout écart entre
  l'oracle et 1.0 est du chevauchement irréductible entre scénarios (au sens
  de ce featurizer), pas un défaut du modèle.

Usage
-----
    python -m experiments.audit2026.oracle_baseline \\
        --dataset data/datasets/ewat_v4_strat --features-root data/features/v4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.audit2026.common import (
    bootstrap_macro_ci,
    build_b2_xy,
    fit_predict_proba,
    load_manifest,
    macro_auroc,
    macro_pr_auc,
    write_results,
)
from utils.seeding import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description="E6 — oracle / borne sup B2")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path, default=Path("experiments/audit2026/oracle"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)

    manifest, classes = load_manifest(args.dataset)
    n_classes = len(classes)
    X_tr, y_tr, _ = build_b2_xy(manifest, args.features_root, "train", classes, k=args.k)
    X_va, y_va, _ = build_b2_xy(manifest, args.features_root, "val", classes, k=args.k)
    X_te, y_te, _ = build_b2_xy(manifest, args.features_root, "test", classes, k=args.k)

    rng = np.random.default_rng(args.seed)
    conditions = {
        "b2_train_only (référence)": (X_tr, y_tr),
        "train+val (borne légale)": (np.concatenate([X_tr, X_va]),
                                     np.concatenate([y_tr, y_va])),
        "oracle fit-on-all (circulaire)": (
            np.concatenate([X_tr, X_va, X_te]),
            np.concatenate([y_tr, y_va, y_te]),
        ),
    }
    results = {}
    for name, (X_fit, y_fit) in conditions.items():
        _, p_te = fit_predict_proba(X_fit, y_fit, X_te, n_classes)
        auroc = macro_auroc(y_te, p_te, n_classes)
        _, lo, hi = bootstrap_macro_ci(y_te, p_te, n_classes,
                                       args.n_bootstrap, rng)
        results[name] = {
            "auroc": auroc,
            "pr_auc": macro_pr_auc(y_te, p_te, n_classes),
            "ci_lo": lo, "ci_hi": hi,
        }
        print(f"{name:34s}: AUROC={auroc:.4f} [{lo:.3f}, {hi:.3f}]  "
              f"PR-AUC={results[name]['pr_auc']:.4f}")

    base = results["b2_train_only (référence)"]["auroc"]
    oracle = results["oracle fit-on-all (circulaire)"]["auroc"]
    gap_total = 1.0 - base
    gap_model = oracle - base
    gap_data = 1.0 - oracle

    lines = [
        "# E6 — Oracle / borne supérieure du dataset (audit 2026-06)",
        "",
        "| Condition | macro-AUROC | IC95 bootstrap | macro-PR-AUC |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['auroc']:.4f} | [{r['ci_lo']:.3f}, "
                     f"{r['ci_hi']:.3f}] | {r['pr_auc']:.4f} |")
    lines += [
        "",
        "## Décomposition du gap vers 1.0",
        "",
        f"- Gap total B2 → 1.0 : **{gap_total:.4f}**",
        f"- dont rattrapable par cette classe de modèle (oracle − B2) : "
        f"**{gap_model:.4f}** ({gap_model / max(gap_total, 1e-9):.0%} du gap)",
        f"- dont chevauchement irréductible des scénarios (1 − oracle) : "
        f"**{gap_data:.4f}**",
        "",
        "L'oracle est circulaire par construction (fit sur test inclus) — il",
        "ne mesure PAS une performance atteignable, mais le plafond du couple",
        "(featurizer B2, LR-OvR) sur ces épisodes.",
    ]
    write_results(args.output, results | {
        "gap_total": gap_total, "gap_model": gap_model, "gap_data": gap_data,
    }, lines)


if __name__ == "__main__":
    main()
