"""D3 (audit 2026-06) — La missingness comme features (B2 + masque).

T(t) est NaN à ~20-25 % de façon STRUCTURELLE sur les crashs (le service ne
trace plus) : la missingness est un signal du type de panne que l'imputation
efface. Ce script compare, en paired bootstrap :

- B2        : flatten (k×N×17) instance-normalized (headline 0.920)
- B2+masque : B2 ⧺ ratios de NaN bruts par (service × modalité) (+N×3 feat)

Protocole de comparaison identique à A5 (paired Δ, mêmes indices bootstrap
pour les deux conditions → IC sur Δ, pas sur les niveaux).

Usage
-----
    python -m experiments.audit2026.mask_features \\
        --dataset data/datasets/ewat_v4_strat --features-root data/features/v4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.audit2026.common import (
    build_b2_xy,
    fit_predict_proba,
    load_manifest,
    macro_auroc,
    macro_pr_auc,
    write_results,
)
from utils.seeding import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description="D3 — missingness features vs B2")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/audit2026/mask_features"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)

    manifest, classes = load_manifest(args.dataset)
    n_classes = len(classes)

    probas = {}
    y_te = None
    for cond, with_miss in (("b2", False), ("b2_mask", True)):
        X_tr, y_tr, _ = build_b2_xy(manifest, args.features_root, "train",
                                    classes, k=args.k, with_missingness=with_miss)
        X_te, y_te, _ = build_b2_xy(manifest, args.features_root, "test",
                                    classes, k=args.k, with_missingness=with_miss)
        _, p_te = fit_predict_proba(X_tr, y_tr, X_te, n_classes)
        probas[cond] = p_te
        print(f"{cond:8s}: dim={X_tr.shape[1]}  "
              f"AUROC={macro_auroc(y_te, p_te, n_classes):.4f}  "
              f"PR-AUC={macro_pr_auc(y_te, p_te, n_classes):.4f}")

    # Paired bootstrap sur Δ (mêmes indices pour les deux conditions — A5)
    rng = np.random.default_rng(args.seed)
    n = len(y_te)
    deltas_auroc, deltas_pr = [], []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a_mask = macro_auroc(y_te[idx], probas["b2_mask"][idx], n_classes)
        a_base = macro_auroc(y_te[idx], probas["b2"][idx], n_classes)
        p_mask = macro_pr_auc(y_te[idx], probas["b2_mask"][idx], n_classes)
        p_base = macro_pr_auc(y_te[idx], probas["b2"][idx], n_classes)
        if not (np.isnan(a_mask) or np.isnan(a_base)):
            deltas_auroc.append(a_mask - a_base)
        if not (np.isnan(p_mask) or np.isnan(p_base)):
            deltas_pr.append(p_mask - p_base)

    def ci(d):
        d = np.asarray(d)
        return (float(d.mean()), float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)), float((d <= 0).mean()))

    da, da_lo, da_hi, pa = ci(deltas_auroc)
    dp, dp_lo, dp_hi, pp = ci(deltas_pr)

    auroc_base = macro_auroc(y_te, probas["b2"], n_classes)
    auroc_mask = macro_auroc(y_te, probas["b2_mask"], n_classes)
    payload = {
        "auroc_b2": auroc_base,
        "auroc_b2_mask": auroc_mask,
        "pr_auc_b2": macro_pr_auc(y_te, probas["b2"], n_classes),
        "pr_auc_b2_mask": macro_pr_auc(y_te, probas["b2_mask"], n_classes),
        "delta_auroc": {"mean": da, "ci_lo": da_lo, "ci_hi": da_hi,
                        "p_delta_leq_0": pa},
        "delta_pr_auc": {"mean": dp, "ci_lo": dp_lo, "ci_hi": dp_hi,
                         "p_delta_leq_0": pp},
        "n_bootstrap": args.n_bootstrap,
        "k": args.k,
    }
    significant = da_lo > 0
    lines = [
        "# D3 — Missingness comme features (audit 2026-06)",
        "",
        "B2 ⧺ ratios de NaN bruts par (service × modalité) vs B2, paired",
        f"bootstrap (n={args.n_bootstrap}, protocole A5).",
        "",
        "| Condition | macro-AUROC | macro-PR-AUC |",
        "|---|---|---|",
        f"| B2 (headline) | {payload['auroc_b2']:.4f} | {payload['pr_auc_b2']:.4f} |",
        f"| B2 + masque | {payload['auroc_b2_mask']:.4f} | {payload['pr_auc_b2_mask']:.4f} |",
        "",
        f"**Δ AUROC (mask − base)** = {da:+.4f}  IC95 [{da_lo:+.4f}, {da_hi:+.4f}]"
        f"  P(Δ≤0)={pa:.3f}",
        f"**Δ PR-AUC**             = {dp:+.4f}  IC95 [{dp_lo:+.4f}, {dp_hi:+.4f}]"
        f"  P(Δ≤0)={pp:.3f}",
        "",
        ("**Verdict : GAIN SIGNIFICATIF** — l'IC paired exclut 0 ; la "
         "missingness porte un signal de type de panne que l'imputation "
         "effaçait." if significant else
         "**Verdict : pas de gain significatif** — l'IC paired contient 0. "
         "Résultat à reporter tel quel (l'information de missingness est "
         "peut-être déjà recouvrable depuis les valeurs imputées à 0 "
         "post-instance-norm)."),
    ]
    write_results(args.output, payload, lines)


if __name__ == "__main__":
    main()
