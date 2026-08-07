"""E9 (audit 2026-06) — Robustesse du headline B2 aux données manquantes.

Injecte une fraction croissante de NaN aléatoires dans la fenêtre
pré-injection AU MOMENT DE L'INFÉRENCE (le train reste propre — simulation
de trous de collecte en production : scrapes manqués, Jaeger/Loki down),
puis mesure la dégradation AUROC/PR-AUC. 5 graines de masque par fraction.

Le NaN injecté traverse le chemin réel : instance norm nan-aware → NaN→0
(= valeur neutre post-normalisation), exactement comme la missingness
résiduelle v4.

Usage
-----
    python -m experiments.audit2026.nan_robustness \\
        --dataset data/datasets/ewat_v4_strat --features-root data/features/v4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
N_MASK_SEEDS = 5


def main() -> None:
    p = argparse.ArgumentParser(description="E9 — robustesse NaN du headline B2")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/audit2026/nan_robustness"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)

    manifest, classes = load_manifest(args.dataset)
    n_classes = len(classes)

    # Train propre, une seule fois (le modèle déployé est fixe).
    X_tr, y_tr, _ = build_b2_xy(manifest, args.features_root, "train", classes, k=args.k)
    clf, _ = fit_predict_proba(X_tr, y_tr, X_tr[:1], n_classes)

    results: dict[str, dict] = {}
    for frac in FRACTIONS:
        aurocs, prs = [], []
        seeds = [args.seed] if frac == 0.0 else \
            [args.seed + i for i in range(N_MASK_SEEDS)]
        for s in seeds:
            rng = np.random.default_rng(s)
            X_te, y_te, _ = build_b2_xy(
                manifest, args.features_root, "test", classes, k=args.k,
                nan_fraction=frac, nan_rng=rng,
            )
            probas = clf.predict_proba(X_te)
            p_full = np.zeros((len(X_te), n_classes))
            for col, cid in enumerate(clf.classes_):
                p_full[:, int(cid)] = probas[:, col]
            aurocs.append(macro_auroc(y_te, p_full, n_classes))
            prs.append(macro_pr_auc(y_te, p_full, n_classes))
        results[f"{frac:.2f}"] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "pr_auc_mean": float(np.mean(prs)),
            "pr_auc_std": float(np.std(prs)),
            "n_mask_seeds": len(seeds),
        }
        print(f"NaN {frac:>4.0%} : AUROC {np.mean(aurocs):.3f}±{np.std(aurocs):.3f}"
              f"  PR-AUC {np.mean(prs):.3f}±{np.std(prs):.3f}")

    # Figure
    fr = [float(f) for f in results]
    au = [results[f]["auroc_mean"] for f in results]
    au_sd = [results[f]["auroc_std"] for f in results]
    pr = [results[f]["pr_auc_mean"] for f in results]
    pr_sd = [results[f]["pr_auc_std"] for f in results]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.errorbar(fr, au, yerr=au_sd, fmt="o-", capsize=3, label="macro-AUROC")
    ax.errorbar(fr, pr, yerr=pr_sd, fmt="s--", capsize=3, label="macro-PR-AUC")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Fraction de NaN injectée (fenêtre test)")
    ax.set_ylabel("Score (test, 15 scénarios)")
    ax.set_title("E9 — Dégradation B2 vs taux de données manquantes")
    ax.legend()
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "nan_robustness.png")
    plt.close(fig)

    base = results["0.00"]["auroc_mean"]
    half = next((f for f in results
                 if results[f]["auroc_mean"] <= 0.5 + (base - 0.5) / 2), None)
    lines = [
        "# E9 — Robustesse NaN du headline B2 (audit 2026-06)",
        "",
        "Train propre ; NaN injectés dans la fenêtre TEST (instance norm",
        "nan-aware → 0 neutre). 5 graines de masque par fraction.",
        "",
        "| NaN | macro-AUROC | macro-PR-AUC |",
        "|---|---|---|",
    ]
    for f, r in results.items():
        lines.append(f"| {float(f):.0%} | {r['auroc_mean']:.3f} ± "
                     f"{r['auroc_std']:.3f} | {r['pr_auc_mean']:.3f} ± "
                     f"{r['pr_auc_std']:.3f} |")
    lines += [
        "",
        f"- Baseline (0 % NaN) : AUROC {base:.3f}",
        f"- Demi-vie du signal (AUROC ≤ 0.5 + (base−0.5)/2) : "
        f"{'≥ 50 % (jamais atteinte)' if half is None else f'{float(half):.0%}'}",
        "- Figure : nan_robustness.png",
    ]
    write_results(args.output, results, lines)


if __name__ == "__main__":
    main()
