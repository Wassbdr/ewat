"""E5 (audit 2026-06) — Variance inter-split du headline B2.

Le multi-seed Phase H/J mesure la variance des graines SUR UN MÊME split
270/60/45 ; l'IC bootstrap de B2 capture la variance intra-split. Ce script
mesure ce qui manquait : la variance due AU CHOIX DU SPLIT lui-même.

Protocole : 5 assemblages stratifiés shuffled (assemble_dataset
--split-mode shuffled --split-seed s — shuffle intra-scénario seedé,
quotas val/test préservés) → B2 (LR lbfgs déterministe) sur chacun.

⚠ Le mode shuffled brise volontairement l'ordre temporel intra-scénario :
les épisodes étant des unités indépendantes pour B2 (pas de fuite
inter-épisode au niveau features), c'est acceptable pour mesurer la variance
d'échantillonnage — à documenter comme protocole distinct du split temporel
officiel.

Usage
-----
    python -m experiments.audit2026.split_variance \\
        --features-root data/features/v4 [--n-splits 5]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
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

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assemble(features_root: Path, out: Path, split_seed: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.assemble_dataset",
        "--features-root", str(features_root),
        "--output", str(out),
        "--symlink-episodes",
        "--split-mode", "shuffled",
        "--split-seed", str(split_seed),
        "--force",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"assemble split_seed={split_seed} failed:\n{proc.stderr}")


def main() -> None:
    p = argparse.ArgumentParser(description="E5 — variance inter-split de B2")
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--reference-dataset", type=Path,
                   default=Path("data/datasets/ewat_v4_strat"),
                   help="split temporel officiel (point de comparaison)")
    p.add_argument("--output", type=Path,
                   default=Path("experiments/audit2026/split_variance"))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)

    def b2_on(dataset_dir: Path) -> tuple[float, float]:
        manifest, classes = load_manifest(dataset_dir)
        n_classes = len(classes)
        X_tr, y_tr, _ = build_b2_xy(manifest, args.features_root, "train",
                                    classes, k=args.k)
        X_te, y_te, _ = build_b2_xy(manifest, args.features_root, "test",
                                    classes, k=args.k)
        _, p_te = fit_predict_proba(X_tr, y_tr, X_te, n_classes)
        return (macro_auroc(y_te, p_te, n_classes),
                macro_pr_auc(y_te, p_te, n_classes))

    ref_auroc, ref_pr = b2_on(args.reference_dataset)
    print(f"référence (split temporel officiel) : AUROC={ref_auroc:.4f} "
          f"PR-AUC={ref_pr:.4f}")

    per_split = []
    with tempfile.TemporaryDirectory(prefix="ewat_e5_") as tmp:
        for s in range(args.n_splits):
            ds = Path(tmp) / f"split_{s}"
            _assemble(args.features_root, ds, split_seed=s)
            auroc, pr = b2_on(ds)
            per_split.append({"split_seed": s, "auroc": auroc, "pr_auc": pr})
            print(f"split_seed={s} : AUROC={auroc:.4f}  PR-AUC={pr:.4f}")

    aurocs = np.array([r["auroc"] for r in per_split])
    prs = np.array([r["pr_auc"] for r in per_split])
    payload = {
        "reference_temporal": {"auroc": ref_auroc, "pr_auc": ref_pr},
        "shuffled_splits": per_split,
        "auroc_mean": float(aurocs.mean()), "auroc_std": float(aurocs.std()),
        "auroc_min": float(aurocs.min()), "auroc_max": float(aurocs.max()),
        "pr_auc_mean": float(prs.mean()), "pr_auc_std": float(prs.std()),
        "n_splits": args.n_splits, "k": args.k,
    }
    lines = [
        "# E5 — Variance inter-split du headline B2 (audit 2026-06)",
        "",
        f"{args.n_splits} assemblages stratifiés shuffled (quotas identiques",
        "au split officiel) ; LR lbfgs déterministe → toute la variance vient",
        "du split. Protocole distinct du split temporel officiel (documenté",
        "dans le script).",
        "",
        "| Split | macro-AUROC | macro-PR-AUC |",
        "|---|---|---|",
        f"| temporel officiel (v4_strat) | {ref_auroc:.4f} | {ref_pr:.4f} |",
    ]
    for r in per_split:
        lines.append(f"| shuffled seed={r['split_seed']} | {r['auroc']:.4f} "
                     f"| {r['pr_auc']:.4f} |")
    lines += [
        "",
        f"**AUROC inter-split : {aurocs.mean():.4f} ± {aurocs.std():.4f}** "
        f"(range [{aurocs.min():.4f}, {aurocs.max():.4f}])",
        f"PR-AUC inter-split : {prs.mean():.4f} ± {prs.std():.4f}",
        "",
        "Lecture : à comparer à l'IC bootstrap intra-split de B2",
        "([0.878, 0.956]) — si l'écart-type inter-split est du même ordre,",
        "l'IC reporté couvre raisonnablement la variance d'échantillonnage ;",
        "s'il est nettement plus large, le headline doit être reporté avec",
        "cette variance en plus (L2.5/L6.1 enfin quantifiée).",
    ]
    write_results(args.output, payload, lines)


if __name__ == "__main__":
    main()
