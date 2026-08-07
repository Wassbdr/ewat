"""E1 (audit 2026-06) — USAD (KDD 2020) évalué au protocole EWAT sur v4_strat.

Deux volets, honnêtes sur ce que chaque méthode prétend faire :

(a) **Détection** (la tâche native d'USAD) : USAD entraîné sur les fenêtres
    du régime normal des épisodes train ; en test, AUROC/PR-AUC du score
    d'anomalie entre fenêtres normales (y=0) et fenêtres d'injection (y=1).
    Comparé au z-score max (baseline naïve du rapport).

(b) **Typage** (la tâche EWAT) : LR-OvR sur les latents USAD des fenêtres
    pré-injection vs B2 (LR-OvR sur les fenêtres brutes flatten). USAD
    n'étant PAS un classifieur multi-classe, ce volet mesure si sa
    représentation non supervisée porte l'information de type — limite
    documentée, pas masquée.

Featurisation identique à B2 partout : fenêtres k steps instance-normalized
aplaties.

Usage
-----
    python -m experiments.sota.usad_eval \\
        --dataset data/datasets/ewat_v4_strat --features-root data/features/v4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ewat.baselines.usad import USADDetector
from experiments.architecture_v2.instance_norm_diagnostic import (
    _instance_normalize,
    _load_signal,
)
from experiments.audit2026.common import (
    build_b2_xy,
    fit_predict_proba,
    load_manifest,
    macro_auroc,
    macro_pr_auc,
    write_results,
)
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Extraction des fenêtres normales / injection
# ---------------------------------------------------------------------------

def _load_regimes(features_root: Path, ep_id: str) -> np.ndarray:
    df = pd.read_parquet(features_root / ep_id / "labels.parquet",
                         columns=["regime"])
    return df["regime"].values


def _windows_from(sig: np.ndarray, normal: np.ndarray, idx: np.ndarray,
                  k: int, stride: int) -> list[np.ndarray]:
    """Fenêtres de k steps consécutifs entièrement dans ``idx``.

    Fenêtres ancrées au début de chaque run contigu du régime (un stride
    absolu depuis t=0 manquait les phases d'injection courtes — ex. 6 steps
    en [31..36] jamais alignés sur les positions 0, 6, 12…).
    """
    wins = []
    if len(idx) == 0:
        return wins
    # runs contigus dans idx
    breaks = np.where(np.diff(idx) > 1)[0]
    run_starts = np.concatenate([[0], breaks + 1])
    run_ends = np.concatenate([breaks, [len(idx) - 1]])
    for rs, re_ in zip(run_starts, run_ends):
        a, b = int(idx[rs]), int(idx[re_])
        for start in range(a, b - k + 2, stride):
            win = sig[start:start + k]
            win = _instance_normalize(sig, normal, win)
            wins.append(np.nan_to_num(win, nan=0.0).reshape(-1))
    return wins


def _detection_sets(
    manifest: dict, features_root: Path, k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_train_normal, X_test, y_test, |z|max_test) au protocole détection."""
    X_tr, X_te, y_te, zmax_te = [], [], [], []
    for ep_id, info in manifest.items():
        sig, normal = _load_signal(features_root, ep_id)
        regimes = _load_regimes(features_root, ep_id)
        norm_idx = np.where(regimes == "normal")[0]
        inj_idx = np.where(np.isin(regimes, ["injection", "drift_anomaly"]))[0]
        if info["split"] == "train":
            X_tr.extend(_windows_from(sig, normal, norm_idx, k, stride=2))
        elif info["split"] == "test":
            for label, idx in ((0, norm_idx), (1, inj_idx)):
                wins = _windows_from(sig, normal, idx, k, stride=k)
                for w in wins:
                    X_te.append(w)
                    y_te.append(label)
                    zmax_te.append(float(np.abs(w).max()))  # z-score baseline
    return (np.array(X_tr, dtype=np.float32), np.array(X_te, dtype=np.float32),
            np.array(y_te, dtype=int), np.array(zmax_te))


def main() -> None:
    p = argparse.ArgumentParser(description="E1 — USAD au protocole EWAT")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path, default=Path("experiments/sota/usad"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--d-latent", type=int, default=40)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)

    manifest, classes = load_manifest(args.dataset)
    n_classes = len(classes)

    # ----- Volet (a) : détection -----
    print("Volet (a) — détection : extraction des fenêtres …")
    X_tr, X_te, y_te, zmax = _detection_sets(manifest, args.features_root, args.k)
    print(f"  train normal: {X_tr.shape} | test: {X_te.shape} "
          f"({int(y_te.sum())} pos / {int((1 - y_te).sum())} neg)")
    det = USADDetector(d_latent=args.d_latent, epochs=args.epochs,
                       seed=args.seed).fit(X_tr)
    scores = det.anomaly_score(X_te)
    det_auroc = float(roc_auc_score(y_te, scores))
    det_pr = float(average_precision_score(y_te, scores))
    z_auroc = float(roc_auc_score(y_te, zmax))
    z_pr = float(average_precision_score(y_te, zmax))
    print(f"  USAD    : AUROC={det_auroc:.4f}  PR-AUC={det_pr:.4f}")
    print(f"  z-score : AUROC={z_auroc:.4f}  PR-AUC={z_pr:.4f}")

    # ----- Volet (b) : typage -----
    print("Volet (b) — typage : latents USAD vs B2 …")
    Xb_tr, yb_tr, _ = build_b2_xy(manifest, args.features_root, "train",
                                  classes, k=args.k)
    Xb_te, yb_te, _ = build_b2_xy(manifest, args.features_root, "test",
                                  classes, k=args.k)
    # USAD pour le typage : ré-entraîné sur les fenêtres pré-injection train
    # (mêmes entrées que B2), latents → LR-OvR
    det_typ = USADDetector(d_latent=args.d_latent, epochs=args.epochs,
                           seed=args.seed).fit(Xb_tr)
    Z_tr, Z_te = det_typ.latent(Xb_tr), det_typ.latent(Xb_te)
    _, p_usad = fit_predict_proba(Z_tr, yb_tr, Z_te, n_classes)
    _, p_b2 = fit_predict_proba(Xb_tr, yb_tr, Xb_te, n_classes)
    typ = {
        "usad_latent_lr": {"auroc": macro_auroc(yb_te, p_usad, n_classes),
                           "pr_auc": macro_pr_auc(yb_te, p_usad, n_classes)},
        "b2_raw_lr": {"auroc": macro_auroc(yb_te, p_b2, n_classes),
                      "pr_auc": macro_pr_auc(yb_te, p_b2, n_classes)},
    }
    for name, r in typ.items():
        print(f"  {name:16s}: AUROC={r['auroc']:.4f}  PR-AUC={r['pr_auc']:.4f}")

    payload = {
        "detection": {
            "usad": {"auroc": det_auroc, "pr_auc": det_pr},
            "zscore_max": {"auroc": z_auroc, "pr_auc": z_pr},
            "n_test_windows": int(len(y_te)),
            "n_train_windows": int(len(X_tr)),
        },
        "typing": typ,
        "config": {"k": args.k, "d_latent": args.d_latent,
                   "epochs": args.epochs, "seed": args.seed},
    }
    lines = [
        "# E1 — USAD (Audibert et al., KDD 2020) au protocole EWAT (audit 2026-06)",
        "",
        "## Volet (a) — Détection binaire normal/injection (tâche native USAD)",
        "",
        "| Méthode | AUROC | PR-AUC |",
        "|---|---|---|",
        f"| USAD (score reconstruction adversariale) | {det_auroc:.4f} | {det_pr:.4f} |",
        f"| z-score max (baseline naïve du rapport) | {z_auroc:.4f} | {z_pr:.4f} |",
        "",
        f"Fenêtres test : {int((1 - y_te).sum())} normales / {int(y_te.sum())} injection.",
        "",
        "## Volet (b) — Typage 15 scénarios (tâche EWAT)",
        "",
        "| Méthode | macro-AUROC | macro-PR-AUC |",
        "|---|---|---|",
        f"| LR-OvR sur latents USAD (d={args.d_latent}) | "
        f"{typ['usad_latent_lr']['auroc']:.4f} | {typ['usad_latent_lr']['pr_auc']:.4f} |",
        f"| **B2** (LR-OvR fenêtres brutes — headline) | "
        f"{typ['b2_raw_lr']['auroc']:.4f} | {typ['b2_raw_lr']['pr_auc']:.4f} |",
        "",
        "Limites du protocole (documentées, pas masquées) : USAD est un",
        "détecteur non supervisé — le volet (b) mesure si sa représentation",
        "latente porte l'information de type, pas une prétention du papier",
        "d'origine. Featurisation identique à B2 (fenêtres k instance-norm).",
    ]
    write_results(args.output, payload, lines)


if __name__ == "__main__":
    main()
