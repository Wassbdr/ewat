"""Socle commun des expériences d'audit 2026-06 (E2, E5, E6, E9, D3).

Reproduit EXACTEMENT la featurisation du headline défensif B2
(`experiments/architecture_v2/chaos_mesh_target.py`) : fenêtre pré-injection
de k steps (position ``last`` sur le régime normal), instance normalization
sur les stats du régime normal, flatten (k × N × d), LogisticRegression-OvR
sur les scénarios Chaos Mesh — afin que les nouvelles mesures (calibration,
robustesse NaN, variance inter-split, oracle, masque de missingness) soient
directement comparables au chiffre B2 = 0.920.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.architecture_v2.instance_norm_diagnostic import (
    _extract_window,
    _instance_normalize,
    _load_signal,
    _load_split_episodes,
    _scenario_to_int,
)

# Tranches modales v4 (S(t) ∈ ℝ^{N×17}) pour les features de missingness D3.
MODALITY_SLICES_V4 = {"M": slice(0, 7), "T": slice(7, 13), "L": slice(13, 17)}

__all__ = [
    "MODALITY_SLICES_V4",
    "bootstrap_macro_ci",
    "build_b2_xy",
    "fit_predict_proba",
    "load_manifest",
    "macro_auroc",
    "macro_pr_auc",
]


def load_manifest(dataset_or_typing_dir: Path) -> tuple[dict, list[str]]:
    """manifest {ep_id → {scenario, split, cluster}} + scénarios triés."""
    return _load_split_episodes(Path(dataset_or_typing_dir))


def build_b2_xy(
    manifest: dict,
    features_root: Path,
    split: str,
    classes: list[str],
    k: int = 6,
    position: str = "last",
    nan_fraction: float = 0.0,
    nan_rng: np.random.Generator | None = None,
    with_missingness: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, y_int, ep_ids) avec la featurisation B2 exacte + options d'audit.

    Parameters
    ----------
    nan_fraction:
        E9 — fraction de cellules de la fenêtre mises à NaN aléatoirement
        AVANT l'instance norm (simule des trous de collecte à l'inférence).
    with_missingness:
        D3 — concatène, par (service × modalité), le ratio de NaN *brut* de
        la fenêtre (avant imputation/injection) : +N×3 features. La
        missingness est structurelle sur crash (T ≈ 20-25 % NaN) — l'imputer
        sans la représenter efface un signal discriminant.
    """
    X, y, eps = [], [], []
    for ep_id, info in manifest.items():
        if info["split"] != split:
            continue
        sig, normal = _load_signal(features_root, ep_id)
        win = _extract_window(sig, normal, k, position)  # (k, N, d)

        miss_feat = None
        if with_missingness:
            raw_nan = np.isnan(win)  # missingness réelle, avant toute injection
            miss_feat = np.concatenate([
                raw_nan[:, :, sl].mean(axis=(0, 2))  # (N,) par modalité
                for sl in MODALITY_SLICES_V4.values()
            ])  # (N*3,)

        if nan_fraction > 0.0:
            assert nan_rng is not None, "nan_rng requis avec nan_fraction>0"
            mask = nan_rng.random(win.shape) < nan_fraction
            win = win.copy()
            win[mask] = np.nan

        win = _instance_normalize(sig, normal, win)
        win = np.nan_to_num(win, nan=0.0)
        feat = win.reshape(-1)  # flatten (k*N*d,) — protocole B2
        if miss_feat is not None:
            feat = np.concatenate([feat, miss_feat.astype(np.float32)])
        X.append(feat)
        y.append(info["scenario"])
        eps.append(ep_id)
    return np.array(X, dtype=np.float32), _scenario_to_int(y, classes), eps


def fit_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    n_classes: int,
    reg_c: float = 1.0,
    max_iter: int = 2000,
) -> tuple[LogisticRegression, np.ndarray]:
    """LR-OvR B2 (lbfgs, déterministe) → (clf, probas alignées sur classes)."""
    clf = LogisticRegression(C=reg_c, max_iter=max_iter, solver="lbfgs")
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_eval)
    p_full = np.zeros((len(X_eval), n_classes), dtype=np.float64)
    for col_idx, cls_id in enumerate(clf.classes_):
        p_full[:, int(cls_id)] = probas[:, col_idx]
    return clf, p_full


def macro_auroc(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    vals = []
    for i in range(n_classes):
        y_bin = (y == i).astype(int)
        if y_bin.sum() < 1 or y_bin.sum() == len(y_bin):
            continue
        try:
            vals.append(float(roc_auc_score(y_bin, p[:, i])))
        except ValueError:
            continue
    return float(np.mean(vals)) if vals else float("nan")


def macro_pr_auc(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    vals = []
    for i in range(n_classes):
        y_bin = (y == i).astype(int)
        if y_bin.sum() < 1 or y_bin.sum() == len(y_bin):
            continue
        vals.append(float(average_precision_score(y_bin, p[:, i])))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_macro_ci(
    y: np.ndarray,
    p: np.ndarray,
    n_classes: int,
    n_boot: int,
    rng: np.random.Generator,
    metric: str = "auroc",
) -> tuple[float, float, float]:
    """(mean, lo, hi) bootstrap 95 % du macro-AUROC ou macro-PR-AUC."""
    fn = macro_auroc if metric == "auroc" else macro_pr_auc
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(fn(y[idx], p[idx], n_classes))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if not len(boots):
        return float("nan"), float("nan"), float("nan")
    return (float(boots.mean()),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def write_results(output: Path, payload: dict, md_lines: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    (output / "results.md").write_text("\n".join(md_lines) + "\n")
    print(f"→ {output / 'results.md'}")
