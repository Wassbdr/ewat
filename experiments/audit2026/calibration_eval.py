"""E2 (audit 2026-06) — Calibration des probabilités : Brier, ECE, reliability.

Deux cibles :

1. **B2** (headline défensif, LR-OvR sur scénarios Chaos Mesh) : les probas
   OvR par scénario sont-elles calibrées ? Avant/après recalibration
   isotonique fittée sur le split val.
2. **Précurseurs** (--precursor-dir) : les probas qui pilotent le seuil
   d'alerte opérationnel (0.7 recommandé dans STATUS) — si ECE > 0.1, le
   tableau seuil/FA/lead-time du rapport repose sur des scores non
   interprétables.

Métriques : Brier score (par classe, moyenné), ECE (10 bins, pondéré),
reliability curves (figure matplotlib vectorielle par cible).

Usage
-----
    python -m experiments.audit2026.calibration_eval \\
        --dataset data/datasets/ewat_v4_strat --features-root data/features/v4 \\
        [--precursor-dir experiments/multiseed/phase_h2/seed_42] \\
        [--k 6] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

from experiments.audit2026.common import (
    build_b2_xy,
    fit_predict_proba,
    load_manifest,
    write_results,
)
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Métriques de calibration
# ---------------------------------------------------------------------------

def ece_score(y_bin: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (bins équilarges, pondérés par effectif)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_bin)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y_bin[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_bin: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_bin) ** 2))


def _ovr_calibration(
    y: np.ndarray, p: np.ndarray, n_classes: int, n_bins: int = 10,
) -> dict:
    """Brier/ECE macro sur l'ensemble des problèmes binaires OvR (pooled)."""
    y_pool, p_pool = [], []
    for c in range(n_classes):
        y_bin = (y == c).astype(int)
        if y_bin.sum() == 0:
            continue
        y_pool.append(y_bin)
        p_pool.append(p[:, c])
    y_pool = np.concatenate(y_pool)
    p_pool = np.concatenate(p_pool)
    return {
        "brier": brier_score(y_pool, p_pool),
        "ece": ece_score(y_pool, p_pool, n_bins),
        "n_points": int(len(y_pool)),
        "_pooled": (y_pool, p_pool),
    }


def _isotonic_recalibrate(
    p_val_pool: np.ndarray, y_val_pool: np.ndarray, p_test: np.ndarray,
) -> np.ndarray:
    """Recalibration isotonique unique fittée sur les probas OvR poolées val.

    Une isotonic par classe serait idéale mais n_pos val ≈ 4/classe est trop
    petit ; le pooling OvR (15 problèmes binaires) donne ~900 points.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val_pool, y_val_pool)
    return iso.transform(p_test.ravel()).reshape(p_test.shape)


def _reliability_plot(curves: dict[str, tuple], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibration parfaite")
    for label, (y_pool, p_pool) in curves.items():
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_pool >= lo) & (p_pool < hi) if hi < 1 else \
                   (p_pool >= lo) & (p_pool <= hi)
            if mask.sum() < 5:
                continue
            xs.append(float(p_pool[mask].mean()))
            ys.append(float(y_pool[mask].mean()))
        ax.plot(xs, ys, "o-", label=label)
    ax.set_xlabel("Probabilité prédite (bins)")
    ax.set_ylabel("Fréquence observée")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cible 1 — B2
# ---------------------------------------------------------------------------

def eval_b2(args: argparse.Namespace, out: Path) -> dict:
    manifest, classes = load_manifest(args.dataset)
    n_classes = len(classes)
    X_tr, y_tr, _ = build_b2_xy(manifest, args.features_root, "train", classes, k=args.k)
    X_va, y_va, _ = build_b2_xy(manifest, args.features_root, "val", classes, k=args.k)
    X_te, y_te, _ = build_b2_xy(manifest, args.features_root, "test", classes, k=args.k)

    _, p_va = fit_predict_proba(X_tr, y_tr, X_va, n_classes)
    _, p_te = fit_predict_proba(X_tr, y_tr, X_te, n_classes)

    raw = _ovr_calibration(y_te, p_te, n_classes)
    y_te_pool, p_te_pool = raw.pop("_pooled")

    val_pool = _ovr_calibration(y_va, p_va, n_classes)
    y_va_pool, p_va_pool = val_pool.pop("_pooled")
    p_te_cal_pool = _isotonic_recalibrate(p_va_pool, y_va_pool, p_te_pool)
    cal = {
        "brier": brier_score(y_te_pool, p_te_cal_pool),
        "ece": ece_score(y_te_pool, p_te_cal_pool),
        "n_points": int(len(y_te_pool)),
    }

    _reliability_plot(
        {"brut": (y_te_pool, p_te_pool),
         "isotonique (fit val)": (y_te_pool, p_te_cal_pool)},
        out / "reliability_b2.png",
        "B2 (LR-OvR Chaos Mesh) — reliability test, probas OvR poolées",
    )
    return {"raw": raw, "isotonic": cal}


# ---------------------------------------------------------------------------
# Cible 2 — Précurseurs (probas du seuil d'alerte)
# ---------------------------------------------------------------------------

def eval_precursors(args: argparse.Namespace, out: Path) -> dict | None:
    """Calibration des PrecursorClassifiers d'un run (ex. phase_h2/seed_42)."""
    import pickle

    import torch

    from ewat.encoder.factory import build_encoder_from_checkpoint
    from ewat.precursor.dataset import PrecursorDataset
    from ewat.typing.siamese import SiameseTyper
    from experiments.precursor.train import _embed_dataset

    prec_dir = args.precursor_dir / "precursor"
    typ_dir = args.precursor_dir / "typing"
    enc_dir = args.precursor_dir / "encoder"
    if not (prec_dir / "results.json").exists():
        print(f"(précurseurs absents de {prec_dir} — cible 2 sautée)")
        return None

    res = json.loads((prec_dir / "results.json").read_text())
    k_opt = {int(c): int(k) for c, k in res["k_optimal"].items()}
    manifest_path = typ_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest = json.loads(manifest_path.read_text())

    device = "cpu"
    enc_ckpt = torch.load(enc_dir / "checkpoints" / "best_encoder.pt",
                          map_location="cpu", weights_only=False)
    encoder = build_encoder_from_checkpoint(enc_ckpt)
    encoder.load_state_dict(enc_ckpt["encoder_state"])
    typer_ckpt = torch.load(typ_dir / "checkpoints" / "best_siamese.pt",
                            map_location="cpu", weights_only=False)
    typer = SiameseTyper(encoder, d_proj=int(typer_ckpt.get("d_proj", 32)))
    typer.load_state_dict(typer_ckpt["typer_state"])
    typer = typer.to(device).eval()
    scaler_path = Path(enc_ckpt.get("scaler_path", enc_dir / "scaler.pkl"))

    y_pool_va, p_pool_va, y_pool_te, p_pool_te = [], [], [], []
    for c, k in k_opt.items():
        ckpt = prec_dir / "checkpoints" / f"classifier_type{c}_k{k}.pkl"
        if not ckpt.exists():
            continue
        with open(ckpt, "rb") as fh:
            clf = pickle.load(fh)
        for split, (y_pool, p_pool) in (("val", (y_pool_va, p_pool_va)),
                                        ("test", (y_pool_te, p_pool_te))):
            ds = PrecursorDataset(cluster_manifest, args.features_root, k=k, split=split)
            if scaler_path.exists():
                ds.load_scaler(scaler_path)
            z, y = _embed_dataset(typer, ds, batch_size=32, device=device)
            proba = clf.predict_proba(z)[:, c]
            y_pool.append((y == c).astype(int))
            p_pool.append(proba)

    if not y_pool_te:
        return None
    y_va, p_va = np.concatenate(y_pool_va), np.concatenate(p_pool_va)
    y_te, p_te = np.concatenate(y_pool_te), np.concatenate(p_pool_te)

    raw = {"brier": brier_score(y_te, p_te), "ece": ece_score(y_te, p_te),
           "n_points": int(len(y_te))}
    p_te_cal = _isotonic_recalibrate(p_va, y_va, p_te)
    cal = {"brier": brier_score(y_te, p_te_cal), "ece": ece_score(y_te, p_te_cal),
           "n_points": int(len(y_te))}

    _reliability_plot(
        {"brut": (y_te, p_te), "isotonique (fit val)": (y_te, p_te_cal)},
        out / "reliability_precursors.png",
        "Précurseurs (probas du seuil d'alerte) — reliability test",
    )
    return {"raw": raw, "isotonic": cal}


def main() -> None:
    p = argparse.ArgumentParser(description="E2 — calibration eval")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--precursor-dir", type=Path, default=None,
                   help="racine d'un run retrain (contenant encoder/typing/precursor)")
    p.add_argument("--output", type=Path,
                   default=Path("experiments/audit2026/calibration"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    b2 = eval_b2(args, args.output)
    prec = eval_precursors(args, args.output) if args.precursor_dir else None

    def fmt(d):
        return (f"Brier={d['brier']:.4f}  ECE={d['ece']:.4f} "
                f"(n={d['n_points']})")

    lines = [
        "# E2 — Calibration des probabilités (audit 2026-06)",
        "",
        "Probas OvR poolées sur les problèmes binaires (1 point par paire",
        "épisode×classe). Recalibration isotonique fittée sur le split val.",
        "Interprétation ECE : < 0.05 bon, 0.05-0.10 acceptable, > 0.10 les",
        "seuils de probabilité ne sont pas interprétables tels quels.",
        "",
        "## B2 (headline, LR-OvR Chaos Mesh, test)",
        f"- brut       : {fmt(b2['raw'])}",
        f"- isotonique : {fmt(b2['isotonic'])}",
        "- figure : reliability_b2.png",
    ]
    if prec:
        lines += [
            "",
            "## Précurseurs (probas du seuil d'alerte opérationnel, test)",
            f"- brut       : {fmt(prec['raw'])}",
            f"- isotonique : {fmt(prec['isotonic'])}",
            "- figure : reliability_precursors.png",
            "",
            "⚠ Si ECE brut > 0.10, le point opérationnel « seuil 0.7 » du",
            "rapport doit être re-déduit après recalibration.",
        ]
    write_results(args.output, {"b2": b2, "precursors": prec}, lines)


if __name__ == "__main__":
    main()
