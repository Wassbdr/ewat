"""Baselines indépendants pour la valeur ajoutée du STGCN — cible : scénarios Chaos Mesh.

Contrairement à B1/B2 (qui prédisent les *labels EWAT*), B3/B4 utilisent les
**scénarios Chaos Mesh** comme vérité terrain indépendante du pipeline, permettant
de trancher si l'encodeur ajoute une discriminabilité prédictive réelle.

  B3 — Features brutes : LR one-vs-rest sur S(t) aplati (N×17 → 102-dim),
                          même fenêtre précurseur k=k_fixed, sans encodeur.
  B4 — Encodeur STGCN  : LR one-vs-rest sur z_e ∈ ℝ^{d_embed} (sortie encodeur
                          AVANT la tête siamoise, pour éviter la circularité
                          z_proj → entraîné sur labels EWAT).

Lecture des résultats
---------------------
  macro-AUROC B4 >> B3  →  l'encodeur ajoute une discriminabilité prédictive.
  macro-AUROC B4 ≈  B3  →  la valeur du STGCN est purement géométrique (H1) ;
                             pas d'apport prédictif supplémentaire.

Usage
-----
    python -m experiments.baselines.scenario_baselines \\
        --typing-dir  experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        --output experiments/baselines \\
        [--k 6] [--n-bootstrap 1000]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ewat.encoder.factory import build_encoder
from ewat.precursor.dataset import PrecursorDataset

# ---------------------------------------------------------------------------
# Helpers — raw-feature flattening (shared with precursor_baselines.py)
# ---------------------------------------------------------------------------

def _flatten_signal(ds: PrecursorDataset) -> tuple[np.ndarray, list[str]]:
    """Mean-over-time S(t) → flat vector; also return scenario list."""
    x_data, scenarios = [], []
    for idx in range(len(ds)):
        item = ds[idx]
        # signal: (k, N, 17) → mean over time → (N*17,)
        sig = item["signal"].numpy()
        feat = sig.mean(axis=0).ravel()
        x_data.append(feat)
        # PrecursorDataset doesn't expose scenario; retrieve from manifest
        ep_id = ds.episodes[idx][0]
        scenarios.append(ep_id)  # placeholder, filled below
    return np.stack(x_data), scenarios


def _flatten_with_scenario(
    ds: PrecursorDataset,
    cluster_manifest: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_flat, y_scenario_int) for a PrecursorDataset."""
    x_data, y = [], []
    for idx in range(len(ds)):
        item = ds[idx]
        sig = item["signal"].numpy()           # (k, N, 17)
        feat = sig.mean(axis=0).ravel()        # (N*17,)
        ep_id = ds.episodes[idx][0]
        y.append(cluster_manifest[ep_id]["scenario"])
        x_data.append(feat)
    return np.stack(x_data), np.array(y)


# ---------------------------------------------------------------------------
# Helpers — STGCN encoder embeddings (z_e, before siamese projection head)
# ---------------------------------------------------------------------------

def _load_encoder(typing_dir: Path, encoder_dir: Path, device: torch.device):
    """Load STGCNEncoder from checkpoint; return encoder in eval mode."""
    enc_ckpt_path = encoder_dir / "checkpoints" / "best_encoder.pt"
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
    arch_meta = enc_ckpt.get("arch", {})
    encoder = build_encoder(
        architecture=arch_meta.get("arch", "stgcn"),
        d_feat=int(arch_meta.get("d_feat", 17)),
        n_nodes=int(arch_meta.get("n_nodes", 6)),
        d_hidden=int(arch_meta.get("d_hidden", 64)),
        d_embed=int(arch_meta.get("d_embed", 64)),
    )
    encoder.load_state_dict(enc_ckpt["encoder_state"])
    return encoder.to(device).eval()


@torch.no_grad()
def _embed_z_e(
    ds: PrecursorDataset,
    encoder,
    cluster_manifest: dict[str, dict],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass pre-injection windows through encoder → z_e ∈ ℝ^{d_embed}; return (Z, y_scenario)."""
    all_z, all_y = [], []

    for idx in range(len(ds)):
        item = ds[idx]
        sig = item["signal"].unsqueeze(0).to(device)   # (1, k, N, 17)
        adj = item["adjacency"].unsqueeze(0).to(device)  # (1, k, N, N, 3)
        lengths = torch.tensor([item["signal"].shape[0]], dtype=torch.long, device=device)

        z_e = encoder(sig, adj, lengths=lengths)        # (1, d_embed)
        all_z.append(z_e.squeeze(0).cpu().numpy())

        ep_id = ds.episodes[idx][0]
        all_y.append(cluster_manifest[ep_id]["scenario"])

    return np.stack(all_z), np.array(all_y)


# ---------------------------------------------------------------------------
# OvR classification helpers
# ---------------------------------------------------------------------------

def _scenario_to_int(scenarios: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map scenario strings → ints; return (y_int, sorted class list)."""
    classes = sorted(set(scenarios.tolist()))
    mapping = {c: i for i, c in enumerate(classes)}
    return np.array([mapping[s] for s in scenarios], dtype=int), classes


def _macro_auroc_ovr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    reg_c: float = 1.0,
    max_iter: int = 1000,
) -> tuple[float, dict[str, float]]:
    """Fit LR multi-class OvR and return (macro_auroc, per_scenario_auroc)."""
    n_classes = len(classes)
    per_class: dict[str, float] = {}

    # Fit a single multi-class LR with OvR strategy
    clf = LogisticRegression(C=reg_c, max_iter=max_iter, solver="lbfgs")
    clf.fit(x_train, y_train)
    probas = clf.predict_proba(x_test)        # (n_test, n_classes)

    aurocs = []
    for i, cls in enumerate(classes):
        y_bin_test = (y_test == i).astype(int)
        if y_bin_test.sum() < 1 or y_bin_test.sum() == len(y_bin_test):
            per_class[cls] = float("nan")
            continue
        auc = float(roc_auc_score(y_bin_test, probas[:, i]))
        per_class[cls] = auc
        aurocs.append(auc)

    macro = float(np.nanmean(aurocs)) if aurocs else float("nan")
    return macro, per_class


def _bootstrap_macro_auroc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    reg_c: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """Bootstrap 95% CI on macro-AUROC for scenario classification."""
    n_test = len(y_test)
    clf = LogisticRegression(C=reg_c, max_iter=max_iter, solver="lbfgs")
    clf.fit(x_train, y_train)
    probas = clf.predict_proba(x_test)        # (n_test, n_classes)

    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_test, size=n_test)
        y_boot = y_test[idx]
        p_boot = probas[idx]
        per = []
        for i in range(len(classes)):
            y_bin = (y_boot == i).astype(int)
            if y_bin.sum() < 1 or y_bin.sum() == len(y_boot):
                continue
            try:
                per.append(float(roc_auc_score(y_bin, p_boot[:, i])))
            except ValueError:
                continue
        if per:
            boot_aurocs.append(float(np.mean(per)))

    point, *_ = (
        float(np.nanmean(
            [roc_auc_score((y_test == i).astype(int), probas[:, i])
             for i in range(len(classes))
             if 0 < (y_test == i).sum() < len(y_test)]
        )),
    )
    ci_lo, ci_hi = (
        float(np.percentile(boot_aurocs, 2.5)),
        float(np.percentile(boot_aurocs, 97.5)),
    ) if boot_aurocs else (float("nan"), float("nan"))
    return {"estimate": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "alpha": 0.05, "n_bootstrap": n_bootstrap}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario baselines B3/B4 — independent ground truth for STGCN value"
    )
    parser.add_argument("--typing-dir",    type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir",   type=Path, default=None,
                        help="Encoder dir (default: typing_dir.parent/encoder)")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output",        type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--k",             type=int,  default=6,
                        help="Precursor window length (steps). Default 6 = dominant k* in EWAT.")
    parser.add_argument("--reg-c",         type=float, default=1.0)
    parser.add_argument("--max-iter",      type=int,   default=1000)
    parser.add_argument("--n-bootstrap",   type=int,   default=1000)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    encoder_dir = args.encoder_dir or (args.typing_dir.parent / "encoder")

    # Load manifest
    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    scaler_path = encoder_dir / "scaler.pkl"
    print(f"k={args.k} steps ({args.k * 30}s = {args.k * 30 // 60}min {args.k * 30 % 60}s)")
    print(f"Device: {device}")

    # Build datasets
    ds_train = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="train")
    ds_val   = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="val")
    ds_test  = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="test")
    if scaler_path.exists():
        for ds in (ds_train, ds_val, ds_test):
            ds.load_scaler(scaler_path)
    print(f"Split sizes — train:{len(ds_train)}  val:{len(ds_val)}  test:{len(ds_test)}")

    # -----------------------------------------------------------------------
    # B3 — Features brutes → scénario Chaos Mesh
    # -----------------------------------------------------------------------
    print("\n=== B3 — Features brutes → scénario Chaos Mesh ===")
    x3_train, y3_train_str = _flatten_with_scenario(ds_train, cluster_manifest)
    x3_val,   y3_val_str   = _flatten_with_scenario(ds_val,   cluster_manifest)
    x3_test,  y3_test_str  = _flatten_with_scenario(ds_test,  cluster_manifest)

    y3_train, classes = _scenario_to_int(y3_train_str)
    y3_val,   _       = _scenario_to_int(y3_val_str)
    y3_test,  _       = _scenario_to_int(y3_test_str)
    print(f"  {len(classes)} scénarios : {classes}")

    b3_macro_val,  b3_per_val  = _macro_auroc_ovr(
        x3_train, y3_train, x3_val, y3_val, classes, args.reg_c, args.max_iter
    )
    b3_macro_test, b3_per_test = _macro_auroc_ovr(
        x3_train, y3_train, x3_test, y3_test, classes, args.reg_c, args.max_iter
    )
    print(f"  macro-AUROC val={b3_macro_val:.4f}  test={b3_macro_test:.4f}")

    rng3 = np.random.default_rng(args.seed)
    b3_ci = _bootstrap_macro_auroc(
        x3_train, y3_train, x3_test, y3_test, classes,
        args.n_bootstrap, rng3, args.reg_c, args.max_iter
    )
    print(f"  Bootstrap CI test : {b3_ci['estimate']:.4f} [{b3_ci['ci_lo']:.4f}, {b3_ci['ci_hi']:.4f}]")

    # -----------------------------------------------------------------------
    # B4 — Encodeur STGCN (z_e ∈ ℝ^{d_embed}) → scénario Chaos Mesh
    # -----------------------------------------------------------------------
    print("\n=== B4 — Encodeur STGCN (z_e, avant tête siamoise) → scénario Chaos Mesh ===")
    encoder = _load_encoder(args.typing_dir, encoder_dir, device)
    print(f"  Encoder chargé depuis {encoder_dir / 'checkpoints' / 'best_encoder.pt'}")
    print(f"  d_embed = {encoder.embedding_dim}")

    x4_train, y4_train_str = _embed_z_e(ds_train, encoder, cluster_manifest, device)
    x4_val,   y4_val_str   = _embed_z_e(ds_val,   encoder, cluster_manifest, device)
    x4_test,  y4_test_str  = _embed_z_e(ds_test,  encoder, cluster_manifest, device)

    y4_train, _ = _scenario_to_int(y4_train_str)
    y4_val,   _ = _scenario_to_int(y4_val_str)
    y4_test,  _ = _scenario_to_int(y4_test_str)

    b4_macro_val,  b4_per_val  = _macro_auroc_ovr(
        x4_train, y4_train, x4_val, y4_val, classes, args.reg_c, args.max_iter
    )
    b4_macro_test, b4_per_test = _macro_auroc_ovr(
        x4_train, y4_train, x4_test, y4_test, classes, args.reg_c, args.max_iter
    )
    print(f"  macro-AUROC val={b4_macro_val:.4f}  test={b4_macro_test:.4f}")

    rng4 = np.random.default_rng(args.seed + 1)
    b4_ci = _bootstrap_macro_auroc(
        x4_train, y4_train, x4_test, y4_test, classes,
        args.n_bootstrap, rng4, args.reg_c, args.max_iter
    )
    print(f"  Bootstrap CI test : {b4_ci['estimate']:.4f} [{b4_ci['ci_lo']:.4f}, {b4_ci['ci_hi']:.4f}]")

    # -----------------------------------------------------------------------
    # Interprétation automatique
    # -----------------------------------------------------------------------
    delta = b4_macro_test - b3_macro_test
    print("\n--- Interprétation ---")
    if delta > 0.03:
        interpretation = (
            f"B4 > B3 de {delta:+.4f} : l'encodeur STGCN apporte une discriminabilité "
            "prédictive indépendante des labels EWAT."
        )
    elif delta < -0.03:
        interpretation = (
            f"B4 < B3 de {delta:+.4f} : l'encodeur STGCN détériore la discriminabilité "
            "des scénarios. La valeur du STGCN est limitée à la structuration (H1)."
        )
    else:
        interpretation = (
            f"B4 ≈ B3 (Δ={delta:+.4f}) : l'encodeur STGCN n'apporte pas de discriminabilité "
            "prédictive au-delà des features brutes. Sa valeur est géométrique (H1) et non prédictive."
        )
    print(f"  Δ(B4-B3) test = {delta:+.4f}")
    print(f"  → {interpretation}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    summary = {
        "k": args.k,
        "n_scenarios": len(classes),
        "classes": classes,
        "b3_raw_features": {
            "macro_auroc_val": b3_macro_val,
            "macro_auroc_test": b3_macro_test,
            "per_scenario_val": b3_per_val,
            "per_scenario_test": b3_per_test,
            "ci_test": b3_ci,
        },
        "b4_stgcn_encoder": {
            "d_embed": int(encoder.embedding_dim),
            "macro_auroc_val": b4_macro_val,
            "macro_auroc_test": b4_macro_test,
            "per_scenario_val": b4_per_val,
            "per_scenario_test": b4_per_test,
            "ci_test": b4_ci,
        },
        "delta_b4_minus_b3_test": delta,
        "interpretation": interpretation,
    }
    out_json = args.output / "scenario_baselines.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Human-readable report
    lines = [
        "# Baselines scénarios Chaos Mesh — B3/B4\n",
        "Vérité terrain indépendante des labels EWAT (15 scénarios × 3 épisodes test).\n",
        f"Fenêtre précurseur k={args.k} steps ({args.k * 30}s).\n",
        "**Lecture** : macro-AUROC OvR sur les 15 scénarios Chaos Mesh.",
        "B4 > B3 → l'encodeur ajoute une discriminabilité prédictive réelle.\n",
        "## Résultats\n",
        "| Condition | macro-AUROC test | IC 95% | Δ vs B3 |",
        "|-----------|-----------------|--------|---------|",
        f"| B3 (features brutes) | {b3_macro_test:.4f} "
        f"| [{b3_ci['ci_lo']:.4f}, {b3_ci['ci_hi']:.4f}] | — |",
        f"| B4 (STGCN z_e, d={encoder.embedding_dim}) | {b4_macro_test:.4f} "
        f"| [{b4_ci['ci_lo']:.4f}, {b4_ci['ci_hi']:.4f}] | {delta:+.4f} |",
        "",
        "## Par scénario (test set)\n",
        "| Scénario | B3 (brut) | B4 (z_e) | Δ |",
        "|----------|-----------|----------|---|",
    ]
    for cls in classes:
        b3v = b3_per_test.get(cls, float("nan"))
        b4v = b4_per_test.get(cls, float("nan"))
        d = (b4v - b3v) if not (np.isnan(b3v) or np.isnan(b4v)) else float("nan")
        b3s = f"{b3v:.4f}" if not np.isnan(b3v) else "NaN"
        b4s = f"{b4v:.4f}" if not np.isnan(b4v) else "NaN"
        ds = f"{d:+.4f}" if not np.isnan(d) else "NaN"
        lines.append(f"| {cls} | {b3s} | {b4s} | {ds} |")
    lines += [
        "",
        "## Interprétation\n",
        interpretation,
        "",
        "## Méthodologie",
        "- B3 : LR (lbfgs, C=1.0, OvR) sur S(t) aplati (mean sur k steps, N×17→102-dim).",
        f"- B4 : LR (lbfgs, C=1.0, OvR) sur z_e ∈ ℝ^{{{encoder.embedding_dim}}} (STGCN avant tête siamoise).",
        "- Bootstrap CI : 1000 rééchantillonnages (stratification empirique).",
        "- **Note** : B3/B4 utilisent les scénarios Chaos Mesh comme cible, indépendamment",
        "  des labels de clustering EWAT (contrairement à B1/B2 dans precursor_baselines.md).",
    ]
    out_md = args.output / "scenario_baselines.md"
    out_md.write_text("\n".join(lines))
    print(f"\nReport : {out_md}")
    print(f"JSON   : {out_json}")


if __name__ == "__main__":
    main()
