"""Vérification corrigée de H1 et H3.

Problèmes identifiés dans les scripts d'origine :
- H1 : AgglomerativeClustering.fit_predict(z_val/z_test) → labels arbitrairement permutés
  par rapport au clustering train. La silhouette est valide pour mesurer la structurabilité,
  mais les IDs cluster sont incohérents cross-split.
- H3 : k* sélectionné sur auroc_table_TEST → data snooping. Val/test labels permutés →
  AUROC calculé sur une correspondance fausse entre classifieur (train IDs) et labels test.

Correction :
1. Réassigner val/test via nearest centroid depuis les centroides train → labels alignés.
2. Recalculer silhouette(test) avec labels corrigés → H1 corrigé.
3. Pour chaque k, ré-entraîner un classifieur LR sur (z_train, y_train) et évaluer
   AUROC sur (z_val_corrigé, y_val_corrigé). Sélectionner k* depuis val.
4. Rapporter AUROC(test_corrigé) à k* val-optimal → H3 corrigé.
5. Sauvegarder un cluster_manifest_corrected.json avec les bons labels.

Usage
-----
    python -m experiments.verification.verify_h1_h3 \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --precursor-dir experiments/precursor \\
        --features-root data/features/v3 \\
        --output experiments/verification
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ewat.precursor.dataset import PrecursorDataset
from ewat.precursor.model import PrecursorClassifier, baseline_auroc
from ewat.typing.siamese import SiameseTyper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_centroid_labels(z: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each embedding to the nearest train centroid (L2 distance)."""
    dists = np.linalg.norm(z[:, None, :] - centroids[None, :, :], axis=2)  # (N, K)
    return np.argmin(dists, axis=1).astype(int)


from ewat.typing.loader import load_typer as _load_typer  # noqa: E402


@torch.no_grad()
def _embed_precursor_dataset(
    typer: SiameseTyper,
    dataset: PrecursorDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings, labels = [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        sig = item["signal"].unsqueeze(0).to(device)
        adj = item["adjacency"].unsqueeze(0).to(device)
        z = typer.embed(sig, adj).cpu().numpy()[0]
        embeddings.append(z)
        labels.append(item["cluster"])
    return np.stack(embeddings), np.array(labels, dtype=int)


def _auroc_per_type(
    clf: PrecursorClassifier,
    z: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
) -> dict[int, float]:
    """AUROC per cluster type (one-vs-rest). Returns NaN if < 2 positives or negatives."""
    proba = clf.predict_proba(z)  # (N, n_clusters)
    results: dict[int, float] = {}
    for c in range(n_clusters):
        y_bin = (y == c).astype(int)
        if y_bin.sum() < 2 or (len(y_bin) - y_bin.sum()) < 2:
            results[c] = float("nan")
            continue
        try:
            results[c] = float(roc_auc_score(y_bin, proba[:, c]))
        except Exception:
            results[c] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérification corrigée H1+H3")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/verification"))
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--reg-c", type=float, default=1.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # 1. Load saved embeddings + centroids + original manifest
    # -----------------------------------------------------------------------
    artifacts_dir = args.typing_dir / "cluster_artifacts"
    centroids = np.load(artifacts_dir / "centroids.npy")          # (K, d_proj)
    z_train   = np.load(artifacts_dir / "embeddings_train.npy")   # (N_tr, d_proj)
    z_val     = np.load(artifacts_dir / "embeddings_val.npy")     # (N_val, d_proj)
    z_test    = np.load(artifacts_dir / "embeddings_test.npy")    # (N_te, d_proj)
    y_train   = np.load(artifacts_dir / "labels_train.npy")       # (N_tr,) — CORRECT

    k_opt_orig = centroids.shape[0]
    n_clusters = k_opt_orig
    print(f"K={n_clusters}  N_train={len(z_train)}  N_val={len(z_val)}  N_test={len(z_test)}")

    # Original (wrong) labels for comparison
    y_val_wrong  = np.load(artifacts_dir / "labels_val.npy")
    y_test_wrong = np.load(artifacts_dir / "labels_test.npy")

    # -----------------------------------------------------------------------
    # 2. Corrected label assignment (nearest centroid)
    # -----------------------------------------------------------------------
    print("\n[1] Réassignation par nearest centroid …")
    y_val_corr  = _nearest_centroid_labels(z_val,  centroids)
    y_test_corr = _nearest_centroid_labels(z_test, centroids)

    # Verify train: nearest centroid should agree with original train labels
    y_train_nc = _nearest_centroid_labels(z_train, centroids)
    train_agreement = float((y_train_nc == y_train).mean())
    print(f"  Train label agreement (NC vs original): {train_agreement:.3f}")

    # -----------------------------------------------------------------------
    # 3. Corrected H1 — silhouette on test with nearest-centroid labels
    # -----------------------------------------------------------------------
    print("\n[2] Silhouette corrigée (nearest centroid) …")
    sil_train_nc = float(silhouette_score(z_train, y_train))
    sil_val_nc   = float(silhouette_score(z_val,   y_val_corr))
    sil_test_nc  = float(silhouette_score(z_test,  y_test_corr))
    sil_val_wrong  = float(silhouette_score(z_val,  y_val_wrong))
    sil_test_wrong = float(silhouette_score(z_test, y_test_wrong))

    print(f"  silhouette(train, original)  = {sil_train_nc:.4f}")
    print(f"  silhouette(val,   original)={sil_val_wrong:.4f}"
          f"  → NC-corrected={sil_val_nc:.4f}")
    print(f"  silhouette(test,  original)={sil_test_wrong:.4f}"
          f"  → NC-corrected={sil_test_nc:.4f}")
    h1_pass_corr = sil_test_nc >= 0.3
    print(f"  H1 corrigé: {'✓ PASS' if h1_pass_corr else '✗ FAIL'} "
          f"(sil_test_NC={sil_test_nc:.4f}, seuil 0.3)")

    # -----------------------------------------------------------------------
    # 4. Corrected manifest (nearest-centroid labels for val/test)
    # -----------------------------------------------------------------------
    print("\n[3] Construction du manifest corrigé …")
    cluster_manifest_orig: dict[str, dict] = json.loads(
        (artifacts_dir / "cluster_manifest.json").read_text()
    )

    # Ordered episode lists from saved embeddings (order matches embedding arrays)
    train_eps = [ep for ep, m in cluster_manifest_orig.items() if m["split"] == "train"]
    val_eps   = [ep for ep, m in cluster_manifest_orig.items() if m["split"] == "val"]
    test_eps  = [ep for ep, m in cluster_manifest_orig.items() if m["split"] == "test"]

    manifest_corr: dict[str, dict] = {}
    for ep, label in zip(train_eps, y_train.tolist()):
        manifest_corr[ep] = {**cluster_manifest_orig[ep], "cluster": int(label)}
    for ep, label in zip(val_eps, y_val_corr.tolist()):
        manifest_corr[ep] = {**cluster_manifest_orig[ep], "cluster": int(label)}
    for ep, label in zip(test_eps, y_test_corr.tolist()):
        manifest_corr[ep] = {**cluster_manifest_orig[ep], "cluster": int(label)}

    manifest_corr_path = artifacts_dir / "cluster_manifest_corrected.json"
    manifest_corr_path.write_text(json.dumps(manifest_corr, indent=2))
    print(f"  Manifest corrigé sauvegardé : {manifest_corr_path}")

    # Check label changes
    val_changed = sum(
        1 for ep in val_eps if manifest_corr[ep]["cluster"] != cluster_manifest_orig[ep]["cluster"]
    )
    test_changed = sum(
        1 for ep in test_eps if manifest_corr[ep]["cluster"] != cluster_manifest_orig[ep]["cluster"]
    )
    print(f"  Labels val modifiés : {val_changed}/{len(val_eps)}")
    print(f"  Labels test modifiés : {test_changed}/{len(test_eps)}")

    # -----------------------------------------------------------------------
    # 5. Load scaler and typer for precursor re-embedding
    # -----------------------------------------------------------------------
    print("\n[4] Chargement typer pour ré-embedding précurseurs …")
    typer = _load_typer(args.typing_dir, args.encoder_dir, device)

    scaler: StandardScaler | None = None
    scaler_path = args.encoder_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    print(f"  Typer chargé. scaler={'yes' if scaler else 'no'}")

    # -----------------------------------------------------------------------
    # 6. Re-embed and recompute AUROC for each k — corrected H3
    # -----------------------------------------------------------------------
    print("\n[5] Re-embedding et AUROC corrigé pour chaque k …")
    auroc_val_corr:  dict[int, dict[int, float]] = {}
    auroc_test_corr: dict[int, dict[int, float]] = {}

    for k in args.k_values:
        ds_train = PrecursorDataset(manifest_corr, args.features_root, k=k, split="train")
        ds_val   = PrecursorDataset(manifest_corr, args.features_root, k=k, split="val")
        ds_test  = PrecursorDataset(manifest_corr, args.features_root, k=k, split="test")

        if scaler is not None:
            ds_train.load_scaler(scaler_path)
            ds_val.load_scaler(scaler_path)
            ds_test.load_scaler(scaler_path)

        z_tr, y_tr = _embed_precursor_dataset(typer, ds_train, device)
        z_va, y_va = _embed_precursor_dataset(typer, ds_val,   device)
        z_te, y_te = _embed_precursor_dataset(typer, ds_test,  device)

        # Fit fresh classifier on (z_tr, y_tr) — labels already correct
        clf = PrecursorClassifier(n_clusters=n_clusters, reg_c=args.reg_c, max_iter=500)
        clf.fit(z_tr, y_tr)

        auroc_va = _auroc_per_type(clf, z_va, y_va, n_clusters)
        auroc_te = _auroc_per_type(clf, z_te, y_te, n_clusters)
        auroc_val_corr[k]  = auroc_va
        auroc_test_corr[k] = auroc_te

        mean_va = float(np.nanmean(list(auroc_va.values())))
        mean_te = float(np.nanmean(list(auroc_te.values())))
        print(f"  k={k:2d}  AUROC_val={mean_va:.3f}  AUROC_test={mean_te:.3f}")

    # -----------------------------------------------------------------------
    # 7. k* from val (corrected), AUROC test (corrected)
    # -----------------------------------------------------------------------
    print("\n[6] Sélection k* depuis val (corrigé) …")
    baseline = baseline_auroc(n_clusters)

    k_optimal_corr: dict[int, int] = {}
    auroc_final: dict[int, float] = {}

    for c in range(n_clusters):
        best_k, best_auc = args.k_values[0], float("-inf")
        for k in args.k_values:
            auc = auroc_val_corr[k].get(c, float("nan"))
            if not np.isnan(auc) and auc > best_auc:
                best_auc, best_k = auc, k
        k_optimal_corr[c] = best_k
        auroc_final[c] = auroc_test_corr[best_k].get(c, float("nan"))

    print(f"\n{'Type':<6} {'k*_val':>6} {'AUROC_val':>10} {'AUROC_test':>11} {'Pass':>6}")
    print("-" * 44)
    h3_per_type: dict[int, bool | None] = {}
    for c in range(n_clusters):
        k = k_optimal_corr[c]
        auc_v = auroc_val_corr[k].get(c, float("nan"))
        auc_t = auroc_final[c]
        if np.isnan(auc_t):
            h3_per_type[c] = None
            sig = "NaN"
        else:
            h3_per_type[c] = bool(auc_t > baseline)
            sig = "✓" if auc_t > baseline else "✗"
        print(f"  C{c:<4} {k:>6}  {auc_v:>10.3f}  {auc_t:>11.3f}  {sig:>6}")

    h3_pass_corr = any(v for v in h3_per_type.values() if v is not None)
    n_pass = sum(1 for v in h3_per_type.values() if v)
    print(f"\nH3 corrigé: {'✓ PASS' if h3_pass_corr else '✗ FAIL'} "
          f"({n_pass}/{n_clusters} types AUROC > {baseline})")

    # -----------------------------------------------------------------------
    # 8. Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE VÉRIFICATION")
    print("=" * 60)
    print("\nH1 — Silhouette(test)")
    print(f"  AVANT (clustering indépendant) : {sil_test_wrong:.4f}")
    print(f"  APRÈS (nearest centroid)        : {sil_test_nc:.4f}")
    print(f"  H1 {'✓ PASS' if h1_pass_corr else '✗ FAIL'} (seuil 0.3)")

    print("\nH3 — AUROC précurseurs")
    print("  Avant : k* sur TEST, labels permutés")
    print("  Après : k* sur VAL, labels nearest-centroid")
    print(f"  H3 {'✓ PASS' if h3_pass_corr else '✗ FAIL'} ({n_pass}/10 types)")

    # Save results
    summary = {
        "h1": {
            "sil_train": sil_train_nc,
            "sil_val_original": sil_val_wrong,
            "sil_val_corrected": sil_val_nc,
            "sil_test_original": sil_test_wrong,
            "sil_test_corrected": sil_test_nc,
            "h1_pass_corrected": h1_pass_corr,
        },
        "manifest_changes": {
            "val_labels_changed": val_changed,
            "test_labels_changed": test_changed,
        },
        "h3": {
            "baseline_auroc": baseline,
            "k_optimal_from_val": {str(c): k for c, k in k_optimal_corr.items()},
            "auroc_test_at_kval_opt": {str(c): v for c, v in auroc_final.items()},
            "h3_per_type": {
                str(c): bool(v) if v is not None else None
                for c, v in h3_per_type.items()
            },
            "h3_pass_corrected": h3_pass_corr,
            "n_pass": n_pass,
        },
    }
    (args.output / "results_h1_h3.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRésultats sauvegardés : {args.output / 'results_h1_h3.json'}")


if __name__ == "__main__":
    main()
