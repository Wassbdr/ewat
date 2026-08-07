"""Baselines pour H3 — valeur ajoutée du STGCN.

Trois conditions de référence comparées au EWAT complet :

  B0 — Aléatoire   : AUROC = 0.5 pour chaque type (référence théorique)
  B1 — Features brutes : LR one-vs-rest sur S(t) aplati (N×17 → 102-dim),
                          sans encodeur STGCN — même protocole k ∈ k_values, k* sur val
  B2 — K-means brut  : k-means (K=n_clusters) sur S(t) aplati → labels → LR précurseur;
                        test si la structure siamoise est nécessaire vs. clustering naïf

Usage
-----
    python -m experiments.baselines.precursor_baselines \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        --output experiments/baselines \\
        [--k-values 2 4 6 8 10 12] [--n-bootstrap 1000]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ewat.precursor.dataset import PrecursorDataset
from ewat.utils.bootstrap import bootstrap_auroc_ci
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Signal flattening helpers
# ---------------------------------------------------------------------------

def _flatten_signal(ds: PrecursorDataset) -> tuple[np.ndarray, np.ndarray]:
    """Load all episodes and return (X_flat, y) where X_flat = mean over time of S(t).

    S(t) ∈ ℝ^{k×N×17} → mean over k → ℝ^{N×17} → flatten → ℝ^{N*17}.
    Using mean-over-time to reduce temporal dimension without an encoder.
    """
    x_data, y = [], []
    for idx in range(len(ds)):
        item = ds[idx]
        # signal: (k, N, 17) — mean over time axis → (N, 17) → flatten
        sig = item["signal"].numpy()           # (k, N, 17)
        feat = sig.mean(axis=0).ravel()        # (N*17,)
        x_data.append(feat)
        y.append(item["cluster"])
    return np.stack(x_data), np.array(y, dtype=int)


# ---------------------------------------------------------------------------
# One-vs-rest LR for a single condition
# ---------------------------------------------------------------------------

def _auroc_one_vs_rest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    n_clusters: int,
    reg_c: float = 1.0,
    max_iter: int = 500,
) -> dict[int, float]:
    """Fit one-vs-rest LR per cluster on x_train, evaluate AUROC on x_eval."""
    auroc: dict[int, float] = {}
    for c in range(n_clusters):
        y_tr = (y_train == c).astype(int)
        y_ev = (y_eval == c).astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            auroc[c] = float("nan")
            continue
        if y_ev.sum() < 2 or (len(y_ev) - y_ev.sum()) < 2:
            auroc[c] = float("nan")
            continue
        clf = LogisticRegression(C=reg_c, max_iter=max_iter, solver="lbfgs")
        clf.fit(x_train, y_tr)
        score = clf.predict_proba(x_eval)[:, 1]
        auroc[c] = float(roc_auc_score(y_ev, score))
    return auroc


def _auroc_raw_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    n_clusters: int,
    reg_c: float = 1.0,
    max_iter: int = 500,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return raw (y_true, y_score) per cluster for bootstrap."""
    raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for c in range(n_clusters):
        y_tr = (y_train == c).astype(int)
        y_ev = (y_eval == c).astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        if y_ev.sum() < 2 or (len(y_ev) - y_ev.sum()) < 2:
            continue
        clf = LogisticRegression(C=reg_c, max_iter=max_iter, solver="lbfgs")
        clf.fit(x_train, y_tr)
        score = clf.predict_proba(x_eval)[:, 1]
        raw[c] = (y_ev, score)
    return raw


def _find_k_star(auroc_table_val: dict[int, dict[int, float]], n_clusters: int) -> dict[int, int]:
    k_values = sorted(auroc_table_val.keys())
    k_star: dict[int, int] = {}
    for c in range(n_clusters):
        best_k, best_auc = k_values[0], -1.0
        for k in k_values:
            auc = auroc_table_val[k].get(c, float("nan"))
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_k = k
        k_star[c] = best_k
    return k_star


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Precursor baselines (B0/B1/B2)")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for AUROC CIs (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1
    print(f"Clusters: {n_clusters}  |  k values: {args.k_values}")

    scaler_path = args.typing_dir.parent / "encoder" / "scaler.pkl"

    # -----------------------------------------------------------------------
    # B0 — Aléatoire (constant 0.5 for all types and all k)
    # -----------------------------------------------------------------------
    print("\n=== B0 — Aléatoire (AUROC = 0.5) ===")
    b0_auroc = {c: 0.5 for c in range(n_clusters)}
    print("  AUROC moyen = 0.500 (par définition)")

    # -----------------------------------------------------------------------
    # B1 — Features brutes (no STGCN)
    # -----------------------------------------------------------------------
    print("\n=== B1 — Features brutes (LR sur S(t) aplati, sans STGCN) ===")
    b1_auroc_val: dict[int, dict[int, float]] = {}
    b1_auroc_test: dict[int, dict[int, float]] = {}

    for k in args.k_values:
        print(f"  k={k} …", end=" ", flush=True)
        ds_train = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="train")
        ds_val = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="val")
        ds_test = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="test")
        if scaler_path.exists():
            ds_train.load_scaler(scaler_path)
            ds_val.load_scaler(scaler_path)
            ds_test.load_scaler(scaler_path)

        x_train, y_train = _flatten_signal(ds_train)
        x_val, y_val = _flatten_signal(ds_val)
        x_test, y_test = _flatten_signal(ds_test)

        b1_auroc_val[k] = _auroc_one_vs_rest(
            x_train, y_train, x_val, y_val, n_clusters, args.reg_c, args.max_iter
        )
        b1_auroc_test[k] = _auroc_one_vs_rest(
            x_train, y_train, x_test, y_test, n_clusters, args.reg_c, args.max_iter
        )
        mean_val = float(np.nanmean(list(b1_auroc_val[k].values())))
        mean_test = float(np.nanmean(list(b1_auroc_test[k].values())))
        print(f"AUROC mean val={mean_val:.3f}  test={mean_test:.3f}")

    b1_k_star = _find_k_star(b1_auroc_val, n_clusters)
    b1_auroc_final = {
        c: b1_auroc_test[b1_k_star[c]].get(c, float("nan"))
        for c in range(n_clusters)
    }
    b1_mean = float(np.nanmean(list(b1_auroc_final.values())))
    print(f"  B1 AUROC@k* test moyen = {b1_mean:.3f}")

    # Bootstrap CIs for B1
    b1_ci: dict[int, dict] = {}
    if args.n_bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        for c in range(n_clusters):
            k_opt = b1_k_star[c]
            ds_train = PrecursorDataset(
                cluster_manifest, args.features_root, k=k_opt, split="train"
            )
            ds_test = PrecursorDataset(
                cluster_manifest, args.features_root, k=k_opt, split="test"
            )
            if scaler_path.exists():
                ds_train.load_scaler(scaler_path)
                ds_test.load_scaler(scaler_path)
            x_train, y_train = _flatten_signal(ds_train)
            x_test, y_test = _flatten_signal(ds_test)
            raw = _auroc_raw_scores(x_train, y_train, x_test, y_test, n_clusters,
                                    args.reg_c, args.max_iter)
            if c in raw:
                y_true, y_score = raw[c]
                ci = bootstrap_auroc_ci(y_true, y_score, n=args.n_bootstrap, rng=rng)
                b1_ci[c] = ci.as_dict()
            else:
                b1_ci[c] = {"estimate": float("nan"), "ci_lo": float("nan"),
                            "ci_hi": float("nan"), "alpha": 0.05,
                            "n_bootstrap": args.n_bootstrap}

    # -----------------------------------------------------------------------
    # B2 — K-means brut + LR
    # -----------------------------------------------------------------------
    print("\n=== B2 — K-means brut (K-means sur S(t), puis LR précurseur) ===")

    # Fit k-means on train at each k, remap labels to match train cluster IDs best
    b2_auroc_val: dict[int, dict[int, float]] = {}
    b2_auroc_test: dict[int, dict[int, float]] = {}

    for k in args.k_values:
        print(f"  k={k} …", end=" ", flush=True)
        ds_train = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="train")
        ds_val = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="val")
        ds_test = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="test")
        if scaler_path.exists():
            ds_train.load_scaler(scaler_path)
            ds_val.load_scaler(scaler_path)
            ds_test.load_scaler(scaler_path)

        x_train, y_train_gt = _flatten_signal(ds_train)
        x_val, y_val_gt = _flatten_signal(ds_val)
        x_test, y_test_gt = _flatten_signal(ds_test)

        # Fit k-means — use EWAT cluster labels as ground truth for the LR
        # (we don't relabel; instead, k-means learns its own clusters and we train
        # LR to predict EWAT cluster labels from k-means centroids, reproducing the
        # same task as EWAT but with a simpler representation)
        km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
        km.fit(x_train)
        x_train_km = km.transform(x_train)   # distances to centroids → (N, K)
        x_val_km = km.transform(x_val)
        x_test_km = km.transform(x_test)

        b2_auroc_val[k] = _auroc_one_vs_rest(
            x_train_km, y_train_gt, x_val_km, y_val_gt, n_clusters, args.reg_c, args.max_iter
        )
        b2_auroc_test[k] = _auroc_one_vs_rest(
            x_train_km, y_train_gt, x_test_km, y_test_gt, n_clusters, args.reg_c, args.max_iter
        )
        mean_val = float(np.nanmean(list(b2_auroc_val[k].values())))
        mean_test = float(np.nanmean(list(b2_auroc_test[k].values())))
        print(f"AUROC mean val={mean_val:.3f}  test={mean_test:.3f}")

    b2_k_star = _find_k_star(b2_auroc_val, n_clusters)
    b2_auroc_final = {
        c: b2_auroc_test[b2_k_star[c]].get(c, float("nan"))
        for c in range(n_clusters)
    }
    b2_mean = float(np.nanmean(list(b2_auroc_final.values())))
    print(f"  B2 AUROC@k* test moyen = {b2_mean:.3f}")

    # Bootstrap CIs for B2
    b2_ci: dict[int, dict] = {}
    if args.n_bootstrap > 0:
        rng = np.random.default_rng(args.seed + 1)
        for c in range(n_clusters):
            k_opt = b2_k_star[c]
            ds_train = PrecursorDataset(
                cluster_manifest, args.features_root, k=k_opt, split="train"
            )
            ds_test = PrecursorDataset(
                cluster_manifest, args.features_root, k=k_opt, split="test"
            )
            if scaler_path.exists():
                ds_train.load_scaler(scaler_path)
                ds_test.load_scaler(scaler_path)
            x_train, y_train_gt = _flatten_signal(ds_train)
            x_test, y_test_gt = _flatten_signal(ds_test)
            km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
            km.fit(x_train)
            x_train_km = km.transform(x_train)
            x_test_km = km.transform(x_test)
            raw = _auroc_raw_scores(x_train_km, y_train_gt, x_test_km, y_test_gt,
                                    n_clusters, args.reg_c, args.max_iter)
            if c in raw:
                y_true, y_score = raw[c]
                ci = bootstrap_auroc_ci(y_true, y_score, n=args.n_bootstrap, rng=rng)
                b2_ci[c] = ci.as_dict()
            else:
                b2_ci[c] = {"estimate": float("nan"), "ci_lo": float("nan"),
                            "ci_hi": float("nan"), "alpha": 0.05,
                            "n_bootstrap": args.n_bootstrap}

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    summary = {
        "n_clusters": n_clusters,
        "k_values": args.k_values,
        "n_bootstrap": args.n_bootstrap,
        "b0": {"auroc_per_type": b0_auroc, "mean_auroc": 0.5},
        "b1_raw_features": {
            "auroc_val_per_k": {str(k): v for k, v in b1_auroc_val.items()},
            "auroc_test_per_k": {str(k): v for k, v in b1_auroc_test.items()},
            "k_star": b1_k_star,
            "auroc_test_at_kstar": b1_auroc_final,
            "mean_auroc": b1_mean,
            "auroc_ci_test": b1_ci,
        },
        "b2_kmeans": {
            "auroc_val_per_k": {str(k): v for k, v in b2_auroc_val.items()},
            "auroc_test_per_k": {str(k): v for k, v in b2_auroc_test.items()},
            "k_star": b2_k_star,
            "auroc_test_at_kstar": b2_auroc_final,
            "mean_auroc": b2_mean,
            "auroc_ci_test": b2_ci,
        },
    }
    (args.output / "precursor_baselines.json").write_text(json.dumps(summary, indent=2))

    # Report
    lines = [
        "# Baselines précurseurs — H3\n",
        "Comparaison EWAT (STGCN) vs. baselines simples.\n",
        "AUROC > 0.5 = meilleur que l'aléatoire (B0).\n",
        f"{'Type':<8}  {'B0':>6}  {'B1 (brut)':>12}  {'B2 (k-means)':>14}",
        "-" * 50,
    ]
    for c in range(n_clusters):
        b1_v = b1_auroc_final.get(c, float("nan"))
        b2_v = b2_auroc_final.get(c, float("nan"))
        b1_s = f"{b1_v:.3f}" if not np.isnan(b1_v) else "  NaN"
        b2_s = f"{b2_v:.3f}" if not np.isnan(b2_v) else "  NaN"
        b1_ci_str = ""
        if b1_ci and c in b1_ci and not np.isnan(b1_ci[c].get("ci_lo", float("nan"))):
            b1_ci_str = f" [{b1_ci[c]['ci_lo']:.3f},{b1_ci[c]['ci_hi']:.3f}]"
        lines.append(f"C{c:<7}  {'0.500':>6}  {b1_s:>6}{b1_ci_str:<16}  {b2_s:>6}")
    lines += [
        "-" * 50,
        f"{'Moyen':<8}  {'0.500':>6}  {b1_mean:>12.3f}  {b2_mean:>14.3f}",
        "",
        "**Lecture** : B1 = LR sur features brutes (sans encodeur).",
        "B2 = LR sur distances aux centres k-means (sans typage siamois).",
        "Si EWAT > B1 et EWAT > B2 → l'encodeur STGCN et le typage siamois",
        "apportent une représentation significativement plus discriminante.",
    ]
    (args.output / "precursor_baselines.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'precursor_baselines.md'}")


if __name__ == "__main__":
    main()
