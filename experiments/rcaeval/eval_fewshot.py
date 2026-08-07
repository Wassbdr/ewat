"""Few-shot transfer evaluation of EWAT on RCAEval RE2-OB.

Stratégie A — Scaler adaptation : re-fit du StandardScaler sur N_few épisodes
RCAEval, encodeur STGCN et classifieurs LR gardés fixes (ewat_v3).

Protocole
---------
Pour chaque n_few ∈ {1, 3, 5, 10, 20, 40} et n_repeats répétitions :
  1. Sélectionner n_few épisodes aléatoires comme ensemble d'adaptation.
  2. Fitter un StandardScaler sur ces n_few épisodes.
  3. Encoder les épisodes restants (test) avec le nouveau scaler.
  4. H1 : silhouette score (nearest-centroid ewat_v3).
  5. H3 : AUROC (injection vs baseline) avec les classifieurs ewat_v3.
  6. Comparer avec zero-shot (scaler ewat_v3) et instance norm.

Question scientifique
---------------------
Quel est le n_few minimal pour que H3 AUROC > 0.7 sur RCAEval ?

Usage
-----
    python -m experiments.rcaeval.eval_fewshot \\
        --features-root data/features/rcaeval \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --precursor-dir experiments/precursor \\
        --n-few 1 3 5 10 20 40 \\
        --n-repeats 5 \\
        --output experiments/rcaeval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ewat.precursor.model import PrecursorClassifier

# ── Model loading ──────────────────────────────────────────────────────────────
from ewat.typing.loader import load_typer as _load_typer_base  # noqa: E402
from ewat.typing.siamese import SiameseTyper


def _load_typer(encoder_dir: Path, typing_dir: Path, device: torch.device) -> SiameseTyper:
    """Wrapper preserving the (encoder_dir, typing_dir) argument order used in this script."""
    return _load_typer_base(typing_dir=typing_dir, encoder_dir=encoder_dir, device=device)


def _load_classifiers(
    precursor_dir: Path, n_clusters: int, k_values: list[int]
) -> dict[int, PrecursorClassifier]:
    """Load one PrecursorClassifier per k (first available type for each k)."""
    clf_per_k: dict[int, PrecursorClassifier] = {}
    for k in k_values:
        for c in range(n_clusters):
            p = precursor_dir / "checkpoints" / f"classifier_type{c}_k{k}.pkl"
            if p.exists():
                clf_per_k[k] = PrecursorClassifier.load(p)
                break
    return clf_per_k


# ── Signal normalization ───────────────────────────────────────────────────────

def _fit_scaler_on_subset(ep_dirs: list[Path]) -> StandardScaler:
    chunks = []
    for ep_dir in ep_dirs:
        sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        T, N, D = sig.shape
        flat = np.nan_to_num(sig.reshape(-1, D), nan=0.0)
        chunks.append(flat)
    X = np.concatenate(chunks, axis=0)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def _apply_scaler(signal: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    T, N, D = signal.shape
    flat = np.nan_to_num(signal.reshape(-1, D), nan=0.0)
    flat = scaler.transform(flat).astype(np.float32)
    return flat.reshape(T, N, D)


# ── Episode encoding ───────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_episode(
    typer: SiameseTyper,
    signal: np.ndarray,     # (T, N, 17) scaled
    adjacency: np.ndarray,  # (T, N, N, 3)
    device: torch.device,
) -> np.ndarray:
    adj = np.nan_to_num(adjacency, nan=0.0).astype(np.float32)
    sig_t = torch.from_numpy(signal).unsqueeze(0).to(device)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(device)
    z = typer.embed(sig_t, adj_t)
    return z.cpu().numpy()[0]


@torch.no_grad()
def _precursor_embed(
    typer: SiameseTyper,
    signal: np.ndarray,    # (T, N, 17) scaled
    adjacency: np.ndarray,
    start: int,
    end: int,
    device: torch.device,
) -> np.ndarray | None:
    if start < 0 or end - start < 1:
        return None
    adj = np.nan_to_num(adjacency[start:end], nan=0.0).astype(np.float32)
    sig_t = torch.from_numpy(signal[start:end]).unsqueeze(0).to(device)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(device)
    z = typer.embed(sig_t, adj_t)
    return z.cpu().numpy()[0]


# ── Core evaluation ────────────────────────────────────────────────────────────

def _evaluate_transfer(
    typer: SiameseTyper,
    centroids: np.ndarray,
    clf_per_k: dict[int, PrecursorClassifier],
    k_values: list[int],
    test_dirs: list[Path],
    test_meta: list[dict],
    scaler: StandardScaler,
    device: torch.device,
) -> dict:
    """Encode test episodes and evaluate H1 + H3 with the given scaler."""
    n_clusters = centroids.shape[0]
    embeddings, cluster_labels = [], []

    signals_scaled = []
    adjacencies = []
    inject_steps = []

    for ep_dir, meta in zip(test_dirs, test_meta):
        sig_raw = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adj = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
        sig = _apply_scaler(sig_raw, scaler)

        inject_step = meta.get("inject_step", -1)
        z_ep = _encode_episode(typer, sig, adj, device)
        dists = np.linalg.norm(centroids - z_ep[None, :], axis=1)
        cluster = int(np.argmin(dists))

        embeddings.append(z_ep)
        cluster_labels.append(cluster)
        signals_scaled.append(sig)
        adjacencies.append(adj)
        inject_steps.append(inject_step)

    Z = np.stack(embeddings)
    Y = np.array(cluster_labels)

    # H1 — silhouette
    unique_y = np.unique(Y)
    h1_sil = float(silhouette_score(Z, Y)) if len(unique_y) >= 2 else float("nan")

    # H3 — precursor AUROC (injection vs baseline, binary)
    h3_by_k: dict[int, float] = {}
    for k in k_values:
        if k not in clf_per_k:
            continue
        clf = clf_per_k[k]
        pos_feats, neg_feats = [], []

        for sig, adj, inject_step in zip(signals_scaled, adjacencies, inject_steps):
            if inject_step < 0:
                continue
            # Positive: k-step window just before injection
            z_pre = _precursor_embed(typer, sig, adj, inject_step - k, inject_step, device)
            if z_pre is not None:
                pos_feats.append(z_pre)
            # Negative: k-step window early in baseline
            baseline_end = max(k, inject_step // 3)
            z_neg = _precursor_embed(typer, sig, adj, baseline_end - k, baseline_end, device)
            if z_neg is not None:
                neg_feats.append(z_neg)

        if len(pos_feats) < 2 or len(neg_feats) < 2:
            continue

        X_all = np.array(pos_feats + neg_feats)
        y_bin = np.array([1] * len(pos_feats) + [0] * len(neg_feats))
        proba = clf.predict_proba(X_all)   # (N, n_clusters)

        aurocs = []
        for c in range(n_clusters):
            try:
                aurocs.append(roc_auc_score(y_bin, proba[:, c]))
            except Exception:
                pass
        h3_by_k[k] = float(np.mean(aurocs)) if aurocs else float("nan")

    valid_h3 = {k: v for k, v in h3_by_k.items() if not np.isnan(v)}
    best_h3 = float(max(valid_h3.values())) if valid_h3 else float("nan")

    return {
        "h1_silhouette": round(h1_sil, 4) if not np.isnan(h1_sil) else None,
        "h3_best_auroc": round(best_h3, 4) if not np.isnan(best_h3) else None,
        "h3_by_k": {str(k): round(v, 4) for k, v in h3_by_k.items()},
        "n_test": len(test_dirs),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot transfer on RCAEval (scaler adaptation)")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/rcaeval"))
    parser.add_argument("--typing-dir",    type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir",   type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--output",        type=Path, default=Path("experiments/rcaeval"))
    parser.add_argument("--n-few",   type=int, nargs="+", default=[1, 3, 5, 10, 20, 40])
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--k-values",  type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    typer = _load_typer(args.encoder_dir, args.typing_dir, device)
    centroids = np.load(args.typing_dir / "cluster_artifacts" / "centroids.npy")
    n_clusters = centroids.shape[0]
    print(f"Encoder+typer loaded (K={n_clusters})")

    precursor_results = json.loads((args.precursor_dir / "results.json").read_text())
    clf_per_k = _load_classifiers(args.precursor_dir, n_clusters, args.k_values)
    print(f"Classifiers loaded for k={sorted(clf_per_k.keys())}")

    # Discover episodes
    ep_dirs = sorted(p for p in args.features_root.iterdir()
                     if p.is_dir() and (p / "signal.npz").exists())
    all_meta: list[dict] = []
    for ep_dir in ep_dirs:
        meta = json.loads((ep_dir / "metadata.json").read_text())
        # Resolve inject_step from labels
        labels_df = pd.read_parquet(ep_dir / "labels.parquet")
        inj_rows = labels_df[labels_df.get("is_injection", pd.Series([False] * len(labels_df)))]
        if len(inj_rows) > 0:
            meta["inject_step"] = int(inj_rows.index[0])
        else:
            meta["inject_step"] = -1
        all_meta.append(meta)
    n_total = len(ep_dirs)
    print(f"Found {n_total} RCAEval episodes")

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    rng_master = np.random.default_rng(args.seed)

    for n_few in args.n_few:
        if n_few >= n_total:
            print(f"n_few={n_few} ≥ n_total={n_total}, skipping")
            continue

        h1_reps, h3_reps = [], []
        seeds = rng_master.integers(0, 10000, size=args.n_repeats)

        print(f"\nn_few={n_few} …")
        for rep_seed in seeds:
            rng = np.random.default_rng(rep_seed)
            idxs = np.arange(n_total)
            adapt_idxs = rng.choice(idxs, size=n_few, replace=False)
            test_idxs = np.setdiff1d(idxs, adapt_idxs)

            adapt_dirs = [ep_dirs[i] for i in adapt_idxs]
            test_dirs = [ep_dirs[i] for i in test_idxs]
            test_meta = [all_meta[i] for i in test_idxs]

            scaler = _fit_scaler_on_subset(adapt_dirs)
            res = _evaluate_transfer(
                typer, centroids, clf_per_k, args.k_values,
                test_dirs, test_meta, scaler, device,
            )
            h1_reps.append(res["h1_silhouette"] or float("nan"))
            h3_reps.append(res["h3_best_auroc"] or float("nan"))

        h1_arr = np.array([v for v in h1_reps if not np.isnan(v)])
        h3_arr = np.array([v for v in h3_reps if not np.isnan(v)])

        row = {
            "n_few": n_few,
            "h1_mean": round(float(np.mean(h1_arr)), 4) if len(h1_arr) > 0 else None,
            "h1_std": round(float(np.std(h1_arr)), 4) if len(h1_arr) > 1 else None,
            "h3_mean": round(float(np.mean(h3_arr)), 4) if len(h3_arr) > 0 else None,
            "h3_std": round(float(np.std(h3_arr)), 4) if len(h3_arr) > 1 else None,
            "h1_pass": bool(np.mean(h1_arr) >= 0.3) if len(h1_arr) > 0 else False,
            "h3_pass": bool(np.mean(h3_arr) >= 0.7) if len(h3_arr) > 0 else False,
            "n_repeats": int(len(h1_arr)),
        }
        all_results.append(row)
        print(f"  H1={row['h1_mean']:.4f}±{row['h1_std']:.4f}  "
              f"H3={row['h3_mean']:.4f}±{row['h3_std']:.4f}  "
              f"H1={'✓' if row['h1_pass'] else '✗'}  H3={'✓' if row['h3_pass'] else '✗'}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(args.output / "fewshot_results.csv", index=False)

    # Write report
    lines = [
        "# Few-shot transfer RCAEval — Stratégie A (scaler adaptation)\n",
        "Scaler StandardScaler re-fité sur N_few épisodes RCAEval.",
        "Encodeur STGCN et classifieurs LR conservés (ewat_v3 checkpoint).\n",
        f"n_repeats={args.n_repeats}  |  k_values={args.k_values}\n",
        "---",
        "",
        "## Courbe d'apprentissage (H1 silhouette + H3 AUROC vs n_few)",
        "",
        f"{'n_few':>6}  {'H1 mean':>9}  {'H1 std':>7}  {'H1 pass':>8}  "
        f"{'H3 mean':>9}  {'H3 std':>7}  {'H3 pass':>8}",
        "-" * 65,
    ]
    for r in all_results:
        h1_str = f"{r['h1_mean']:.4f}" if r["h1_mean"] is not None else "  N/A  "
        h1s_str = f"{r['h1_std']:.4f}" if r["h1_std"] is not None else "  N/A "
        h3_str = f"{r['h3_mean']:.4f}" if r["h3_mean"] is not None else "  N/A  "
        h3s_str = f"{r['h3_std']:.4f}" if r["h3_std"] is not None else "  N/A "
        lines.append(
            f"{r['n_few']:>6}  {h1_str:>9}  {h1s_str:>7}  "
            f"{'✓' if r['h1_pass'] else '✗':>8}  {h3_str:>9}  {h3s_str:>7}  "
            f"{'✓' if r['h3_pass'] else '✗':>8}"
        )

    # Find minimal n_few for H3 ≥ 0.7
    passing_h3 = [r["n_few"] for r in all_results if r["h3_pass"]]
    if passing_h3:
        lines += [
            "",
            f"**n_few minimal pour H3 AUROC ≥ 0.7 : n_few = {min(passing_h3)}**",
        ]
    else:
        lines += [
            "",
            "**H3 AUROC < 0.7 pour tous les n_few testés** — Stratégie A insuffisante.",
            "Envisager la Stratégie B (fine-tuning de l'encodeur).",
        ]

    lines += [
        "",
        "---",
        "",
        "## Références",
        "",
        "- Zero-shot (scaler ewat_v3) : H1≈0.24, H3≈0.50 (résultats de eval_zeroshot.py)",
        "- Zero-shot (instance norm + M_only) : H1=0.684, H3≈0.50",
        "- Cible : H3 AUROC > 0.7 (discrimination de types, au-delà de la détection générique)",
    ]

    (args.output / "fewshot_results.md").write_text("\n".join(lines))
    print(f"\nOutputs: {args.output}/fewshot_results.md + fewshot_results.csv")

    # Minimal n_few summary
    if passing_h3:
        print(f"\nn_few minimal pour H3 ≥ 0.7 : n_few = {min(passing_h3)}")
    else:
        print("\nH3 < 0.7 pour tous n_few — Stratégie A insuffisante.")


if __name__ == "__main__":
    main()
