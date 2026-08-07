"""Zero-shot transfer evaluation of EWAT on RCAEval RE2-OB.

Applies the trained EWAT encoder + SiameseTyper (ewat_v3) to RCAEval episodes
without retraining. Evaluates:
  H1 — silhouette score of RCAEval embeddings under ewat_v3 centroids
  H3 — precursor AUROC (injection vs baseline steps) using trained classifiers

Zero-shot protocol:
  1. Encode each RCAEval episode with the STGCN encoder trained on ewat_v3
  2. Assign cluster label via nearest-centroid from ewat_v3 centroids
  3. Compute silhouette score on all RCAEval episode embeddings → H1
  4. For each cluster type C_i, apply its trained precursor classifier to
     the pre-injection window embeddings; evaluate AUROC separating
     injection vs baseline timesteps → H3

Usage
-----
    python -m experiments.rcaeval.eval_zeroshot \\
        --features-root data/features/rcaeval \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --precursor-dir experiments/precursor \\
        --output experiments/rcaeval
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, silhouette_score

from ewat.encoder.stgcn import STGCNEncoder
from ewat.precursor.model import PrecursorClassifier
from ewat.typing.siamese import SiameseTyper

STEP_S = 30.0


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_scaler(encoder_dir: Path):
    scaler_path = encoder_dir / "scaler.pkl"
    if not scaler_path.exists():
        return None
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def _instance_normalize(signal: np.ndarray, inject_step: int) -> np.ndarray:
    """Normalize signal per episode: (x - baseline_mean) / (baseline_std + ε).

    Uses the pre-injection steps as 'baseline' so the normalized signal
    measures deviation from normal. This is transferable across clusters
    because relative deviations (×10 latency) are comparable even when
    absolute values differ (50ms vs 200ms baseline).
    """
    baseline = signal[:inject_step] if inject_step > 2 else signal[:max(2, len(signal) // 3)]
    mu = np.nanmean(baseline.reshape(-1, signal.shape[-1]), axis=0)  # (D,)
    sigma = np.nanstd(baseline.reshape(-1, signal.shape[-1]), axis=0)  # (D,)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    out = (signal - mu[None, None, :]) / sigma[None, None, :]
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _fit_scaler_on_rcaeval(ep_dirs: list[Path]):
    """Fit a StandardScaler on all RCAEval signal data (replaces ewat_v3 scaler).

    Exploration benchmark only — not production methodology.
    The canonical scaler is the one fitted on ewat_v3 training data.
    """
    from sklearn.preprocessing import StandardScaler
    chunks = []
    for ep_dir in ep_dirs:
        with np.load(ep_dir / "signal.npz") as z:
            sig = z["signal"]   # (T, N, 17)
        T, N, D = sig.shape
        flat = sig.reshape(-1, D)
        flat = np.nan_to_num(flat, nan=0.0)
        chunks.append(flat)
    X = np.concatenate(chunks, axis=0)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def _load_typer(encoder_dir: Path, typing_dir: Path, device: torch.device) -> SiameseTyper:
    enc_ckpt = torch.load(
        encoder_dir / "checkpoints" / "best_encoder.pt",
        map_location="cpu", weights_only=False,
    )
    encoder = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    encoder.load_state_dict(enc_ckpt["encoder_state"])

    typer_ckpt = torch.load(
        typing_dir / "checkpoints" / "best_siamese.pt",
        map_location="cpu", weights_only=False,
    )
    typer = SiameseTyper(encoder, d_proj=32)
    typer.load_state_dict(typer_ckpt["typer_state"])
    return typer.to(device).eval()


# ── Episode encoding ──────────────────────────────────────────────────────────

def _apply_scaler(signal: np.ndarray, scaler) -> np.ndarray:
    """Apply StandardScaler fitted on ewat_v3 training data."""
    if scaler is None:
        return np.nan_to_num(signal, nan=0.0).astype(np.float32)
    T, N, D = signal.shape
    flat = signal.reshape(-1, D)
    # nan → 0 before scaler (same as EpisodeDataset)
    flat = np.nan_to_num(flat, nan=0.0)
    flat = scaler.transform(flat).astype(np.float32)
    return flat.reshape(T, N, D)


@torch.no_grad()
def _encode_episode(
    typer: SiameseTyper,
    signal: np.ndarray,   # (T, N, 17)
    adjacency: np.ndarray,  # (T, N, N, 3)
    device: torch.device,
    scaler=None,
) -> np.ndarray:
    """Encode entire episode → (d_proj,) embedding."""
    signal = _apply_scaler(signal, scaler)
    adjacency = np.nan_to_num(adjacency, nan=0.0).astype(np.float32)

    sig_t = torch.from_numpy(signal).unsqueeze(0).to(device)       # (1, T, N, 17)
    adj_t = torch.from_numpy(adjacency).unsqueeze(0).to(device)    # (1, T, N, N, 3)

    z = typer.embed(sig_t, adj_t)   # (1, d_proj) — episode-level embedding
    return z.cpu().numpy()[0]       # (d_proj,)


def _nearest_centroid(z: np.ndarray, centroids: np.ndarray) -> int:
    dists = np.linalg.norm(centroids - z[None, :], axis=1)
    return int(np.argmin(dists))


# ── Precursor feature: mean embedding of pre-injection window ─────────────────

@torch.no_grad()
def _precursor_feature(
    typer: SiameseTyper,
    signal: np.ndarray,   # (T, N, 17) — raw (unscaled)
    adjacency: np.ndarray,
    inject_step: int,
    k: int,
    device: torch.device,
    scaler=None,
) -> np.ndarray | None:
    """Embed the window [inject_step-k, inject_step) → precursor feature vector."""
    start = inject_step - k
    if start < 0:
        return None
    sig_win = _apply_scaler(signal[start:inject_step], scaler)
    adj_win = np.nan_to_num(adjacency[start:inject_step], nan=0.0).astype(np.float32)

    sig_t = torch.from_numpy(sig_win).unsqueeze(0).to(device)
    adj_t = torch.from_numpy(adj_win).unsqueeze(0).to(device)
    z = typer.embed(sig_t, adj_t)
    return z.cpu().numpy()[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EWAT zero-shot transfer on RCAEval")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/rcaeval"))
    parser.add_argument("--typing-dir",    type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir",   type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--output",        type=Path, default=Path("experiments/rcaeval"))
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--scaler", choices=["ewat", "rcaeval", "instance", "none"],
                        default="ewat",
                        help="ewat=scaler ewat_v3, rcaeval=scaler réajusté RCAEval, "
                             "instance=normalisation par épisode vs baseline, none=brut")
    parser.add_argument("--features", choices=["all", "M_only"],
                        default="all",
                        help="all=17 features, M_only=7 métriques seulement (T/L zeroed)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model and centroids ──────────────────────────────────────────────
    typer = _load_typer(args.encoder_dir, args.typing_dir, device)
    centroids = np.load(args.typing_dir / "cluster_artifacts" / "centroids.npy")
    n_clusters = centroids.shape[0]
    print(f"Loaded encoder+typer (K={n_clusters} ewat_v3 clusters)")

    # Scaler selection — resolved after ep_dirs is known for rcaeval mode
    ep_dirs = sorted(p for p in args.features_root.iterdir()
                     if p.is_dir() and (p / "signal.npz").exists())
    print(f"Found {len(ep_dirs)} RCAEval episodes in {args.features_root}")

    if args.scaler == "ewat":
        scaler = _load_scaler(args.encoder_dir)
        print(f"Scaler: ewat_v3 (from {args.encoder_dir / 'scaler.pkl'})")
    elif args.scaler == "rcaeval":
        print("Scaler: fitting on RCAEval episodes …")
        scaler = _fit_scaler_on_rcaeval(ep_dirs)
        print("Scaler: fitted on RCAEval (mean/std from 90 épisodes)")
    elif args.scaler == "instance":
        scaler = None  # handled per-episode inside the encoding loop
        print("Scaler: instance (z-score per episode vs pre-injection baseline)")
    else:
        scaler = None
        print("Scaler: none (raw values)")

    precursor_results = json.loads((args.precursor_dir / "results.json").read_text())
    k_optimal: dict[int, int] = {int(k): int(v)
                                  for k, v in precursor_results["k_optimal"].items()}

    # ── Discover episodes (already done above) ────────────────────────────────

    # ── Encode all episodes ───────────────────────────────────────────────────
    embeddings: list[np.ndarray] = []
    cluster_labels: list[int] = []
    fault_types: list[str] = []
    fault_services: list[str] = []
    inject_steps: list[int] = []
    episode_ids: list[str] = []
    all_signals: list[np.ndarray] = []
    all_adjacencies: list[np.ndarray] = []

    print("\n[1] Encoding episodes …")
    for ep_dir in ep_dirs:
        meta = json.loads((ep_dir / "metadata.json").read_text())
        ep_id = meta["episode_id"]

        with np.load(ep_dir / "signal.npz") as z:
            signal = z["signal"]        # (T, N, 17)
        with np.load(ep_dir / "adjacency.npz") as z:
            adjacency = z["adjacency"]  # (T, N, N, 3)

        # Find inject step from metadata (needed before instance normalization)
        inject_t = meta.get("inject_time")
        labels_path = ep_dir / "labels.parquet"
        inject_step = -1
        if inject_t is not None:
            import pandas as pd
            labels_df = pd.read_parquet(labels_path)
            inj_rows = labels_df[labels_df["is_injection"]]
            inject_step = int(inj_rows.index[0]) if len(inj_rows) else -1

        sig_use = signal.copy()
        if args.features == "M_only":
            sig_use[:, :, 7:] = 0.0  # zero out T(t) and L(t), keep M(t)

        if args.scaler == "instance":
            signal_enc = _instance_normalize(sig_use, inject_step if inject_step > 2 else 10)
            z_ep = _encode_episode(typer, signal_enc, adjacency, device, scaler=None)
        else:
            z_ep = _encode_episode(typer, sig_use, adjacency, device, scaler)
        cluster = _nearest_centroid(z_ep, centroids)

        embeddings.append(z_ep)
        cluster_labels.append(cluster)
        fault_types.append(meta.get("fault_type", "unknown"))
        fault_services.append(meta.get("fault_service", "unknown"))
        inject_steps.append(inject_step)
        episode_ids.append(ep_id)
        all_signals.append(signal)
        all_adjacencies.append(adjacency)

    Z = np.stack(embeddings)       # (N_ep, d_proj)
    Y = np.array(cluster_labels)   # (N_ep,)

    # ── H1: silhouette on all RCAEval episodes ────────────────────────────────
    print("\n[2] H1 — Silhouette (nearest-centroid labels) …")
    unique_labels = np.unique(Y)
    if len(unique_labels) < 2:
        print("  Only one cluster assigned — silhouette undefined")
        sil_rcaeval = float("nan")
    else:
        sil_rcaeval = float(silhouette_score(Z, Y))
    h1_pass = not np.isnan(sil_rcaeval) and sil_rcaeval >= 0.3
    print(f"  silhouette(RCAEval, NC) = {sil_rcaeval:.4f}  "
          f"H1: {'✓ PASS' if h1_pass else '✗ FAIL'} (seuil 0.3)")

    # Cluster distribution
    from collections import Counter
    dist = Counter(Y.tolist())
    print(f"  Cluster distribution: {dict(sorted(dist.items()))}")

    # ── H3: precursor AUROC ───────────────────────────────────────────────────
    print("\n[3] H3 — Precursor AUROC (injection vs baseline) …")

    # Pre-load one PrecursorClassifier per k (they're saved per-type but contain all n_clusters)
    clf_per_k: dict[int, PrecursorClassifier] = {}
    for k in args.k_values:
        for c in range(n_clusters):
            clf_path = args.precursor_dir / "checkpoints" / f"classifier_type{c}_k{k}.pkl"
            if clf_path.exists():
                clf_per_k[k] = PrecursorClassifier.load(clf_path)
                break  # one file per k is enough — all contain same trained model

    h3_results: dict[str, dict] = {}

    for k in args.k_values:
        if k not in clf_per_k:
            continue  # no classifier saved at this horizon

        clf = clf_per_k[k]
        pos_feats, neg_feats = [], []

        for i, inject_step in enumerate(inject_steps):
            if inject_step < 0:
                continue
            signal = all_signals[i].copy()
            if args.features == "M_only":
                signal[:, :, 7:] = 0.0
            if args.scaler == "instance":
                signal = _instance_normalize(signal, inject_step if inject_step > 2 else 10)
                _scaler = None
            else:
                _scaler = scaler
            adjacency = all_adjacencies[i]

            # Positive: embedding of k-step window just before injection
            z_pre = _precursor_feature(typer, signal, adjacency, inject_step, k, device, _scaler)
            if z_pre is not None:
                pos_feats.append(z_pre)

            # Negative: embedding of k-step window early in the baseline
            baseline_end = max(k, inject_step // 3)
            z_neg = _precursor_feature(typer, signal, adjacency, baseline_end, k, device, _scaler)
            if z_neg is not None:
                neg_feats.append(z_neg)

        if len(pos_feats) < 2 or len(neg_feats) < 2:
            continue

        X_all = np.array(pos_feats + neg_feats)
        y_inj = np.array([1] * len(pos_feats) + [0] * len(neg_feats))
        proba = clf.predict_proba(X_all)  # (N, n_clusters)

        auroc_per_type: dict[int, float] = {}
        for c in range(n_clusters):
            p_c = proba[:, c]
            if y_inj.sum() < 2 or (len(y_inj) - y_inj.sum()) < 2:
                continue
            try:
                auroc_per_type[c] = float(roc_auc_score(y_inj, p_c))
            except Exception:
                auroc_per_type[c] = float("nan")

        valid = {c: v for c, v in auroc_per_type.items() if not np.isnan(v)}
        mean_auroc = float(np.mean(list(valid.values()))) if valid else float("nan")
        print(f"  k={k:2d}: mean AUROC={mean_auroc:.4f}  ({len(valid)}/{n_clusters} types)  "
              f"[pos={len(pos_feats)}, neg={len(neg_feats)}]")
        h3_results[str(k)] = {"auroc_per_type": auroc_per_type, "mean_auroc": mean_auroc}

    # Best k
    valid_ks = {k: v["mean_auroc"] for k, v in h3_results.items()
                if not np.isnan(v["mean_auroc"])}
    if valid_ks:
        k_best = max(valid_ks, key=valid_ks.get)
        best_auroc = valid_ks[k_best]
        h3_pass = best_auroc > 0.5
        print(f"\n  Best k={k_best}: AUROC={best_auroc:.4f}  "
              f"H3: {'✓ PASS' if h3_pass else '✗ FAIL'} (seuil 0.5)")
    else:
        h3_pass = False
        best_auroc = float("nan")
        k_best = None

    # ── Fault-type cluster mapping ────────────────────────────────────────────
    print("\n[4] Fault type → dominant cluster mapping …")
    from collections import defaultdict
    ft_to_clusters: dict[str, list[int]] = defaultdict(list)
    for ft, cl in zip(fault_types, cluster_labels):
        ft_to_clusters[ft].append(cl)

    ft_purity: dict[str, dict] = {}
    for ft, cls_list in sorted(ft_to_clusters.items()):
        cnt = Counter(cls_list)
        dom = cnt.most_common(1)[0]
        purity = dom[1] / len(cls_list)
        ft_purity[ft] = {"n": len(cls_list), "dominant_cluster": dom[0],
                         "purity": round(purity, 3)}
        print(f"  {ft:10s}: n={len(cls_list)}  dominant=C{dom[0]}  purity={purity:.2f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "dataset": "rcaeval_re2_ob",
        "n_episodes": len(ep_dirs),
        "n_clusters_ewat_v3": n_clusters,
        "H1": {
            "silhouette_rcaeval": sil_rcaeval,
            "h1_pass": h1_pass,
            "threshold": 0.3,
            "cluster_distribution": {str(k): int(v) for k, v in dist.items()},
        },
        "H3": {
            "h3_results_per_k": h3_results,
            "best_k": k_best,
            "best_auroc": best_auroc,
            "h3_pass": h3_pass,
        },
        "fault_type_cluster_mapping": ft_purity,
    }
    out_path = args.output / "zeroshot_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    print(f"\n{'='*60}")
    print("ZERO-SHOT TRANSFER SUMMARY")
    print(f"{'='*60}")
    print(f"  H1 silhouette : {sil_rcaeval:.4f}  {'✓ PASS' if h1_pass else '✗ FAIL'}")
    print(f"  H3 best AUROC : {best_auroc:.4f}  {'✓ PASS' if h3_pass else '✗ FAIL'} (k={k_best})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
