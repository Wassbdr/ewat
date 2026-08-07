"""Stratégie B1 — Re-clustering RCAEval avec encodeur ewat_v3 frozen.

Hypothèse : même si tous les épisodes RCAEval se projettent proches du cluster
C2 d'ewat_v3, il existe une sous-structure dans l'espace latent (CPU vs mémoire
vs réseau). En re-clustant RCAEval indépendamment, on peut récupérer cette
structure sans modifier l'encodeur.

Protocol
--------
1. Charger l'encodeur ewat_v3 (frozen) + scaler ewat_v3.
2. Extraire z_RCAEval ∈ ℝ^{d_proj} pour les 90 épisodes.
3. Sweeper K ∈ {5, 10, 15, 20, 25, 30} — sélection par silhouette interne.
4. Évaluer alignement externe : NMI(clusters, fault_type_RCAEval).
5. Split 70/15/15 stratifié par fault_type → entraîner LR/RF classifiers.
6. Évaluer H3 sur test RCAEval.

Usage
-----
    python -m experiments.rcaeval.eval_stratb1 \\
        --encoder-dir experiments/encoder \\
        --typing-dir experiments/typing \\
        --rcaeval-root data/rcaeval \\
        --output experiments/rcaeval/stratb1 \\
        [--k-min 5] [--k-max 30] \\
        [--classifier lr|rf] \\
        [--linkage average] [--metric cosine] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit

import mlflow
from ewat.encoder.dataset import EpisodeDataset
from ewat.encoder.factory import build_encoder
from ewat.precursor.model import PrecursorClassifier
from ewat.typing.clustering import cluster_embeddings
from ewat.typing.siamese import SiameseTyper
from utils.seeding import seed_everything

MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "file:///home/wassimbadraoui/repos/ewat/mlruns",
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_rcaeval_episodes(
    rcaeval_root: Path,
    encoder_dir: Path,
    typing_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all RCAEval episodes and extract embeddings using ewat_v3 encoder.

    Returns
    -------
    z : (N_ep, d_proj) L2-normalised embeddings from SiameseTyper
    fault_types_int : (N_ep,) integer fault type labels (0..n_types-1)
    fault_type_names : list of fault type name strings
    """
    # Locate split file for RCAEval
    split_json = rcaeval_root / "split.json"
    if not split_json.exists():
        # Fallback: look for episode directories and build a pseudo-split
        split_json = _build_rcaeval_split(rcaeval_root, seed)

    # Load encoder
    enc_ckpt_path = encoder_dir / "checkpoints" / "best_encoder.pt"
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
    arch = enc_ckpt.get("arch", {})
    state = enc_ckpt["encoder_state"]
    use_layer_norm = any("tcn_blocks" in k and "norm" in k for k in state)
    encoder = build_encoder(
        arch.get("architecture", "stgcn"),
        d_feat=int(arch.get("d_feat", 17)),
        n_nodes=int(arch.get("n_nodes", 6)),
        d_hidden=int(arch.get("d_hidden", 64)),
        d_embed=int(arch.get("d_embed", 64)),
        use_layer_norm=use_layer_norm,
    )
    encoder.load_state_dict(state)

    # Load typer
    typ_ckpt_path = typing_dir / "checkpoints" / "best_siamese.pt"
    typ_ckpt = torch.load(typ_ckpt_path, map_location="cpu", weights_only=False)
    d_proj = int(typ_ckpt.get("d_proj", 32))
    typer = SiameseTyper(encoder, d_proj=d_proj)
    typer.load_state_dict(typ_ckpt["typer_state"])
    typer = typer.to(device).eval()

    # Load scaler (ewat_v3 scaler — intentional domain mismatch, see B1 rationale)
    default_scaler = str(encoder_dir / "scaler.pkl")
    scaler_path = Path(enc_ckpt.get("scaler_path", default_scaler))

    # Load all RCAEval episodes
    features_root = rcaeval_root / "features"
    if not features_root.exists():
        features_root = rcaeval_root  # fallback: root contains episode dirs

    dataset = EpisodeDataset(split_json, features_root, split="all")
    if scaler_path.exists():
        dataset.load_scaler(scaler_path)

    embeddings, fault_types_raw = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        sig = item["signal"].unsqueeze(0).to(device)
        adj = item["adjacency"].unsqueeze(0).to(device)
        with torch.no_grad():
            z = typer.embed(sig, adj).cpu().numpy()[0]
        embeddings.append(z)
        fault_types_raw.append(item.get("scenario", item.get("fault_type", "unknown")))

    z = np.stack(embeddings)

    # Map fault type strings → integers
    unique_faults = sorted(set(fault_types_raw))
    fault2int = {f: i for i, f in enumerate(unique_faults)}
    fault_types_int = np.array([fault2int[f] for f in fault_types_raw], dtype=int)

    print(f"RCAEval embeddings: {z.shape}, fault types: {len(unique_faults)}")
    return z, fault_types_int, unique_faults


def _build_rcaeval_split(rcaeval_root: Path, seed: int) -> Path:
    """Build a split.json from RCAEval episode directories (all episodes → split=all)."""
    episode_dirs = sorted(d for d in rcaeval_root.iterdir() if d.is_dir()
                          and (d / "signal.npz").exists())
    split_data = {
        "train": [], "val": [], "test": [],
        "all": [d.name for d in episode_dirs],
    }
    # Assign all to train/val/test for StratifiedShuffleSplit (used later externally)
    split_data["train"] = split_data["all"]
    split_path = rcaeval_root / "split_b1.json"
    split_path.write_text(json.dumps(split_data, indent=2))
    return split_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RCAEval Strategy B1 — re-clustering")
    p.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    p.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    p.add_argument("--rcaeval-root", type=Path, default=Path("data/rcaeval"))
    p.add_argument("--output", type=Path, default=Path("experiments/rcaeval/stratb1"))
    p.add_argument("--k-min", type=int, default=5, help="Min K for clustering search")
    p.add_argument("--k-max", type=int, default=30, help="Max K (exclusive) for clustering search")
    p.add_argument("--classifier", default="lr", choices=["lr", "lr_tuned", "rf"],
                   help="Classifier type for precursors")
    p.add_argument("--linkage", default="average",
                   choices=["ward", "average", "complete"])
    p.add_argument("--metric", default="cosine",
                   choices=["euclidean", "cosine"])
    p.add_argument("--test-size", type=float, default=0.15,
                   help="Test fraction for RCAEval split")
    p.add_argument("--val-size", type=float, default=0.15,
                   help="Val fraction for RCAEval split")
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    # ---- Load embeddings ----
    z, fault_int, fault_names = _load_rcaeval_episodes(
        args.rcaeval_root, args.encoder_dir, args.typing_dir, args.seed, device,
    )
    n_fault_types = len(fault_names)
    n_ep = len(z)
    print(f"Episodes: {n_ep}, fault types: {n_fault_types}")

    # ---- Split RCAEval into train/val/test ----
    # Stratified by fault_type so each type appears in all splits
    rng = np.random.default_rng(args.seed)

    # First: test split
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_val_idx, test_idx = next(sss_test.split(z, fault_int))

    # Then: val split from train_val
    val_frac_of_train_val = args.val_size / (1 - args.test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_train_val, random_state=args.seed + 1
    )
    train_idx, val_idx = next(sss_val.split(z[train_val_idx], fault_int[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    z_train, y_train_fault = z[train_idx], fault_int[train_idx]
    z_val, y_val_fault = z[val_idx], fault_int[val_idx]
    z_test, y_test_fault = z[test_idx], fault_int[test_idx]
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ---- MLflow ----
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment("ewat_improvements")
        mlflow_run = mlflow.start_run(run_name=f"rcaeval_stratb1_s{args.seed}")
        mlflow.log_params({
            "seed": args.seed, "k_min": args.k_min, "k_max": args.k_max,
            "linkage": args.linkage, "metric": args.metric,
            "classifier": args.classifier, "n_ep": n_ep,
        })
    except Exception:
        mlflow_run = None

    # ---- Re-clustering on train RCAEval ----
    linkage = args.linkage
    metric = args.metric
    if linkage == "ward":
        metric = "euclidean"

    k_range = range(max(2, args.k_min), min(args.k_max + 1, len(train_idx)))
    result = cluster_embeddings(
        z_train,
        k_range=k_range,
        n_gap_refs=5,
        random_state=args.seed,
        linkage=linkage,
        metric=metric,
    )
    k_opt = result.k_optimal
    sil_train_internal = result.silhouette_scores[k_opt]
    print(f"Re-clustering: K={k_opt}, sil_internal={sil_train_internal:.3f}")

    # External evaluation: NMI(cluster labels, fault types)
    nmi_train = float(normalized_mutual_info_score(y_train_fault, result.labels))
    print(f"NMI(clusters, fault_types) on train = {nmi_train:.3f}")

    # Assign val/test via nearest centroid (in clustering metric space)
    centroids = np.zeros((k_opt, z_train.shape[1]), dtype=np.float32)
    for c in range(k_opt):
        mask = result.labels == c
        if mask.any():
            centroids[c] = z_train[mask].mean(axis=0)

    def _nearest_centroid(z: np.ndarray) -> np.ndarray:
        # Use Euclidean on unit-sphere (equivalent to cosine for L2-normalized)
        dists = np.linalg.norm(z[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1).astype(int)

    y_val_cluster = _nearest_centroid(z_val)
    y_test_cluster = _nearest_centroid(z_test)

    # Silhouette on val/test (H1 internal criterion)
    sil_metric = metric
    sil_val = float(silhouette_score(z_val, y_val_cluster, metric=sil_metric)) if len(set(y_val_cluster)) >= 2 else -1.0
    sil_test = float(silhouette_score(z_test, y_test_cluster, metric=sil_metric)) if len(set(y_test_cluster)) >= 2 else -1.0
    h1_pass = sil_test >= 0.3
    print(f"H1 sil_test={sil_test:.3f} ({'PASS' if h1_pass else 'FAIL'})")

    # NMI on val/test
    nmi_val = float(normalized_mutual_info_score(y_val_fault, y_val_cluster))
    nmi_test = float(normalized_mutual_info_score(y_test_fault, y_test_cluster))
    print(f"NMI val={nmi_val:.3f}  test={nmi_test:.3f}")

    # ---- Train precursor classifiers ----

    clf = PrecursorClassifier(
        n_clusters=k_opt,
        classifier_type=args.classifier,
    )
    clf.fit(z_train, result.labels)

    auroc_val = clf.auroc_per_type(z_val, y_val_cluster)
    auroc_test = clf.auroc_per_type(z_test, y_test_cluster)

    mean_auroc_val = float(np.nanmean(list(auroc_val.values())))
    mean_auroc_test = float(np.nanmean(list(auroc_test.values())))
    n_valid_types = sum(1 for v in auroc_test.values() if v == v)  # not NaN
    h3_pass = mean_auroc_test > 0.5 and n_valid_types > 0
    print(f"H3 mean AUROC test={mean_auroc_test:.3f} ({n_valid_types}/{k_opt} types, "
          f"{'PASS' if h3_pass else 'FAIL'})")

    # ---- Results ----
    results = {
        "strategy": "B1_recluster",
        "n_episodes": n_ep,
        "n_fault_types_rcaeval": n_fault_types,
        "k_optimal": k_opt,
        "linkage": linkage,
        "metric": metric,
        "classifier": args.classifier,
        "sil_internal_train": sil_train_internal,
        "sil_test": sil_test,
        "h1_pass": h1_pass,
        "nmi_train": nmi_train,
        "nmi_val": nmi_val,
        "nmi_test": nmi_test,
        "auroc_val": auroc_val,
        "auroc_test": auroc_test,
        "mean_auroc_val": mean_auroc_val,
        "mean_auroc_test": mean_auroc_test,
        "n_valid_types": n_valid_types,
        "h3_pass": h3_pass,
    }

    (args.output / "results.json").write_text(json.dumps(results, indent=2, default=str))

    lines = [
        "# RCAEval Stratégie B1 — Re-clustering\n",
        "Strategy   : re-cluster with frozen ewat_v3 encoder",
        f"Clustering : {linkage} + {metric}, K={k_opt}",
        f"Classifier : {args.classifier}",
        "",
        "## H1 — Silhouette (RCAEval test set)",
        f"sil_test = {sil_test:.3f}  → {'PASS' if h1_pass else 'FAIL'} (threshold 0.3)",
        f"NMI(clusters, fault_types) = {nmi_test:.3f}",
        "",
        "## H3 — AUROC précurseurs (RCAEval test set)",
        f"Mean AUROC = {mean_auroc_test:.3f}  ({n_valid_types}/{k_opt} types évaluables)",
        f"H3 : {'PASS' if h3_pass else 'FAIL'} (AUROC > 0.5)",
        "",
        "## AUROC par cluster (test)",
    ] + [
        f"  C{c}: {auroc_test.get(c, float('nan')):.3f}" for c in range(k_opt)
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"Report: {args.output / 'results.md'}")

    if mlflow_run is not None:
        try:
            mlflow.log_metrics({
                "sil_test": sil_test,
                "h1_pass": float(h1_pass),
                "nmi_test": nmi_test,
                "mean_auroc_test": mean_auroc_test,
                "h3_pass": float(h3_pass),
                "k_optimal": float(k_opt),
            })
            mlflow.end_run()
        except Exception:
            pass

    return results


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
