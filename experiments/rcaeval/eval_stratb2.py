"""Stratégie B2 — Fine-tuning de la tête de projection sur des paires labellisées RCAEval.

Garde le backbone STGCN frozen, fine-tune uniquement la couche de projection
(siamese head) sur n_few paires contrastives labellisées RCAEval.

Motivation
----------
Si B1 échoue (NMI faible → pas de structure récupérable dans les embeddings
ewat_v3 figés), B2 force la séparation via une supervision faible :
paires positives = même fault_type, paires négatives = fault_type différent.

Protocol
--------
1. Charger encodeur ewat_v3 (FROZEN) + siamese head.
2. Fine-tuner la siamese head avec ContrastiveLoss sur n_few épisodes labellisés.
3. Re-cluster les embeddings fine-tunés (K=auto, cosine+average).
4. Entraîner LR classifiers et évaluer H3 sur le test set RCAEval.
5. Comparer B2 vs B1 vs zero-shot.

Usage
-----
    python -m experiments.rcaeval.eval_stratb2 \\
        --encoder-dir experiments/encoder \\
        --typing-dir experiments/typing \\
        --rcaeval-root data/rcaeval \\
        --output experiments/rcaeval/stratb2 \\
        [--n-few 10 20 40] \\
        [--fine-tune-epochs 10] \\
        [--fine-tune-lr 1e-5] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
from ewat.typing.siamese import ContrastiveLoss, SiameseTyper
from utils.seeding import seed_everything

MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "file:///home/wassimbadraoui/repos/ewat/mlruns",
)


# ---------------------------------------------------------------------------
# Episode loading (RCAEval)
# ---------------------------------------------------------------------------

def _load_rcaeval(
    rcaeval_root: Path,
    encoder_dir: Path,
    typing_dir: Path,
    device: torch.device,
) -> tuple[SiameseTyper, list[dict]]:
    """Load SiameseTyper and all RCAEval episode tensors.

    Returns
    -------
    typer   : SiameseTyper (encoder frozen, head trainable)
    episodes: list of {"signal": Tensor, "adjacency": Tensor, "fault_type": str}
    """
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

    typ_ckpt_path = typing_dir / "checkpoints" / "best_siamese.pt"
    typ_ckpt = torch.load(typ_ckpt_path, map_location="cpu", weights_only=False)
    d_proj = int(typ_ckpt.get("d_proj", 32))
    # Encoder FROZEN, only head is trainable
    typer = SiameseTyper(encoder, d_proj=d_proj, freeze_encoder=True)
    typer.load_state_dict(typ_ckpt["typer_state"])
    typer = typer.to(device)

    # Load scaler (ewat_v3)
    default_scaler = str(encoder_dir / "scaler.pkl")
    scaler_path = Path(enc_ckpt.get("scaler_path", default_scaler))

    split_json = rcaeval_root / "split.json"
    if not split_json.exists():
        split_json = _build_split(rcaeval_root)
    features_root = rcaeval_root / "features"
    if not features_root.exists():
        features_root = rcaeval_root

    dataset = EpisodeDataset(split_json, features_root, split="all")
    if scaler_path.exists():
        dataset.load_scaler(scaler_path)

    episodes = []
    for i in range(len(dataset)):
        item = dataset[i]
        episodes.append({
            "signal": item["signal"],
            "adjacency": item["adjacency"],
            "fault_type": item.get("scenario", item.get("fault_type", "unknown")),
        })

    return typer, episodes


def _build_split(rcaeval_root: Path) -> Path:
    episode_dirs = sorted(d for d in rcaeval_root.iterdir()
                          if d.is_dir() and (d / "signal.npz").exists())
    split_data = {"train": [d.name for d in episode_dirs], "val": [], "test": [], "all": [d.name for d in episode_dirs]}
    p = rcaeval_root / "split_b2.json"
    p.write_text(json.dumps(split_data, indent=2))
    return p


# ---------------------------------------------------------------------------
# Fine-tuning helpers
# ---------------------------------------------------------------------------

def _build_contrastive_pairs(
    episodes: list[dict],
    indices: list[int],
    n_neg_per_anchor: int = 5,
    seed: int = 42,
) -> list[tuple[int, int, bool]]:
    """Build contrastive pairs from a subset of episode indices."""
    rng = random.Random(seed)
    fault_types = [episodes[i]["fault_type"] for i in indices]
    pairs: list[tuple[int, int, bool]] = []

    for pos, idx_i in enumerate(indices):
        ft_i = fault_types[pos]
        pos_candidates = [j for j, idx_j in enumerate(indices) if j != pos and fault_types[j] == ft_i]
        neg_candidates = [j for j, idx_j in enumerate(indices) if fault_types[j] != ft_i]

        if pos_candidates:
            pairs.append((idx_i, indices[rng.choice(pos_candidates)], True))
        for _ in range(min(n_neg_per_anchor, len(neg_candidates))):
            j = rng.choice(neg_candidates)
            pairs.append((idx_i, indices[j], False))

    return pairs


@torch.no_grad()
def _embed_all(typer: SiameseTyper, episodes: list[dict], device: torch.device) -> np.ndarray:
    typer.eval()
    zs = []
    for ep in episodes:
        sig = ep["signal"].unsqueeze(0).to(device)
        adj = ep["adjacency"].unsqueeze(0).to(device)
        z = typer.embed(sig, adj).cpu().numpy()[0]
        zs.append(z)
    return np.stack(zs)


def _fine_tune(
    typer: SiameseTyper,
    episodes: list[dict],
    few_shot_indices: list[int],
    n_epochs: int,
    lr: float,
    margin: float,
    device: torch.device,
    seed: int,
) -> SiameseTyper:
    """Fine-tune the siamese head (encoder remains frozen) on n_few episodes."""
    pairs = _build_contrastive_pairs(episodes, few_shot_indices, n_neg_per_anchor=5, seed=seed)
    loss_fn = ContrastiveLoss(margin=margin)
    # Only optimize projection head parameters (encoder is frozen)
    trainable_params = [p for p in typer.parameters() if p.requires_grad]
    if not trainable_params:
        print("  Warning: no trainable parameters found — is encoder frozen?")
        return typer
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)

    typer.train()
    for epoch in range(1, n_epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        for idx_i, idx_j, is_same in pairs:
            sig_i = episodes[idx_i]["signal"].unsqueeze(0).to(device)
            adj_i = episodes[idx_i]["adjacency"].unsqueeze(0).to(device)
            sig_j = episodes[idx_j]["signal"].unsqueeze(0).to(device)
            adj_j = episodes[idx_j]["adjacency"].unsqueeze(0).to(device)
            z_i = typer.embed(sig_i, adj_i)
            z_j = typer.embed(sig_j, adj_j)
            dist = typer.distance(z_i, z_j)
            loss = loss_fn(dist, torch.tensor([is_same], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == n_epochs:
            print(f"    epoch {epoch:03d}/{n_epochs}  loss={total_loss/max(len(pairs),1):.4f}")

    typer.eval()
    return typer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RCAEval Strategy B2 — head fine-tuning")
    p.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    p.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    p.add_argument("--rcaeval-root", type=Path, default=Path("data/rcaeval"))
    p.add_argument("--output", type=Path, default=Path("experiments/rcaeval/stratb2"))
    p.add_argument("--n-few", type=int, nargs="+", default=[10, 20, 40],
                   help="Number of labeled episodes for few-shot fine-tuning")
    p.add_argument("--fine-tune-epochs", type=int, default=10)
    p.add_argument("--fine-tune-lr", type=float, default=1e-5)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--k-max", type=int, default=30)
    p.add_argument("--n-repeats", type=int, default=3,
                   help="Repeats per n_few (different random subsets)")
    p.add_argument("--seed", type=int, default=42)
    return p


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    typer_orig, episodes = _load_rcaeval(
        args.rcaeval_root, args.encoder_dir, args.typing_dir, device,
    )

    n_ep = len(episodes)
    fault_types_raw = [ep["fault_type"] for ep in episodes]
    unique_faults = sorted(set(fault_types_raw))
    fault2int = {f: i for i, f in enumerate(unique_faults)}
    fault_int = np.array([fault2int[f] for f in fault_types_raw], dtype=int)
    n_fault_types = len(unique_faults)
    print(f"RCAEval: {n_ep} episodes, {n_fault_types} fault types")

    # Fixed test split (stratified, never fine-tuned on)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    trainval_idx, test_idx = next(sss.split(np.arange(n_ep), fault_int))
    z_test_orig = _embed_all(typer_orig, [episodes[i] for i in test_idx], device)
    fault_test = fault_int[test_idx]

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment("ewat_improvements")
        mlflow_run = mlflow.start_run(run_name=f"rcaeval_stratb2_s{args.seed}")
        mlflow.log_params({
            "seed": args.seed, "n_few_list": str(args.n_few),
            "fine_tune_epochs": args.fine_tune_epochs, "fine_tune_lr": args.fine_tune_lr,
        })
    except Exception:
        mlflow_run = None

    all_results: dict[int, list[dict]] = {}

    for n_few in args.n_few:
        print(f"\n--- n_few={n_few} ---")
        results_nfew = []

        for repeat in range(args.n_repeats):
            rep_seed = args.seed + repeat * 1000
            rng = np.random.default_rng(rep_seed)

            # Sample n_few episodes from trainval (stratified attempt)
            few_idx_local = _stratified_sample(fault_int[trainval_idx], n_few, rep_seed)
            few_idx_global = [trainval_idx[i] for i in few_idx_local]
            few_eps = [episodes[i] for i in few_idx_global]

            # Deep-copy typer state for this run
            import copy
            typer = copy.deepcopy(typer_orig)
            typer = typer.to(device)

            print(f"  Repeat {repeat+1}/{args.n_repeats}: fine-tuning on {n_few} episodes …")
            typer = _fine_tune(
                typer, episodes, few_idx_global,
                n_epochs=args.fine_tune_epochs, lr=args.fine_tune_lr,
                margin=args.margin, device=device, seed=rep_seed,
            )

            # Re-cluster all RCAEval with fine-tuned head
            z_all = _embed_all(typer, episodes, device)
            z_trainval = z_all[trainval_idx]
            z_test = z_all[test_idx]

            k_range = range(2, min(args.k_max + 1, len(trainval_idx)))
            cluster_result = cluster_embeddings(
                z_trainval, k_range=k_range, n_gap_refs=5,
                random_state=rep_seed, linkage="average", metric="cosine",
            )
            k_opt = cluster_result.k_optimal
            sil_int = cluster_result.silhouette_scores[k_opt]

            # Nearest centroid for test
            centroids = np.zeros((k_opt, z_trainval.shape[1]), dtype=np.float32)
            for c in range(k_opt):
                mask = cluster_result.labels == c
                if mask.any():
                    centroids[c] = z_trainval[mask].mean(axis=0)

            dists = np.linalg.norm(z_test[:, None, :] - centroids[None, :, :], axis=2)
            y_test_cluster = np.argmin(dists, axis=1)

            sil_test = float(silhouette_score(z_test, y_test_cluster, metric="cosine")) if len(set(y_test_cluster)) >= 2 else -1.0
            nmi_test = float(normalized_mutual_info_score(fault_test, y_test_cluster))
            h1_pass = sil_test >= 0.3

            # Train classifiers
            clf = PrecursorClassifier(n_clusters=k_opt, classifier_type="lr")
            clf.fit(z_trainval, cluster_result.labels)
            auroc_test = clf.auroc_per_type(z_test, y_test_cluster)
            mean_auroc_test = float(np.nanmean(list(auroc_test.values())))
            h3_pass = mean_auroc_test > 0.5

            print(f"    K={k_opt}, sil_test={sil_test:.3f}, NMI={nmi_test:.3f}, "
                  f"AUROC={mean_auroc_test:.3f} ({'H3-PASS' if h3_pass else 'H3-FAIL'})")

            results_nfew.append({
                "n_few": n_few, "repeat": repeat, "seed": rep_seed,
                "k_optimal": k_opt, "sil_internal": sil_int,
                "sil_test": sil_test, "h1_pass": h1_pass,
                "nmi_test": nmi_test, "mean_auroc_test": mean_auroc_test, "h3_pass": h3_pass,
            })

        all_results[n_few] = results_nfew

        mean_sil = float(np.mean([r["sil_test"] for r in results_nfew]))
        mean_auc = float(np.mean([r["mean_auroc_test"] for r in results_nfew]))
        print(f"  n_few={n_few}: mean sil_test={mean_sil:.3f}, mean AUROC={mean_auc:.3f}")

        if mlflow_run is not None:
            try:
                mlflow.log_metrics({
                    f"sil_test_nfew{n_few}": mean_sil,
                    f"auroc_nfew{n_few}": mean_auc,
                }, step=n_few)
            except Exception:
                pass

    # ---- Save results ----
    flat_results = [r for rlist in all_results.values() for r in rlist]
    (args.output / "results.json").write_text(json.dumps({
        "strategy": "B2_finetune_head",
        "n_few_list": args.n_few,
        "n_repeats": args.n_repeats,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_lr": args.fine_tune_lr,
        "results": flat_results,
    }, indent=2))

    lines = [
        "# RCAEval Stratégie B2 — Fine-tuning de la tête siamoise\n",
        "Strategy : fine-tune siamese head (encoder frozen) on n_few labeled RCAEval pairs",
        "",
        f"{'n_few':<8}  {'sil_test (mean)':<18}  {'AUROC (mean)':<14}  {'H1':<6}  {'H3':<6}",
        "-" * 55,
    ]
    for n_few, rlist in sorted(all_results.items()):
        sils = [r["sil_test"] for r in rlist]
        aucs = [r["mean_auroc_test"] for r in rlist]
        h1 = "PASS" if float(np.mean([r["h1_pass"] for r in rlist])) > 0.5 else "FAIL"
        h3 = "PASS" if float(np.mean([r["h3_pass"] for r in rlist])) > 0.5 else "FAIL"
        lines.append(
            f"{n_few:<8}  {np.mean(sils):.3f} ± {np.std(sils):.3f}  "
            f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}  {h1:<6}  {h3:<6}"
        )

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"Report: {args.output / 'results.md'}")

    if mlflow_run is not None:
        try:
            mlflow.end_run()
        except Exception:
            pass

    return {"all_results": all_results}


def _stratified_sample(fault_int: np.ndarray, n: int, seed: int) -> list[int]:
    """Sample at most n indices, approximately stratified by fault type."""
    rng = np.random.default_rng(seed)
    unique = np.unique(fault_int)
    n_per_type = max(1, n // len(unique))
    selected = []
    for u in unique:
        idxs = np.where(fault_int == u)[0]
        take = min(n_per_type, len(idxs))
        selected.extend(rng.choice(idxs, size=take, replace=False).tolist())
    # If under budget, sample randomly to fill
    if len(selected) < n:
        remaining = [i for i in range(len(fault_int)) if i not in set(selected)]
        extra = min(n - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return selected[:n]


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
