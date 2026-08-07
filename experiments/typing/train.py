"""Siamese typing training — fine-tunes STGCNEncoder with contrastive loss.

Pipeline
--------
1. Load pre-trained encoder checkpoint (from experiments/encoder/train.py).
2. Wrap with SiameseTyper (ProjectionHead on top).
3. Train with EpisodePairSampler + ContrastiveLoss.
4. Embed all train episodes → AgglomerativeClustering → K types.
5. SHAP per cluster → JSON fiches.
6. Evaluate on val/test: silhouette score (H1 criterion: ≥0.3).

Usage
-----
    python -m experiments.typing.train \\
        --dataset data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --encoder-checkpoint experiments/encoder/checkpoints/best_encoder.pt \\
        --output experiments/typing \\
        [--freeze-encoder] [--epochs 50] [--d-proj 32] [--margin 1.0]
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

import mlflow
from ewat.encoder.dataset import EpisodeDataset
from ewat.typing.clustering import cluster_embeddings
from ewat.typing.pairs import EpisodePairSampler
from ewat.typing.saliency_explainer import (
    compute_cluster_saliency,
    write_cluster_fiches,
)
from ewat.typing.siamese import ContrastiveLoss, SiameseTyper
from ewat.utils.bootstrap import bootstrap_silhouette_ci
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Pair DataLoader
# ---------------------------------------------------------------------------

class _PairDataset(torch.utils.data.Dataset):
    """Dataset of (idx_i, idx_j, is_same) triples."""

    def __init__(self, pairs: list[tuple[int, int, bool]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, bool]:
        return self.pairs[idx]


def _load_all_episodes(dataset: EpisodeDataset, device: torch.device) -> list[dict]:
    """Pre-load all episodes into memory (signal + adjacency tensors)."""
    items = []
    for i in range(len(dataset)):
        item = dataset[i]
        items.append({
            "signal": item["signal"].to(device),       # (T, N, 17)
            "adjacency": item["adjacency"].to(device),  # (T, N, N, 3)
            "scenario": item["scenario"],
            "episode_id": item["episode_id"],
        })
    return items


def _collate_pairs(batch: list[tuple[int, int, bool]], episodes: list[dict]) -> dict:
    """Build a batch of episode pairs from index triples."""
    sigs_i, adjs_i, sigs_j, adjs_j, same_labels = [], [], [], [], []
    max_t = max(
        max(episodes[i]["signal"].shape[0], episodes[j]["signal"].shape[0])
        for i, j, _ in batch
    )
    for i, j, same in batch:
        def _pad(t: torch.Tensor) -> torch.Tensor:
            pad = max_t - t.shape[0]
            return torch.cat([t, torch.zeros(pad, *t.shape[1:], device=t.device)]) if pad else t

        sigs_i.append(_pad(episodes[i]["signal"]))
        adjs_i.append(_pad(episodes[i]["adjacency"]))
        sigs_j.append(_pad(episodes[j]["signal"]))
        adjs_j.append(_pad(episodes[j]["adjacency"]))
        same_labels.append(same)

    return {
        "signal_i": torch.stack(sigs_i),
        "adjacency_i": torch.stack(adjs_i),
        "signal_j": torch.stack(sigs_j),
        "adjacency_j": torch.stack(adjs_j),
        "is_same": torch.tensor(same_labels, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Embedding all episodes
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_all(
    typer: SiameseTyper,
    episodes: list[dict],
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Embed all episodes and return (N_ep, d_proj) numpy array."""
    typer.eval()
    embeddings = []
    for start in range(0, len(episodes), batch_size):
        batch = episodes[start: start + batch_size]
        max_t = max(ep["signal"].shape[0] for ep in batch)

        def _pad(t: torch.Tensor) -> torch.Tensor:
            p = max_t - t.shape[0]
            return torch.cat([t, torch.zeros(p, *t.shape[1:], device=t.device)]) if p else t

        sig = torch.stack([_pad(ep["signal"]) for ep in batch]).to(device)
        adj = torch.stack([_pad(ep["adjacency"]) for ep in batch]).to(device)
        z = typer.embed(sig, adj)
        embeddings.append(z.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Per-epoch validation silhouette (M6, audit 2026-06)
# ---------------------------------------------------------------------------

def _val_silhouette(
    typer: SiameseTyper,
    train_eps: list[dict],
    val_eps: list[dict],
    batch_size: int,
    device: torch.device,
    linkage: str,
    metric: str,
    k_range_max: int,
    seed: int,
    fixed_k: int | None = None,
    k_selection: str = "silhouette",
) -> tuple[float, int]:
    """Silhouette val sous le modèle courant (critère de checkpoint M6).

    Reproduit exactement le protocole d'évaluation final : clustering sur les
    embeddings train, assignation val par nearest-centroid, silhouette val.
    ``n_gap_refs`` est réduit à 2 (monitoring par époque, pas un diagnostic) ;
    avec ``fixed_k`` (recommandé), aucun gap n'est calculé du tout.
    """
    from sklearn.metrics import silhouette_score

    z_train = _embed_all(typer, train_eps, batch_size, device)
    result = cluster_embeddings(
        z_train,
        k_range=range(2, k_range_max),
        n_gap_refs=2,
        random_state=seed,
        linkage=linkage,
        metric=metric,
        k_selection_method=k_selection,
        fixed_k=fixed_k,
    )
    k = result.k_optimal
    centroids = np.zeros((k, z_train.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = result.labels == c
        if mask.any():
            centroids[c] = z_train[mask].mean(axis=0)

    z_val = _embed_all(typer, val_eps, batch_size, device)
    dists = np.linalg.norm(z_val[:, None, :] - centroids[None, :, :], axis=2)
    labels_val = np.argmin(dists, axis=1).astype(int)
    if len(set(labels_val)) < 2:
        return -1.0, k
    return float(silhouette_score(z_val, labels_val, metric=metric)), k


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def _epoch(
    typer: SiameseTyper,
    loss_fn: ContrastiveLoss,
    pairs: list[tuple[int, int, bool]],
    episodes: list[dict],
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    device: torch.device,
) -> float:
    training = optimizer is not None
    typer.train(training)

    random.shuffle(pairs)
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start: start + batch_size]
            if not batch_pairs:
                continue
            batch = _collate_pairs(batch_pairs, episodes)
            sig_i = batch["signal_i"].to(device)
            adj_i = batch["adjacency_i"].to(device)
            sig_j = batch["signal_j"].to(device)
            adj_j = batch["adjacency_j"].to(device)
            is_same = batch["is_same"].to(device)

            z_i = typer.embed(sig_i, adj_i)
            z_j = typer.embed(sig_j, adj_j)
            dist = typer.distance(z_i, z_j)
            loss = loss_fn(dist, is_same)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Siamese typing training")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True,
                        help="Path to best_encoder.pt from encoder pre-training")
    parser.add_argument("--output", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    # M6 (audit 2026-06): la sélection du checkpoint sur la val loss
    # contrastive ne suit pas la métrique cible (silhouette). best_epoch~3
    # persistait (L10) parce que la loss val convergeait immédiatement alors
    # que la géométrie du clustering continuait d'évoluer. Le défaut devient
    # la silhouette val (nearest-centroid depuis les centroïdes train).
    parser.add_argument("--checkpoint-criterion", default="silhouette",
                        choices=["silhouette", "loss"],
                        help="critère de sélection du checkpoint : 'silhouette' "
                             "(défaut, métrique cible H1) ou 'loss' (ancien "
                             "comportement, val loss contrastive)")
    # M8/T3 (audit 2026-06): K-selection instable sur n≈270 (Phase K :
    # range [9,15], accord silhouette/Tibshirani 4/10). K fixe recommandé
    # pour la comparabilité multi-graines.
    parser.add_argument("--fixed-k", type=int, default=None,
                        help="fixe K (court-circuite la sélection) — "
                             "recommandé : 10 (cf. Phase K)")
    parser.add_argument("--k-selection", default="silhouette",
                        choices=["silhouette", "gap_tibshirani", "hdbscan"],
                        help="méthode de sélection de K quand --fixed-k absent")
    # Step 6 fix 6.1 (audit 2026-05-26): margin 2.0 was identified by sweep
    # as the optimal value (STATUS §config optimisée), but the default kept
    # being 1.0 → most v3 runs used a suboptimal margin. Default raised to 2.0.
    # Step 6 fix 6.3-bis: d_proj=64 is the swept-optimal projection dim
    # (vs 32 historical). H1 silhouette gain: +0.26 (0.519 → 0.782).
    parser.add_argument("--d-proj", type=int, default=64,
                        help="Projection head output dim (config optimisée: 64)")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="Contrastive hinge margin (config optimisée: 2.0)")
    parser.add_argument("--n-neg-per-anchor", type=int, default=5)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder weights, only train ProjectionHead")
    parser.add_argument("--k-range-max", type=int, default=16,
                        help="Upper bound (exclusive) for K search")
    parser.add_argument("--n-gap-refs", type=int, default=10)
    parser.add_argument("--n-shap-bg", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for silhouette CIs (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load existing checkpoint and recompute artifacts")
    # Step 6 fix 6.2 (audit 2026-05-26): hard-negative mining is identified
    # as the fix for the surentraînement siamois sur ewat_v4 (best_epoch=2-7
    # → cf. STATUS C-4). Default switched from 'random' to 'semi-hard' to
    # exploit harder pairs once warmup is over.
    parser.add_argument("--mining", default="semi-hard",
                        choices=["random", "hard", "semi-hard"],
                        help="Negative mining strategy for EpisodePairSampler "
                             "(default 'semi-hard' since audit 2026-05-26)")
    parser.add_argument("--mining-warmup-epochs", type=int, default=3,
                        help="Use 'random' mining for the first N epochs before "
                             "switching to --mining (only relevant for hard/semi-hard)")
    parser.add_argument("--mining-pool-size", type=int, default=0,
                        help="Cap negative-candidate pool size (0 = scan all)")
    parser.add_argument("--clustering-linkage", default="average",
                        choices=["ward", "average", "complete", "single"],
                        help="Agglomerative clustering linkage criterion")
    parser.add_argument("--clustering-metric", default="cosine",
                        choices=["euclidean", "cosine", "manhattan"],
                        help="Clustering distance metric (ward forces euclidean)")
    return parser


def run(args: argparse.Namespace) -> None:
    """Run the full siamese typing pipeline. Pure function — no argv parsing.

    Used both by the legacy argparse ``main()`` and by the Hydra entry point
    in ``experiments.typing.train_hydra``.
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    split_json = args.dataset / "split.json"

    # --- Load datasets ---
    train_ds = EpisodeDataset(split_json, args.features_root, split="train")
    val_ds = EpisodeDataset(split_json, args.features_root, split="val")
    test_ds = EpisodeDataset(split_json, args.features_root, split="test")

    # Load scaler from encoder checkpoint directory
    ckpt_data = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    scaler_path = Path(ckpt_data.get("scaler_path", "experiments/encoder/scaler.pkl"))
    if scaler_path.exists():
        train_ds.load_scaler(scaler_path)
        val_ds.load_scaler(scaler_path)
        test_ds.load_scaler(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    else:
        print(f"Warning: scaler not found at {scaler_path}; using no normalisation")

    # --- Load encoder ---
    # Step 5 fix 5.3 (audit 2026-05-26): use build_encoder_from_checkpoint to
    # auto-detect use_layer_norm from state_dict keys. Previously this site
    # passed default use_layer_norm=False, which silently failed to load v3
    # checkpoints or new use_layer_norm=True training runs.
    from ewat.encoder.factory import build_encoder_from_checkpoint
    encoder = build_encoder_from_checkpoint(ckpt_data)
    encoder.load_state_dict(ckpt_data["encoder_state"])
    enc_architecture = (ckpt_data.get("arch") or {}).get("architecture", "stgcn")
    print(f"Encoder ({enc_architecture}) loaded from {args.encoder_checkpoint} (epoch {ckpt_data.get('epoch','?')})")

    # --- Build siamese typer ---
    typer = SiameseTyper(encoder, d_proj=args.d_proj, freeze_encoder=args.freeze_encoder)
    typer = typer.to(device)
    n_params = sum(p.numel() for p in typer.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,} ({'frozen encoder' if args.freeze_encoder else 'full'})")

    # --- Pre-load episodes ---
    print("Loading train episodes into memory …")
    train_eps = _load_all_episodes(train_ds, device)
    print("Loading val episodes into memory …")
    val_eps = _load_all_episodes(val_ds, device)

    # --- Pair sampler ---
    mining = getattr(args, "mining", "random")
    mining_warmup = int(getattr(args, "mining_warmup_epochs", 0))
    mining_pool = getattr(args, "mining_pool_size", 0) or None

    pair_sampler = EpisodePairSampler(
        train_ds,
        n_neg_per_anchor=args.n_neg_per_anchor,
        seed=args.seed,
        mining=mining,
        margin=args.margin,
        candidate_pool_size=mining_pool,
    )
    train_pairs = list(pair_sampler)
    print(f"Training pairs: {len(train_pairs)} ({sum(s for _,_,s in train_pairs)} pos, "
          f"{sum(not s for _,_,s in train_pairs)} neg)")
    if mining != "random":
        print(f"Mining: {mining} (warmup={mining_warmup} epochs, pool={mining_pool})")

    val_sampler = EpisodePairSampler(val_ds, n_neg_per_anchor=3, seed=args.seed)
    val_pairs = list(val_sampler)

    # --- Loss + optimiser ---
    loss_fn = ContrastiveLoss(margin=args.margin)
    params = [p for p in typer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # --- MLflow (local file store by default — no server required) ---
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", str(args.output / "mlruns"))
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        mlflow.set_experiment("ewat")
        run = mlflow.start_run(run_name="typing_siamese")
        mlflow.log_params({
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "d_proj": args.d_proj, "margin": args.margin,
            "freeze_encoder": args.freeze_encoder,
            "n_neg_per_anchor": args.n_neg_per_anchor, "seed": args.seed,
            "clustering_linkage": getattr(args, "clustering_linkage", "average"),
            "clustering_metric": getattr(args, "clustering_metric", "cosine"),
        })
    except Exception:
        run = None

    # --- Training loop ---
    # M6 (audit 2026-06): le checkpoint est sélectionné par défaut sur la
    # silhouette val (métrique cible H1), pas sur la val loss contrastive qui
    # convergeait dès l'époque ~3 (L10) alors que la géométrie continuait
    # d'évoluer. --checkpoint-criterion loss restaure l'ancien comportement.
    use_sil_criterion = args.checkpoint_criterion == "silhouette"
    best_val_loss = float("inf")
    best_val_sil = -float("inf")

    clustering_linkage = getattr(args, "clustering_linkage", "average")
    clustering_metric = getattr(args, "clustering_metric", "cosine")
    if clustering_linkage == "ward":
        clustering_metric = "euclidean"

    if not args.eval_only:
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            if mining != "random" and epoch > mining_warmup:
                z_cache = _embed_all(typer, train_eps, args.batch_size, device)
                pair_sampler.update_embeddings(z_cache)
                train_pairs = list(pair_sampler)

            train_loss = _epoch(typer, loss_fn, train_pairs, train_eps, optimizer,
                                args.batch_size, device)
            val_loss = _epoch(typer, loss_fn, val_pairs, val_eps, None,
                              args.batch_size, device)

            val_sil = float("nan")
            k_epoch = -1
            if use_sil_criterion:
                val_sil, k_epoch = _val_silhouette(
                    typer, train_eps, val_eps, args.batch_size, device,
                    linkage=clustering_linkage, metric=clustering_metric,
                    k_range_max=args.k_range_max, seed=args.seed,
                    fixed_k=args.fixed_k, k_selection=args.k_selection,
                )
                print(f"Epoch {epoch:03d}/{args.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_sil={val_sil:.4f} (K={k_epoch})")
            else:
                print(f"Epoch {epoch:03d}/{args.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if run is not None:
                try:
                    metrics = {"contrastive_train_loss": train_loss,
                               "contrastive_val_loss": val_loss}
                    if use_sil_criterion:
                        metrics["val_silhouette"] = val_sil
                    mlflow.log_metrics(metrics, step=epoch)
                except Exception:
                    pass

            improved = (val_sil > best_val_sil) if use_sil_criterion \
                else (val_loss < best_val_loss)
            if improved:
                best_val_loss = min(best_val_loss, val_loss)
                if use_sil_criterion:
                    best_val_sil = val_sil
                patience_counter = 0
                torch.save({
                    "typer_state": typer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_silhouette": val_sil if use_sil_criterion else None,
                    "checkpoint_criterion": args.checkpoint_criterion,
                    "args": vars(args),
                    "d_proj": args.d_proj,
                }, ckpt_dir / "best_siamese.pt")
                crit = (f"val_sil={val_sil:.4f}" if use_sil_criterion
                        else f"val_loss={val_loss:.4f}")
                print(f"  ✓ Checkpoint saved ({crit})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    else:
        print("--eval-only: skipping training, loading existing checkpoint …")

    # Load best checkpoint for clustering + SHAP
    best = torch.load(ckpt_dir / "best_siamese.pt", map_location="cpu", weights_only=False)
    typer.load_state_dict(best["typer_state"])
    typer = typer.to(device).eval()
    print(f"\nBest checkpoint loaded (epoch {best['epoch']}, val_loss={best['val_loss']:.4f})")

    # --- Clustering on train embeddings ---
    print("Embedding train episodes …")
    z_train = _embed_all(typer, train_eps, args.batch_size, device)
    print(f"Train embeddings: {z_train.shape}")

    result = cluster_embeddings(
        z_train,
        k_range=range(2, args.k_range_max),
        n_gap_refs=args.n_gap_refs,
        random_state=args.seed,
        linkage=clustering_linkage,
        metric=clustering_metric,
        k_selection_method=args.k_selection,
        fixed_k=args.fixed_k,
    )
    k_opt = result.k_optimal
    sil_train = result.silhouette_scores[k_opt]
    print(f"Clustering: K={k_opt}, silhouette(train)={sil_train:.3f}")

    # --- Evaluate silhouette on val + test via nearest centroid (no data leakage) ---
    # Val/test labels are assigned to the nearest train centroid so that cluster IDs
    # are consistent across splits.  Running fit_predict on val/test independently
    # produces arbitrarily permuted IDs and inflates silhouette.
    from sklearn.metrics import silhouette_score

    centroids = np.zeros((k_opt, z_train.shape[1]), dtype=np.float32)
    for c in range(k_opt):
        mask = result.labels == c
        if mask.any():
            centroids[c] = z_train[mask].mean(axis=0)

    def _nearest_centroid(z: np.ndarray, ctrs: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(z[:, None, :] - ctrs[None, :, :], axis=2)
        return np.argmin(dists, axis=1).astype(int)

    print("Embedding val episodes …")
    z_val = _embed_all(typer, val_eps, args.batch_size, device)
    labels_val = _nearest_centroid(z_val, centroids)
    sil_metric = clustering_metric
    sil_val = float(silhouette_score(z_val, labels_val, metric=sil_metric)) if len(set(labels_val)) >= 2 else -1.0
    print(f"Silhouette(val)={sil_val:.3f}  [H1 threshold: 0.3, metric={sil_metric}]")

    print("Loading test episodes …")
    test_eps = _load_all_episodes(test_ds, device)
    z_test = _embed_all(typer, test_eps, args.batch_size, device)
    labels_test = _nearest_centroid(z_test, centroids)
    sil_test = float(silhouette_score(z_test, labels_test, metric=sil_metric)) if len(set(labels_test)) >= 2 else -1.0
    print(f"Silhouette(test)={sil_test:.3f}  [H1 threshold: 0.3]")
    h1_pass = sil_test >= 0.3
    print(f"H1 {'✓ PASS' if h1_pass else '✗ FAIL'} (silhouette_test={sil_test:.3f})")

    # Bootstrap CIs on silhouette (val and test)
    sil_ci_val: dict = {}
    sil_ci_test: dict = {}
    if args.n_bootstrap > 0:
        print(f"Bootstrap silhouette CIs (n={args.n_bootstrap}) …")
        rng = np.random.default_rng(args.seed)
        ci_val = bootstrap_silhouette_ci(z_val, labels_val, n=args.n_bootstrap, rng=rng)
        ci_test = bootstrap_silhouette_ci(z_test, labels_test, n=args.n_bootstrap, rng=rng)
        sil_ci_val = ci_val.as_dict()
        sil_ci_test = ci_test.as_dict()
        print(f"  sil_val  = {ci_val}")
        print(f"  sil_test = {ci_test}")

    # M9 (audit 2026-06): modèle nul — silhouette test sous labels permutés.
    # H1 « sil ≥ 0.3 » ne dit rien si une partition aléatoire de ces
    # embeddings atteint déjà ce niveau ; on reporte Δ(sil − sil_null) et la
    # p-value empirique (Phipson–Smyth).
    sil_null: dict = {}
    if len(set(labels_test)) >= 2:
        rng_null = np.random.default_rng(args.seed + 1)
        null_scores = []
        for _ in range(200):
            perm = rng_null.permutation(labels_test)
            if len(set(perm)) < 2:
                continue
            null_scores.append(
                float(silhouette_score(z_test, perm, metric=sil_metric))
            )
        null_arr = np.asarray(null_scores)
        n_geq = int((null_arr >= sil_test).sum())
        sil_null = {
            "null_mean": float(null_arr.mean()),
            "null_std": float(null_arr.std()),
            "null_p95": float(np.percentile(null_arr, 95)),
            "delta_vs_null": float(sil_test - null_arr.mean()),
            "p_value": float((1 + n_geq) / (1 + len(null_arr))),
            "n_permutations": int(len(null_arr)),
        }
        print(f"  null model (200 perms): sil_null={sil_null['null_mean']:.3f}"
              f"±{sil_null['null_std']:.3f}  Δ={sil_null['delta_vs_null']:.3f}"
              f"  p={sil_null['p_value']:.4f}")

    # --- Saliency fiches (gradient × input — fast proxy for SHAP) ---
    print("Computing saliency per cluster …")
    try:
        cluster_imp = compute_cluster_saliency(
            typer.encoder, train_ds, result.labels,
            device=device, seed=args.seed,
        )
        write_cluster_fiches(
            cluster_imp, result.labels, train_ds, args.output, method="saliency",
        )
        print(f"Fiches written to {args.output / 'fiches'}/")
    except Exception as e:
        print(f"Warning: saliency failed ({e}); skipping fiches")

    # --- Save cluster artifacts for ontology ---
    print("Saving cluster artifacts …")
    artifacts_dir = args.output / "cluster_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    np.save(artifacts_dir / "labels_train.npy", result.labels)
    np.save(artifacts_dir / "embeddings_train.npy", z_train)
    np.save(artifacts_dir / "labels_val.npy", labels_val)
    np.save(artifacts_dir / "embeddings_val.npy", z_val)
    np.save(artifacts_dir / "labels_test.npy", labels_test)
    np.save(artifacts_dir / "embeddings_test.npy", z_test)

    centroids = np.zeros((k_opt, z_train.shape[1]), dtype=np.float32)
    for c in range(k_opt):
        mask = result.labels == c
        if mask.any():
            centroids[c] = z_train[mask].mean(axis=0)
    np.save(artifacts_dir / "centroids.npy", centroids)

    manifest: dict[str, dict] = {}
    for ep, label in zip(train_eps, result.labels.tolist()):
        manifest[ep["episode_id"]] = {
            "cluster": int(label), "split": "train", "scenario": ep["scenario"],
        }
    for ep, label in zip(val_eps, labels_val.tolist()):
        manifest[ep["episode_id"]] = {
            "cluster": int(label), "split": "val", "scenario": ep["scenario"],
        }
    for ep, label in zip(test_eps, labels_test.tolist()):
        manifest[ep["episode_id"]] = {
            "cluster": int(label), "split": "test", "scenario": ep["scenario"],
        }
    (artifacts_dir / "cluster_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Cluster artifacts saved to {artifacts_dir}/")

    # --- Final report ---
    summary = {
        "k_optimal": k_opt,
        "silhouette_train": sil_train,
        "silhouette_val": sil_val,
        "silhouette_test": sil_test,
        "silhouette_ci_val": sil_ci_val,
        "silhouette_ci_test": sil_ci_test,
        "silhouette_null_test": sil_null,  # M9 audit 2026-06
        "silhouette_metric": sil_metric,
        "clustering_linkage": clustering_linkage,
        "clustering_metric": clustering_metric,
        "n_bootstrap": args.n_bootstrap,
        "h1_pass": h1_pass,
        "best_val_loss": best_val_loss,
        "epochs_trained": best["epoch"],
        # clé attendue par run_phase_h (était absente → best_epoch=None dans
        # les summaries multiseed)
        "best_epoch": best["epoch"],
        "best_val_silhouette": best.get("val_silhouette"),
        "checkpoint_criterion": best.get("checkpoint_criterion", "loss"),
        "silhouette_scores": result.silhouette_scores,
        "gap_stats": result.gap_stats,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    ci_val_str = (
        f" [{sil_ci_val['ci_lo']:.3f}, {sil_ci_val['ci_hi']:.3f}]"
        if sil_ci_val else ""
    )
    ci_test_str = (
        f" [{sil_ci_test['ci_lo']:.3f}, {sil_ci_test['ci_hi']:.3f}]"
        if sil_ci_test else ""
    )
    report_lines = [
        "# Siamese Typing — Results\n",
        f"K optimal: **{k_opt}**",
        f"Silhouette train: {sil_train:.3f}",
        f"Silhouette val:   {sil_val:.3f}{ci_val_str}",
        f"Silhouette test:  {sil_test:.3f}{ci_test_str}",
        f"H1: {'✓ PASS' if h1_pass else '✗ FAIL'} (threshold 0.3)",
        "",
        "## Silhouette scores by K",
        *[f"  K={k}: {s:.3f}" for k, s in sorted(result.silhouette_scores.items())],
    ]
    (args.output / "results.md").write_text("\n".join(report_lines))
    print(f"Report: {args.output / 'results.md'}")

    if run is not None:
        try:
            mlflow.log_metrics({
                "silhouette_train": sil_train,
                "silhouette_val": sil_val,
                "silhouette_test": sil_test,
                "k_optimal": k_opt,
            })
            mlflow.end_run()
        except Exception:
            pass


def main() -> None:
    """Argparse entry point (legacy)."""
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
