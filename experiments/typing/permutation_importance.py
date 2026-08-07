"""Permutation importance par cluster — remplacement des fiches gradient×input.

Pour chaque feature j (0..16) :
  1. Mélanger la colonne j sur l'ensemble des épisodes train.
  2. Repasser le signal perturbé dans l'encodeur STGCN → z_e perturbés.
  3. Calculer la silhouette moyenne *par cluster* sur les embeddings perturbés.
  4. Importance(j, C_i) = silhouette_baseline(C_i) − silhouette_perturbed(C_i)
     (moyennée sur n_shuffles répétitions)

Normalisation par cluster : divisé par la somme des importances positives.
Features dont le shuffle *augmente* la silhouette reçoivent importance=0
(l'encodeur n'exploite pas cette feature pour ce cluster).

Usage
-----
    python -m experiments.typing.permutation_importance \\
        --typing-dir  experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        --n-shuffles 50 \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import silhouette_samples

from ewat.encoder.dataset import EpisodeDataset
from ewat.encoder.factory import build_encoder
from ewat.encoder.stgcn import STGCNEncoder

FEAT_NAMES = [
    "cpu_util", "ram_util", "latency_p99", "error_rate_http",
    "net_sat", "disk_io", "queue_depth",
    "span_dur_p99", "abnormal_span_rate", "trace_depth",
    "fan_out", "retry_rate", "latency_cv",
    "log_error_rate", "log_warn_rate", "semantic_anomaly",
    "lexical_entropy",
]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_batch(
    encoder: STGCNEncoder,
    signals: list[torch.Tensor],
    adjacencies: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Embed a list of variable-length episodes; return (N, d_embed) ndarray."""
    embeddings = []
    for sig, adj in zip(signals, adjacencies):
        s = sig.unsqueeze(0).to(device)   # (1, T, N, 17)
        a = adj.unsqueeze(0).to(device)   # (1, T, N, N, 3)
        z = encoder(s, a)                 # (1, d_embed)
        embeddings.append(z.squeeze(0).cpu().numpy())
    return np.stack(embeddings)           # (N, d_embed)


def _load_train_signals(
    cluster_manifest: dict[str, dict],
    features_root: Path,
    split_json: Path,
    scaler_path: Path | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], np.ndarray, list[str]]:
    """Load train split raw signals, adjacencies, cluster labels, and episode ids."""
    dataset = EpisodeDataset(split_json, features_root, split="train")
    if scaler_path and scaler_path.exists():
        import pickle
        with open(scaler_path, "rb") as f:
            dataset.scaler = pickle.load(f)

    signals, adjacencies, labels, ep_ids = [], [], [], []
    for idx in range(len(dataset)):
        item = dataset[int(idx)]
        signals.append(item["signal"])       # (T, N, 17) tensor
        adjacencies.append(item["adjacency"])
        ep_id = dataset.episode_ids[idx]
        labels.append(int(cluster_manifest[ep_id]["cluster"]))
        ep_ids.append(ep_id)

    return signals, adjacencies, np.array(labels, dtype=int), ep_ids


# ---------------------------------------------------------------------------
# Permutation importance core
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    encoder: STGCNEncoder,
    signals: list[torch.Tensor],
    adjacencies: list[torch.Tensor],
    cluster_labels: np.ndarray,
    n_shuffles: int = 50,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[int, np.ndarray]:
    """Return per-cluster permutation importance (17,), normalised.

    Returns
    -------
    Dict {cluster_id → (17,) array}, each normalised to [0,1] summing to 1.
    Feature importance = max(0, silhouette_drop) when that feature is shuffled.
    """
    if device is None:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)
    n_ep = len(signals)
    cluster_ids = sorted(set(int(c) for c in cluster_labels))
    n_features = 17

    # --- Baseline embeddings and per-cluster silhouette ---
    print("  Computing baseline embeddings …", flush=True)
    Z_base = _embed_batch(encoder, signals, adjacencies, device)   # (N, d)
    sil_base_samples = silhouette_samples(Z_base, cluster_labels)  # (N,)
    sil_base = {
        cid: float(sil_base_samples[cluster_labels == cid].mean())
        for cid in cluster_ids
    }
    print(f"  Baseline silhouette per cluster: "
          f"{', '.join(f'C{c}={v:.3f}' for c,v in sorted(sil_base.items()))}")

    # --- Per-feature permutation ---
    # importance_accum[cid][feat] = list of silhouette drops
    importance_accum: dict[int, dict[int, list[float]]] = {
        cid: {j: [] for j in range(n_features)} for cid in cluster_ids
    }

    for feat_j in range(n_features):
        print(f"  Feature {feat_j:2d}/{n_features-1} ({FEAT_NAMES[feat_j]}) …",
              end="  ", flush=True)
        drops_per_cluster: dict[int, list[float]] = {c: [] for c in cluster_ids}

        for _ in range(n_shuffles):
            perm_idx = rng.permutation(n_ep)

            # Build shuffled signals: copy signal list, replace feature j with
            # the time-averaged value from a permuted episode (handles variable T).
            shuffled_signals = []
            for ep_i in range(n_ep):
                sig = signals[ep_i].clone()               # (T_i, N, 17)
                sig_perm = signals[perm_idx[ep_i]]        # (T_j, N, 17)
                # Mean over time from permuted episode → (N,), broadcast to (T_i, N)
                feat_mean = sig_perm[..., feat_j].mean(dim=0)  # (N,)
                sig[..., feat_j] = feat_mean.unsqueeze(0).expand(sig.shape[0], -1)
                shuffled_signals.append(sig)

            Z_shuf = _embed_batch(encoder, shuffled_signals, adjacencies, device)
            sil_shuf = silhouette_samples(Z_shuf, cluster_labels)

            for cid in cluster_ids:
                mask = cluster_labels == cid
                drop = sil_base[cid] - float(sil_shuf[mask].mean())
                drops_per_cluster[cid].append(drop)

        for cid in cluster_ids:
            importance_accum[cid][feat_j] = drops_per_cluster[cid]
        mean_drops = {c: np.mean(importance_accum[c][feat_j]) for c in cluster_ids}
        print("drop " + " ".join(f"C{c}:{v:+.3f}" for c, v in sorted(mean_drops.items())))

    # --- Aggregate and normalise ---
    out: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        raw = np.array([
            max(0.0, float(np.mean(importance_accum[cid][j])))
            for j in range(n_features)
        ], dtype=np.float32)
        total = raw.sum()
        if total > 0:
            raw = raw / total
        out[cid] = raw

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permutation importance per cluster — replace gradient×input fiches"
    )
    parser.add_argument("--typing-dir",    type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir",   type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--dataset",       type=Path, default=Path("data/datasets/ewat_v3"),
                        help="Dataset dir containing split.json")
    parser.add_argument("--n-shuffles",    type=int,  default=50)
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    encoder_dir = args.encoder_dir or (args.typing_dir.parent / "encoder")

    # --- Load encoder ---
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
    encoder = encoder.to(device).eval()
    print(f"Encoder loaded from {enc_ckpt_path}")

    # --- Load cluster manifest ---
    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    scaler_path = encoder_dir / "scaler.pkl"

    split_json = args.dataset / "split.json"

    # --- Load train signals ---
    print("Loading train episodes …")
    signals, adjacencies, cluster_labels, ep_ids = _load_train_signals(
        cluster_manifest, args.features_root, split_json, scaler_path
    )
    print(f"  {len(signals)} train episodes, {len(set(cluster_labels.tolist()))} clusters")

    # --- Compute permutation importance ---
    print(f"\nComputing permutation importance (n_shuffles={args.n_shuffles}) …")
    cluster_importance = compute_permutation_importance(
        encoder, signals, adjacencies, cluster_labels,
        n_shuffles=args.n_shuffles, seed=args.seed, device=device,
    )

    # --- Write/update fiches ---
    fiches_dir = args.typing_dir / "fiches"
    fiches_dir.mkdir(exist_ok=True)

    cluster_ids = sorted(cluster_importance.keys())
    for cid in cluster_ids:
        importance = cluster_importance[cid]
        ep_idxs = np.where(cluster_labels == cid)[0]

        # Scenario distribution
        scenarios: dict[str, int] = {}
        for i in ep_idxs:
            sc = cluster_manifest[ep_ids[i]]["scenario"]
            scenarios[sc] = scenarios.get(sc, 0) + 1

        ranked = sorted(
            zip(FEAT_NAMES, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        fiche = {
            "cluster_id": cid,
            "method": "permutation_importance",
            "n_shuffles": args.n_shuffles,
            "n_episodes": int(len(ep_idxs)),
            "scenario_distribution": scenarios,
            "feature_importance": {name: float(val) for name, val in ranked},
            "top5_features": [name for name, _ in ranked[:5]],
        }
        fiche_path = fiches_dir / f"cluster_{cid}.json"
        fiche_path.write_text(json.dumps(fiche, indent=2))
        top5 = ", ".join(fiche["top5_features"])
        print(f"  C{cid}: top5 = [{top5}]  → {fiche_path}")

    # --- Summary report ---
    report_lines = [
        "# Permutation importance par cluster\n",
        f"Méthode : permutation des features dans S(t) brut → silhouette drop moyen (n_shuffles={args.n_shuffles}).\n",
        "**Note** : remplace les fiches gradient×input (invalidées par ρ_Spearman=−0.34 avec permutation).\n\n",
        "| Cluster | n_ép | Top-5 features (par importance) |",
        "|---------|------|--------------------------------|",
    ]
    for cid in cluster_ids:
        fiche_path = fiches_dir / f"cluster_{cid}.json"
        d = json.loads(fiche_path.read_text())
        n = d["n_episodes"]
        top5 = " > ".join(d["top5_features"])
        report_lines.append(f"| C{cid} | {n} | {top5} |")

    report_path = args.typing_dir / "permutation_importance.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport : {report_path}")
    print("Fiches mises à jour avec method='permutation_importance'.")


if __name__ == "__main__":
    main()
