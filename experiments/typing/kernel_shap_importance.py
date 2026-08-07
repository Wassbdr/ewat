"""KernelSHAP importance per cluster — validation croisée avec permutation_importance.

Calcule les valeurs Shapley (shap.KernelExplainer) par cluster et les compare
aux fiches permutation_importance existantes.

Question scientifique
---------------------
ρ_Spearman(SHAP, permutation_importance) > 0 pour ≥ 7/10 clusters ?
- Si oui : les deux méthodes concordent → fiches permutation_importance validées.
- Si non : SHAP et permutation capturent des aspects différents de l'importance.

Outputs
-------
- ``fiches/cluster_{i}_shap.json``  — fiches SHAP (sans écraser les permutation fiches)
- ``shap_vs_permutation_correlation.csv`` — ρ Spearman + p-value par cluster
- ``kernel_shap_importance.md`` — rapport de synthèse

Usage
-----
    python -m experiments.typing.kernel_shap_importance \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        --dataset data/datasets/ewat_v3 \\
        [--n-bg 20] [--n-samples 64] [--max-eps 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from ewat.encoder.dataset import EpisodeDataset
from ewat.encoder.factory import build_encoder
from ewat.typing.saliency_explainer import compute_cluster_kernel_shap

FEAT_NAMES = [
    "cpu_util", "ram_util", "latency_p99", "error_rate_http",
    "net_sat", "disk_io", "queue_depth",
    "span_dur_p99", "abnormal_span_rate", "trace_depth",
    "fan_out", "retry_rate", "latency_cv",
    "log_error_rate", "log_warn_rate", "semantic_anomaly",
    "lexical_entropy",
]


def _load_encoder(encoder_dir: Path, device: torch.device) -> torch.nn.Module:
    ckpt_path = encoder_dir / "checkpoints" / "best_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", {})
    encoder = build_encoder(
        architecture=arch.get("arch", "stgcn"),
        d_feat=int(arch.get("d_feat", 17)),
        n_nodes=int(arch.get("n_nodes", 6)),
        d_hidden=int(arch.get("d_hidden", 64)),
        d_embed=int(arch.get("d_embed", 64)),
    )
    encoder.load_state_dict(ckpt["encoder_state"])
    return encoder.to(device).eval()


def _build_dataset_with_labels(
    cluster_manifest: dict[str, dict],
    features_root: Path,
    split_json: Path,
    scaler_path: Path,
) -> tuple[EpisodeDataset, np.ndarray]:
    dataset = EpisodeDataset(split_json, features_root, split="train")
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            dataset.scaler = pickle.load(f)
    labels = np.array(
        [int(cluster_manifest[ep_id]["cluster"]) for ep_id in dataset.episode_ids],
        dtype=int,
    )
    return dataset, labels


def _load_permutation_fiches(fiches_dir: Path) -> dict[int, np.ndarray]:
    """Load existing permutation_importance fiches → {cluster_id: (17,) array}."""
    perm: dict[int, np.ndarray] = {}
    for fiche_path in sorted(fiches_dir.glob("cluster_[0-9]*.json")):
        # Skip SHAP fiches (cluster_*_shap.json)
        if "_shap" in fiche_path.name:
            continue
        data = json.loads(fiche_path.read_text())
        cid = int(data["cluster_id"])
        importance = np.array(
            [data["feature_importance"].get(f, 0.0) for f in FEAT_NAMES],
            dtype=np.float32,
        )
        perm[cid] = importance
    return perm


def _compare_correlations(
    shap_importance: dict[int, np.ndarray],
    perm_importance: dict[int, np.ndarray],
) -> list[dict]:
    rows = []
    for cid in sorted(shap_importance):
        shap_vec = shap_importance[cid]
        perm_vec = perm_importance.get(cid)
        if perm_vec is None:
            rows.append({"cluster": cid, "spearman_rho": None, "p_value": None})
            continue
        rho, pval = spearmanr(shap_vec, perm_vec)
        rows.append({
            "cluster": cid,
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(pval), 4),
            "concordant": bool(rho > 0),
        })
    return rows


def _write_shap_fiches(
    shap_importance: dict[int, np.ndarray],
    cluster_labels: np.ndarray,
    cluster_manifest: dict[str, dict],
    dataset: EpisodeDataset,
    fiches_dir: Path,
) -> None:
    fiches_dir.mkdir(parents=True, exist_ok=True)
    ep_ids = dataset.episode_ids

    for cid, importance in sorted(shap_importance.items()):
        ep_idxs = np.where(cluster_labels == cid)[0]
        scenarios: dict[str, int] = {}
        for i in ep_idxs:
            sc = cluster_manifest.get(ep_ids[i], {}).get("scenario", "?")
            scenarios[sc] = scenarios.get(sc, 0) + 1

        ranked = sorted(
            zip(FEAT_NAMES, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        fiche = {
            "cluster_id": cid,
            "method": "kernel_shap",
            "n_episodes": int(len(ep_idxs)),
            "scenario_distribution": scenarios,
            "feature_importance": {name: float(val) for name, val in ranked},
            "top5_features": [name for name, _ in ranked[:5]],
        }
        (fiches_dir / f"cluster_{cid}_shap.json").write_text(json.dumps(fiche, indent=2))


def _write_report(
    typing_dir: Path,
    correlations: list[dict],
    shap_importance: dict[int, np.ndarray],
    perm_importance: dict[int, np.ndarray],
    elapsed_s: float,
) -> None:
    n_concordant = sum(1 for r in correlations if r.get("concordant"))
    n_total = len(correlations)
    verdict = "✓ VALIDÉES" if n_concordant >= 7 else "⚠️ DISCORDANTES"

    lines = [
        "# KernelSHAP vs Permutation importance — validation croisée\n",
        f"Temps d'exécution : {elapsed_s:.0f} s\n",
        "---",
        "",
        "## Résultat principal",
        "",
        f"Fiches permutation_importance : **{verdict}**  "
        f"({n_concordant}/{n_total} clusters avec ρ > 0)\n",
        "",
        "---",
        "",
        "## Corrélation Spearman par cluster",
        "",
        f"{'C':>2}  {'ρ Spearman':>11}  {'p-value':>9}  {'Concordant':>11}  Top-5 SHAP",
        "-" * 80,
    ]

    for r in correlations:
        cid = r["cluster"]
        rho_str = f"{r['spearman_rho']:+.4f}" if r["spearman_rho"] is not None else "  N/A"
        p_str = f"{r['p_value']:.4f}" if r["p_value"] is not None else "   N/A"
        conc = "✓" if r.get("concordant") else "✗"
        top5 = shap_importance.get(cid)
        if top5 is not None:
            top5_names = [FEAT_NAMES[i] for i in np.argsort(top5)[::-1][:5]]
            top5_str = " > ".join(top5_names)
        else:
            top5_str = "N/A"
        lines.append(f"C{cid:<1}  {rho_str:>11}  {p_str:>9}  {conc:>11}  {top5_str}")

    lines += [
        "",
        "---",
        "",
        "## Top-5 features SHAP vs Permutation — comparaison",
        "",
        f"{'C':>2}  {'Top-5 SHAP':<55}  {'Top-5 Permutation':<55}",
        "-" * 115,
    ]
    for cid in sorted(shap_importance):
        shap_vec = shap_importance[cid]
        perm_vec = perm_importance.get(cid, np.zeros(17))
        shap_top5 = " > ".join([FEAT_NAMES[i] for i in np.argsort(shap_vec)[::-1][:5]])
        perm_top5 = " > ".join([FEAT_NAMES[i] for i in np.argsort(perm_vec)[::-1][:5]])
        lines.append(f"C{cid:<1}  {shap_top5:<55}  {perm_top5:<55}")

    lines += [
        "",
        "---",
        "",
        "## Interprétation",
        "",
        f"- {n_concordant}/{n_total} clusters concordants (ρ > 0)",
        "- ρ > 0 signifie que les deux méthodes classent les features dans le même ordre.",
        "- Un ρ < 0 révèle une discordance : SHAP et permutation capturent",
        "  des aspects différents de la contribution des features à la géométrie des embeddings.",
        "- Les fiches `cluster_*_shap.json` sont stockées séparément pour ne pas",
        "  écraser les fiches permutation_importance existantes.",
    ]

    (typing_dir / "kernel_shap_importance.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelSHAP importance per cluster + comparison with permutation_importance"
    )
    parser.add_argument("--typing-dir",    type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir",   type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--dataset",       type=Path, default=Path("data/datasets/ewat_v3"))
    parser.add_argument("--n-bg",          type=int,  default=20,
                        help="Background samples for KernelExplainer")
    parser.add_argument("--n-samples",     type=int,  default=64,
                        help="nsamples per episode for KernelSHAP")
    parser.add_argument("--max-eps",       type=int,  default=5,
                        help="Max episodes per cluster for SHAP")
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    encoder_dir = args.encoder_dir or (args.typing_dir.parent / "encoder")

    print(f"Device: {device}")

    encoder = _load_encoder(encoder_dir, device)
    print(f"Encoder loaded from {encoder_dir / 'checkpoints' / 'best_encoder.pt'}")

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    scaler_path = encoder_dir / "scaler.pkl"
    split_json = args.dataset / "split.json"

    print("Loading train episodes …")
    dataset, cluster_labels = _build_dataset_with_labels(
        cluster_manifest, args.features_root, split_json, scaler_path
    )
    n_clusters = len(set(cluster_labels.tolist()))
    print(f"  {len(dataset)} train episodes, {n_clusters} clusters")

    print(f"\nRunning KernelSHAP (n_bg={args.n_bg}, n_samples={args.n_samples}, "
          f"max_eps_per_cluster={args.max_eps}) …")
    t0 = time.time()
    shap_importance = compute_cluster_kernel_shap(
        encoder=encoder,
        dataset=dataset,
        cluster_labels=cluster_labels,
        n_bg=args.n_bg,
        n_samples_per_episode=args.n_samples,
        max_episodes_per_cluster=args.max_eps,
        device=device,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"KernelSHAP done in {elapsed:.0f} s")

    fiches_dir = args.typing_dir / "fiches"
    perm_importance = _load_permutation_fiches(fiches_dir)
    print(f"Loaded {len(perm_importance)} permutation_importance fiches")

    correlations = _compare_correlations(shap_importance, perm_importance)

    _write_shap_fiches(
        shap_importance, cluster_labels, cluster_manifest, dataset, fiches_dir
    )

    _write_report(args.typing_dir, correlations, shap_importance, perm_importance, elapsed)

    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(args.typing_dir / "shap_vs_permutation_correlation.csv", index=False)

    # Summary
    print("\n=== Corrélation Spearman SHAP vs Permutation ===")
    for r in correlations:
        rho_str = f"{r['spearman_rho']:+.4f}" if r["spearman_rho"] is not None else "N/A"
        conc = "✓" if r.get("concordant") else "✗"
        print(f"  C{r['cluster']}: ρ={rho_str}  p={r.get('p_value', 'N/A')}  {conc}")

    n_concordant = sum(1 for r in correlations if r.get("concordant"))
    verdict = "VALIDÉES" if n_concordant >= 7 else "DISCORDANTES"
    print(f"\nFiches permutation_importance : {verdict} ({n_concordant}/{len(correlations)} ρ > 0)")
    print(f"Outputs: {args.typing_dir}/kernel_shap_importance.md + shap_vs_permutation_correlation.csv")


if __name__ == "__main__":
    main()
