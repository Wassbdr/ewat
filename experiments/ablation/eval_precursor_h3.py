"""Ablation features sur H3 — impact du masquage sur l'AUROC des précurseurs.

Pour chaque condition d'ablation (7 modalités + 17 leave-one-out), masque les
features correspondantes dans les fenêtres pré-injection, ré-infère avec le
SiameseTyper existant, et évalue l'AUROC des PrecursorClassifiers pré-entraînés.

Distinct de l'ablation H1 (silhouette clustering dans experiments/ablation/run.py).
Ici la métrique est l'AUROC précurseur H3 au k* val-optimal.

Note méthodologique
-------------------
Ablation par masquage à l'inférence (pas réentraînement). Mesure la sensibilité
du classifieur LR pré-entraîné à l'absence d'une feature — pas l'importance
causale. À interpréter comme : « quelles features le modèle exploite-t-il pour
la prédiction ? » et non « quelles features seraient optimales si réentraîné ? »

Usage
-----
    python -m experiments.ablation.eval_precursor_h3 \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --precursor-dir experiments/precursor \\
        --features-root data/features/v3 \\
        [--output experiments/ablation] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ewat.encoder.factory import build_encoder
from ewat.precursor.dataset import PrecursorDataset
from ewat.precursor.model import PrecursorClassifier
from ewat.typing.siamese import SiameseTyper

FEAT_NAMES = [
    "cpu_util", "ram_util", "latency_p99", "error_rate_http",
    "net_sat", "disk_io", "queue_depth",                         # M: 0-6
    "span_dur_p99", "abnormal_span_rate", "trace_depth",
    "fan_out", "retry_rate", "latency_cv",                       # T: 7-12
    "log_error_rate", "log_warn_rate", "semantic_anomaly",
    "lexical_entropy",                                            # L: 13-16
]
IDX_M = list(range(0, 7))
IDX_T = list(range(7, 13))
IDX_L = list(range(13, 17))

MODALITY_CONDITIONS: dict[str, list[int]] = {
    "full":   IDX_M + IDX_T + IDX_L,
    "M_only": IDX_M,
    "T_only": IDX_T,
    "L_only": IDX_L,
    "M+T":    IDX_M + IDX_T,
    "M+L":    IDX_M + IDX_L,
    "T+L":    IDX_T + IDX_L,
}


def _loo_conditions() -> dict[str, list[int]]:
    """Leave-one-out: mask feature i, keep all others."""
    all_feats = list(range(17))
    return {
        f"drop_{FEAT_NAMES[i]}": [j for j in all_feats if j != i]
        for i in range(17)
    }


def _load_typer(typing_dir: Path, encoder_dir: Path, device: torch.device) -> SiameseTyper:
    enc_ckpt_path = encoder_dir / "checkpoints" / "best_encoder.pt"
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
    arch = enc_ckpt.get("arch") or {}
    encoder = build_encoder(
        arch.get("architecture", "stgcn"),
        d_feat=int(arch.get("d_feat", 17)),
        n_nodes=int(arch.get("n_nodes", 6)),
        d_hidden=int(arch.get("d_hidden", 64)),
        d_embed=int(arch.get("d_embed", 64)),
    )
    encoder.load_state_dict(enc_ckpt["encoder_state"])

    typer_ckpt = torch.load(
        typing_dir / "checkpoints" / "best_siamese.pt",
        map_location="cpu", weights_only=False,
    )
    typer = SiameseTyper(encoder, d_proj=32)
    typer.load_state_dict(typer_ckpt["typer_state"])
    return typer.to(device).eval()


@torch.no_grad()
def _embed_masked(
    typer: SiameseTyper,
    dataset: PrecursorDataset,
    active_feats: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed test episodes with feature mask. Returns (z_test, y_test)."""
    mask = torch.zeros(17, dtype=torch.float32)
    mask[active_feats] = 1.0

    embeddings, labels = [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        sig_masked = item["signal"] * mask          # (k, N, 17) broadcast
        sig = sig_masked.unsqueeze(0).to(device)    # (1, k, N, 17)
        adj = item["adjacency"].unsqueeze(0).to(device)
        z = typer.embed(sig, adj).cpu().numpy()[0]  # (d_proj,)
        embeddings.append(z)
        labels.append(int(item["cluster"]))

    return np.stack(embeddings), np.array(labels, dtype=int)


def _macro_auroc(auroc_dict: dict[int, float]) -> float:
    vals = [v for v in auroc_dict.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _evaluate_condition(
    typer: SiameseTyper,
    datasets: dict[int, PrecursorDataset],  # k → dataset
    classifiers: dict[int, PrecursorClassifier],  # c → clf at k_optimal[c]
    k_optimal: dict[int, int],
    active_feats: list[int],
    device: torch.device,
) -> dict:
    """Evaluate AUROC for all clusters with the given feature mask.

    Each cluster c is evaluated at its val-optimal k. Returns per-cluster
    and macro AUROC.
    """
    # Embed test set for each unique k used in k_optimal
    unique_ks = sorted(set(k_optimal.values()))
    z_by_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in unique_ks:
        z_by_k[k] = _embed_masked(typer, datasets[k], active_feats, device)

    auroc_per_cluster: dict[int, float] = {}
    for c, clf in classifiers.items():
        k = k_optimal[c]
        z_test, y_test = z_by_k[k]
        auc = clf.auroc_per_type(z_test, y_test)
        auroc_per_cluster[c] = auc.get(c, float("nan"))

    return {
        "auroc_per_cluster": auroc_per_cluster,
        "macro_auroc": _macro_auroc(auroc_per_cluster),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H3 ablation — feature masking impact on precursor AUROC"
    )
    parser.add_argument("--typing-dir",    type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir",   type=Path, default=None)
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output",        type=Path, default=Path("experiments/ablation"))
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    encoder_dir = args.encoder_dir or (args.typing_dir.parent / "encoder")

    print(f"Device: {device}")

    # --- Load manifest and precursor results ---
    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    precursor_results = json.loads((args.precursor_dir / "results.json").read_text())
    k_optimal: dict[int, int] = {
        int(c): int(k) for c, k in precursor_results["k_optimal"].items()
    }
    n_clusters = precursor_results["n_clusters"]
    print(f"k_optimal: {k_optimal}")

    # --- Load SiameseTyper ---
    typer = _load_typer(args.typing_dir, encoder_dir, device)
    print("SiameseTyper loaded")

    # --- Load scaler ---
    scaler_path = encoder_dir / "scaler.pkl"

    # --- Build test PrecursorDataset for each unique k in k_optimal ---
    unique_ks = sorted(set(k_optimal.values()))
    datasets: dict[int, PrecursorDataset] = {}
    for k in unique_ks:
        ds = PrecursorDataset(cluster_manifest, args.features_root, k=k, split="test")
        if scaler_path.exists():
            ds.load_scaler(scaler_path)
        datasets[k] = ds
        print(f"  PrecursorDataset k={k}: {len(ds)} test episodes")

    # --- Load PrecursorClassifiers (val-optimal k per cluster) ---
    classifiers: dict[int, PrecursorClassifier] = {}
    ckpt_dir = args.precursor_dir / "checkpoints"
    for c in range(n_clusters):
        k = k_optimal[c]
        ckpt_path = ckpt_dir / f"classifier_type{c}_k{k}.pkl"
        if ckpt_path.exists():
            classifiers[c] = PrecursorClassifier.load(ckpt_path)
        else:
            print(f"  WARNING: checkpoint not found for C{c} k={k}: {ckpt_path}")
    print(f"Loaded {len(classifiers)}/{n_clusters} PrecursorClassifiers")

    # --- Define all ablation conditions ---
    conditions: dict[str, list[int]] = {}
    conditions.update(MODALITY_CONDITIONS)
    conditions.update(_loo_conditions())
    print(f"\nRunning {len(conditions)} ablation conditions …")

    # --- Evaluate each condition ---
    results: list[dict] = []
    for cond_name, active_feats in conditions.items():
        print(f"  {cond_name:35s} ({len(active_feats)} features) … ", end="", flush=True)
        res = _evaluate_condition(
            typer, datasets, classifiers, k_optimal, active_feats, device
        )
        macro = res["macro_auroc"]
        print(f"macro_AUROC = {macro:.4f}")
        results.append({
            "condition": cond_name,
            "n_active_feats": len(active_feats),
            "macro_auroc": macro,
            **{f"C{c}": v for c, v in res["auroc_per_cluster"].items()},
        })

    # --- Compute Δ vs full ---
    full_row = next(r for r in results if r["condition"] == "full")
    full_macro = full_row["macro_auroc"]
    for r in results:
        r["delta_macro"] = round(r["macro_auroc"] - full_macro, 4)

    # --- Save CSV ---
    df = pd.DataFrame(results)
    df_sorted = pd.concat([
        df[df["condition"].isin(MODALITY_CONDITIONS)],
        df[~df["condition"].isin(MODALITY_CONDITIONS)].sort_values("delta_macro"),
    ]).reset_index(drop=True)
    df_sorted.to_csv(args.output / "results_h3_ablation.csv", index=False)

    # --- Write report ---
    lines = [
        "# Ablation H3 — Impact du masquage sur l'AUROC précurseur\n",
        "Masquage à l'inférence sur PrecursorClassifiers pré-entraînés (k* val-optimal).",
        "Ne pas confondre avec l'ablation H1 (silhouette sur encodeur).\n",
        f"AUROC full (référence) : **{full_macro:.4f}**\n",
        "---",
        "",
        "## Ablation par modalité",
        "",
        f"{'Condition':<15}  {'Feats':>5}  {'Macro-AUROC':>11}  {'Δ vs full':>10}",
        "-" * 50,
    ]
    for r in results:
        if r["condition"] not in MODALITY_CONDITIONS:
            continue
        delta_str = f"{r['delta_macro']:+.4f}"
        lines.append(
            f"{r['condition']:<15}  {r['n_active_feats']:>5}  "
            f"{r['macro_auroc']:>11.4f}  {delta_str:>10}"
        )

    lines += [
        "",
        "---",
        "",
        "## Leave-one-out (features ↓ Δ AUROC = plus critiques pour H3)",
        "",
        f"{'Feature retirée':<26}  {'Macro-AUROC':>11}  {'Δ vs full':>10}",
        "-" * 52,
    ]
    loo_rows = sorted(
        [r for r in results if r["condition"].startswith("drop_")],
        key=lambda r: r["delta_macro"],
    )
    for r in loo_rows:
        feat = r["condition"].replace("drop_", "")
        delta_str = f"{r['delta_macro']:+.4f}"
        lines.append(
            f"{feat:<26}  {r['macro_auroc']:>11.4f}  {delta_str:>10}"
        )

    lines += [
        "",
        "---",
        "",
        "## Note méthodologique",
        "",
        "Δ négatif = masquer cette feature fait baisser l'AUROC → feature importante pour H3.",
        "Δ positif = masquer cette feature améliore l'AUROC → feature redondante ou bruitée.",
        "Interprétation causale limitée : ablation par masquage, pas réentraînement.",
        "Comparer avec l'ablation H1 (silhouette) pour voir si les importances diffèrent.",
    ]

    (args.output / "results_h3_ablation.md").write_text("\n".join(lines))
    print(f"\nOutputs: {args.output}/results_h3_ablation.md + results_h3_ablation.csv")
    print(f"AUROC full = {full_macro:.4f}")

    # Print top-5 most critical features for H3
    print("\nTop-5 features les plus critiques pour H3 (Δ le plus négatif) :")
    for r in loo_rows[:5]:
        feat = r["condition"].replace("drop_", "")
        print(f"  {feat:<26}  Δ={r['delta_macro']:+.4f}")


if __name__ == "__main__":
    main()
