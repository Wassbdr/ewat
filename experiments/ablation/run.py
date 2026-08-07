"""Ablation study — impact de chaque modalité et feature sur la structurabilité.

Stratégie : masquage à l'inférence (zéros) sur le checkpoint existant.
Pas de réentraînement — mesure l'effet sur la géométrie des embeddings.

Conditions testées
------------------
1. Ablation par modalité (7 conditions) :
   - full, M_only, T_only, L_only, M+T, M+L, T+L

2. Leave-one-out feature (17 conditions) :
   - mask_feat_{i} pour i ∈ {0..16}

3. Redondance : matrice de corrélation de Spearman sur les données train.
   Paires avec |ρ| > 0.9 → features candidates à la suppression.

Métrique principale : silhouette score sur le test set (sklearn.silhouette_samples).
Test statistique : Wilcoxon signé unilatéral (masked vs. full), p-threshold=0.05.

Usage
-----
    python -m experiments.ablation.run \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        --output experiments/ablation \\
        [--split test]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from ewat.encoder.stgcn import STGCNEncoder
from ewat.ontology.cooccurrence import benjamini_hochberg, holm_bonferroni
from ewat.typing.siamese import SiameseTyper

# Feature index ranges (0-based, dimension 17)
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
    "full":  IDX_M + IDX_T + IDX_L,
    "M_only": IDX_M,
    "T_only": IDX_T,
    "L_only": IDX_L,
    "M+T":   IDX_M + IDX_T,
    "M+L":   IDX_M + IDX_L,
    "T+L":   IDX_T + IDX_L,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_typer(typing_dir: Path, encoder_dir: Path, device: torch.device) -> SiameseTyper:
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


def _load_scaler(encoder_dir: Path) -> StandardScaler | None:
    import pickle
    scaler_path = encoder_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


def _load_episodes(
    cluster_manifest: dict[str, dict],
    features_root: Path,
    split: str,
    scaler: StandardScaler | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Returns (signals, adjacencies, cluster_labels) for the given split."""
    signals, adjacencies, labels = [], [], []
    for ep_id, meta in cluster_manifest.items():
        if meta.get("split") != split:
            continue
        ep_dir = features_root / ep_id
        if not ep_dir.exists():
            continue
        sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adj = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
        adj = np.nan_to_num(adj, nan=0.0)

        if scaler is not None:
            t_len, n_nodes, d = sig.shape
            flat = sig.reshape(-1, d)
            flat = np.where(np.isnan(flat), 0.0, flat)
            flat = scaler.transform(flat).astype(np.float32)
            sig = flat.reshape(t_len, n_nodes, d)
        else:
            sig = np.nan_to_num(sig, nan=0.0)

        signals.append(sig)
        adjacencies.append(adj)
        labels.append(int(meta["cluster"]))
    return signals, adjacencies, labels


@torch.no_grad()
def _embed_masked(
    typer: SiameseTyper,
    signals: list[np.ndarray],
    adjacencies: list[np.ndarray],
    active_feats: list[int],
    device: torch.device,
) -> np.ndarray:
    """Embed all episodes with a feature mask. Returns (N, d_proj) array."""
    mask = np.zeros(17, dtype=np.float32)
    mask[active_feats] = 1.0

    embeddings = []
    for sig, adj in zip(signals, adjacencies):
        sig_masked = sig * mask[np.newaxis, np.newaxis, :]  # (T, N, 17) * (17,)
        sig_t = torch.from_numpy(sig_masked).unsqueeze(0).to(device)
        adj_t = torch.from_numpy(adj).unsqueeze(0).to(device)
        z = typer.embed(sig_t, adj_t).cpu().numpy()[0]
        embeddings.append(z)
    return np.stack(embeddings)


def _silhouette(z: np.ndarray, labels: list[int]) -> tuple[float, np.ndarray]:
    y = np.array(labels)
    if len(np.unique(y)) < 2:
        return float("nan"), np.full(len(y), float("nan"))
    scores = silhouette_samples(z, y)
    return float(silhouette_score(z, y)), scores


def _wilcoxon(full_scores: np.ndarray, masked_scores: np.ndarray) -> float:
    """One-sided Wilcoxon: H₀ full ≤ masked, H₁ full > masked (masking hurts)."""
    diff = full_scores - masked_scores
    if np.all(diff == 0):
        return 1.0
    try:
        result = stats.wilcoxon(diff, alternative="greater")
        return float(result.pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def _correlation_matrix(
    cluster_manifest: dict[str, dict],
    features_root: Path,
    split: str = "train",
) -> pd.DataFrame:
    rows = []
    for ep_id, meta in cluster_manifest.items():
        if meta.get("split") != split:
            continue
        ep_dir = features_root / ep_id
        if not ep_dir.exists():
            continue
        sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        flat = sig.reshape(-1, 17)
        rows.append(flat)
    x_all = np.concatenate(rows, axis=0)
    df = pd.DataFrame(x_all, columns=FEAT_NAMES)
    return df.corr(method="spearman")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/ablation"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Split: {args.split}")

    # Load cluster manifest
    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    typer = _load_typer(args.typing_dir, args.encoder_dir, device)
    scaler = _load_scaler(args.encoder_dir)
    signals, adjacencies, labels = _load_episodes(
        cluster_manifest, args.features_root, args.split, scaler
    )
    print(f"Episodes: {len(signals)}  |  scaler={'yes' if scaler else 'no'}")

    # -----------------------------------------------------------------------
    # Full model baseline
    # -----------------------------------------------------------------------
    print("\n[baseline] full model …")
    z_full = _embed_masked(typer, signals, adjacencies, IDX_M + IDX_T + IDX_L, device)
    sil_full, scores_full = _silhouette(z_full, labels)
    print(f"  silhouette={sil_full:.4f}")

    results: list[dict] = []

    # -----------------------------------------------------------------------
    # 1. Modality ablation
    # -----------------------------------------------------------------------
    print("\n[modality ablation]")
    for cond_name, active in MODALITY_CONDITIONS.items():
        z = _embed_masked(typer, signals, adjacencies, active, device)
        sil, scores = _silhouette(z, labels)
        delta = sil - sil_full
        p = _wilcoxon(scores_full, scores) if cond_name != "full" else float("nan")
        results.append({
            "condition": cond_name,
            "type": "modality",
            "active_feats": active,
            "silhouette": sil,
            "delta_vs_full": delta,
            "p_wilcoxon": p,
            # Adjusted p-values are filled in below, after the family is closed.
            "p_holm": float("nan"),
            "p_bh": float("nan"),
            "significant_raw": bool(p < 0.05) if not np.isnan(p) else None,
            "significant_holm": None,
            "significant_bh": None,
        })

    # -----------------------------------------------------------------------
    # 2. Leave-one-out feature ablation
    # -----------------------------------------------------------------------
    print("\n[leave-one-out]")
    all_feats = list(range(17))
    for i in range(17):
        active = [j for j in all_feats if j != i]
        z = _embed_masked(typer, signals, adjacencies, active, device)
        sil, scores = _silhouette(z, labels)
        delta = sil - sil_full
        p = _wilcoxon(scores_full, scores)
        results.append({
            "condition": f"mask_{FEAT_NAMES[i]}",
            "type": "leave_one_out",
            "masked_feat_idx": i,
            "masked_feat_name": FEAT_NAMES[i],
            "active_feats": active,
            "silhouette": sil,
            "delta_vs_full": delta,
            "p_wilcoxon": p,
            "p_holm": float("nan"),
            "p_bh": float("nan"),
            "significant_raw": bool(p < 0.05),
            "significant_holm": None,
            "significant_bh": None,
        })

    # -----------------------------------------------------------------------
    # Multiple-testing correction over the joint family
    # (modality conditions other than "full" + 17 leave-one-out tests)
    # -----------------------------------------------------------------------
    family = [r for r in results if not np.isnan(r["p_wilcoxon"])]
    family_pvals = [r["p_wilcoxon"] for r in family]
    holm_adj = holm_bonferroni(family_pvals)
    bh_adj = benjamini_hochberg(family_pvals)
    for r, p_h, p_b in zip(family, holm_adj, bh_adj):
        r["p_holm"] = float(p_h)
        r["p_bh"] = float(p_b)
        r["significant_holm"] = bool(p_h < 0.05)
        r["significant_bh"] = bool(p_b < 0.05)

    print("\n[multiplicity-corrected significance]")
    print(f"  family size = {len(family)}  |  α = 0.05")
    for r in family:
        sig_h = "✓" if r["significant_holm"] else "✗"
        sig_b = "✓" if r["significant_bh"] else "✗"
        print(
            f"  {r['condition']:<32}  p_raw={r['p_wilcoxon']:.3f}  "
            f"p_holm={r['p_holm']:.3f} {sig_h}  p_bh={r['p_bh']:.3f} {sig_b}"
        )

    # -----------------------------------------------------------------------
    # 3. Correlation / redondance
    # -----------------------------------------------------------------------
    print("\n[correlation — train set]")
    corr = _correlation_matrix(cluster_manifest, args.features_root, split="train")
    redundant_pairs = []
    for i in range(17):
        for j in range(i + 1, 17):
            rho = float(corr.iloc[i, j])
            if abs(rho) >= 0.9:
                redundant_pairs.append({
                    "feat_a": FEAT_NAMES[i], "feat_b": FEAT_NAMES[j], "rho": round(rho, 4)
                })
                print(f"  |ρ|≥0.9 : {FEAT_NAMES[i]} ↔ {FEAT_NAMES[j]}  ρ={rho:.4f}")
    if not redundant_pairs:
        print("  Aucune paire redondante (|ρ| < 0.9)")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    summary = {
        "split": args.split,
        "n_episodes": len(signals),
        "silhouette_full": sil_full,
        "multiple_testing": {
            "family_size": len(family),
            "alpha": 0.05,
            "methods": ["holm", "bh"],
            "note": (
                "Holm controls FWER; Benjamini–Hochberg controls FDR. "
                "p_wilcoxon is the raw p-value before correction."
            ),
        },
        "modality_ablation": [r for r in results if r["type"] == "modality"],
        "leave_one_out": [r for r in results if r["type"] == "leave_one_out"],
        "redundant_pairs": redundant_pairs,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2, default=str))

    # Human-readable report
    mod_results = [r for r in results if r["type"] == "modality"]
    loo_results = sorted(
        [r for r in results if r["type"] == "leave_one_out"],
        key=lambda r: r["delta_vs_full"],
    )

    lines = [
        "# Ablation EWAT — Impact modalités & features\n",
        f"Split : {args.split}  |  N={len(signals)}  |  silhouette(full)={sil_full:.4f}\n",
        f"Famille de tests : {len(family)} (modalités + LOO).  "
        "Corrections multiples : Holm (FWER) et Benjamini–Hochberg (FDR).\n",
        "## 1. Ablation par modalité\n",
        f"{'Condition':<10} {'Silhouette':>11} {'Δ':>8} {'p_raw':>7} "
        f"{'p_holm':>8} {'p_bh':>7} {'H':>2} {'B':>2}",
        "-" * 64,
    ]
    for r in mod_results:
        p_raw = (
            f"{r['p_wilcoxon']:.3f}" if not np.isnan(r["p_wilcoxon"]) else "  —"
        )
        p_h = f"{r['p_holm']:.3f}" if not np.isnan(r["p_holm"]) else "  —"
        p_b = f"{r['p_bh']:.3f}" if not np.isnan(r["p_bh"]) else "  —"
        sig_h = "✓" if r["significant_holm"] else (
            "✗" if r["significant_holm"] is not None else "—"
        )
        sig_b = "✓" if r["significant_bh"] else ("✗" if r["significant_bh"] is not None else "—")
        lines.append(
            f"{r['condition']:<10} {r['silhouette']:>11.4f} {r['delta_vs_full']:>+8.3f}"
            f" {p_raw:>7} {p_h:>8} {p_b:>7} {sig_h:>2} {sig_b:>2}"
        )

    lines += [
        "\n## 2. Leave-one-out (impact négatif croissant)\n",
        f"{'Feature':<30} {'Silhouette':>11} {'Δ':>8} {'p_raw':>7} "
        f"{'p_holm':>8} {'p_bh':>7} {'H':>2} {'B':>2}",
        "-" * 84,
    ]
    for r in loo_results:
        p_h = f"{r['p_holm']:.3f}" if not np.isnan(r["p_holm"]) else "  —"
        p_b = f"{r['p_bh']:.3f}" if not np.isnan(r["p_bh"]) else "  —"
        sig_h = "✓" if r["significant_holm"] else "✗"
        sig_b = "✓" if r["significant_bh"] else "✗"
        lines.append(
            f"{r['masked_feat_name']:<30} {r['silhouette']:>11.4f}"
            f" {r['delta_vs_full']:>+8.3f} {r['p_wilcoxon']:>7.3f} {p_h:>8} {p_b:>7}"
            f" {sig_h:>2} {sig_b:>2}"
        )

    if redundant_pairs:
        lines += ["\n## 3. Paires redondantes (|ρ| ≥ 0.9)\n"]
        for p in redundant_pairs:
            lines.append(f"  {p['feat_a']} ↔ {p['feat_b']}  ρ={p['rho']}")
    else:
        lines.append("\n## 3. Redondance\n\nAucune paire (|ρ| ≥ 0.9).")

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


if __name__ == "__main__":
    main()
