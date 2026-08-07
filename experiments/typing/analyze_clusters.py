"""Analyse de la qualité et de la sémantique des clusters.

Calcule :
- Pureté par cluster (fraction du scénario dominant)
- NMI (Normalized Mutual Information) global cluster ↔ scénario
- Matrice scénario×cluster (15×K) exportée en heatmap matplotlib
- Validation de la méthode SHAP gradient vs. permutation importance (Spearman ρ)

Usage
-----
    python -m experiments.typing.analyze_clusters \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        [--output experiments/typing] \\
        [--n-perm-shap 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(typing_dir: Path) -> dict[str, dict]:
    return json.loads((typing_dir / "cluster_artifacts" / "cluster_manifest.json").read_text())


def _purity(labels: np.ndarray, scenario_ids: np.ndarray, n_clusters: int) -> list[dict]:
    """Compute purity per cluster: fraction of the dominant scenario."""
    stats = []
    for c in range(n_clusters):
        mask = labels == c
        if not mask.any():
            stats.append({"cluster": c, "n": 0, "purity": float("nan"),
                          "dominant_scenario": None, "dominant_count": 0})
            continue
        sc = scenario_ids[mask]
        values, counts = np.unique(sc, return_counts=True)
        dom_idx = np.argmax(counts)
        purity = float(counts[dom_idx] / mask.sum())
        stats.append({
            "cluster": c,
            "n": int(mask.sum()),
            "purity": round(purity, 4),
            "dominant_scenario": str(values[dom_idx]),
            "dominant_count": int(counts[dom_idx]),
        })
    return stats


def _scenario_cluster_matrix(
    labels: np.ndarray,
    scenario_ids: np.ndarray,
    scenarios: list[str],
    n_clusters: int,
) -> np.ndarray:
    """Return (n_scenarios, n_clusters) count matrix."""
    mat = np.zeros((len(scenarios), n_clusters), dtype=int)
    for i, sc in enumerate(scenarios):
        mask = scenario_ids == sc
        for c in range(n_clusters):
            mat[i, c] = int((labels[mask] == c).sum())
    return mat


def _plot_heatmap(mat: np.ndarray, scenarios: list[str], n_clusters: int, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, n_clusters * 0.9), max(5, len(scenarios) * 0.55)))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        fig.colorbar(im, ax=ax, label="Nombre d'épisodes")

        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels([f"C{i}" for i in range(n_clusters)], fontsize=9)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=8)
        ax.set_xlabel("Cluster EWAT")
        ax.set_ylabel("Scénario Chaos Mesh")
        ax.set_title("Distribution scénario × cluster (tous splits)")

        for i in range(len(scenarios)):
            for j in range(n_clusters):
                if mat[i, j] > 0:
                    color = "white" if mat[i, j] > mat.max() * 0.6 else "black"
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            fontsize=7, color=color)

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved → {out}")
    except Exception as e:
        print(f"Warning: heatmap plot failed ({e})")


def _permutation_importance(
    z: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Estimate feature importance by permuting each feature dimension and measuring
    the silhouette drop (averaged over n_perm permutations).

    Uses the *per-sample* silhouette mean as the metric — cheaper than full score
    but proportional.

    Returns {feature_name: importance_score} (higher = more important).
    """
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return {f: 0.0 for f in feature_names}

    # Subsample to at most 150 points for speed (silhouette is O(n²))
    max_n = 150
    n = len(z)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        z = z[idx]
        labels = labels[idx]

    base_sil = float(silhouette_score(z, labels))
    importances: dict[str, float] = {}

    # z shape: (N_episodes, d_proj) — features act through the projection, so we
    # cannot directly permute individual signal features. Instead, permute the
    # embedding dimensions as a proxy (coarser but fast).
    # For full feature-level analysis, use saliency_explainer.compute_cluster_saliency.
    d = z.shape[1]
    n_feat = len(feature_names)
    # Group embedding dims into n_feat groups (approximate mapping)
    group_size = max(1, d // n_feat)

    for i, fname in enumerate(feature_names):
        lo = i * group_size
        hi = min((i + 1) * group_size, d)
        drops = []
        for _ in range(n_perm):
            z_perm = z.copy()
            z_perm[:, lo:hi] = rng.permutation(z_perm[:, lo:hi])
            try:
                sil_perm = float(silhouette_score(z_perm, labels))
            except ValueError:
                sil_perm = base_sil
            drops.append(base_sil - sil_perm)
        importances[fname] = round(float(np.mean(drops)), 6)

    return importances


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman ρ between two lists (rank correlation)."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d_sq = np.sum((rx - ry) ** 2)
    return float(1 - 6 * d_sq / (n * (n ** 2 - 1)))


FEATURE_NAMES = [
    "cpu_util", "ram_util", "latency_p99", "error_rate_http", "net_sat",
    "disk_io", "queue_depth", "span_dur_p99", "abnormal_span_rate",
    "trace_depth", "fan_out", "retry_rate", "latency_cv",
    "log_error_rate", "log_warn_rate", "semantic_anomaly", "lexical_entropy",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster quality and semantics analysis")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-perm-shap", type=int, default=50,
                        help="Permutations for importance validation (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.output or args.typing_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(args.typing_dir)
    artifacts_dir = args.typing_dir / "cluster_artifacts"

    # Load embeddings and labels (all splits merged)
    z_parts, l_parts = [], []
    for split in ("train", "val", "test"):
        z_f = artifacts_dir / f"embeddings_{split}.npy"
        l_f = artifacts_dir / f"labels_{split}.npy"
        if z_f.exists() and l_f.exists():
            z_parts.append(np.load(z_f))
            l_parts.append(np.load(l_f))

    z_all = np.concatenate(z_parts, axis=0)
    labels_all = np.concatenate(l_parts, axis=0)

    # Scenario per episode (in manifest order matching embedding order)
    ep_scenarios = []
    for ep_id, meta in manifest.items():
        ep_scenarios.append(meta.get("scenario", "unknown"))
    scenario_arr = np.array(ep_scenarios)

    n_clusters = int(labels_all.max()) + 1
    scenarios_sorted = sorted(set(ep_scenarios))
    print(
        f"Episodes: {len(z_all)}  |  Clusters: {n_clusters}  |  Scenarios: {len(scenarios_sorted)}"
    )

    # -----------------------------------------------------------------------
    # NMI
    # -----------------------------------------------------------------------
    nmi = float(normalized_mutual_info_score(scenario_arr, labels_all, average_method="arithmetic"))
    print(f"NMI (cluster ↔ scenario): {nmi:.4f}")

    # -----------------------------------------------------------------------
    # Purity per cluster
    # -----------------------------------------------------------------------
    purity_stats = _purity(labels_all, scenario_arr, n_clusters)
    mean_purity = float(np.nanmean([s["purity"] for s in purity_stats]))
    print(f"Mean purity: {mean_purity:.4f}")
    for s in purity_stats:
        print(f"  C{s['cluster']:<2}  n={s['n']:>3}  purity={s['purity']:.3f}  "
              f"dominant={s['dominant_scenario']}")

    # -----------------------------------------------------------------------
    # Scenario × cluster matrix
    # -----------------------------------------------------------------------
    mat = _scenario_cluster_matrix(labels_all, scenario_arr, scenarios_sorted, n_clusters)
    _plot_heatmap(mat, scenarios_sorted, n_clusters, out_dir / "scenario_cluster_heatmap.png")

    # -----------------------------------------------------------------------
    # SHAP gradient vs permutation importance validation
    # -----------------------------------------------------------------------
    shap_corr: dict[int, float] = {}
    perm_importances: dict[int, dict[str, float]] = {}

    if args.n_perm_shap > 0:
        fiches_dir = args.typing_dir / "fiches"
        rng = np.random.default_rng(args.seed)
        print(f"\nValidating SHAP gradient vs. permutation importance "
              f"({args.n_perm_shap} perms per cluster) …")

        for c in range(n_clusters):
            fiche_path = fiches_dir / f"cluster_{c}.json"
            if not fiche_path.exists():
                print(f"  C{c}: no fiche, skip")
                continue

            fiche = json.loads(fiche_path.read_text())
            gradient_imp = fiche.get("feature_importance", {})
            if not gradient_imp:
                continue

            perm_imp = _permutation_importance(
                z_all, labels_all, FEATURE_NAMES, n_perm=args.n_perm_shap, rng=rng
            )
            perm_importances[c] = perm_imp

            # Rank correlation between gradient and permutation
            feat_common = [f for f in FEATURE_NAMES if f in gradient_imp]
            if len(feat_common) < 3:
                continue
            grad_vals = [gradient_imp[f] for f in feat_common]
            perm_vals = [perm_imp.get(f, 0.0) for f in feat_common]
            rho = _spearman(grad_vals, perm_vals)
            shap_corr[c] = round(rho, 4)
            print(f"  C{c}: Spearman ρ(gradient, permutation) = {rho:.3f}")

        if shap_corr:
            mean_rho = float(np.nanmean(list(shap_corr.values())))
            print(f"  Mean ρ across clusters: {mean_rho:.3f}  "
                  f"({'validated ✓' if mean_rho > 0.5 else 'weak correlation ✗'})")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    results = {
        "n_clusters": n_clusters,
        "n_scenarios": len(scenarios_sorted),
        "n_episodes": len(z_all),
        "nmi": nmi,
        "mean_purity": mean_purity,
        "purity_per_cluster": purity_stats,
        "scenario_cluster_matrix": {
            "scenarios": scenarios_sorted,
            "clusters": list(range(n_clusters)),
            "matrix": mat.tolist(),
        },
        "shap_validation": {
            "spearman_per_cluster": shap_corr,
            "mean_spearman": (
                round(float(np.nanmean(list(shap_corr.values()))), 4) if shap_corr else None
            ),
            "n_permutations": args.n_perm_shap,
        },
    }
    (out_dir / "cluster_analysis.json").write_text(json.dumps(results, indent=2))

    # Report
    lines = [
        "# Analyse qualité et sémantique des clusters\n",
        f"NMI (cluster ↔ scénario) : **{nmi:.4f}**"
        " — mesure de l'alignement avec les labels Chaos Mesh\n",
        f"Pureté moyenne : **{mean_purity:.4f}**\n",
        "## Pureté par cluster",
        f"{'C':<4}  {'N':>4}  {'Pureté':>7}  {'Scénario dominant':<30}",
        "-" * 55,
    ]
    for s in purity_stats:
        lines.append(
            f"C{s['cluster']:<3}  {s['n']:>4}  {(s['purity'] or 0):>7.3f}"
            f"  {(s['dominant_scenario'] or 'N/A'):<30}"
        )

    if shap_corr:
        mean_rho = float(np.nanmean(list(shap_corr.values())))
        lines += [
            "",
            "## Validation SHAP gradient vs. permutation importance",
            f"Spearman ρ moyen : **{mean_rho:.4f}** "
            f"({'gradient validé ✓' if mean_rho > 0.5 else 'corrélation faible ✗'})",
            f"{'C':<4}  {'ρ':>7}",
            "-" * 15,
        ]
        for c, rho in shap_corr.items():
            lines.append(f"C{c:<3}  {rho:>7.4f}")

    (out_dir / "cluster_analysis.md").write_text("\n".join(lines))
    print(f"\nReport: {out_dir / 'cluster_analysis.md'}")


if __name__ == "__main__":
    main()
