"""Ablation rigoureuse — réentraînement complet par condition.

Contrairement à ``experiments.ablation.run`` qui masque les features à
l'inférence sur le checkpoint existant, ce script **réentraîne** l'encodeur
puis le typer pour chaque condition. C'est la version honnête de l'ablation :
elle mesure ce que l'on perd quand le modèle n'a jamais vu les features
ablées, pas seulement quand on les met à zéro après coup.

Conditions
----------
* 7 modalités (full, M_only, T_only, L_only, M+T, M+L, T+L).
* 17 leave-one-out features.
* 5 graines (par défaut) — donc :math:`24 \\times 5 = 120` runs complets.

Pipeline par run
----------------
1. **Mask au niveau dataset** : on reconstruit dynamiquement un sous-ensemble
   du signal qui ne contient que les features actives (les autres sont mises
   à zéro AVANT le scaler — ce qui désactive l'information côté backprop).
2. **Encoder retrain** : reconstruction self-supervised (cf.
   ``experiments.encoder.train.run``).
3. **Typing retrain** : siamese contrastif (cf.
   ``experiments.typing.train.run``).
4. **Eval** : silhouette test + IC bootstrap BCa.

Coût compute
------------
Largement dominé par le forward STGCN (≈ ``epochs_enc + epochs_typ`` epochs
par condition × seed). Avec les hyperparamètres par défaut (epochs réduits)
le tour complet se fait en quelques heures sur GPU. Ajuster ``epochs_enc``
et ``epochs_typ`` pour le compromis fidélité/temps.

Usage
-----
    python -m experiments.ablation.retrain_run \\
        --dataset data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --output experiments/ablation_retrain \\
        --seeds 5 --epochs-enc 30 --epochs-typ 20

Le résultat agrégé inclut, par condition, la moyenne ± SE du silhouette test
sur les graines, et le test de Wilcoxon (avec correction Holm/BH) contre la
condition ``full``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ewat.ontology.cooccurrence import benjamini_hochberg, holm_bonferroni
from experiments.encoder.train import run as encoder_run
from experiments.typing.train import run as typing_run

FEAT_NAMES = [
    "cpu_util", "ram_util", "latency_p99", "error_rate_http",
    "net_sat", "disk_io", "queue_depth",
    "span_dur_p99", "abnormal_span_rate", "trace_depth",
    "fan_out", "retry_rate", "latency_cv",
    "log_error_rate", "log_warn_rate", "semantic_anomaly", "lexical_entropy",
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


# --------------------------------------------------------------------------- #
# Mask propagation — patch the feature store for one condition
# --------------------------------------------------------------------------- #

def _build_masked_feature_store(
    src_features_root: Path,
    dst_features_root: Path,
    active_feats: list[int],
) -> None:
    """Copy ``src_features_root`` to ``dst_features_root`` while zero-ing the
    inactive feature columns of every ``signal.npz``. ``adjacency.npz`` and
    ``labels.parquet`` are passed through unchanged.

    Hard-linking is used when possible to keep disk usage bounded.
    """
    mask = np.zeros(17, dtype=np.float32)
    for i in active_feats:
        mask[i] = 1.0

    if dst_features_root.exists():
        shutil.rmtree(dst_features_root)
    dst_features_root.mkdir(parents=True)

    for ep_dir in sorted(src_features_root.iterdir()):
        if not ep_dir.is_dir():
            continue
        out_dir = dst_features_root / ep_dir.name
        out_dir.mkdir()

        sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        sig_masked = sig * mask[None, None, :]
        np.savez(out_dir / "signal.npz", signal=sig_masked)

        for fname in ("adjacency.npz", "labels.parquet"):
            src = ep_dir / fname
            if not src.exists():
                continue
            dst = out_dir / fname
            try:
                dst.hardlink_to(src)
            except (OSError, AttributeError):
                shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# Single condition × seed run
# --------------------------------------------------------------------------- #

def _run_one(
    *,
    cond_name: str,
    active_feats: list[int],
    seed: int,
    dataset: Path,
    features_root: Path,
    work_root: Path,
    epochs_enc: int,
    epochs_typ: int,
    batch_size: int,
    n_bootstrap: int,
) -> dict:
    """Retrain encoder + typing for one (condition, seed) pair and return
    silhouette metrics.
    """
    cond_dir = work_root / f"{cond_name}__seed{seed}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    masked_features = cond_dir / "features"
    _build_masked_feature_store(features_root, masked_features, active_feats)

    encoder_dir = cond_dir / "encoder"
    typing_dir = cond_dir / "typing"

    enc_args = argparse.Namespace(
        dataset=dataset,
        features_root=masked_features,
        output=encoder_dir,
        epochs=epochs_enc,
        lr=1e-3,
        batch_size=batch_size,
        patience=max(5, epochs_enc // 4),
        d_hidden=64,
        d_embed=64,
        seed=seed,
    )
    encoder_run(enc_args)

    typ_args = argparse.Namespace(
        dataset=dataset,
        features_root=masked_features,
        encoder_checkpoint=encoder_dir / "checkpoints" / "best_encoder.pt",
        output=typing_dir,
        epochs=epochs_typ,
        lr=1e-4,
        batch_size=batch_size,
        patience=max(5, epochs_typ // 4),
        d_proj=32,
        margin=1.0,
        n_neg_per_anchor=5,
        freeze_encoder=False,
        k_range_max=12,
        n_gap_refs=5,
        n_shap_bg=10,
        n_bootstrap=n_bootstrap,
        seed=seed,
        eval_only=False,
        mining="random",
        mining_warmup_epochs=0,
        mining_pool_size=0,
    )
    typing_run(typ_args)

    results_path = typing_dir / "results.json"
    summary = json.loads(results_path.read_text())

    out = {
        "condition": cond_name,
        "seed": seed,
        "active_feats": active_feats,
        "k_optimal": summary.get("k_optimal"),
        "silhouette_train": summary.get("silhouette_train"),
        "silhouette_val": summary.get("silhouette_val"),
        "silhouette_test": summary.get("silhouette_test"),
        "h1_pass": summary.get("h1_pass"),
        "silhouette_ci_test": summary.get("silhouette_ci_test", {}),
    }

    if cond_name != "full":
        labels_test = np.load(typing_dir / "cluster_artifacts" / "labels_test.npy")
        z_test = np.load(typing_dir / "cluster_artifacts" / "embeddings_test.npy")
        from sklearn.metrics import silhouette_samples
        if len(np.unique(labels_test)) >= 2:
            out["silhouette_samples_test"] = silhouette_samples(z_test, labels_test).tolist()
        else:
            out["silhouette_samples_test"] = []
    return out


# --------------------------------------------------------------------------- #
# Aggregation across seeds
# --------------------------------------------------------------------------- #

def _aggregate(per_seed_results: list[dict], rng: np.random.Generator) -> dict:
    sils = np.asarray(
        [r["silhouette_test"] for r in per_seed_results if r["silhouette_test"] is not None],
        dtype=float,
    )
    sils = sils[~np.isnan(sils)]
    n = len(sils)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "se": float("nan")}
    mean = float(sils.mean())
    std = float(sils.std(ddof=1)) if n > 1 else float("nan")
    se = float(std / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "std": std, "se": se, "values": sils.tolist()}


def _wilcoxon_vs_full(
    full_per_seed: list[dict], cond_per_seed: list[dict]
) -> float:
    from scipy import stats
    aligned_full, aligned_cond = [], []
    by_seed_full = {r["seed"]: r["silhouette_test"] for r in full_per_seed}
    for r in cond_per_seed:
        s = r["seed"]
        if s in by_seed_full and r["silhouette_test"] is not None:
            sf = by_seed_full[s]
            sc = r["silhouette_test"]
            if sf is not None and not (np.isnan(sf) or np.isnan(sc)):
                aligned_full.append(sf)
                aligned_cond.append(sc)
    if len(aligned_full) < 3:
        return float("nan")
    diff = np.asarray(aligned_full) - np.asarray(aligned_cond)
    if np.all(diff == 0):
        return 1.0
    try:
        result = stats.wilcoxon(diff, alternative="greater")
        return float(result.pvalue)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation rigoureuse (retrain)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/ablation_retrain"))
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds per condition")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="seeds = [seed_base, seed_base+1, …]")
    parser.add_argument("--epochs-enc", type=int, default=30)
    parser.add_argument("--epochs-typ", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap resamples for silhouette CIs in sub-runs")
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run (default: all)")
    parser.add_argument("--skip-loo", action="store_true",
                        help="Skip the 17 leave-one-out runs (modality only)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_base + i for i in range(args.seeds)]
    work_root = args.output / "runs"
    work_root.mkdir(exist_ok=True)

    # Build the full condition list.
    conditions: list[tuple[str, list[int]]] = list(MODALITY_CONDITIONS.items())
    if not args.skip_loo:
        all_feats = list(range(17))
        for i in range(17):
            cond_name = f"mask_{FEAT_NAMES[i]}"
            active = [j for j in all_feats if j != i]
            conditions.append((cond_name, active))

    if args.conditions:
        conditions = [(n, a) for n, a in conditions if n in args.conditions]
        if not conditions:
            raise SystemExit(f"no condition matched --conditions {args.conditions}")

    # ----- Run grid -----
    grid: dict[str, list[dict]] = {name: [] for name, _ in conditions}
    print(f"Running {len(conditions)} conditions × {len(seeds)} seeds = "
          f"{len(conditions) * len(seeds)} retrains")

    for cond_name, active in conditions:
        print(f"\n=== Condition: {cond_name} (|active|={len(active)}) ===")
        for seed in seeds:
            print(f"  -- seed {seed}")
            try:
                row = _run_one(
                    cond_name=cond_name,
                    active_feats=active,
                    seed=seed,
                    dataset=args.dataset,
                    features_root=args.features_root,
                    work_root=work_root,
                    epochs_enc=args.epochs_enc,
                    epochs_typ=args.epochs_typ,
                    batch_size=args.batch_size,
                    n_bootstrap=args.n_bootstrap,
                )
            except Exception as exc:
                print(f"     [error] {exc}")
                row = {
                    "condition": cond_name,
                    "seed": seed,
                    "active_feats": active,
                    "error": str(exc),
                    "silhouette_test": float("nan"),
                }
            grid[cond_name].append(row)

    # ----- Aggregate per condition -----
    rng = np.random.default_rng(args.bootstrap_seed)
    aggregated: dict[str, dict] = {}
    for cond_name, runs in grid.items():
        agg = _aggregate(runs, rng)
        aggregated[cond_name] = agg

    # ----- Pairwise tests vs full + multiple-testing correction -----
    full_runs = grid.get("full", [])
    family_names: list[str] = []
    family_pvals: list[float] = []
    for cond_name, runs in grid.items():
        if cond_name == "full":
            aggregated[cond_name]["p_wilcoxon"] = float("nan")
            aggregated[cond_name]["p_holm"] = float("nan")
            aggregated[cond_name]["p_bh"] = float("nan")
            continue
        p = _wilcoxon_vs_full(full_runs, runs)
        aggregated[cond_name]["p_wilcoxon"] = p
        if not np.isnan(p):
            family_names.append(cond_name)
            family_pvals.append(p)

    holm_adj = holm_bonferroni(family_pvals)
    bh_adj = benjamini_hochberg(family_pvals)
    for name, p_h, p_b in zip(family_names, holm_adj, bh_adj):
        aggregated[name]["p_holm"] = float(p_h)
        aggregated[name]["p_bh"] = float(p_b)
        aggregated[name]["significant_holm"] = bool(p_h < 0.05)
        aggregated[name]["significant_bh"] = bool(p_b < 0.05)

    # ----- Save -----
    payload = {
        "config": {
            "seeds": seeds,
            "epochs_enc": args.epochs_enc,
            "epochs_typ": args.epochs_typ,
            "batch_size": args.batch_size,
            "n_bootstrap": args.n_bootstrap,
        },
        "per_run": grid,
        "aggregated": aggregated,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, default=str))

    # ----- Markdown report -----
    lines = [
        "# Ablation rigoureuse (retrain) — résumé\n",
        f"Seeds: {seeds}",
        f"Epochs encoder: {args.epochs_enc}  |  Epochs typing: {args.epochs_typ}\n",
        "## Silhouette test moyenne ± SE par condition (Wilcoxon vs full)\n",
        f"{'Condition':<32} {'n':>3} {'mean':>8} {'SE':>8} "
        f"{'p_raw':>7} {'p_holm':>8} {'p_bh':>7} {'H':>2} {'B':>2}",
        "-" * 88,
    ]
    full_mean = aggregated.get("full", {}).get("mean", float("nan"))
    for cond_name, agg in aggregated.items():
        n = agg.get("n", 0)
        mean = agg.get("mean", float("nan"))
        se = agg.get("se", float("nan"))
        p_raw = agg.get("p_wilcoxon", float("nan"))
        p_h = agg.get("p_holm", float("nan"))
        p_b = agg.get("p_bh", float("nan"))
        sig_h = agg.get("significant_holm")
        sig_b = agg.get("significant_bh")

        def _fmt_p(v: float) -> str:
            return f"{v:.3f}" if not np.isnan(v) else "  —"

        h = "✓" if sig_h else ("✗" if sig_h is False else "—")
        b = "✓" if sig_b else ("✗" if sig_b is False else "—")
        lines.append(
            f"{cond_name:<32} {n:>3} "
            f"{mean:>8.4f} {se:>8.4f} {_fmt_p(p_raw):>7} "
            f"{_fmt_p(p_h):>8} {_fmt_p(p_b):>7} {h:>2} {b:>2}"
        )

    lines.append(
        f"\nNote: H = Holm (FWER<0.05), B = BH-FDR<0.05. "
        f"Référence: silhouette(full) moyenne = {full_mean:.4f}."
    )
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nResults: {args.output / 'results.md'}")


if __name__ == "__main__":
    main()
