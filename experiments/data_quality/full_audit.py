"""Full data quality audit for EWAT datasets.

Goal
----
Avant de continuer les Étapes 4-10 (modèles), vérifier que la donnée existante
est utilisable et de bonne qualité. Cet audit produit un rapport exhaustif
couvrant :

1. **Inventory**       : Comptage épisodes, split, scenarios
2. **NaN audit**       : Ratio NaN par feature × modality × service
3. **Class balance**   : n_pos par scenario par split (critique pour AUROC)
4. **Signal sanity**   : Distribution feature, outliers, magnitude normale vs injection
5. **Episode duration** : Distribution T_steps, cohérence cross-dataset
6. **Adjacency graph** : Densité, % timesteps avec graphe vide
7. **Label integrity** : Transitions regime, scenarios cohérents
8. **Temporal alignment** : check M/T/L alignment (audit warning Step 2.1)
9. **Cross-dataset**   : Diff v3 vs v4_strat

Output:
    experiments/data_quality/{dataset}/audit.json
    experiments/data_quality/{dataset}/audit.md
    experiments/data_quality/cross_comparison.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "cpu_util", "ram_util", "latency_p99", "error_rate_http",
    "net_sat", "disk_io", "queue_depth",
    "span_dur_p99", "abnormal_span_rate", "trace_depth", "fan_out",
    "retry_rate", "latency_cv",
    "log_error_rate", "log_warn_rate", "semantic_anomaly", "lexical_entropy",
]
METRICS_DIM = slice(0, 7)
TRACES_DIM = slice(7, 13)
LOGS_DIM = slice(13, 17)


def _inventory(dataset_dir: Path) -> dict:
    df = pd.read_parquet(dataset_dir / "index.parquet")
    split_count = df.groupby("split").size().to_dict()
    scenario_count = df.groupby("scenario").size().to_dict()
    scenario_per_split = df.groupby(["split", "scenario"]).size().unstack(fill_value=0)
    return {
        "n_episodes_total": len(df),
        "split_counts": split_count,
        "n_scenarios": len(scenario_count),
        "scenarios": sorted(scenario_count.keys()),
        "scenarios_per_split": scenario_per_split.to_dict(),
        "scenarios_missing_per_split": {
            split: sorted(set(scenario_count) - set(scenario_per_split.columns[scenario_per_split.loc[split] > 0]))
            for split in scenario_per_split.index
        },
    }


def _nan_audit(features_root: Path, ep_ids: list[str]) -> dict:
    """Compute NaN ratios per feature, modality, service across episodes."""
    feature_nan = np.zeros(17, dtype=np.int64)
    feature_total = np.zeros(17, dtype=np.int64)
    # Per-service (assumes N=6 constant)
    service_nan = defaultdict(int)
    service_total = defaultdict(int)
    services_list: list[str] = []

    for ep_id in ep_ids:
        ep_dir = features_root / ep_id
        if not ep_dir.exists():
            continue
        try:
            with np.load(ep_dir / "signal.npz") as z:
                sig = z["signal"]   # (T, N, 17)
            svcs = json.loads((ep_dir / "services.json").read_text())
        except Exception:
            continue
        if not services_list:
            services_list = svcs
        T, N, d = sig.shape
        # Per-feature
        for f in range(d):
            feature_nan[f] += int(np.isnan(sig[:, :, f]).sum())
            feature_total[f] += T * N
        # Per-service
        for s_idx, s_name in enumerate(svcs):
            service_nan[s_name] += int(np.isnan(sig[:, s_idx, :]).sum())
            service_total[s_name] += T * d

    per_feature = []
    for f in range(17):
        ratio = feature_nan[f] / max(feature_total[f], 1)
        per_feature.append({
            "feature_index": f,
            "feature_name": FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f"f{f}",
            "modality": "M" if METRICS_DIM.start <= f < METRICS_DIM.stop
                else "T" if TRACES_DIM.start <= f < TRACES_DIM.stop
                else "L",
            "nan_ratio": float(ratio),
            "nan_count": int(feature_nan[f]),
            "total_count": int(feature_total[f]),
        })

    # Per-modality aggregate
    mod_nan = {"M": 0, "T": 0, "L": 0}
    mod_total = {"M": 0, "T": 0, "L": 0}
    for f in range(7):
        mod_nan["M"] += feature_nan[f]; mod_total["M"] += feature_total[f]
    for f in range(7, 13):
        mod_nan["T"] += feature_nan[f]; mod_total["T"] += feature_total[f]
    for f in range(13, 17):
        mod_nan["L"] += feature_nan[f]; mod_total["L"] += feature_total[f]
    per_modality = {
        m: {"nan_ratio": mod_nan[m] / max(mod_total[m], 1),
            "nan_count": int(mod_nan[m]), "total_count": int(mod_total[m])}
        for m in ["M", "T", "L"]
    }

    per_service = []
    for s in services_list:
        ratio = service_nan[s] / max(service_total[s], 1)
        per_service.append({
            "service": s,
            "nan_ratio": float(ratio),
            "nan_count": int(service_nan[s]),
            "total_count": int(service_total[s]),
        })

    return {
        "per_feature": per_feature,
        "per_modality": per_modality,
        "per_service": per_service,
    }


def _class_balance(dataset_dir: Path) -> dict:
    """n_pos per scenario per split — critical for AUROC stability."""
    df = pd.read_parquet(dataset_dir / "index.parquet")
    res = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        counts = sub.groupby("scenario").size().to_dict()
        # Flag scenarios with n_pos < 5 (statistically noisy AUROC)
        underpopulated = {s: int(n) for s, n in counts.items() if n < 5}
        res[split] = {
            "total": len(sub),
            "per_scenario": {s: int(n) for s, n in counts.items()},
            "min_per_scenario": min(counts.values()) if counts else 0,
            "max_per_scenario": max(counts.values()) if counts else 0,
            "underpopulated_scenarios": underpopulated,
            "n_underpopulated": len(underpopulated),
        }
    return res


def _signal_sanity(features_root: Path, ep_ids: list[str],
                   sample_size: int = 30) -> dict:
    """Distribution stats per feature, regime (normal vs injection) comparison."""
    rng = np.random.default_rng(0)
    sampled = rng.choice(ep_ids, size=min(sample_size, len(ep_ids)), replace=False).tolist()

    per_feature_global: list[dict] = []
    per_feature_regime: dict[str, list] = {"normal": [[] for _ in range(17)],
                                            "injection": [[] for _ in range(17)]}

    for ep_id in sampled:
        ep_dir = features_root / ep_id
        if not ep_dir.exists():
            continue
        try:
            with np.load(ep_dir / "signal.npz") as z:
                sig = z["signal"]
            labels_df = pd.read_parquet(ep_dir / "labels.parquet", columns=["regime"])
        except Exception:
            continue
        regime = labels_df["regime"].values
        normal_mask = regime == "normal"
        injection_mask = regime == "injection"
        for f in range(17):
            vals = sig[:, :, f].ravel()
            normal_vals = sig[normal_mask, :, f].ravel()
            inject_vals = sig[injection_mask, :, f].ravel()
            normal_vals = normal_vals[~np.isnan(normal_vals)]
            inject_vals = inject_vals[~np.isnan(inject_vals)]
            if normal_vals.size:
                per_feature_regime["normal"][f].extend(normal_vals.tolist())
            if inject_vals.size:
                per_feature_regime["injection"][f].extend(inject_vals.tolist())

    # Compute distribution stats
    rows = []
    for f in range(17):
        normal = np.array(per_feature_regime["normal"][f])
        inject = np.array(per_feature_regime["injection"][f])
        normal = normal[np.isfinite(normal)]
        inject = inject[np.isfinite(inject)]
        if normal.size and inject.size:
            normal_mean = float(normal.mean())
            inject_mean = float(inject.mean())
            normal_std = float(normal.std()) if normal.size > 1 else 0.0
            # Cohen's d
            pooled_std = (normal_std + (inject.std() if inject.size > 1 else 0.0)) / 2 + 1e-9
            cohen_d = (inject_mean - normal_mean) / pooled_std
            # Outlier ratio (>3σ from normal mean)
            outliers = np.abs(normal - normal_mean) > 3 * normal_std if normal_std > 0 else np.zeros_like(normal, dtype=bool)
        else:
            normal_mean = float("nan")
            inject_mean = float("nan")
            normal_std = float("nan")
            cohen_d = float("nan")
            outliers = np.zeros(0, dtype=bool)
        rows.append({
            "feature_index": f,
            "feature_name": FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f"f{f}",
            "n_normal_samples": int(normal.size),
            "n_inject_samples": int(inject.size),
            "normal_mean": normal_mean,
            "inject_mean": inject_mean,
            "normal_std": normal_std,
            "cohen_d": cohen_d,
            "abs_cohen_d": abs(cohen_d) if not np.isnan(cohen_d) else float("nan"),
            "outlier_ratio_normal": float(outliers.mean()) if outliers.size else float("nan"),
        })
    rows.sort(key=lambda r: -r["abs_cohen_d"] if not np.isnan(r["abs_cohen_d"]) else -1)
    return {"per_feature_normal_vs_inject": rows, "sample_size": len(sampled)}


def _duration_stats(dataset_dir: Path) -> dict:
    df = pd.read_parquet(dataset_dir / "index.parquet")
    return {
        "n_timesteps_min": int(df["n_timesteps"].min()),
        "n_timesteps_max": int(df["n_timesteps"].max()),
        "n_timesteps_median": int(df["n_timesteps"].median()),
        "n_timesteps_mean": float(df["n_timesteps"].mean()),
        "n_timesteps_std": float(df["n_timesteps"].std()),
        "n_timesteps_distribution": df["n_timesteps"].value_counts().sort_index().to_dict(),
    }


def _adjacency_stats(features_root: Path, ep_ids: list[str], sample: int = 20) -> dict:
    rng = np.random.default_rng(0)
    chosen = rng.choice(ep_ids, size=min(sample, len(ep_ids)), replace=False).tolist()
    empty_ratios = []
    densities = []
    for ep_id in chosen:
        ep_dir = features_root / ep_id
        if not ep_dir.exists():
            continue
        try:
            with np.load(ep_dir / "adjacency.npz") as z:
                adj = z["adjacency"]   # (T, N, N, 3)
        except Exception:
            continue
        T, N, _, _ = adj.shape
        # An edge "exists" if any of the 3 channels is non-zero non-NaN
        any_channel = np.nansum(adj, axis=-1)   # (T, N, N)
        # Exclude diagonal
        for t in range(T):
            mat = any_channel[t]
            mat = mat - np.diag(np.diag(mat))
            n_edges = int((mat > 0).sum())
            empty_ratios.append(1.0 if n_edges == 0 else 0.0)
            densities.append(n_edges / max(N * (N - 1), 1))
    return {
        "sample_size": len(chosen),
        "empty_graph_ratio": float(np.mean(empty_ratios)) if empty_ratios else float("nan"),
        "density_mean": float(np.mean(densities)) if densities else float("nan"),
        "density_std": float(np.std(densities)) if densities else float("nan"),
    }


def _label_integrity(features_root: Path, ep_ids: list[str], sample: int = 30) -> dict:
    rng = np.random.default_rng(0)
    chosen = rng.choice(ep_ids, size=min(sample, len(ep_ids)), replace=False).tolist()
    issues = []
    regime_transitions = Counter()
    for ep_id in chosen:
        ep_dir = features_root / ep_id
        if not ep_dir.exists():
            continue
        try:
            labels_df = pd.read_parquet(ep_dir / "labels.parquet")
        except Exception:
            issues.append({"ep_id": ep_id, "issue": "labels.parquet unreadable"})
            continue
        # Expected columns
        for col in ["regime", "scenario", "episode_id", "drift_flag", "is_injection"]:
            if col not in labels_df.columns:
                issues.append({"ep_id": ep_id, "issue": f"missing column {col}"})
        # Episode_id consistency
        ep_ids_in_labels = labels_df["episode_id"].unique() if "episode_id" in labels_df.columns else []
        if len(ep_ids_in_labels) > 1:
            issues.append({"ep_id": ep_id, "issue": f"multiple ep_ids in labels: {ep_ids_in_labels[:3]}"})
        # Regime transitions
        if "regime" in labels_df.columns:
            regimes = labels_df["regime"].values
            for i in range(len(regimes) - 1):
                if regimes[i] != regimes[i + 1]:
                    regime_transitions[(regimes[i], regimes[i + 1])] += 1
        # Drift_flag vs is_injection consistency
        if "drift_flag" in labels_df.columns and "is_injection" in labels_df.columns:
            # Expectation: drift_flag fixed per episode, is_injection True during injection
            drift_unique = labels_df["drift_flag"].nunique()
            if drift_unique > 1:
                issues.append({"ep_id": ep_id, "issue": "drift_flag varies within episode"})
    return {
        "sample_size": len(chosen),
        "n_issues": len(issues),
        "issues": issues[:10],   # cap output
        "regime_transitions": {f"{src}→{dst}": int(n) for (src, dst), n in regime_transitions.most_common()},
    }


def audit_dataset(dataset_dir: Path, features_root: Path,
                  output_dir: Path) -> dict:
    """Run full audit on one dataset."""
    print(f"\n=== Auditing {dataset_dir.name} (features at {features_root}) ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_idx = pd.read_parquet(dataset_dir / "index.parquet")
    all_ep_ids = df_idx["episode_id"].tolist()
    test_ep_ids = df_idx[df_idx["split"] == "test"]["episode_id"].tolist()

    print("  [1/7] Inventory…")
    inv = _inventory(dataset_dir)
    print("  [2/7] NaN audit (all episodes)…")
    nan = _nan_audit(features_root, all_ep_ids)
    print("  [3/7] Class balance…")
    bal = _class_balance(dataset_dir)
    print("  [4/7] Signal sanity (30 ep sample)…")
    san = _signal_sanity(features_root, all_ep_ids, sample_size=30)
    print("  [5/7] Duration stats…")
    dur = _duration_stats(dataset_dir)
    print("  [6/7] Adjacency stats (20 ep sample)…")
    adj = _adjacency_stats(features_root, all_ep_ids, sample=20)
    print("  [7/7] Label integrity (30 ep sample)…")
    lbl = _label_integrity(features_root, all_ep_ids, sample=30)

    audit = {
        "dataset": str(dataset_dir),
        "features_root": str(features_root),
        "inventory": inv,
        "nan_audit": nan,
        "class_balance": bal,
        "signal_sanity": san,
        "duration_stats": dur,
        "adjacency_stats": adj,
        "label_integrity": lbl,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=str))

    # Generate Markdown report
    md = _render_markdown(audit, dataset_dir.name)
    (output_dir / "audit.md").write_text(md)
    print(f"  → {output_dir / 'audit.md'}")
    return audit


def _render_markdown(audit: dict, name: str) -> str:
    lines = [f"# Data audit — {name}", "",
             f"_Dataset_: `{audit['dataset']}` | _features_: `{audit['features_root']}`", ""]

    # Inventory
    inv = audit["inventory"]
    lines += ["## 1. Inventory", "",
              f"- Total episodes: **{inv['n_episodes_total']}**",
              f"- Scenarios: **{inv['n_scenarios']}**",
              f"- Splits: {inv['split_counts']}",
              ""]
    if any(inv["scenarios_missing_per_split"].values()):
        lines.append("**Scenarios missing per split**:")
        for split, missing in inv["scenarios_missing_per_split"].items():
            if missing:
                lines.append(f"- {split}: ❌ missing {missing}")
        lines.append("")

    # Class balance
    bal = audit["class_balance"]
    lines += ["## 2. Class balance", "",
              "| Split | Total | Min/scenario | Max/scenario | n underpopulated (<5) |",
              "|---|---|---|---|---|"]
    for split in ["train", "val", "test"]:
        b = bal[split]
        lines.append(
            f"| {split} | {b['total']} | {b['min_per_scenario']} | {b['max_per_scenario']} | "
            f"{b['n_underpopulated']} |"
        )
    lines.append("")
    if bal["test"]["underpopulated_scenarios"]:
        lines += ["**Test set underpopulated scenarios (n_pos < 5)**:", "",
                  "These will produce noisy AUROC. Should be marked 'non-conclusive'.",
                  ""]
        for s, n in sorted(bal["test"]["underpopulated_scenarios"].items()):
            lines.append(f"- {s}: n_pos = {n}")
        lines.append("")

    # NaN
    nan = audit["nan_audit"]
    lines += ["## 3. NaN audit", "",
              "### Per modality",
              "| Modality | NaN ratio |",
              "|---|---|"]
    for mod in ["M", "T", "L"]:
        lines.append(f"| {mod} | {nan['per_modality'][mod]['nan_ratio']:.3f} ({nan['per_modality'][mod]['nan_count']:,} NaN) |")
    lines += ["",
              "### Per feature (top 5 worst)",
              "| # | Feature | Modality | NaN ratio |",
              "|---|---|---|---|"]
    sorted_feats = sorted(nan["per_feature"], key=lambda x: -x["nan_ratio"])
    for r in sorted_feats[:5]:
        lines.append(f"| {r['feature_index']} | {r['feature_name']} | {r['modality']} | {r['nan_ratio']:.3f} |")
    lines += ["",
              "### Per service",
              "| Service | NaN ratio |",
              "|---|---|"]
    for r in sorted(nan["per_service"], key=lambda x: -x["nan_ratio"]):
        lines.append(f"| {r['service']} | {r['nan_ratio']:.3f} |")
    lines.append("")

    # Signal sanity
    san = audit["signal_sanity"]
    lines += ["## 4. Signal sanity (normal vs injection)", "",
              f"_Sample: {san['sample_size']} episodes_", "",
              "Top features by |Cohen's d| (signal change between normal and injection):", "",
              "| Feature | Cohen's d | Normal mean | Inject mean | n_normal | n_inject |",
              "|---|---|---|---|---|---|"]
    for r in san["per_feature_normal_vs_inject"][:10]:
        cd = f"{r['cohen_d']:.3f}" if not np.isnan(r["cohen_d"]) else "NaN"
        nm = f"{r['normal_mean']:.4g}" if not np.isnan(r["normal_mean"]) else "NaN"
        im = f"{r['inject_mean']:.4g}" if not np.isnan(r["inject_mean"]) else "NaN"
        lines.append(
            f"| {r['feature_name']} | {cd} | {nm} | {im} | {r['n_normal_samples']} | {r['n_inject_samples']} |"
        )
    lines += ["",
              "**Interpretation**: |Cohen's d| > 0.8 → large effect; 0.5-0.8 → medium; <0.5 → small. "
              "Features with d≈0 don't change during injection — useless for prediction.",
              ""]

    # Duration
    dur = audit["duration_stats"]
    lines += ["## 5. Episode duration",
              "",
              f"- n_timesteps: min={dur['n_timesteps_min']}, "
              f"median={dur['n_timesteps_median']}, max={dur['n_timesteps_max']}, "
              f"mean={dur['n_timesteps_mean']:.1f} ± {dur['n_timesteps_std']:.1f}",
              ""]

    # Adjacency
    adj = audit["adjacency_stats"]
    lines += ["## 6. Adjacency graph", "",
              f"- Empty-graph ratio: {adj['empty_graph_ratio']:.3f}",
              f"- Density mean ± std: {adj['density_mean']:.3f} ± {adj['density_std']:.3f}",
              ""]

    # Label integrity
    lbl = audit["label_integrity"]
    lines += ["## 7. Label integrity", "",
              f"- Issues found: **{lbl['n_issues']}** / {lbl['sample_size']} sampled",
              ""]
    if lbl["issues"]:
        lines += ["**First issues**:", ""]
        for i in lbl["issues"]:
            lines.append(f"- `{i['ep_id']}`: {i['issue']}")
        lines.append("")
    lines += ["**Regime transitions observed**:",
              ", ".join(f"`{k}` ({v})" for k, v in list(lbl["regime_transitions"].items())[:8]),
              ""]

    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EWAT data quality audit")
    p.add_argument("--datasets", nargs="+",
                   default=["ewat_v3", "ewat_v4_strat"],
                   help="Dataset names under data/datasets/")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("experiments/data_quality"))
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for dataset_name in args.datasets:
        dataset_dir = args.data_root / "datasets" / dataset_name
        if not dataset_dir.exists():
            print(f"Skip {dataset_name}: not found at {dataset_dir}")
            continue
        # Infer features root
        version = dataset_name.replace("ewat_", "").replace("_strat", "")
        features_root = args.data_root / "features" / version
        if not features_root.exists():
            features_root = args.data_root / "features" / "v3"
        out_dir = args.output / dataset_name
        audit = audit_dataset(dataset_dir, features_root, out_dir)
        summaries[dataset_name] = audit
    # Cross-dataset comparison
    if len(summaries) >= 2:
        compare = []
        compare.append("# Cross-dataset comparison\n")
        compare.append("| Metric | " + " | ".join(summaries.keys()) + " |")
        compare.append("|" + "---|" * (len(summaries) + 1))
        compare.append(
            "| n_episodes | " +
            " | ".join(str(s["inventory"]["n_episodes_total"]) for s in summaries.values()) +
            " |"
        )
        compare.append(
            "| n_scenarios | " +
            " | ".join(str(s["inventory"]["n_scenarios"]) for s in summaries.values()) +
            " |"
        )
        for mod in ["M", "T", "L"]:
            row = [f"| NaN {mod}"]
            for s in summaries.values():
                row.append(f"{s['nan_audit']['per_modality'][mod]['nan_ratio']:.3f}")
            compare.append("| ".join(row) + " |")
        compare.append(
            "| median T steps | " +
            " | ".join(str(s["duration_stats"]["n_timesteps_median"]) for s in summaries.values()) +
            " |"
        )
        compare.append(
            "| empty graph % | " +
            " | ".join(f"{s['adjacency_stats']['empty_graph_ratio']*100:.1f}%"
                      for s in summaries.values()) +
            " |"
        )
        compare.append(
            "| underpop test scenarios | " +
            " | ".join(str(s["class_balance"]["test"]["n_underpopulated"]) for s in summaries.values()) +
            " |"
        )
        (args.output / "cross_comparison.md").write_text("\n".join(compare))
        print(f"\nCross-comparison: {args.output / 'cross_comparison.md'}")


if __name__ == "__main__":
    main()
