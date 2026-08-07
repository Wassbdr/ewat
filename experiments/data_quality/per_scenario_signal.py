"""Deep dive: per-scenario × per-target-service Cohen's d analysis.

The global Cohen's d (~0.2-0.3) hides massive per-scenario variation. Many chaos
types (crash, kill) make services *fail* → latency DROPS to 0 instead of
increasing. Pooling cancels these inverse signals.

This script computes Cohen's d **per scenario** to find scenarios where the
signal is genuinely strong vs where it's weak/inverse.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = (a.std() + b.std()) / 2 + 1e-9
    return (b.mean() - a.mean()) / pooled


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/data_quality/per_scenario"))
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df_idx = pd.read_parquet(args.dataset / "index.parquet")
    print(f"Loaded {len(df_idx)} episodes from {args.dataset}")

    # Collect per-scenario × per-feature: normal vs injection
    scenarios = sorted(df_idx["scenario"].unique())
    per_sc_feat_normal: dict = defaultdict(lambda: defaultdict(list))
    per_sc_feat_inject: dict = defaultdict(lambda: defaultdict(list))
    # Also: per-target-service (the service being attacked)
    per_sc_target_normal: dict = defaultdict(lambda: defaultdict(list))
    per_sc_target_inject: dict = defaultdict(lambda: defaultdict(list))

    for _, row in df_idx.iterrows():
        ep_id = row["episode_id"]
        scenario = row["scenario"]
        ep_dir = args.features_root / ep_id
        if not ep_dir.exists():
            continue
        try:
            with np.load(ep_dir / "signal.npz") as z:
                sig = z["signal"]   # (T, N, 17)
            labels_df = pd.read_parquet(ep_dir / "labels.parquet")
            services = json.loads((ep_dir / "services.json").read_text())
        except Exception:
            continue
        regime = labels_df["regime"].values
        # Get target services from labels
        try:
            target_list = labels_df["target_services"].iloc[0]
            if isinstance(target_list, str):
                # JSON-encoded
                try:
                    target_list = json.loads(target_list)
                except Exception:
                    target_list = [target_list]
            target_indices = [services.index(t) for t in target_list if t in services]
        except Exception:
            target_indices = []
        normal_mask = regime == "normal"
        inject_mask = regime == "injection"
        for f in range(17):
            n_vals = sig[normal_mask, :, f].ravel()
            i_vals = sig[inject_mask, :, f].ravel()
            per_sc_feat_normal[scenario][f].extend(n_vals.tolist())
            per_sc_feat_inject[scenario][f].extend(i_vals.tolist())
            # Restricted to target services if available
            if target_indices:
                n_t = sig[normal_mask][:, target_indices, f].ravel()
                i_t = sig[inject_mask][:, target_indices, f].ravel()
                per_sc_target_normal[scenario][f].extend(n_t.tolist())
                per_sc_target_inject[scenario][f].extend(i_t.tolist())

    # Compute per-scenario × per-feature Cohen's d
    rows_all = []
    for sc in scenarios:
        for f in range(17):
            d_pooled = _cohens_d(
                np.array(per_sc_feat_normal[sc][f]),
                np.array(per_sc_feat_inject[sc][f]),
            )
            d_target = _cohens_d(
                np.array(per_sc_target_normal[sc][f]),
                np.array(per_sc_target_inject[sc][f]),
            ) if sc in per_sc_target_normal else float("nan")
            rows_all.append({
                "scenario": sc,
                "feature": FEATURE_NAMES[f],
                "feature_idx": f,
                "cohens_d_pooled": d_pooled,
                "cohens_d_target_only": d_target,
                "abs_d_pooled": abs(d_pooled) if not np.isnan(d_pooled) else float("nan"),
                "abs_d_target": abs(d_target) if not np.isnan(d_target) else float("nan"),
            })

    df_out = pd.DataFrame(rows_all)
    df_out.to_csv(args.output / "per_scenario_cohens_d.csv", index=False)

    # Per scenario: best feature by |d| target-only
    summary = []
    for sc in scenarios:
        sub = df_out[df_out["scenario"] == sc]
        # Filter NaN
        sub_target = sub.dropna(subset=["cohens_d_target_only"])
        sub_pooled = sub.dropna(subset=["cohens_d_pooled"])
        best_target = sub_target.nlargest(1, "abs_d_target") if not sub_target.empty else None
        best_pooled = sub_pooled.nlargest(1, "abs_d_pooled") if not sub_pooled.empty else None
        summary.append({
            "scenario": sc,
            "best_feature_target": best_target.iloc[0]["feature"] if best_target is not None and not best_target.empty else None,
            "best_d_target": float(best_target.iloc[0]["cohens_d_target_only"]) if best_target is not None and not best_target.empty else float("nan"),
            "best_feature_pooled": best_pooled.iloc[0]["feature"] if best_pooled is not None and not best_pooled.empty else None,
            "best_d_pooled": float(best_pooled.iloc[0]["cohens_d_pooled"]) if best_pooled is not None and not best_pooled.empty else float("nan"),
            "max_abs_d_target": float(sub_target["abs_d_target"].max()) if not sub_target.empty else float("nan"),
            "max_abs_d_pooled": float(sub_pooled["abs_d_pooled"].max()) if not sub_pooled.empty else float("nan"),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output / "per_scenario_summary.csv", index=False)

    # Markdown report
    lines = [
        f"# Per-scenario signal strength — {args.dataset.name}",
        "",
        "## Cohen's d per scenario : best feature (target-restricted vs pooled)",
        "",
        "**Reminder** : |d| < 0.2 = trivial, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large.",
        "",
        "**Target-restricted** means we measure only on the service(s) actually attacked. "
        "This is the *real* signal-to-noise ratio for that scenario.",
        "**Pooled** averages across all 6 services — non-attacked services dilute the signal.",
        "",
        "| Scenario | Best feature (target) | d_target | Best feature (pooled) | d_pooled | max |d| target |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary:
        bft = r["best_feature_target"] or "-"
        bfp = r["best_feature_pooled"] or "-"
        dt = f"{r['best_d_target']:+.3f}" if not np.isnan(r["best_d_target"]) else "NaN"
        dp = f"{r['best_d_pooled']:+.3f}" if not np.isnan(r["best_d_pooled"]) else "NaN"
        mxt = f"{r['max_abs_d_target']:.3f}" if not np.isnan(r["max_abs_d_target"]) else "NaN"
        lines.append(f"| {r['scenario']} | {bft} | {dt} | {bfp} | {dp} | {mxt} |")

    # Top 20 (scenario, feature) by |d|target
    top20 = df_out.dropna(subset=["abs_d_target"]).nlargest(20, "abs_d_target")
    lines += ["", "## Top 20 (scenario, feature) by |Cohen's d| restricted to target services",
              "", "| # | Scenario | Feature | Cohen's d (target) | Cohen's d (pooled) |",
              "|---|---|---|---|---|"]
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        lines.append(
            f"| {i} | {r['scenario']} | {r['feature']} | "
            f"{r['cohens_d_target_only']:+.3f} | {r['cohens_d_pooled']:+.3f} |"
        )

    # Worst 10 scenarios — least signal
    summary_sorted = sorted(summary, key=lambda x: x["max_abs_d_target"] if not np.isnan(x["max_abs_d_target"]) else -1)
    lines += ["", "## Scenarios with WEAKEST signal (target-restricted)",
              "", "| # | Scenario | max |d| target | best feature |",
              "|---|---|---|---|"]
    for i, r in enumerate(summary_sorted[:5], 1):
        mxt = f"{r['max_abs_d_target']:.3f}" if not np.isnan(r["max_abs_d_target"]) else "NaN"
        lines.append(f"| {i} | {r['scenario']} | {mxt} | {r['best_feature_target']} |")

    # Aggregate
    arr_pool = df_out["abs_d_pooled"].dropna().values
    arr_targ = df_out["abs_d_target"].dropna().values
    lines += ["",
              "## Aggregate distribution of |Cohen's d|", "",
              f"- **Pooled** (n={arr_pool.size}): mean={arr_pool.mean():.3f}, "
              f"median={np.median(arr_pool):.3f}, max={arr_pool.max():.3f}",
              f"- **Target-restricted** (n={arr_targ.size}): mean={arr_targ.mean():.3f}, "
              f"median={np.median(arr_targ):.3f}, max={arr_targ.max():.3f}",
              "",
              f"- |d| ≥ 0.8 (large effect): pooled={int((arr_pool>=0.8).sum())}/{arr_pool.size} "
              f"vs target={int((arr_targ>=0.8).sum())}/{arr_targ.size}",
              f"- |d| ≥ 0.5 (medium+): pooled={int((arr_pool>=0.5).sum())}/{arr_pool.size} "
              f"vs target={int((arr_targ>=0.5).sum())}/{arr_targ.size}",
              f"- |d| ≥ 0.2 (small+): pooled={int((arr_pool>=0.2).sum())}/{arr_pool.size} "
              f"vs target={int((arr_targ>=0.2).sum())}/{arr_targ.size}",
              ""]
    (args.output / "per_scenario_signal.md").write_text("\n".join(lines))
    print(f"Wrote {args.output / 'per_scenario_signal.md'}")
    print(f"\nBest |d| target-restricted: {arr_targ.max():.3f}")
    print(f"Mean |d| target-restricted: {arr_targ.mean():.3f}")


if __name__ == "__main__":
    main()
