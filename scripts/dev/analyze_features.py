"""EWAT — Phase 2 feature quality audit.

Reads all feature episodes under a features root and produces:
  - experiments/data_audit/nan_report.csv   (per-episode NaN + graph stats)
  - experiments/data_audit/nan_by_scenario.csv (aggregated per scenario)
  - experiments/data_audit/excluded.json    (episodes failing quality gates)
  - experiments/data_audit/nan_heatmap.png
  - experiments/data_audit/timesteps_boxplot.png
  - experiments/data_audit/graph_density_boxplot.png

Usage
-----
::

    python -m scripts.analyze_features --features-root data/features/v1
    python -m scripts.analyze_features --features-root data/features/v1 \\
        --max-nan-M 0.60 --max-nan-T 0.80 --max-nan-L 0.80 --min-graph-density 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Feature slice indices in S(t) ∈ ℝ^{T×N×17}
_SLICE_M = slice(0, 7)
_SLICE_T = slice(7, 13)
_SLICE_L = slice(13, 17)

_FEATURE_NAMES = [
    "cpu_util", "ram_util", "latency_p99", "error_rate_http",
    "net_sat", "disk_io", "queue_depth",                          # M indices 0-6
    "span_dur_med", "abnormal_span_rate", "trace_depth",
    "fan_out", "retry_rate", "latency_cv",                        # T indices 7-12
    "log_error_rate", "log_warn_rate", "semantic_anomaly",
    "lexical_entropy",                                             # L indices 13-16
]


def _scenario_from_episode_id(episode_id: str) -> str:
    parts = episode_id.split("_")
    # strip leading "episode_" and trailing rep + timestamp
    # episode_cpu_starvation_000_20260430T... → cpu_starvation
    if parts[0] == "episode":
        parts = parts[1:]
    # last two parts are rep number and timestamp
    return "_".join(parts[:-2])


def _audit_episode(ep_dir: Path) -> dict | None:
    signal_path = ep_dir / "signal.npz"
    adj_path = ep_dir / "adjacency.npz"
    if not signal_path.exists() or not adj_path.exists():
        return None

    signal = np.load(signal_path)["signal"]  # (T, N, 17)
    adjacency = np.load(adj_path)["adjacency"]  # (T, N, N, 3)

    T = signal.shape[0]
    nan_total = float(np.isnan(signal).mean())
    nan_M = float(np.isnan(signal[:, :, _SLICE_M]).mean())
    nan_T = float(np.isnan(signal[:, :, _SLICE_T]).mean())
    nan_L = float(np.isnan(signal[:, :, _SLICE_L]).mean())

    # Per-feature NaN (averaged over T and N)
    per_feature_nan = np.isnan(signal).mean(axis=(0, 1))  # (17,)

    # Graph density: fraction of timesteps with at least one edge (volume > 0)
    edge_vol = adjacency[:, :, :, 0]  # (T, N, N)
    has_edge = (edge_vol > 0).any(axis=(1, 2))  # (T,)
    graph_density = float(has_edge.mean())

    episode_id = ep_dir.name
    scenario = _scenario_from_episode_id(episode_id)

    row = {
        "episode_id": episode_id,
        "scenario": scenario,
        "n_timesteps": T,
        "nan_total": nan_total,
        "nan_M": nan_M,
        "nan_T": nan_T,
        "nan_L": nan_L,
        "graph_density": graph_density,
    }
    for i, name in enumerate(_FEATURE_NAMES):
        row[f"nan_{name}"] = float(per_feature_nan[i])

    return row


def _apply_quality_gates(
    df: pd.DataFrame,
    max_nan_M: float,
    max_nan_T: float,
    max_nan_L: float,
    min_graph_density: float,
) -> pd.DataFrame:
    reasons: list[list[str]] = []
    for _, row in df.iterrows():
        r: list[str] = []
        if row["nan_M"] > max_nan_M:
            r.append(f"nan_M={row['nan_M']:.2f}>{max_nan_M}")
        if row["nan_T"] > max_nan_T:
            r.append(f"nan_T={row['nan_T']:.2f}>{max_nan_T}")
        if row["nan_L"] > max_nan_L:
            r.append(f"nan_L={row['nan_L']:.2f}>{max_nan_L}")
        if row["graph_density"] < min_graph_density:
            r.append(f"graph_density={row['graph_density']:.2f}<{min_graph_density}")
        reasons.append(r)
    df = df.copy()
    df["reject_reasons"] = [";".join(r) for r in reasons]
    df["keep"] = [len(r) == 0 for r in reasons]
    return df


def _print_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_keep = df["keep"].sum()
    print(f"\n{'='*60}")
    print(f"Episodes total : {n_total}")
    print(f"Kept           : {n_keep}")
    print(f"Excluded       : {n_total - n_keep}")
    print(f"{'='*60}")
    print(f"\nNaN rates (all episodes):")
    print(f"  M(t)  : {df['nan_M'].mean()*100:.1f}% ± {df['nan_M'].std()*100:.1f}%  "
          f"[{df['nan_M'].min()*100:.1f}% – {df['nan_M'].max()*100:.1f}%]")
    print(f"  T(t)  : {df['nan_T'].mean()*100:.1f}% ± {df['nan_T'].std()*100:.1f}%  "
          f"[{df['nan_T'].min()*100:.1f}% – {df['nan_T'].max()*100:.1f}%]")
    print(f"  L(t)  : {df['nan_L'].mean()*100:.1f}% ± {df['nan_L'].std()*100:.1f}%  "
          f"[{df['nan_L'].min()*100:.1f}% – {df['nan_L'].max()*100:.1f}%]")

    print(f"\nPer-feature NaN (mean across all episodes × services × timesteps):")
    feat_cols = [c for c in df.columns if c.startswith("nan_") and c[4:] in _FEATURE_NAMES]
    feat_means = df[feat_cols].mean().sort_values(ascending=False)
    for col, val in feat_means.items():
        name = col[4:]
        modality = "M" if name in _FEATURE_NAMES[:7] else ("T" if name in _FEATURE_NAMES[7:13] else "L")
        print(f"  [{modality}] {name:<25} {val*100:.1f}%")

    print(f"\nKept episodes per scenario:")
    summary = df.groupby("scenario").agg(
        total=("keep", "count"),
        kept=("keep", "sum"),
        nan_M_mean=("nan_M", "mean"),
        nan_T_mean=("nan_T", "mean"),
        nan_L_mean=("nan_L", "mean"),
        graph_density_mean=("graph_density", "mean"),
    )
    for scenario, row in summary.iterrows():
        flag = "⚠️ " if row["kept"] < 10 else "   "
        print(f"  {flag}{scenario:<30} {int(row['kept']):2d}/{int(row['total'])}  "
              f"M={row['nan_M_mean']*100:.0f}%  T={row['nan_T_mean']*100:.0f}%  "
              f"L={row['nan_L_mean']*100:.0f}%  G={row['graph_density_mean']*100:.0f}%")


def _make_figures(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figures", file=sys.stderr)
        return

    scenarios = sorted(df["scenario"].unique())

    # 1. Heatmap NaN par scénario × modalité
    agg = df.groupby("scenario")[["nan_M", "nan_T", "nan_L"]].mean()
    agg = agg.reindex(scenarios)
    fig, ax = plt.subplots(figsize=(6, max(4, len(scenarios) * 0.5)))
    im = ax.imshow(agg.values * 100, aspect="auto", vmin=0, vmax=100, cmap="YlOrRd")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["M(t)\nPrometheus", "T(t)\nJaeger", "L(t)\nLoki"])
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    for i in range(len(scenarios)):
        for j in range(3):
            ax.text(j, i, f"{agg.values[i, j]*100:.0f}%", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="NaN %")
    ax.set_title("NaN par scénario × modalité")
    fig.tight_layout()
    fig.savefig(out_dir / "nan_heatmap.png", dpi=150)
    plt.close(fig)

    # 2. Boxplot n_timesteps par scénario
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df["scenario"] == s]["n_timesteps"].values for s in scenarios]
    ax.boxplot(data, labels=scenarios, vert=True)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Timesteps par épisode")
    ax.set_title("Distribution des timesteps par scénario")
    fig.tight_layout()
    fig.savefig(out_dir / "timesteps_boxplot.png", dpi=150)
    plt.close(fig)

    # 3. Boxplot graph density par scénario
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df["scenario"] == s]["graph_density"].values * 100 for s in scenarios]
    ax.boxplot(data, labels=scenarios, vert=True)
    ax.axhline(10, color="red", linestyle="--", linewidth=0.8, label="seuil min 10%")
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Graph density (%)")
    ax.set_title("Fraction de timesteps avec ≥1 arête — par scénario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "graph_density_boxplot.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {out_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="EWAT Phase 2 feature quality audit")
    parser.add_argument("--features-root", default="data/features/v1")
    parser.add_argument("--output-dir", default="experiments/data_audit")
    parser.add_argument("--max-nan-M", type=float, default=0.60)
    parser.add_argument("--max-nan-T", type=float, default=0.80)
    parser.add_argument("--max-nan-L", type=float, default=0.80)
    parser.add_argument("--min-graph-density", type=float, default=0.10)
    args = parser.parse_args()

    features_root = Path(args.features_root)
    if not features_root.is_absolute():
        features_root = REPO_ROOT / features_root
    if not features_root.exists():
        print(f"features root not found: {features_root}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(
        d for d in features_root.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )
    if not episode_dirs:
        print(f"no episode directories found under {features_root}", file=sys.stderr)
        return 2

    print(f"Auditing {len(episode_dirs)} episodes …")
    rows = []
    for ep_dir in episode_dirs:
        row = _audit_episode(ep_dir)
        if row is not None:
            rows.append(row)
        else:
            print(f"  SKIP {ep_dir.name} (missing signal.npz or adjacency.npz)", file=sys.stderr)

    df = pd.DataFrame(rows)
    df = _apply_quality_gates(
        df,
        max_nan_M=args.max_nan_M,
        max_nan_T=args.max_nan_T,
        max_nan_L=args.max_nan_L,
        min_graph_density=args.min_graph_density,
    )

    # Save reports
    df.to_csv(out_dir / "nan_report.csv", index=False)

    by_scenario = df.groupby("scenario").agg(
        n_total=("keep", "count"),
        n_kept=("keep", "sum"),
        nan_M_mean=("nan_M", "mean"),
        nan_M_max=("nan_M", "max"),
        nan_T_mean=("nan_T", "mean"),
        nan_T_max=("nan_T", "max"),
        nan_L_mean=("nan_L", "mean"),
        nan_L_max=("nan_L", "max"),
        graph_density_mean=("graph_density", "mean"),
        graph_density_min=("graph_density", "min"),
        n_timesteps_mean=("n_timesteps", "mean"),
    ).reset_index()
    by_scenario.to_csv(out_dir / "nan_by_scenario.csv", index=False)

    excluded = df[~df["keep"]][["episode_id", "scenario", "reject_reasons"]].to_dict(orient="records")
    (out_dir / "excluded.json").write_text(
        json.dumps(excluded, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_summary(df)
    print(f"\nReports saved to {out_dir}/")
    _make_figures(df, out_dir)

    n_kept = df["keep"].sum()
    return 0 if n_kept > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
