"""Smoke analysis for EWAT labeled dataset runs.

Reports NaN distribution, class balance, signal statistics, and graph health.

Usage
-----
    python -m scripts.smoke_analysis data/raw/run_20260416_112413
    python -m scripts.smoke_analysis data/raw/campaign_smoke_11 --save-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the loader from validate_dataset to stay consistent.
from scripts.validate_dataset import _load_artifacts  # noqa: PLC2701

# ── Feature registry ─────────────────────────────────────────────────────────

FEATURE_NAMES: list[tuple[int, str, str]] = [
    # (index, name, modality)
    (0,  "cpu_util",           "M"),
    (1,  "ram_util",           "M"),
    (2,  "latency_p99",        "M"),
    (3,  "error_rate_http",    "M"),
    (4,  "net_sat",            "M"),
    (5,  "disk_io",            "M"),
    (6,  "queue_len",          "M"),
    (7,  "span_duration",      "T"),
    (8,  "span_anomaly_rate",  "T"),
    (9,  "trace_depth",        "T"),
    (10, "fan_out",            "T"),
    (11, "retry_rate",         "T"),
    (12, "latency_cv",         "T"),
    (13, "log_error_rate",     "L"),
    (14, "log_warn_rate",      "L"),
    (15, "semantic_anomaly",   "L"),
    (16, "lexical_entropy",    "L"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_SEP = "─" * 72


def _sep(title: str = "") -> None:
    if title:
        pad = max(0, 70 - len(title))
        print(f"\n{'─' * 3} {title} {'─' * pad}")
    else:
        print(_SEP)


def _pct(ratio: float) -> str:
    return f"{ratio * 100:6.2f}%"


def _flag(ratio: float, threshold: float = 0.5) -> str:
    return "  !" if ratio > threshold else "   "


# ── Sections ──────────────────────────────────────────────────────────────────

def report_metadata(metadata: dict, signal: np.ndarray, services: list[str]) -> None:
    _sep("1. Metadata")
    print(f"  run_id           : {metadata.get('run_id', 'n/a')}")
    print(f"  n_timestamps     : {signal.shape[0]}")
    print(f"  n_services       : {len(services)}  {services}")
    print(f"  signal_dim       : {signal.shape[2] if signal.ndim == 3 else 'n/a'}")
    print(f"  git_commit       : {metadata.get('git_commit', 'n/a')}")

    trace_stats = metadata.get("trace_collection_stats", {})
    if trace_stats:
        considered = float(trace_stats.get("services_considered", 0))
        timed_out = float(trace_stats.get("services_timed_out", 0))
        timeout_ratio = timed_out / considered if considered > 0 else 0.0
        empty_ratio = float(trace_stats.get("traces_empty_window_ratio", 0.0))
        print(f"  trace_timeout    : {_pct(timeout_ratio)}  ({int(timed_out)}/{int(considered)} services)")
        print(f"  trace_empty_wins : {_pct(empty_ratio)}")


def report_nan_by_feature(signal: np.ndarray, mask: np.ndarray) -> None:
    """NaN ratio per feature across all services and timestamps."""
    _sep("2. NaN by feature  (! = >50%)")
    # signal shape: (T, N, F)
    print(f"  {'idx':>3}  {'feature':<20}  {'mod':>3}  {'nan_ratio':>10}")
    print(f"  {'───':>3}  {'─' * 20}  {'───':>3}  {'──────────':>10}")
    for idx, name, mod in FEATURE_NAMES:
        if signal.ndim == 3 and idx < signal.shape[2]:
            col = mask[:, :, idx].ravel()
            nan_ratio = float(col.mean())
        else:
            nan_ratio = float("nan")
        flag = _flag(nan_ratio)
        print(f"  {idx:>3}  {name:<20}  {mod:>3}  {_pct(nan_ratio)}{flag}")


def report_nan_by_service(signal: np.ndarray, mask: np.ndarray, services: list[str]) -> None:
    """Mean NaN ratio per service across all features and timestamps."""
    _sep("3. NaN by service  (sorted desc,  ! = >50%)")
    print(f"  {'service':<25}  {'nan_ratio':>10}")
    print(f"  {'─' * 25}  {'──────────':>10}")
    # mask shape: (T, N, F)
    per_service = mask.mean(axis=(0, 2))  # shape (N,)
    order = np.argsort(per_service)[::-1]
    for i in order:
        name = services[i] if i < len(services) else str(i)
        flag = _flag(float(per_service[i]))
        print(f"  {name:<25}  {_pct(float(per_service[i]))}{flag}")


def report_nan_by_regime(signal: np.ndarray, mask: np.ndarray, labels: pd.DataFrame) -> None:
    """Mean NaN ratio per operational regime."""
    _sep("4. NaN by regime")
    print(f"  {'regime':<12}  {'n_timestamps':>12}  {'nan_ratio':>10}")
    print(f"  {'─' * 12}  {'──────────────':>14}  {'──────────':>10}")

    ts_to_mask: dict[float, np.ndarray] = {}
    if "timestamp" in labels.columns and mask.ndim == 3:
        timestamps = labels["timestamp"].unique()
        if len(timestamps) == mask.shape[0]:
            for t_idx, ts in enumerate(sorted(timestamps)):
                ts_to_mask[float(ts)] = mask[t_idx]

    regime_col = "regime" if "regime" in labels.columns else None
    if regime_col is None:
        print("  (no regime column in labels)")
        return

    for regime, group in labels.groupby(regime_col):
        ts_vals = group["timestamp"].unique() if "timestamp" in group.columns else np.array([])
        n_ts = len(ts_vals)
        if ts_to_mask and n_ts > 0:
            idxs = [i for i, ts in enumerate(sorted(labels["timestamp"].unique())) if ts in set(ts_vals)]
            if idxs:
                nan_ratio = float(mask[idxs].mean())
            else:
                nan_ratio = float("nan")
        else:
            nan_ratio = float("nan")
        flag = _flag(nan_ratio) if not np.isnan(nan_ratio) else "   "
        n_str = str(n_ts) if n_ts > 0 else "n/a"
        print(f"  {regime:<12}  {n_str:>14}  {_pct(nan_ratio) if not np.isnan(nan_ratio) else '   n/a    '}{flag}")


def report_class_distribution(labels: pd.DataFrame, signal: np.ndarray) -> None:
    """Episode count and timestamp count per scenario."""
    _sep("5. Class distribution")
    inj = labels[labels["regime"] == "injection"]
    if inj.empty:
        print("  (no injection labels)")
        return

    total_ts = signal.shape[0]
    inj_ts = len(labels[labels["regime"] == "injection"])

    print(f"  total timestamps  : {total_ts}")
    print(f"  injection timestamps: {inj_ts}  ({_pct(inj_ts / total_ts).strip()} of total)")
    print()

    summary = (
        inj.groupby(["scenario", "category"])
        .agg(n_episodes=("episode_id", "nunique"), n_timestamps=("episode_id", "count"))
        .reset_index()
        .sort_values("n_episodes", ascending=False)
    )
    print(f"  {'scenario':<30}  {'cat':<12}  {'episodes':>8}  {'inj_ts':>8}")
    print(f"  {'─' * 30}  {'─' * 12}  {'─' * 8}  {'─' * 8}")
    for _, row in summary.iterrows():
        print(
            f"  {row['scenario']:<30}  {row['category']:<12}  "
            f"{row['n_episodes']:>8}  {row['n_timestamps']:>8}"
        )


def report_signal_stats(signal: np.ndarray, mask: np.ndarray) -> None:
    """Percentile statistics per feature for valid (non-NaN) values."""
    _sep("6. Signal stats (non-NaN values)")
    print(f"  {'idx':>3}  {'feature':<20}  {'mean':>8}  {'std':>8}  {'p5':>8}  {'p50':>8}  {'p95':>8}  {'n_valid':>9}")
    print(f"  {'───':>3}  {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 9}")

    for idx, name, _mod in FEATURE_NAMES:
        if signal.ndim != 3 or idx >= signal.shape[2]:
            continue
        col = signal[:, :, idx].ravel()
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            print(f"  {idx:>3}  {name:<20}  {'all NaN':>56}")
            continue
        p5, p50, p95 = np.percentile(valid, [5, 50, 95])
        print(
            f"  {idx:>3}  {name:<20}  "
            f"{valid.mean():>8.4f}  {valid.std():>8.4f}  "
            f"{p5:>8.4f}  {p50:>8.4f}  {p95:>8.4f}  {valid.size:>9,}"
        )


def report_graph_health(graph_stats: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Graph edge statistics per regime."""
    _sep("7. Graph health")
    if graph_stats.empty or "n_edges" not in graph_stats.columns:
        print("  (graph_stats empty or missing columns)")
        return

    # Attach regime if possible
    if "regime" in graph_stats.columns:
        df = graph_stats
    elif "regime" in labels.columns and "timestamp" in labels.columns and "timestamp" in graph_stats.columns:
        regime_map = labels.drop_duplicates("timestamp").set_index("timestamp")["regime"]
        df = graph_stats.copy()
        df["regime"] = df["timestamp"].map(regime_map)
    else:
        df = graph_stats
        df["regime"] = "all"

    cols = ["n_edges", "density", "avg_degree"]
    available = [c for c in cols if c in df.columns]
    header = f"  {'regime':<12}  {'n_rows':>6}  " + "  ".join(f"{'mean_' + c:>12}  {'std_' + c:>12}" for c in available)
    print(header)
    print(f"  {'─' * 12}  {'─' * 6}  " + "  ".join(["─" * 12 + "  " + "─" * 12] * len(available)))

    for regime, grp in df.groupby("regime"):
        parts = [f"  {regime:<12}  {len(grp):>6}"]
        for col in available:
            parts.append(f"  {grp[col].mean():>12.4f}  {grp[col].std():>12.4f}")
        print("".join(parts))


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(
    signal: np.ndarray,
    mask: np.ndarray,
    labels: pd.DataFrame,
    services: list[str],
    run_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [plots] matplotlib/seaborn not available — skipping plots.")
        return

    out_dir = run_dir / "smoke_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: NaN heatmap (services × features)
    nan_matrix = mask.mean(axis=0)  # (N, F)
    feature_labels = [f[1] for f in FEATURE_NAMES if f[0] < nan_matrix.shape[1]]
    fig, ax = plt.subplots(figsize=(min(18, len(feature_labels) * 0.8 + 2), max(4, len(services) * 0.4 + 1)))
    sns.heatmap(
        nan_matrix[:, : len(feature_labels)],
        annot=False,
        fmt=".0%",
        xticklabels=feature_labels,
        yticklabels=services,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("NaN ratio — services × features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Service")
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_nan_heatmap.png", dpi=150)
    plt.close(fig)

    # Figure 2: Scenario distribution barplot
    inj = labels[labels["regime"] == "injection"]
    if not inj.empty:
        counts = inj.groupby("scenario")["episode_id"].nunique().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.6 + 2), 4))
        counts.plot.bar(ax=ax, color="steelblue")
        ax.set_title("Episode count per scenario (injection regime)")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("# episodes")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(out_dir / "fig2_scenario_distribution.png", dpi=150)
        plt.close(fig)

    # Figure 3: Boxplot of 7 M-features per regime
    m_indices = [f[0] for f in FEATURE_NAMES if f[2] == "M"]
    m_names = [f[1] for f in FEATURE_NAMES if f[2] == "M"]
    if signal.ndim == 3 and "regime" in labels.columns and "timestamp" in labels.columns:
        unique_ts = sorted(labels["timestamp"].unique())
        ts_to_regime = labels.drop_duplicates("timestamp").set_index("timestamp")["regime"].to_dict()
        if len(unique_ts) == signal.shape[0]:
            records = []
            for t_idx, ts in enumerate(unique_ts):
                regime = ts_to_regime.get(ts, "unknown")
                for s_idx in range(signal.shape[1]):
                    for fi, fname in zip(m_indices, m_names):
                        val = signal[t_idx, s_idx, fi]
                        if not np.isnan(val):
                            records.append({"regime": regime, "feature": fname, "value": val})
            if records:
                df_plot = pd.DataFrame(records)
                fig, axes = plt.subplots(1, len(m_names), figsize=(max(14, len(m_names) * 2), 5), sharey=False)
                if len(m_names) == 1:
                    axes = [axes]
                regimes = sorted(df_plot["regime"].unique())
                for ax, fname in zip(axes, m_names):
                    data = [
                        df_plot[df_plot["feature"] == fname][df_plot["regime"] == r]["value"].dropna().values
                        for r in regimes
                    ]
                    ax.boxplot(data, labels=regimes, showfliers=False)
                    ax.set_title(fname, fontsize=8)
                    ax.tick_params(axis="x", rotation=45, labelsize=7)
                fig.suptitle("M-features distribution per regime", y=1.02)
                plt.tight_layout()
                fig.savefig(out_dir / "fig3_metric_features_by_regime.png", dpi=150)
                plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke analysis for an EWAT labeled dataset run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", type=Path, help="Path to a run directory (contains signal.npz etc.)")
    parser.add_argument("--save-plots", action="store_true", help="Save matplotlib figures to <run_dir>/smoke_plots/")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir: Path = args.run_dir

    if not run_dir.exists():
        raise SystemExit(f"[ERROR] run_dir does not exist: {run_dir}")

    print(f"\nSMOKE ANALYSIS — {run_dir}")
    _sep()

    signal, mask, adjacency, labels, graph_stats, metadata, services = _load_artifacts(run_dir)

    report_metadata(metadata, signal, services)
    report_nan_by_feature(signal, mask)
    report_nan_by_service(signal, mask, services)
    report_nan_by_regime(signal, mask, labels)
    report_class_distribution(labels, signal)
    report_signal_stats(signal, mask)
    report_graph_health(graph_stats, labels)

    if args.save_plots:
        _sep("Plots")
        save_plots(signal, mask, labels, services, run_dir)

    _sep()
    print("Done.\n")


if __name__ == "__main__":
    main()
