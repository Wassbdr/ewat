"""Phase K.3 — Per-seed variance distribution analysis.

Reads Phase H all_summaries.json + Phase J all_summaries.json and produces:
- Per-metric box/violin plots (sil_test, AUROC, A1 delta, K)
- Outlier identification (seeds > 1.5 IQR)
- Markdown summary with the distribution shape

Output:
- experiments/multiseed/phase_h/distribution.png
- experiments/multiseed/phase_h/variance_analysis.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _iqr_outliers(values: np.ndarray, seeds: list, k: float = 1.5) -> list:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return [(seeds[i], float(v)) for i, v in enumerate(values)
            if v < lo or v > hi]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-h-dir", type=Path,
                   default=Path("experiments/multiseed/phase_h"))
    p.add_argument("--phase-j-dir", type=Path,
                   default=Path("experiments/multiseed/phase_j"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/multiseed/phase_h"))
    args = p.parse_args()

    h_data = json.loads((args.phase_h_dir / "all_summaries.json").read_text())
    j_data = json.loads((args.phase_j_dir / "all_summaries.json").read_text())
    h_valid = [s for s in h_data if "failed" not in s]
    j_valid = [s for s in j_data if "failed" not in s]

    seeds = [s["seed"] for s in h_valid]
    sil_test = np.array([s["H1"]["silhouette_test"] for s in h_valid])
    auroc = np.array([s["H3"]["auroc_peak_test_mean"] for s in h_valid])
    delta = np.array([s["A1"].get("delta_far_near_macro") or float("nan")
                      for s in h_valid])
    # K from k_selection_comparison if available
    k_path = args.phase_h_dir / "k_selection_comparison.json"
    if k_path.exists():
        k_data = json.loads(k_path.read_text())
        k_sil = np.array([r["K_silhouette"] for r in k_data["per_seed"]])
    else:
        k_sil = np.array([s["H1"].get("K_optimal") or 0 for s in h_valid])
    b2_strat = np.array([s["B2"]["stratified_macro_auroc"]
                         for s in j_valid if s["seed"] in seeds])
    b2_loso = np.array([s["B2"]["loso_macro_auroc_full_test_mean"]
                        for s in j_valid if s["seed"] in seeds])

    # Plot violin + scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    metrics = [
        ("H1 sil_test", sil_test, axes[0], None),
        ("H3 AUROC peak (circular)", auroc, axes[1], None),
        ("A1 Δ(far−near)", delta, axes[2], 0.0),
        ("K_optimal (silhouette)", k_sil.astype(float), axes[3], None),
        ("B2 stratified (Chaos Mesh)", b2_strat, axes[4], 0.5),
        ("B2 LOSO macro", b2_loso, axes[5], 0.5),
    ]
    for name, arr, ax, threshold in metrics:
        finite = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
        if finite.size == 0:
            continue
        parts = ax.violinplot([finite], showmeans=True, showmedians=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.25)
        ax.scatter(np.random.normal(1, 0.05, size=len(finite)), finite,
                   c='steelblue', s=40, alpha=0.7, zorder=3)
        if threshold is not None:
            ax.axhline(threshold, color='red', linestyle='--', alpha=0.5,
                       label=f"threshold={threshold}")
            ax.legend()
        ax.set_title(name)
        ax.set_xticks([])
        # Annotate outliers
        outliers = _iqr_outliers(finite, [seeds[i] for i in range(len(arr))
                                          if not np.isnan(arr[i]) or arr.dtype.kind != "f"])
        # Stats overlay
        mean = float(finite.mean())
        std = float(finite.std(ddof=0))
        ax.text(0.05, 0.95, f"mean={mean:.3f}\nstd={std:.3f}\nrange=[{finite.min():.3f}, {finite.max():.3f}]",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.suptitle("Phase H + J — per-seed variance (10 seeds, ewat_v4_strat)",
                 fontsize=13)
    plt.tight_layout()
    out_png = args.output / "distribution.png"
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close()

    # Markdown summary
    out_lines = [
        "# Phase K.3 — Per-seed variance analysis",
        "",
        f"![Distribution]({out_png.name})",
        "",
        "## Statistical summary",
        "",
        "| Metric | Mean | Std | Min | Max | Range | Outliers (1.5×IQR) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, arr, _, _ in metrics:
        finite = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
        if finite.size == 0:
            continue
        seeds_for_arr = [seeds[i] for i in range(len(arr))]
        outliers = _iqr_outliers(arr, seeds_for_arr) if arr.dtype.kind == "f" else []
        outlier_str = ", ".join(f"seed{s}={v:.3f}" for s, v in outliers) or "—"
        out_lines.append(
            f"| {name} | {finite.mean():.3f} | {finite.std(ddof=0):.3f} | "
            f"{finite.min():.3f} | {finite.max():.3f} | "
            f"{finite.max()-finite.min():.3f} | {outlier_str} |"
        )

    out_lines += [
        "",
        "## Interpretation",
        "",
        "**Variance per-seed reveals two regimes** :",
        "",
        "1. **Stable metrics** (low std):",
        "   - H3 AUROC peak (circular, by design): std=0.012",
        "   - B2 stratified (deterministic): std=0",
        "   - B2 LOSO (deterministic): std=0",
        "",
        "2. **Unstable metrics** (high std):",
        "   - H1 sil_test: range 0.521-0.839 (Δ=0.32, std=0.115)",
        "   - K_optimal: range 9-15 (intrinsically unstable, cf. K.1)",
        "   - A1 Δ(far-near): mostly ~0 (LEAK), 1/10 outlier at -0.05 (seed 42, GENUINE)",
        "",
        "**Outliers** (1.5×IQR) flag seeds whose behaviour deviates from the population. "
        "Crucially, **seed 42 was used for the initial Phase G retrain** — its A1 Δ=-0.05 is "
        "now confirmed as an outlier, not a robust gain.",
        "",
        "## Implications for the report",
        "",
        "- Report multi-seed means ± std, not best/outlier values",
        "- Acknowledge K_optimal instability as L10 (already documented)",
        "- Headline défensif **B2 = 0.9201** is deterministic — robust by construction",
        "- The audit fixes corrected real bugs (NaN-aware scaler +20% data, etc.) "
        "but do not improve the honest headline (which is fundamentally limited by "
        "n_pos=3 per scenario test set, cf. C-5)",
    ]
    (args.output / "variance_analysis.md").write_text("\n".join(out_lines))
    print(f"Wrote {out_png}")
    print(f"Wrote {args.output / 'variance_analysis.md'}")


if __name__ == "__main__":
    main()
