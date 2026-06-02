"""Generate vector (PDF) figures for the EWAT paper (docs/paper/figures/).

Reads numbers from the project's experiment logs where available; a few small
distant-window curves are taken verbatim from STATUS.md (cited in the paper text).
All figures are written as PDF for crisp inclusion in LaTeX.

Usage: python -m scripts.export_paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def fig_pipeline() -> None:
    """Stage 0 -> 3 schematic."""
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    stages = [
        ("S(t)\nN x 17", "#eeeeee"),
        ("Stage 0\nMMD-RFF\nlook-through", "#cfe8ff"),
        ("Stage 1\nSTGCN\nz in R^64", "#cfe8ff"),
        ("Stage 2/2b\nsiamese typing\n+ OWL ontology", "#ffe2c2"),
        ("Stage 3\ntyped\nprecursors", "#cfe8ff"),
        ("Alert(t)\nC_i, p_i, k*", "#d6f5d6"),
    ]
    n = len(stages)
    w = 1.42
    gap = (10 - n * w) / (n + 1)
    x = gap
    centers = []
    for label, color in stages:
        box = FancyBboxPatch(
            (x, 0.55), w, 0.95,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, 1.02, label, ha="center", va="center", fontsize=7.5)
        centers.append(x + w / 2)
        x += w + gap
    for i in range(n - 1):
        ax.add_patch(
            FancyArrowPatch(
                (centers[i] + w / 2, 1.02), (centers[i + 1] - w / 2, 1.02),
                arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#333333",
            )
        )
    ax.text(5, 0.2, "online: stages 0,1,3  (p95 = 13 ms)   |   offline: stage 2/2b",
            ha="center", va="center", fontsize=7, style="italic", color="#555555")
    fig.savefig(OUT / "pipeline_architecture.pdf")
    plt.close(fig)


def fig_multiseed() -> None:
    """Per-seed distributions on ewat_v4_strat (Phase H, 10 seeds)."""
    data = json.loads((ROOT / "experiments/multiseed/phase_h/all_summaries.json").read_text())
    sil = [s["H1"]["silhouette_test"] for s in data]
    auroc = [s["H3"]["auroc_peak_test_mean"] for s in data]
    kopt = [s["H1"].get("K_optimal") for s in data]
    kopt = [k for k in kopt if k is not None]
    delta = [s["A1"]["delta_far_near_macro"] for s in data]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    axes[0].boxplot(sil, widths=0.5)
    axes[0].axhline(0.3, ls="--", color="red", lw=0.8)
    axes[0].set_title("silhouette$_{test}$")
    axes[0].set_xticks([])

    axes[1].boxplot(auroc, widths=0.5)
    axes[1].set_title("circular AUROC")
    axes[1].set_xticks([])

    axes[2].hist(kopt, bins=range(8, 17), color="#cfe8ff", edgecolor="#333")
    axes[2].set_title("selected $K$")
    axes[2].set_xlabel("K")

    axes[3].axhline(0.0, ls="-", color="#999", lw=0.6)
    axes[3].axhline(-0.04, ls="--", color="red", lw=0.8)
    axes[3].scatter(np.random.default_rng(0).normal(0, 0.04, len(delta)), delta,
                    s=14, color="#1f77b4")
    axes[3].set_title(r"$\Delta_{far-near}$")
    axes[3].set_xticks([])
    fig.tight_layout()
    fig.savefig(OUT / "multiseed_distribution.pdf")
    plt.close(fig)


def fig_distant_window() -> None:
    """A1 (circular, ewat_v3) vs C2 (independent end-to-end, v4_strat).

    Values from STATUS.md (Stress tests H3 / Phase C2).
    """
    positions = ["first\n(far)", "middle", "last\n(near)"]
    a1 = [0.897, 0.907, 0.904]      # A1 distant-window, circular target
    c2 = [0.759, 0.813, 0.876]      # C2 distant-window, independent target
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(x, a1, "-o", color="#888888", label=r"A1 circular ($\Delta=-0.007$)")
    ax.plot(x, c2, "-s", color="#1f77b4", label=r"C2 independent ($\Delta=-0.116$)")
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel("macro-AUROC")
    ax.set_ylim(0.7, 0.95)
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(OUT / "distant_window.pdf")
    plt.close(fig)


def fig_heatmap() -> None:
    """Scenario x cluster alignment heatmap (ewat_v3)."""
    d = json.loads((ROOT / "experiments/typing/cluster_analysis.json").read_text())
    block = d.get("scenario_cluster_matrix")
    if block is None or "matrix" not in block:
        print("  [skip] scenario_cluster_matrix missing")
        return
    arr = np.array(block["matrix"], dtype=float)
    scenarios = block["scenarios"]
    clusters = block["clusters"]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xlabel("cluster")
    ax.set_ylabel("scenario")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=6)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="episodes")
    fig.savefig(OUT / "scenario_cluster_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    fig_pipeline()
    print("  wrote pipeline_architecture.pdf")
    fig_multiseed()
    print("  wrote multiseed_distribution.pdf")
    fig_distant_window()
    print("  wrote distant_window.pdf")
    fig_heatmap()
    print("  wrote scenario_cluster_heatmap.pdf")


if __name__ == "__main__":
    main()
