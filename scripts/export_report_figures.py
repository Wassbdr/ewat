"""Génère les schémas d'architecture du rapport (sans graphviz).

Produit deux PNG dans docs/rapport/figures/ :
- pipeline_architecture.png : pipeline complet S(t) -> etape 0..3 -> Alert
- pipeline_operational_v2.png : chaine operationnelle S(t) -> instance norm -> LR-OvR -> OpenMax

Usage : python -m scripts.export_report_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path("docs/rapport/figures")
BLUE = "#2b6cb0"
GREY = "#4a5568"
GREEN = "#2f855a"
LIGHT = "#ebf2fa"
LIGHTG = "#e6f4ea"


def _box(ax, x, y, w, h, title, sub, edge=BLUE, face=LIGHT):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=edge)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=8.2, color=GREY)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREY))


def pipeline_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")

    w, h, y = 1.55, 1.4, 0.9
    xs = [0.15, 2.05, 3.95, 5.85, 7.75]
    steps = [
        ("Étape 0", "Détection drift\nMMD-RFF", BLUE, LIGHT),
        ("Étape 1", "Encodeur\nSTGCN", BLUE, LIGHT),
        ("Étape 2", "Typage siamois\n+ clustering", BLUE, LIGHT),
        ("Étape 2b", "Ontologie\n(hors ligne)", GREEN, LIGHTG),
        ("Étape 3", "Précurseurs\ntypés", BLUE, LIGHT),
    ]
    for (x, (t, s, e, f)) in zip(xs, steps):
        _box(ax, x, y, w, h, t, s, edge=e, face=f)
    # entrée S(t)
    ax.text(0.15 + w / 2, y + h + 0.28, "S(t) ∈ ℝ^{N×17}", ha="center",
            fontsize=9.5, color=GREY)
    # flèches horizontales (0->1->2->3 ; 2->2b en pointillé)
    for i in range(4):
        if i == 2:
            continue
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    _arrow(ax, xs[2] + w, y + h / 2, xs[3], y + h / 2)
    _arrow(ax, xs[3] + w, y + h / 2, xs[4], y + h / 2)
    # sortie Alert
    _box(ax, 9.45, y, 1.4, h, "Alert(t)", "(C_i, p̂_i, k*_i,\nfiche)", edge=GREY, face="#f7fafc")
    _arrow(ax, xs[4] + w, y + h / 2, 9.45, y + h / 2)

    fig.tight_layout()
    fig.savefig(OUT / "pipeline_architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def pipeline_operational_v2():
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    w, h, y = 1.9, 1.1, 0.5
    xs = [0.2, 2.45, 4.7, 6.95]
    blocks = [
        ("S(t)", "signal brut", GREY, "#f7fafc"),
        ("Instance norm", "par épisode", BLUE, LIGHT),
        ("LR one-vs-rest", "15 scénarios", BLUE, LIGHT),
        ("OpenMax", "signal nouveauté", GREEN, LIGHTG),
    ]
    for (x, (t, s, e, f)) in zip(xs, blocks):
        _box(ax, x, y, w, h, t, s, edge=e, face=f)
    for i in range(3):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    _box(ax, 9.0, y, 0.9, h, "Alerte", "", edge=GREY, face="#f7fafc")
    _arrow(ax, xs[3] + w, y + h / 2, 9.0, y + h / 2)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_operational_v2.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline_architecture()
    pipeline_operational_v2()
    print("OK ->", OUT / "pipeline_architecture.png")
    print("OK ->", OUT / "pipeline_operational_v2.png")
