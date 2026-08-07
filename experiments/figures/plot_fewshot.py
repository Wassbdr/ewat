"""Generate few-shot transfer learning curve (Strategy A)."""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
CSV = ROOT / "experiments" / "rcaeval" / "fewshot_results.csv"
OUT = ROOT / "docs" / "rapport" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, metric, ylabel, threshold, title in [
    (axes[0], "h1", "Silhouette H1", 0.3, "H1 — Structurabilité"),
    (axes[1], "h3", "AUROC H3", 0.7, "H3 — Prédictibilité"),
]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    n_few = df["n_few"].values
    mean = df[mean_col].values
    std = df[std_col].values

    ax.plot(n_few, mean, "o-", color="#2E75B6", linewidth=1.8, markersize=6, label="Stratégie A")
    ax.fill_between(n_few, mean - std, mean + std, alpha=0.2, color="#2E75B6")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Seuil PASS ({threshold})")
    ax.set_xscale("log")
    ax.set_xticks(n_few)
    ax.set_xticklabels(n_few)
    ax.set_xlabel("$n_\\text{few}$ (épisodes d'adaptation)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

fig.suptitle("Courbe d'apprentissage — Transfert few-shot RCAEval (Stratégie A)",
             fontsize=11)
plt.tight_layout()

out_path = OUT / "fewshot_learning_curve.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
