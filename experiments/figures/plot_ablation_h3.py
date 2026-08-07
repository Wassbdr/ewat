"""Generate ablation H3 heatmap: conditions x clusters x AUROC."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
CSV = ROOT / "experiments" / "ablation" / "results_h3_ablation.csv"
OUT = ROOT / "docs" / "rapport" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Keep modality ablations only (first 7 rows) for readability
modality_conditions = ["full", "M_only", "T_only", "L_only", "M+T", "M+L", "T+L"]
df_mod = df[df["condition"].isin(modality_conditions)].set_index("condition")
df_mod = df_mod.reindex(modality_conditions)

# Top LOO conditions (most critical: delta < -0.03)
loo_conditions = [c for c in df["condition"].values
                  if c.startswith("drop_") and df.loc[df["condition"] == c, "delta_macro"].values[0] < -0.03]
loo_conditions = sorted(loo_conditions,
                        key=lambda c: df.loc[df["condition"] == c, "delta_macro"].values[0])[:8]
df_loo = df[df["condition"].isin(loo_conditions)].set_index("condition")
df_loo = df_loo.reindex(loo_conditions)

cluster_cols = [f"C{i}" for i in range(10)]
cluster_labels = [
    "C0\nfail_slow", "C1\ntraffic", "C2\nresource", "C3\nnoisy",
    "C4\ncrash", "C5\nrolling", "C6\nconfig", "C7\ncpu", "C8\nfaulty", "C9\nscale",
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                         gridspec_kw={"width_ratios": [len(modality_conditions), len(loo_conditions)]})

cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0.4, vmax=1.0)

for ax, df_sub, title in [
    (axes[0], df_mod, "Ablation par modalité"),
    (axes[1], df_loo, "Leave-one-out (top 8 critiques)"),
]:
    mat = df_sub[cluster_cols].values.astype(float)
    im = ax.imshow(mat.T, aspect="auto", cmap=cmap, norm=norm, origin="upper")

    ax.set_xticks(range(len(df_sub)))
    ax.set_xticklabels(df_sub.index, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(cluster_labels, fontsize=7.5)
    ax.set_title(title, fontsize=10, pad=6)

    for row in range(len(df_sub)):
        for col in range(10):
            v = mat[row, col]
            if not np.isnan(v):
                txt = f"{v:.2f}"
                color = "white" if v < 0.65 else "black"
                ax.text(row, col, txt, ha="center", va="center", fontsize=6.5, color=color)
            else:
                ax.text(row, col, "—", ha="center", va="center", fontsize=7, color="#888888")

fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.015, pad=0.02,
             label="AUROC test")
fig.suptitle("Sensibilité de l'AUROC H3 par masquage à l'inférence (graine 42)",
             fontsize=11, y=1.01)
plt.tight_layout()

out_path = OUT / "ablation_h3_heatmap.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
