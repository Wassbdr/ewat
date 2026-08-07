"""Generate AUROC H3 per cluster bar chart with 95% bootstrap CI."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
PRECURSOR_DIR = ROOT / "experiments" / "precursor"
OUT = ROOT / "docs" / "rapport" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(PRECURSOR_DIR / "results.json") as f:
    res = json.load(f)

k_opt = res["k_optimal"]
auroc_test = res["auroc_test"]
ci = res.get("auroc_ci_95", {})

cluster_names = {
    "0": "C0\nfail\\_slow\\_cpu",
    "1": "C1\ntraffic\\_ramp",
    "2": "C2\nresource\\_leak",
    "3": "C3\nnoisy\\_neighbor",
    "4": "C4\ncrash",
    "5": "C5\nrolling\\_deploy",
    "6": "C6\nconfig\\_change",
    "7": "C7\ncpu\\_starvation",
    "8": "C8\nfaulty\\_deploy",
    "9": "C9\nscale\\_up",
}

drift_clusters = {"1", "5", "6", "9"}

aurocs, lo_errs, hi_errs, labels, colors = [], [], [], [], []
for c in range(10):
    cs = str(c)
    k = str(k_opt.get(cs, 6))
    val = auroc_test.get(k, {}).get(cs, None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        val = np.nan
    aurocs.append(val)
    lo = ci.get(cs, {}).get("lo", np.nan)
    hi = ci.get(cs, {}).get("hi", np.nan)
    lo_errs.append(val - lo if not np.isnan(lo) else 0)
    hi_errs.append(hi - val if not np.isnan(hi) else 0)
    labels.append(cluster_names[cs])
    colors.append("#5B9BD5" if cs in drift_clusters else "#ED7D31")

x = np.arange(10)
fig, ax = plt.subplots(figsize=(11, 4.5))

valid = ~np.isnan(aurocs)
xerr = np.array([lo_errs, hi_errs])
for i in x:
    if valid[i]:
        ax.bar(i, aurocs[i], color=colors[i], width=0.6, zorder=3)
        if lo_errs[i] > 0 or hi_errs[i] > 0:
            ax.errorbar(i, aurocs[i], yerr=[[lo_errs[i]], [hi_errs[i]]],
                        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    else:
        ax.bar(i, 0, color="#CCCCCC", width=0.6, zorder=3)
        ax.text(i, 0.02, "NaN\n($n_{+}<2$)", ha="center", va="bottom",
                fontsize=7.5, color="#666666")

ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Baseline aléatoire (0.5)")
ax.axhline(0.3, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1.08)
ax.set_ylabel("AUROC test (IC 95\\,\\% bootstrap)", fontsize=10)
ax.set_xlabel("Cluster", fontsize=10)
ax.set_title("AUROC précurseur H3 par cluster (graine 42, $k^*$ sur val)", fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

drift_patch = mpatches.Patch(color="#5B9BD5", label="Drift bénin")
anomaly_patch = mpatches.Patch(color="#ED7D31", label="Anomalie vraie")
ax.legend(handles=[anomaly_patch, drift_patch,
                   plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1,
                              label="Baseline aléatoire (0.5)")],
          fontsize=9, loc="lower right")

plt.tight_layout()
out_path = OUT / "auroc_h3_per_cluster.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
