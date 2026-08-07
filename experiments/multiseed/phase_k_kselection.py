"""Phase K.1 + K.3 — K-selection diagnostic + variance per-seed analysis.

K.1: Recompute K_optimal on each seed's train embeddings using two strategies:
- ``silhouette``      (current default — argmax silhouette)
- ``gap_tibshirani``  (Step 6 fix 6.4 — smallest K s.t. gap(K) ≥ gap(K+1) − s(K+1))

K.3: Aggregate per-seed metrics and write distribution summary.

Output:
- experiments/multiseed/phase_h/k_selection_comparison.md
- experiments/multiseed/phase_h/k_selection_comparison.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ewat.typing.clustering import cluster_embeddings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-h-dir", type=Path,
                   default=Path("experiments/multiseed/phase_h"))
    p.add_argument("--k-range-max", type=int, default=16)
    p.add_argument("--n-gap-refs", type=int, default=10)
    args = p.parse_args()

    seed_dirs = sorted(args.phase_h_dir.glob("seed_*"))
    print(f"Found {len(seed_dirs)} seed directories")

    rows = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed_", ""))
        artifacts = sd / "typing" / "cluster_artifacts"
        if not artifacts.exists():
            continue
        z_train = np.load(artifacts / "embeddings_train.npy")
        print(f"\nSeed {seed:5d}: embeddings (N={z_train.shape[0]}, d={z_train.shape[1]})")

        # 1) silhouette (current default)
        res_sil = cluster_embeddings(
            z_train, k_range=range(2, args.k_range_max),
            n_gap_refs=args.n_gap_refs, random_state=42,
            linkage="average", metric="cosine",
            k_selection_method="silhouette",
        )
        # 2) gap_tibshirani
        res_tib = cluster_embeddings(
            z_train, k_range=range(2, args.k_range_max),
            n_gap_refs=args.n_gap_refs, random_state=42,
            linkage="average", metric="cosine",
            k_selection_method="gap_tibshirani",
        )
        rows.append({
            "seed": seed,
            "K_silhouette": int(res_sil.k_optimal),
            "K_tibshirani": int(res_tib.k_optimal),
            "max_sil_score": max(res_sil.silhouette_scores.values()),
            "gap_at_K_silhouette": res_sil.gap_stats.get(res_sil.k_optimal),
            "gap_at_K_tibshirani": res_tib.gap_stats.get(res_tib.k_optimal),
            "gap_se_at_K_tib": res_tib.gap_se.get(res_tib.k_optimal),
        })
        print(f"  K_silhouette={rows[-1]['K_silhouette']}  K_tibshirani={rows[-1]['K_tibshirani']}")

    if not rows:
        print("No seeds with embeddings found.")
        return

    k_sil = [r["K_silhouette"] for r in rows]
    k_tib = [r["K_tibshirani"] for r in rows]
    sil_dist = Counter(k_sil)
    tib_dist = Counter(k_tib)
    n_agree = sum(1 for r in rows if r["K_silhouette"] == r["K_tibshirani"])

    summary = {
        "n_seeds": len(rows),
        "silhouette": {
            "distribution": dict(sil_dist),
            "mode": sil_dist.most_common(1)[0],
            "mean": float(np.mean(k_sil)),
            "std": float(np.std(k_sil, ddof=0)),
            "min": min(k_sil),
            "max": max(k_sil),
        },
        "tibshirani": {
            "distribution": dict(tib_dist),
            "mode": tib_dist.most_common(1)[0],
            "mean": float(np.mean(k_tib)),
            "std": float(np.std(k_tib, ddof=0)),
            "min": min(k_tib),
            "max": max(k_tib),
        },
        "agreement": {
            "n_agree": n_agree,
            "n_total": len(rows),
            "rate": n_agree / len(rows),
        },
        "per_seed": rows,
    }
    out = args.phase_h_dir / "k_selection_comparison.json"
    out.write_text(json.dumps(summary, indent=2))

    # Markdown
    lines = [
        "# Phase K.1 — K selection comparison (silhouette vs gap_tibshirani)",
        "",
        f"_Generated from {args.phase_h_dir} embeddings (10 seeds)_",
        "",
        "## Method",
        "",
        "Two strategies for choosing the optimal K from train embeddings:",
        "- **silhouette** (current default): K = argmax silhouette(K), fragile when curve is flat",
        "- **gap_tibshirani** (Step 6 fix): K = smallest K s.t. gap(K) ≥ gap(K+1) − s(K+1)",
        "",
        f"## Distribution comparison (n={len(rows)} seeds)",
        "",
        "| Strategy | Mode (count) | Mean ± Std | Range | Distribution |",
        "|---|---|---|---|---|",
        f"| silhouette | K={sil_dist.most_common(1)[0][0]} ({sil_dist.most_common(1)[0][1]}/{len(rows)}) | "
        f"{summary['silhouette']['mean']:.1f} ± {summary['silhouette']['std']:.1f} | "
        f"[{summary['silhouette']['min']}, {summary['silhouette']['max']}] | "
        f"{dict(sorted(sil_dist.items()))} |",
        f"| gap_tibshirani | K={tib_dist.most_common(1)[0][0]} ({tib_dist.most_common(1)[0][1]}/{len(rows)}) | "
        f"{summary['tibshirani']['mean']:.1f} ± {summary['tibshirani']['std']:.1f} | "
        f"[{summary['tibshirani']['min']}, {summary['tibshirani']['max']}] | "
        f"{dict(sorted(tib_dist.items()))} |",
        "",
        f"**Agreement** : {n_agree}/{len(rows)} seeds chose the same K with both strategies ({n_agree/len(rows)*100:.0f}%)",
        "",
        "## Per-seed table",
        "",
        "| Seed | K_silhouette | K_tibshirani | Agree? | gap@K_tib | gap_se@K_tib |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        agree = "✅" if r["K_silhouette"] == r["K_tibshirani"] else "❌"
        gap = f"{r['gap_at_K_tibshirani']:.3f}" if r['gap_at_K_tibshirani'] is not None else "—"
        se = f"{r['gap_se_at_K_tib']:.3f}" if r['gap_se_at_K_tib'] is not None else "—"
        lines.append(
            f"| {r['seed']} | {r['K_silhouette']} | {r['K_tibshirani']} | {agree} | {gap} | {se} |"
        )

    # Verdict
    sil_dominant = summary['silhouette']['mode'][1] >= 6
    tib_dominant = summary['tibshirani']['mode'][1] >= 6
    tib_better = tib_dominant and not sil_dominant
    lines += [
        "",
        "## Verdict",
        "",
        f"- silhouette mode K={sil_dist.most_common(1)[0][0]} dominates ≥6/10 seeds: "
        f"{'✅' if sil_dominant else '❌'}",
        f"- gap_tibshirani mode K={tib_dist.most_common(1)[0][0]} dominates ≥6/10 seeds: "
        f"{'✅' if tib_dominant else '❌'}",
        "",
    ]
    if tib_better:
        lines.append(
            f"**Recommendation**: switch default to gap_tibshirani. K={tib_dist.most_common(1)[0][0]} "
            f"becomes stable across seeds, reducing variance in downstream H1/H3 metrics."
        )
    elif sil_dominant:
        lines.append("**Recommendation**: keep silhouette default; both strategies agree well.")
    else:
        lines.append(
            "**Recommendation**: K is intrinsically unstable on this dataset. Consider "
            "fixing K manually (e.g., 10 = number of natural categories) or use HDBSCAN "
            "density-based clustering for v5."
        )

    md_out = args.phase_h_dir / "k_selection_comparison.md"
    md_out.write_text("\n".join(lines))
    print(f"\nWrote {out}")
    print(f"Wrote {md_out}")
    print(f"\nsilhouette: mode K={sil_dist.most_common(1)[0][0]} ({sil_dist.most_common(1)[0][1]}/{len(rows)})")
    print(f"tibshirani: mode K={tib_dist.most_common(1)[0][0]} ({tib_dist.most_common(1)[0][1]}/{len(rows)})")
    print(f"agreement : {n_agree}/{len(rows)}")


if __name__ == "__main__":
    main()
