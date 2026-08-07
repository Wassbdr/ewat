"""Phase H aggregator: produce mean ± std + distributions from all seeds.

Reads ``experiments/multiseed/phase_h/all_summaries.json`` (or per-seed
``summary.json``) and emits ``aggregate.json`` + ``results.md``.

Usage
-----
    python -m experiments.multiseed.aggregate_phase_h \\
        --output experiments/multiseed/phase_h
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _safe_mean_std(values: list[float]) -> tuple[float, float, int]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)],
                   dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate Phase H multi-seed results")
    p.add_argument("--output", type=Path,
                   default=Path("experiments/multiseed/phase_h"))
    args = p.parse_args()

    summary_path = args.output / "all_summaries.json"
    if not summary_path.exists():
        raise SystemExit(f"Not found: {summary_path}. Run run_phase_h.py first.")
    summaries: list[dict] = json.loads(summary_path.read_text())
    if not summaries:
        raise SystemExit("Empty summaries")

    valid = [s for s in summaries if "failed" not in s]
    failed = [s for s in summaries if "failed" in s]

    sil_test = [s["H1"]["silhouette_test"] for s in valid]
    sil_val = [s["H1"]["silhouette_val"] for s in valid]
    sil_train = [s["H1"]["silhouette_train"] for s in valid]
    K_opt = [s["H1"]["K_optimal"] for s in valid]
    best_epoch = [s["H1"]["best_epoch_siamese"] for s in valid]
    h1_pass = sum(1 for s in valid if s["H1"]["h1_pass"])

    auroc_peak = [s["H3"]["auroc_peak_test_mean"] for s in valid]
    h3_pass = sum(1 for s in valid if s["H3"]["h3_pass"])
    # E3+E8 (audit 2026-06): PR-AUC + filtre reportable n_pos≥5 (absents des
    # summaries pré-audit → NaN tolérés par _safe_mean_std)
    pr_peak = [s["H3"].get("pr_auc_peak_test_mean", float("nan")) for s in valid]
    auroc_rep = [s["H3"].get("auroc_peak_test_mean_reportable", float("nan"))
                 for s in valid]
    pr_rep = [s["H3"].get("pr_auc_peak_test_mean_reportable", float("nan"))
              for s in valid]

    delta_far_near = [s["A1"]["delta_far_near_macro"] for s in valid
                      if s["A1"] and s["A1"].get("delta_far_near_macro") is not None]
    a1_verdicts = Counter([s["A1"]["verdict"] for s in valid
                          if s["A1"] and s["A1"].get("verdict")])

    sil_test_mean, sil_test_std, sil_test_n = _safe_mean_std(sil_test)
    auroc_mean, auroc_std, auroc_n = _safe_mean_std(auroc_peak)
    delta_mean, delta_std, delta_n = _safe_mean_std(delta_far_near)
    best_epoch_mean, best_epoch_std, _ = _safe_mean_std(best_epoch)

    k_dist = Counter(K_opt)
    k_mode = k_dist.most_common(1)[0] if k_dist else (None, 0)
    delta_below_neg004 = sum(1 for d in delta_far_near if d is not None and d <= -0.04)

    aggregate = {
        "n_seeds_total": len(summaries),
        "n_seeds_valid": len(valid),
        "n_seeds_failed": len(failed),
        "failed_seeds": [{"seed": s["seed"], "step": s["failed"]} for s in failed],
        "H1": {
            "silhouette_test_mean": sil_test_mean,
            "silhouette_test_std": sil_test_std,
            "silhouette_test_min": min(sil_test) if sil_test else None,
            "silhouette_test_max": max(sil_test) if sil_test else None,
            "silhouette_val_mean": _safe_mean_std(sil_val)[0],
            "silhouette_train_mean": _safe_mean_std(sil_train)[0],
            "h1_pass_count": h1_pass,
            "h1_pass_rate": h1_pass / max(len(valid), 1),
            "best_epoch_mean": best_epoch_mean,
            "best_epoch_std": best_epoch_std,
            "K_distribution": dict(k_dist),
            "K_mode": k_mode[0],
            "K_mode_count": k_mode[1],
        },
        "H3": {
            "auroc_peak_test_mean": auroc_mean,
            "auroc_peak_test_std": auroc_std,
            "h3_pass_count": h3_pass,
            "h3_pass_rate": h3_pass / max(len(valid), 1),
            # E3+E8 (audit 2026-06)
            "pr_auc_peak_test_mean": _safe_mean_std(pr_peak)[0],
            "pr_auc_peak_test_std": _safe_mean_std(pr_peak)[1],
            "auroc_peak_reportable_mean": _safe_mean_std(auroc_rep)[0],
            "auroc_peak_reportable_std": _safe_mean_std(auroc_rep)[1],
            "pr_auc_peak_reportable_mean": _safe_mean_std(pr_rep)[0],
            "pr_auc_peak_reportable_std": _safe_mean_std(pr_rep)[1],
        },
        "A1": {
            "delta_far_near_mean": delta_mean,
            "delta_far_near_std": delta_std,
            "n_seeds_with_a1": delta_n,
            "verdicts": dict(a1_verdicts),
            "n_seeds_delta_below_neg004": delta_below_neg004,
        },
        "per_seed": valid,
    }

    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    # Markdown report
    lines = [
        "# Phase H — Multi-seed retrain aggregate",
        "",
        f"_Generated 2026-05-26 from {args.output}/all_summaries.json_",
        "",
        f"- Seeds total : {aggregate['n_seeds_total']}",
        f"- Seeds valid : {aggregate['n_seeds_valid']}",
        f"- Seeds failed : {aggregate['n_seeds_failed']}",
        "",
        "## H1 — Silhouette",
        "",
        "| Stat | Train | Val | Test |",
        "|---|---|---|---|",
        f"| Mean | {_safe_mean_std(sil_train)[0]:.3f} | {_safe_mean_std(sil_val)[0]:.3f} | **{sil_test_mean:.3f}** |",
        f"| Std  | {_safe_mean_std(sil_train)[1]:.3f} | {_safe_mean_std(sil_val)[1]:.3f} | **{sil_test_std:.3f}** |",
        f"| Min  | — | — | {min(sil_test) if sil_test else float('nan'):.3f} |",
        f"| Max  | — | — | {max(sil_test) if sil_test else float('nan'):.3f} |",
        "",
        f"**H1 PASS rate** : {h1_pass}/{len(valid)} ({h1_pass/max(len(valid),1)*100:.0f}%)",
        f"**K distribution** : {dict(k_dist)} (mode K={k_mode[0]}, n={k_mode[1]})",
        f"**Best epoch siamese** : {best_epoch_mean:.1f} ± {best_epoch_std:.1f}",
        "",
        "## H3 — Precursor AUROC",
        "",
        f"- **AUROC peak test mean ± std** : {auroc_mean:.3f} ± {auroc_std:.3f}",
        f"- **PR-AUC peak test mean ± std** : {_safe_mean_std(pr_peak)[0]:.3f} "
        f"± {_safe_mean_std(pr_peak)[1]:.3f}  _(E3 audit 2026-06)_",
        f"- AUROC reportable (n_pos≥5) : {_safe_mean_std(auroc_rep)[0]:.3f} "
        f"± {_safe_mean_std(auroc_rep)[1]:.3f}  _(E8)_",
        f"- PR-AUC reportable (n_pos≥5) : {_safe_mean_std(pr_rep)[0]:.3f} "
        f"± {_safe_mean_std(pr_rep)[1]:.3f}",
        f"- H3 PASS rate : {h3_pass}/{len(valid)}",
        "",
        "## A1 — Distant-window",
        "",
        f"- **Δ(far − near) mean ± std** : {delta_mean:+.4f} ± {delta_std:.4f}",
        f"- Verdicts : {dict(a1_verdicts)}",
        f"- Seeds with Δ ≤ −0.04 (GENUINE_PRECURSION threshold) : {delta_below_neg004}/{delta_n}",
        "",
        "## Per-seed table",
        "",
        "| Seed | sil_test | K | best_epoch | AUROC_peak | Δ(far-near) | A1 verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in valid:
        a1 = s.get("A1") or {}
        delta = a1.get("delta_far_near_macro")
        verdict = a1.get("verdict") or "—"
        delta_str = f"{delta:+.4f}" if delta is not None else "—"
        lines.append(
            f"| {s['seed']} | {s['H1']['silhouette_test']:.3f} | "
            f"{s['H1']['K_optimal']} | "
            f"{s['H1']['best_epoch_siamese']} | "
            f"{s['H3']['auroc_peak_test_mean']:.3f} | "
            f"{delta_str} | {verdict} |"
        )

    if failed:
        lines += ["", "## Failed seeds", ""]
        for s in failed:
            lines.append(f"- seed {s['seed']} failed at step **{s['failed']}**")

    # Acceptance criteria
    sil_ok = sil_test_mean >= 0.6 and sil_test_std <= 0.15
    auroc_ok = auroc_mean >= 0.95 and auroc_std <= 0.05
    a1_ok = delta_below_neg004 >= 8
    k_ok = k_mode[1] >= 6
    lines += [
        "",
        "## Acceptance criteria (plan H.3)",
        "",
        f"- {'✅' if sil_ok else '❌'} sil_test mean ≥ 0.6 and std ≤ 0.15 → got {sil_test_mean:.3f} ± {sil_test_std:.3f}",
        f"- {'✅' if auroc_ok else '❌'} H3 AUROC peak mean ≥ 0.95 and std ≤ 0.05 → got {auroc_mean:.3f} ± {auroc_std:.3f}",
        f"- {'✅' if a1_ok else '❌'} A1 Δ ≤ −0.04 on ≥ 8/10 seeds → got {delta_below_neg004}/{delta_n}",
        f"- {'✅' if k_ok else '❌'} K mode dominates ≥ 6/10 seeds → K={k_mode[0]} on {k_mode[1]}/{aggregate['n_seeds_valid']}",
    ]

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"Wrote {args.output / 'aggregate.json'}")
    print(f"Wrote {args.output / 'results.md'}")
    print()
    print(f"H1 sil_test : {sil_test_mean:.3f} ± {sil_test_std:.3f} (n={sil_test_n})")
    print(f"H3 AUROC    : {auroc_mean:.3f} ± {auroc_std:.3f}")
    print(f"A1 Δ        : {delta_mean:+.4f} ± {delta_std:.4f} (n={delta_n})")


if __name__ == "__main__":
    main()
