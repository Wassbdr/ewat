"""B2 — Train directly on Chaos Mesh labels (15-way scenario classification).

Question
--------
Plutôt que de prédire des labels EWAT auto-référents (circulaire), peut-on
construire un modèle qui prédit directement les 15 scénarios Chaos Mesh
(vérité terrain indépendante) avec une AUROC honnête > 0.85 ?

Configuration
-------------
- Cible : 15 scénarios Chaos Mesh (one-vs-rest).
- Features : signal brut pré-injection, fenêtre last k steps, **instance
  normalized** (révèle la dynamique pré-injection, cf. B1 diagnostic).
- Modèle : LogisticRegression-OvR sur (k × N × 17) features aplaties — *sans*
  encodeur STGCN. C'est le compétiteur direct de B3.
- Évaluation : (i) stratified (status quo), (ii) LOSO-CV scénario par scénario
  pour mesurer la vraie généralisation.

Usage
-----
    python -m experiments.architecture_v2.chaos_mesh_target \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        [--k 6] [--n-bootstrap 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.architecture_v2.instance_norm_diagnostic import (
    _extract_window,
    _instance_normalize,
    _load_signal,
    _load_split_episodes,
    _scenario_to_int,
)
from utils.seeding import seed_everything


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 — Chaos Mesh target")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/architecture_v2/chaos_mesh_target"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-flatten-time", action="store_true",
                        help="Average over k instead of flatten (B3-style baseline).")
    return parser


def _macro_auroc(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    aurocs = []
    for i in range(n_classes):
        y_bin = (y == i).astype(int)
        if y_bin.sum() < 1 or y_bin.sum() == len(y_bin):
            continue
        try:
            aurocs.append(float(roc_auc_score(y_bin, p[:, i])))
        except ValueError:
            continue
    return float(np.mean(aurocs)) if aurocs else float("nan")


def _per_scenario_metrics(
    y: np.ndarray, p: np.ndarray, classes: list[str],
) -> list[dict]:
    """Return per-scenario {AUROC, PR-AUC, n_pos, prevalence} for transparent reporting.

    M-6 fix: macro-AUROC masks heterogeneity. Per-scenario PR-AUC is more
    informative on imbalanced classes (n_pos << n_neg).
    """
    rows = []
    n = len(y)
    for i, cls in enumerate(classes):
        y_bin = (y == i).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = n - n_pos
        if n_pos < 1 or n_neg < 1:
            rows.append({
                "scenario": cls, "n_pos": n_pos, "prevalence": n_pos / n,
                "auroc": float("nan"), "pr_auc": float("nan"),
            })
            continue
        try:
            auroc = float(roc_auc_score(y_bin, p[:, i]))
        except ValueError:
            auroc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_bin, p[:, i]))
        except ValueError:
            pr_auc = float("nan")
        rows.append({
            "scenario": cls, "n_pos": n_pos, "prevalence": n_pos / n,
            "auroc": auroc, "pr_auc": pr_auc,
        })
    return rows


def _bootstrap_ci(y, p, n_classes, n_boot, rng) -> tuple[float, float, float]:
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(_macro_auroc(y[idx], p[idx], n_classes))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if not len(boots):
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(boots)),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def _build_x_y(
    manifest: dict, features_root: Path, k: int, split: str,
    classes: list[str], flatten_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, scenarios = [], [], []
    for ep_id, info in manifest.items():
        if info["split"] != split:
            continue
        sig, normal = _load_signal(features_root, ep_id)
        win = _extract_window(sig, normal, k, "last")
        win = _instance_normalize(sig, normal, win)
        win = np.nan_to_num(win, nan=0.0)
        if flatten_time:
            feat = win.reshape(-1)   # (k * N * 17,)
        else:
            feat = win.mean(axis=0).reshape(-1)   # (N * 17,)
        X.append(feat)
        y.append(info["scenario"])
        scenarios.append(info["scenario"])
    y_int = _scenario_to_int(y, classes)
    return np.array(X, dtype=np.float32), y_int, scenarios


def _fit_predict_proba(
    X_train, y_train, X_test, classes, reg_c, max_iter
) -> np.ndarray:
    n_classes = len(classes)
    clf = LogisticRegression(C=reg_c, max_iter=max_iter, solver="lbfgs")
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    p_full = np.zeros((len(X_test), n_classes), dtype=np.float64)
    for col_idx, cls_id in enumerate(clf.classes_):
        p_full[:, int(cls_id)] = probas[:, col_idx]
    return p_full


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, classes = _load_split_episodes(args.typing_dir)
    n_classes = len(classes)
    flatten_time = not args.no_flatten_time
    print(f"{len(manifest)} ep | {n_classes} scenarios | k={args.k} | "
          f"flatten_time={flatten_time}")

    # ----- Stratified (status quo) -----
    X_train, y_train, _ = _build_x_y(
        manifest, args.features_root, args.k, "train", classes, flatten_time
    )
    X_val, y_val, _ = _build_x_y(
        manifest, args.features_root, args.k, "val", classes, flatten_time
    )
    X_test, y_test, sc_test = _build_x_y(
        manifest, args.features_root, args.k, "test", classes, flatten_time
    )
    print(f"Stratified shapes — train {X_train.shape}, val {X_val.shape}, "
          f"test {X_test.shape}")

    probas_test = _fit_predict_proba(
        X_train, y_train, X_test, classes, args.reg_c, args.max_iter
    )
    macro_strat = _macro_auroc(y_test, probas_test, n_classes)
    rng = np.random.default_rng(args.seed)
    mean_b, lo_b, hi_b = _bootstrap_ci(y_test, probas_test, n_classes,
                                       args.n_bootstrap, rng)
    print("\n** STRATIFIED (status quo) **")
    print(f"  macro-AUROC test = {macro_strat:.3f}  | "
          f"bootstrap mean {mean_b:.3f} [95% {lo_b:.3f}, {hi_b:.3f}]")

    # Per-scenario detail (AUROC + PR-AUC + prevalence — M-6 fix)
    per_scenario_full = _per_scenario_metrics(y_test, probas_test, classes)
    per_scenario = {r["scenario"]: r["auroc"] for r in per_scenario_full}
    per_scenario_pr = {r["scenario"]: r["pr_auc"] for r in per_scenario_full}

    # ----- LOSO-CV -----
    # For each held-out scenario s: train on (train ∪ val) without s, evaluate
    # on the full test set's prediction for s. Combine.
    loso_results = {}
    print("\n** LOSO-CV (15 scenarios) **")
    X_full_train = np.concatenate([X_train, X_val], axis=0)
    y_full_train = np.concatenate([y_train, y_val], axis=0)
    sc_full_train = []
    for ep_id, info in manifest.items():
        if info["split"] in ("train", "val"):
            sc_full_train.append(info["scenario"])

    # Per-scenario LOSO: train on full_train without scenario s, predict on test set
    loso_macros = []
    loso_top1 = []
    for s_idx, s in enumerate(classes):
        sc_mask = np.array([sc != s for sc in sc_full_train])
        if not sc_mask.any():
            continue
        X_tr_s = X_full_train[sc_mask]
        y_tr_s = y_full_train[sc_mask]
        probas_loso = _fit_predict_proba(
            X_tr_s, y_tr_s, X_test, classes, args.reg_c, args.max_iter
        )
        macro_loso_full = _macro_auroc(y_test, probas_loso, n_classes)

        # top-1 on held-out scenario's test episodes
        s_test_mask = np.array([sc_t == s for sc_t in sc_test])
        if s_test_mask.any():
            argmax = np.argmax(probas_loso[s_test_mask], axis=1)
            top1 = float(np.mean(argmax == y_test[s_test_mask]))
        else:
            top1 = float("nan")

        loso_macros.append(macro_loso_full)
        loso_top1.append(top1)
        loso_results[s] = {
            "macro_auroc_full_test": macro_loso_full,
            "top1_held_out": top1,
        }
        print(f"  hold-out {s:<28s} | macro(full)={macro_loso_full:.3f} | "
              f"top1(s)={top1:.3f}")

    loso_macros_arr = np.array([m for m in loso_macros if not np.isnan(m)])
    loso_top1_arr = np.array([t for t in loso_top1 if not np.isnan(t)])
    summary = {
        "k": args.k,
        "target": "chaos_mesh_scenario_15way",
        "normalization": "instance_per_episode",
        "flatten_time": flatten_time,
        "stratified_macro_auroc": macro_strat,
        "stratified_ci": [lo_b, hi_b],
        "per_scenario_stratified": per_scenario,
        "per_scenario_full_stratified": per_scenario_full,
        "loso_macro_auroc_full_test_mean": float(np.mean(loso_macros_arr)),
        "loso_macro_auroc_full_test_std": float(np.std(loso_macros_arr)),
        "loso_top1_held_out_mean": float(np.mean(loso_top1_arr)),
        "loso_top1_held_out_std": float(np.std(loso_top1_arr)),
        "per_scenario_loso": loso_results,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    print("\n** AGGREGATE LOSO **")
    print(f"  macro-AUROC (full test, 15 folds) = "
          f"{summary['loso_macro_auroc_full_test_mean']:.3f} ± "
          f"{summary['loso_macro_auroc_full_test_std']:.3f}")
    print(f"  top-1 acc (held-out s) = "
          f"{summary['loso_top1_held_out_mean']:.3f} ± "
          f"{summary['loso_top1_held_out_std']:.3f}")

    # Markdown
    lines = [
        "# B2 — Chaos Mesh target (instance norm + temporal flatten)",
        "",
        f"Cible : 15 scénarios Chaos Mesh (vérité terrain indépendante). "
        f"k = {args.k} | features = signal pré-injection (last k) instance-normalized | "
        f"modèle = LR-OvR | {'flatten' if flatten_time else 'mean'} over k.",
        "",
        "## Headline (honest)",
        "",
        f"- **Stratified macro-AUROC test = {macro_strat:.3f}** "
        f"[95% CI {lo_b:.3f}, {hi_b:.3f}]",
        f"- **LOSO macro-AUROC full test (15 folds) = "
        f"{summary['loso_macro_auroc_full_test_mean']:.3f} ± "
        f"{summary['loso_macro_auroc_full_test_std']:.3f}**",
        f"- LOSO top-1 acc on held-out scenario = "
        f"{summary['loso_top1_held_out_mean']:.3f} ± "
        f"{summary['loso_top1_held_out_std']:.3f}",
        "",
        "## Per-scenario AUROC + PR-AUC (stratified, M-6)",
        "",
        "AUROC peut être trompeur sur classes déséquilibrées. PR-AUC (Average Precision) "
        "est plus informatif quand n_pos << n_neg. La table ci-dessous expose la "
        "redistribution de difficulté par scénario.",
        "",
        "| scenario | n_pos | prevalence | AUROC | PR-AUC |",
        "|---|---|---|---|---|",
    ]
    for r in per_scenario_full:
        auc_s = f"{r['auroc']:.3f}" if not np.isnan(r['auroc']) else "NaN"
        pr_s = f"{r['pr_auc']:.3f}" if not np.isnan(r['pr_auc']) else "NaN"
        lines.append(
            f"| {r['scenario']} | {r['n_pos']} | {r['prevalence']:.3f} | "
            f"{auc_s} | {pr_s} |"
        )
    lines += [
        "",
        "## Per-scenario LOSO",
        "",
        "| held-out | macro-AUROC (full test) | top-1 acc (3 ép) |",
        "|---|---|---|",
    ]
    for s in classes:
        r = loso_results.get(s)
        if r is None:
            continue
        lines.append(f"| {s} | {r['macro_auroc_full_test']:.3f} | "
                     f"{r['top1_held_out']:.3f} |")

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
