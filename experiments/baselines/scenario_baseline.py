"""Scenario-classification baseline (15 classes, raw signal).

The EWAT pipeline learns an unsupervised embedding then discovers an empirical
type ontology by clustering. A natural sanity check is: how well can a simple
classifier predict the **chaos scenario** directly from the raw signal,
without any of EWAT's machinery? If the answer is "very well" then EWAT's
added value lies elsewhere — namely in the structurability of the embedding
(H1) and in the early-warning lead time (H3) — not in raw scenario
discrimination.

This baseline trains, on the train split:

* Logistic Regression (multinomial) on time-averaged + std features;
* Random Forest on the same feature vector.

It reports macro-F1, top-1 accuracy and per-scenario F1 + a confusion matrix
on val + test splits, plus paired bootstrap CIs on accuracy.

Usage
-----
    python -m experiments.baselines.scenario_baseline \\
        --dataset data/datasets/ewat_v3 \\
        --features-root data/features/v3 \\
        --output experiments/baselines/scenario \\
        [--n-bootstrap 1000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ewat.encoder.dataset import EpisodeDataset
from ewat.utils.bootstrap import bootstrap_proportion_ci

# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def _extract_features(ds: EpisodeDataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert each episode into a fixed-size feature vector.

    For an episode signal ``S ∈ ℝ^{T×N×17}`` we compute, on the time axis,
    the mean and std per (node, feature) and concatenate. This is a simple
    yet competitive baseline used in TS classification literature.

    Returns ``(X, y, ids)`` where ``y`` are scenario labels (strings) and
    ``ids`` are episode IDs in the same order as rows of ``X``.
    """
    x_data, y, ids = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        sig = item["signal"].numpy()  # (T, N, 17)
        sig = np.nan_to_num(sig, nan=0.0)
        mean = sig.mean(axis=0)        # (N, 17)
        std = sig.std(axis=0)          # (N, 17)
        feat = np.concatenate([mean.ravel(), std.ravel()], axis=0)
        x_data.append(feat)
        y.append(item["scenario"])
        ids.append(item["episode_id"])
    return np.stack(x_data), np.asarray(y, dtype=object), ids


# --------------------------------------------------------------------------- #
# Per-model evaluation
# --------------------------------------------------------------------------- #

def _evaluate(
    name: str,
    model,
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    label_order: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict:
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    metrics = {
        "model": name,
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "val_macro_f1": float(f1_score(
            y_val, y_val_pred, average="macro", labels=label_order, zero_division=0,
        )),
        "test_macro_f1": float(f1_score(
            y_test, y_test_pred, average="macro", labels=label_order, zero_division=0,
        )),
        "test_per_label_f1": {
            label: float(score) for label, score in zip(
                label_order,
                f1_score(y_test, y_test_pred, labels=label_order, average=None, zero_division=0),
            )
        },
        "test_classification_report": classification_report(
            y_test, y_test_pred, labels=label_order, output_dict=True, zero_division=0,
        ),
        "test_confusion_matrix": confusion_matrix(
            y_test, y_test_pred, labels=label_order,
        ).tolist(),
    }

    if n_bootstrap > 0:
        successes = (y_test_pred == y_test).astype(int)
        ci = bootstrap_proportion_ci(successes, n=n_bootstrap, rng=rng)
        metrics["test_accuracy_ci"] = ci.as_dict()

    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario-classification baseline")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/baselines/scenario"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    split_json = args.dataset / "split.json"
    train_ds = EpisodeDataset(split_json, args.features_root, split="train")
    val_ds = EpisodeDataset(split_json, args.features_root, split="val")
    test_ds = EpisodeDataset(split_json, args.features_root, split="test")

    print("Fitting StandardScaler on train …")
    train_ds.fit_scaler()
    val_ds.scaler = train_ds.scaler
    test_ds.scaler = train_ds.scaler

    x_tr, y_tr, _ = _extract_features(train_ds)
    x_va, y_va, _ = _extract_features(val_ds)
    x_te, y_te, ids_te = _extract_features(test_ds)
    print(f"Train: {x_tr.shape}  Val: {x_va.shape}  Test: {x_te.shape}")

    label_order = sorted(set(y_tr.tolist()) | set(y_va.tolist()) | set(y_te.tolist()))
    print(f"Scenarios ({len(label_order)}): {label_order}")

    rng = np.random.default_rng(args.seed)
    models = [
        ("logreg", LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            n_jobs=None,
            random_state=args.seed,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=args.seed,
            class_weight="balanced",
        )),
    ]

    results = {
        "config": {
            "n_train": int(x_tr.shape[0]),
            "n_val": int(x_va.shape[0]),
            "n_test": int(x_te.shape[0]),
            "feature_dim": int(x_tr.shape[1]),
            "n_scenarios": len(label_order),
            "labels": label_order,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "models": [],
    }

    for name, model in models:
        print(f"\n=== {name} ===")
        metrics = _evaluate(
            name, model, x_tr, y_tr, x_va, y_va, x_te, y_te,
            label_order, rng, args.n_bootstrap,
        )
        results["models"].append(metrics)
        ci = metrics.get("test_accuracy_ci")
        ci_str = ""
        if ci:
            ci_str = f"  CI=[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]"
        print(f"  test acc = {metrics['test_accuracy']:.3f}{ci_str}  "
              f"macro-F1 = {metrics['test_macro_f1']:.3f}")

    (args.output / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Scenario classification baseline (raw signal, 15 classes)\n",
        f"Scénarios : {len(label_order)}",
        f"Train / Val / Test : {x_tr.shape[0]} / {x_va.shape[0]} / {x_te.shape[0]}",
        f"Dim feature : {x_tr.shape[1]} (mean+std sur T → N·17·2)\n",
        "## Performances test\n",
        f"{'Model':<10} {'Acc':>8} {'CI':>20} {'MacroF1':>9}",
        "-" * 50,
    ]
    for m in results["models"]:
        ci = m.get("test_accuracy_ci")
        ci_s = f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]" if ci else "n/a"
        lines.append(
            f"{m['model']:<10} {m['test_accuracy']:>8.3f} {ci_s:>20} "
            f"{m['test_macro_f1']:>9.3f}"
        )

    lines.append("\n## F1 par scénario (modèle le plus performant)\n")
    best = max(results["models"], key=lambda r: r["test_macro_f1"])
    f1s = best["test_per_label_f1"]
    sorted_labels = sorted(f1s.items(), key=lambda kv: -kv[1])
    lines.append(f"Modèle : **{best['model']}** (macro-F1 = {best['test_macro_f1']:.3f})\n")
    lines.append(f"{'Scenario':<28} {'F1':>6}")
    for label, f1 in sorted_labels:
        lines.append(f"{label:<28} {f1:>6.3f}")

    lines.append(
        "\n## Lecture\n\n"
        "Ce baseline est complémentaire des baselines précurseur "
        "(`precursor_baselines.py`) : il prédit le **scénario chaos** depuis "
        "le signal brut, sans encodeur STGCN ni clustering siamois. Si la "
        "macro-F1 de RandomForest atteint ~0.7+, EWAT n'apporte pas une "
        "discrimination scénario/scénario fondamentalement nouvelle — sa "
        "contribution se mesure plutôt sur la structurabilité (H1) et le "
        "lead time précurseur (H3). Inversement, une F1 < 0.4 valide la "
        "nécessité d'une représentation apprise."
    )

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"Report: {args.output / 'results.md'}")


if __name__ == "__main__":
    main()
