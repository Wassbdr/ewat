"""B1 — Instance normalization diagnostic.

Question
--------
A1 showed that a window placed at the *beginning* of the normal regime
(`window_position='first'`) yields the same AUROC as right before injection
(`'last'`). The model exploits *static scenario signature* (absolute baselines
of CPU/RAM/latency per service) rather than pre-injection dynamics.

If we *instance-normalize* the raw signal — subtract each episode's own mean
(over the normal regime) and divide by its own std — we destroy the static
baseline, leaving only the relative dynamics. If the AUROC then collapses,
the headline number was entirely scenario-baseline. If some signal survives,
there is real dynamic content.

Method
------
For each (window_position ∈ {last, middle, first}, norm_mode ∈ {global, instance}):
  - Extract window of length k from the test episodes.
  - For instance norm: per-episode, per-feature, per-node z-score using the
    full normal regime as reference statistics, applied to the window.
  - Flatten and fit LR (one-vs-rest) on Chaos Mesh scenario labels.
  - Macro-AUROC + bootstrap CI.

Cible : **scénarios Chaos Mesh** (15 classes) — vérité terrain indépendante,
pas les labels EWAT auto-référents.

Usage
-----
    python -m experiments.architecture_v2.instance_norm_diagnostic \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        [--k 6] [--n-bootstrap 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Window extraction with normalization mode
# ---------------------------------------------------------------------------

def _load_signal(features_root: Path, ep_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (signal, normal_mask) for a single episode."""
    ep_dir = features_root / ep_id
    sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)  # (T, N, 17)
    labels_df = pd.read_parquet(ep_dir / "labels.parquet",
                                columns=["regime"])
    normal = (labels_df["regime"] == "normal").values
    return sig, normal


def _extract_window(
    sig: np.ndarray, normal_mask: np.ndarray, k: int, position: str
) -> np.ndarray:
    """Return (k, N, 17) window — left-padded with zeros if window too short."""
    normal_idx = np.where(normal_mask)[0]
    if len(normal_idx) == 0:
        normal_idx = np.arange(min(k, sig.shape[0]))
    n = len(normal_idx)
    if position == "last":
        sel = normal_idx[-k:]
    elif position == "first":
        sel = normal_idx[:k]
    else:  # middle
        if n <= k:
            sel = normal_idx
        else:
            start = (n - k) // 2
            sel = normal_idx[start: start + k]
    win = sig[sel]  # (actual, N, 17)
    actual = win.shape[0]
    if actual < k:
        pad = np.zeros((k - actual, *win.shape[1:]), dtype=np.float32)
        win = np.concatenate([pad, win], axis=0)
    return win


def _instance_normalize(
    sig: np.ndarray, normal_mask: np.ndarray, win: np.ndarray
) -> np.ndarray:
    """Per-episode z-score: use NORMAL regime as reference statistics.

    sig:         (T, N, 17) full episode (used for ref stats).
    normal_mask: bool (T,)  — which timesteps are normal regime.
    win:         (k, N, 17) the extracted window to normalize.

    Stats are per-feature (across nodes & normal timesteps) so we remove the
    *global episode-level* baseline. Per-node-per-feature stats would remove
    even service identity — keep nodes intact so the GNN structure remains
    meaningful.
    """
    normal_idx = np.where(normal_mask)[0]
    if len(normal_idx) < 2:
        return win  # not enough samples, return as is
    ref = sig[normal_idx]  # (n_normal, N, 17)
    # Compute mean/std per-feature, ignoring NaN
    mu = np.nanmean(ref, axis=(0, 1), keepdims=True)   # (1, 1, 17)
    sd = np.nanstd(ref, axis=(0, 1), keepdims=True) + 1e-6
    return (win - mu) / sd


def _global_normalize(win: np.ndarray, scaler_mu: np.ndarray, scaler_sd: np.ndarray) -> np.ndarray:
    """Apply train-wide StandardScaler statistics."""
    return (win - scaler_mu) / (scaler_sd + 1e-6)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B1 — Instance normalization diagnostic")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/architecture_v2/instance_norm_diagnostic"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
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


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, n_classes: int,
                  n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(_macro_auroc(y[idx], p[idx], n_classes))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if not len(boots):
        return float("nan"), float("nan")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _load_split_episodes(typing_dir: Path) -> tuple[dict, list[str]]:
    """Load episode → {scenario, split} from cluster_manifest.json if present,
    otherwise from a dataset's index.parquet (treats --typing-dir as dataset
    root). Cluster info absent in the index.parquet fallback.
    """
    manifest_path = typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        scenarios = sorted({v["scenario"] for v in manifest.values()})
        return manifest, scenarios
    index_path = typing_dir / "index.parquet"
    if index_path.exists():
        import pandas as pd
        df = pd.read_parquet(index_path)
        manifest = {
            row["episode_id"]: {
                "scenario": row["scenario"], "split": row["split"], "cluster": -1,
            }
            for _, row in df.iterrows()
        }
        scenarios = sorted(df["scenario"].unique().tolist())
        return manifest, scenarios
    raise FileNotFoundError(
        f"Neither {manifest_path} nor {index_path} exists — pass either a typing-dir "
        "containing cluster_manifest.json or a dataset dir containing index.parquet."
    )


def _build_features(
    manifest: dict, features_root: Path, k: int, position: str,
    norm_mode: str, scaler_stats: tuple[np.ndarray, np.ndarray] | None,
    split: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X_flat, y_scenario, ep_ids) for the given split & config."""
    X, y, eps = [], [], []
    for ep_id, info in manifest.items():
        if info["split"] != split:
            continue
        sig, normal = _load_signal(features_root, ep_id)
        win = _extract_window(sig, normal, k, position)        # (k, N, 17)
        if norm_mode == "instance":
            win = _instance_normalize(sig, normal, win)
        elif norm_mode == "global" and scaler_stats is not None:
            mu, sd = scaler_stats
            win = _global_normalize(win, mu, sd)
        win = np.nan_to_num(win, nan=0.0)
        # Aggregate over time (mean) to get a fixed-size feature vector — same
        # as scenario_baselines does (k × N × 17 → N × 17 → flatten).
        feat = win.mean(axis=0).reshape(-1)   # (N*17,)
        X.append(feat)
        y.append(info["scenario"])
        eps.append(ep_id)
    return np.array(X, dtype=np.float32), y, eps


def _scenario_to_int(scenarios: list[str], classes: list[str]) -> np.ndarray:
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    return np.array([cls_to_idx[s] for s in scenarios], dtype=int)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, classes = _load_split_episodes(args.typing_dir)
    n_classes = len(classes)
    print(f"{len(manifest)} ep | {n_classes} scenarios | k={args.k}")

    # Compute global scaler stats from train split, per-feature (17-dim)
    print("Computing global train scaler statistics …")
    train_feats = []
    for ep_id, info in manifest.items():
        if info["split"] != "train":
            continue
        sig, normal = _load_signal(args.features_root, ep_id)
        win = _extract_window(sig, normal, args.k, "last")
        train_feats.append(win)
    arr = np.concatenate(train_feats, axis=0)   # (n_ep*k, N, 17)
    flat = arr.reshape(-1, arr.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    scaler_mu = np.nanmean(flat[mask], axis=0)   # (17,)
    scaler_sd = np.nanstd(flat[mask], axis=0)
    scaler_stats = (scaler_mu, scaler_sd)

    rng = np.random.default_rng(args.seed)
    rows = []
    for position in ["last", "middle", "first"]:
        for norm_mode in ["global", "instance"]:
            print(f"\n--- position={position} norm={norm_mode} ---")
            X_train, y_train_str, _ = _build_features(
                manifest, args.features_root, args.k, position,
                norm_mode, scaler_stats, "train"
            )
            X_test, y_test_str, _ = _build_features(
                manifest, args.features_root, args.k, position,
                norm_mode, scaler_stats, "test"
            )
            y_train = _scenario_to_int(y_train_str, classes)
            y_test = _scenario_to_int(y_test_str, classes)

            clf = LogisticRegression(C=args.reg_c, max_iter=args.max_iter, solver="lbfgs")
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_test)
            probas_full = np.zeros((len(y_test), n_classes), dtype=np.float64)
            for col_idx, cls_id in enumerate(clf.classes_):
                probas_full[:, int(cls_id)] = probas[:, col_idx]
            macro = _macro_auroc(y_test, probas_full, n_classes)
            ci_lo, ci_hi = _bootstrap_ci(
                y_test, probas_full, n_classes, args.n_bootstrap, rng
            )
            print(f"  macro-AUROC = {macro:.3f} | 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]")
            rows.append({
                "position": position,
                "norm_mode": norm_mode,
                "macro_auroc": macro,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            })

    # Critical comparisons
    g_last = next(r for r in rows if r["position"] == "last" and r["norm_mode"] == "global")
    g_first = next(r for r in rows if r["position"] == "first" and r["norm_mode"] == "global")
    i_last = next(r for r in rows if r["position"] == "last" and r["norm_mode"] == "instance")
    i_first = next(r for r in rows if r["position"] == "first" and r["norm_mode"] == "instance")

    delta_window_global = g_first["macro_auroc"] - g_last["macro_auroc"]
    delta_window_instance = i_first["macro_auroc"] - i_last["macro_auroc"]
    delta_norm_last = i_last["macro_auroc"] - g_last["macro_auroc"]
    delta_norm_first = i_first["macro_auroc"] - g_first["macro_auroc"]

    summary = {
        "k": args.k,
        "target": "chaos_mesh_scenario_15way",
        "rows": rows,
        "diagnostic": {
            "global_norm_far_minus_near": delta_window_global,
            "instance_norm_far_minus_near": delta_window_instance,
            "instance_minus_global_at_last": delta_norm_last,
            "instance_minus_global_at_first": delta_norm_first,
        },
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# B1 — Instance normalization diagnostic",
        "",
        f"Cible : 15 scénarios Chaos Mesh (vérité terrain indépendante). "
        f"k = {args.k} | LR (raw features × position × norm)",
        "",
        "## Macro-AUROC test (45 ép)",
        "",
        "| position | norm_mode | macro-AUROC | 95% CI |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['position']} | {r['norm_mode']} | {r['macro_auroc']:.3f} | "
            f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |"
        )

    lines += [
        "",
        "## Diagnostic",
        "",
        f"- Δ(far − near) **avec global norm** = "
        f"**{delta_window_global:+.3f}** (status quo, A1-style)",
        f"- Δ(far − near) **avec instance norm** = "
        f"**{delta_window_instance:+.3f}** "
        f"(si grand et négatif → l'instance norm révèle la dynamique pré-injection)",
        f"- Δ(instance − global) à `last` = {delta_norm_last:+.3f} "
        f"(combien on perd en éliminant les baselines statiques)",
        f"- Δ(instance − global) à `first` = {delta_norm_first:+.3f}",
        "",
        "## Lecture",
        "",
        "- **Instance norm = global norm** → l'information n'était pas dans les "
        "baselines (= il y avait une dynamique réelle). Bonne nouvelle.",
        "- **Instance norm ≪ global norm** → la quasi-totalité de la "
        "discriminabilité venait des baselines statiques. Confirme A1 : pas de "
        "vraie précursion. À combiner avec une nouvelle architecture pour "
        "obtenir un modèle viable.",
        "- Si Δ(far − near) reste petit même en instance norm → le résidu de "
        "discriminabilité provient d'autres invariants statiques (mix de "
        "charge, topologie des arêtes).",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
