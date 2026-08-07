"""H2 — Validation du mécanisme look-through sur le test set.

Compare deux détecteurs de drift sur les épisodes de test :
- Look-through : DriftDetector avec fenêtre de confirmation (post=3)
- Baseline : seuil simple MMD² ≥ ε, sans fenêtre de confirmation

Métriques :
- TPR (épisodes drift) : % correctement flagués DRIFT
- FPR (épisodes anomalie) : % incorrectement flagués DRIFT

H2 PASS si FPR_lookthrough < FPR_baseline (test Student unilatéral, p < 0.05).

Usage
-----
    python -m experiments.h2_lookthrough.eval \\
        --features-root data/features/v3 \\
        --typing-dir experiments/typing \\
        --output experiments/h2_lookthrough \\
        [--epsilon 0.5226] [--window-ref 5] [--window-cur 5] [--post-window 3]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_TRACKING_SILENT", "true")

import numpy as np
import pandas as pd
from scipy import stats

import mlflow
from ewat.drift.detector import DriftDetector
from ewat.drift.mmd import RFFKernel

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

DRIFT_SCENARIOS = {
    "drift_config_change", "drift_rolling_deploy",
    "drift_scale_up", "drift_traffic_ramp",
}


# ---------------------------------------------------------------------------
# Baseline (no look-through)
# ---------------------------------------------------------------------------


def _baseline_drift_at_injection(
    signal: np.ndarray,
    injection_t: int | None,
    epsilon: float,
    window_ref: int,
    window_cur: int,
    seed: int = 42,
) -> bool:
    """True if MMD² ≥ ε in sliding window at or near injection point."""
    t_total = signal.shape[0]
    flat = signal.reshape(t_total, -1).astype(np.float64)  # (T, N*d)

    kernel = RFFKernel(rff_dim=256, seed=seed)
    if flat.shape[0] < window_ref:
        return False
    kernel.fit_sigma(flat[:window_ref])

    # Check timesteps around injection (or all if drift episode)
    check_range = range(window_ref + window_cur, t_total)
    for t in check_range:
        ref = flat[t - window_cur - window_ref: t - window_cur]
        cur = flat[t - window_cur: t]
        mmd2 = float(kernel.mmd_squared(ref, cur))
        if mmd2 >= epsilon:
            if injection_t is None or t >= injection_t:
                return True
    return False


# ---------------------------------------------------------------------------
# Look-through detector
# ---------------------------------------------------------------------------


def _lookthrough_drift_at_injection(
    signal: np.ndarray,
    injection_t: int | None,
    epsilon: float,
    window_ref: int,
    window_cur: int,
    post_window: int,
    seed: int = 42,
) -> bool:
    """True if DriftDetector (look-through) flags DRIFT at or after injection."""
    t_total = signal.shape[0]
    flat = signal.reshape(t_total, -1).astype(np.float64)

    kernel = RFFKernel(rff_dim=256, seed=seed)
    # Pre-seed sigma from first window_ref steps
    if flat.shape[0] < window_ref:
        return False
    kernel.fit_sigma(flat[:window_ref])

    detector = DriftDetector(
        kernel=kernel,
        epsilon_drift=epsilon,
        window_ref_size=window_ref,
        window_cur_size=window_cur,
        post_drift_window_s=post_window,
    )
    detector.load_reference(flat[:window_ref])

    for t in range(t_total):
        result = detector.update(flat[t])
        if result.flag:
            if injection_t is None or t >= injection_t:
                return True
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_test_episodes(
    *,
    dataset_dir: Path | None,
    typing_dir: Path,
) -> list[tuple[str, dict]]:
    """Return (episode_id, meta) for test split.

    Prefer ``--dataset`` (assembled split + index) so H2 can run on ewat_v4
    without retraining the siamese typer. Falls back to typing cluster manifest.
    """
    if dataset_dir is not None:
        split_path = dataset_dir / "split.json"
        index_path = dataset_dir / "index.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"split.json not found: {split_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"index.parquet not found: {index_path}")
        split = json.loads(split_path.read_text())
        test_ids = set(split.get("test", []))
        index = pd.read_parquet(index_path)
        test_rows = index[index["episode_id"].isin(test_ids)]
        episodes: list[tuple[str, dict]] = []
        for _, row in test_rows.iterrows():
            ep_id = str(row["episode_id"])
            meta = {
                "split": "test",
                "scenario": row.get("scenario", row.get("scenario_name", "")),
            }
            if isinstance(meta["scenario"], float) and np.isnan(meta["scenario"]):
                meta["scenario"] = ""
            episodes.append((ep_id, meta))
        return episodes

    manifest_path = typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cluster manifest not found: {manifest_path}")
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())
    return [
        (ep_id, meta)
        for ep_id, meta in cluster_manifest.items()
        if meta.get("split") == "test"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 look-through validation")
    parser.add_argument("--features-root", type=Path, default=None,
                        help="Feature store (default: inferred from --dataset or v3)")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Assembled dataset (ewat_v3/v4) — uses split.json test set")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"),
                        help="Fallback episode list when --dataset is omitted")
    parser.add_argument("--output", type=Path, default=Path("experiments/h2_lookthrough"))
    parser.add_argument("--epsilon", type=float, default=0.5226)
    parser.add_argument("--window-ref", type=int, default=5,
                        help="Reference window (same as calibration)")
    parser.add_argument("--window-cur", type=int, default=5,
                        help="Current window (same as calibration)")
    parser.add_argument("--post-window", type=int, default=3,
                        help="Post-drift confirmation window (look-through only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.features_root is None:
        if args.dataset is not None:
            ds_name = args.dataset.name
            version = ds_name[len("ewat_"):] if ds_name.startswith("ewat_") else ds_name
            args.features_root = Path("data/features") / version
        else:
            args.features_root = Path("data/features/v3")

    test_episodes = _load_test_episodes(
        dataset_dir=args.dataset,
        typing_dir=args.typing_dir,
    )
    print(f"Test episodes: {len(test_episodes)}  features={args.features_root}")

    records: list[dict] = []

    for ep_id, meta in test_episodes:
        ep_dir = args.features_root / ep_id
        if not ep_dir.exists():
            print(f"  skip {ep_id} (features not found)")
            continue

        signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        signal = np.nan_to_num(signal, nan=0.0)
        labels = pd.read_parquet(ep_dir / "labels.parquet")

        scenario = meta.get("scenario", "")
        is_drift_ep = scenario in DRIFT_SCENARIOS

        # Find injection timestep
        non_normal = labels[labels["regime"] != "normal"]
        injection_t: int | None = int(non_normal.index[0]) if not non_normal.empty else None

        # Run both detectors
        drift_lt = _lookthrough_drift_at_injection(
            signal, injection_t, args.epsilon,
            args.window_ref, args.window_cur, args.post_window, args.seed,
        )
        drift_bl = _baseline_drift_at_injection(
            signal, injection_t, args.epsilon,
            args.window_ref, args.window_cur, args.seed,
        )

        records.append({
            "episode_id": ep_id,
            "scenario": scenario,
            "is_drift": is_drift_ep,
            "injection_t": injection_t,
            "drift_lookthrough": drift_lt,
            "drift_baseline": drift_bl,
        })
        print(f"  {ep_id:40s}  drift={is_drift_ep}  lt={drift_lt}  bl={drift_bl}")

    df = pd.DataFrame(records)
    df.to_csv(args.output / "per_episode.csv", index=False)

    # Compute TPR and FPR
    drift_eps = df[df["is_drift"]]
    anomaly_eps = df[~df["is_drift"]]

    tpr_lt = float(drift_eps["drift_lookthrough"].mean()) if len(drift_eps) else float("nan")
    tpr_bl = float(drift_eps["drift_baseline"].mean()) if len(drift_eps) else float("nan")
    fpr_lt = float(anomaly_eps["drift_lookthrough"].mean()) if len(anomaly_eps) else float("nan")
    fpr_bl = float(anomaly_eps["drift_baseline"].mean()) if len(anomaly_eps) else float("nan")

    # Paired one-sided Student t-test: H2 = FPR_lt < FPR_baseline
    h2_pass = False
    p_value = float("nan")
    if len(anomaly_eps) >= 2:
        lt_vals = anomaly_eps["drift_lookthrough"].astype(float).values
        bl_vals = anomaly_eps["drift_baseline"].astype(float).values
        # One-sided: alternative = "less" → FPR_lt < FPR_bl
        result = stats.ttest_rel(lt_vals, bl_vals, alternative="less")
        p_value = float(result.pvalue)
        h2_pass = bool(fpr_lt < fpr_bl and p_value < 0.05)

    print("\n--- H2 Summary ---")
    print(f"TPR (drift)    : lt={tpr_lt:.3f}  baseline={tpr_bl:.3f}")
    print(f"FPR (anomaly)  : lt={fpr_lt:.3f}  baseline={fpr_bl:.3f}")
    print(f"p-value (paired t, one-sided): {p_value:.4f}")
    print(f"H2 {'✓ PASS' if h2_pass else '✗ FAIL'}")

    summary = {
        "epsilon": args.epsilon,
        "window_ref": args.window_ref,
        "window_cur": args.window_cur,
        "post_window": args.post_window,
        "n_drift_episodes": int(len(drift_eps)),
        "n_anomaly_episodes": int(len(anomaly_eps)),
        "tpr_lookthrough": tpr_lt,
        "tpr_baseline": tpr_bl,
        "fpr_lookthrough": fpr_lt,
        "fpr_baseline": fpr_bl,
        "p_value": p_value,
        "h2_pass": h2_pass,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# H2 — Validation look-through (test set)\n",
        f"ε = {args.epsilon}  |  W_ref={args.window_ref}  W_cur={args.window_cur}  "
        f"W_post={args.post_window}\n",
        f"H2 : {'✓ PASS' if h2_pass else '✗ FAIL'}"
        f"  (p={p_value:.4f}, seuil 0.05)\n",
        f"{'':12} {'Look-through':>14} {'Baseline':>10}",
        f"{'TPR (drift)':12} {tpr_lt:>14.3f} {tpr_bl:>10.3f}",
        f"{'FPR (anomaly)':12} {fpr_lt:>14.3f} {fpr_bl:>10.3f}",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"Report: {args.output / 'results.md'}")

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("ewat_h2_lookthrough")
        with mlflow.start_run(run_name="h2_lookthrough"):
            mlflow.log_params({
                "epsilon": args.epsilon,
                "window_ref": args.window_ref,
                "window_cur": args.window_cur,
                "post_window": args.post_window,
                "seed": args.seed,
                "n_drift_episodes": len(drift_eps),
                "n_anomaly_episodes": len(anomaly_eps),
            })
            mlflow.log_metrics({
                "tpr_lookthrough": tpr_lt,
                "tpr_baseline": tpr_bl,
                "fpr_lookthrough": fpr_lt,
                "fpr_baseline": fpr_bl,
                "p_value": p_value,
                "h2_pass": float(h2_pass),
            })
    except Exception:
        pass


if __name__ == "__main__":
    main()
