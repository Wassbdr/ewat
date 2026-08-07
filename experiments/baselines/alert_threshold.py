"""Baseline alerte — détecteur z-score simple.

Compare avec le pipeline EWAT complet sur les mêmes 45 épisodes test.

Méthode
-------
- Fenêtre de référence : 5 premiers steps de chaque épisode (phase normale)
- Alerte : dès qu'au moins une feature dépasse μ + n_sigma * σ sur la
  fenêtre courante (derniers w steps, agrégée par max)
- Métriques : detection rate, false alarm rate, lead time (comparables à eval.py)

Usage
-----
    python -m experiments.baselines.alert_threshold \\
        --typing-dir experiments/typing \\
        --features-root data/features/v3 \\
        --output experiments/baselines \\
        [--n-sigma 3.0] [--n-bootstrap 1000]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import numpy as np
import pandas as pd

from ewat.utils.bootstrap import bootstrap_mean_ci, bootstrap_proportion_ci

DRIFT_SCENARIOS = {
    "drift_config_change", "drift_rolling_deploy",
    "drift_scale_up", "drift_traffic_ramp",
}
STEP_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def _load_episode(
    features_root: Path, ep_id: str
) -> tuple[np.ndarray, pd.DataFrame]:
    ep_dir = features_root / ep_id
    signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
    labels = pd.read_parquet(ep_dir / "labels.parquet")
    return signal, labels


def _injection_step(labels: pd.DataFrame) -> int | None:
    non_normal = labels[labels["regime"] != "normal"]
    return int(non_normal.index[0]) if not non_normal.empty else None


# ---------------------------------------------------------------------------
# Z-score detector
# ---------------------------------------------------------------------------

def _eval_zscore(
    signal: np.ndarray,
    labels: pd.DataFrame,
    ep_id: str,
    scenario: str,
    n_sigma: float,
    ref_steps: int,
    cur_window: int,
) -> dict:
    """Evaluate a z-score detector on one episode.

    Reference statistics computed from the first `ref_steps` steps.
    Alert fires when any feature in the current window exceeds ref_mu + n_sigma * ref_std.

    Parameters
    ----------
    signal:     (T, N, 17) array.
    n_sigma:    Alert threshold in standard deviations.
    ref_steps:  Number of initial steps used to compute reference statistics.
    cur_window: Size of the sliding window for the current observation.
    """
    is_drift = scenario in DRIFT_SCENARIOS
    injection_t = _injection_step(labels)
    t_total = signal.shape[0]

    # Compute reference stats from first ref_steps steps
    # Shape: (ref_steps, N, 17) → flatten to (ref_steps, N*17)
    ref = signal[:ref_steps].reshape(ref_steps, -1)
    ref_mu = ref.mean(axis=0)       # (N*17,)
    ref_std = ref.std(axis=0) + 1e-8

    end_t = t_total if is_drift else (
        (injection_t + 1) if injection_t is not None else t_total
    )

    first_alert_step: int | None = None
    start = ref_steps + cur_window

    for t in range(start, end_t):
        # Current window: last cur_window steps
        window = signal[t - cur_window:t].reshape(cur_window, -1)  # (W, N*17)
        cur_max = window.max(axis=0)                                # (N*17,) — max over window
        z = (cur_max - ref_mu) / ref_std
        if np.any(z > n_sigma):
            first_alert_step = t
            break

    result: dict = {
        "episode_id": ep_id,
        "scenario": scenario,
        "is_drift": is_drift,
        "injection_t": injection_t,
        "n_sigma": n_sigma,
        "first_alert_step": first_alert_step,
        "lead_time_steps": None,
        "lead_time_min": None,
        "tp": False,
        "false_alarm": False,
    }

    if is_drift:
        result["false_alarm"] = first_alert_step is not None
    else:
        if injection_t is not None and first_alert_step is not None:
            if first_alert_step <= injection_t:
                lead = injection_t - first_alert_step
                result["lead_time_steps"] = lead
                result["lead_time_min"] = round(lead * STEP_SECONDS / 60.0, 2)
                result["tp"] = True

    return result


def _aggregate(records: list[dict], n_bootstrap: int = 0, seed: int = 42) -> dict:
    anomaly = [r for r in records if not r["is_drift"]]
    drift = [r for r in records if r["is_drift"]]

    rng = np.random.default_rng(seed) if n_bootstrap > 0 else None

    tp = sum(1 for r in anomaly if r["tp"])
    fa = sum(1 for r in drift if r["false_alarm"])
    leads = [r["lead_time_steps"] for r in anomaly if r["lead_time_steps"] is not None]

    n_anom, n_drift = len(anomaly), len(drift)

    out: dict = {
        "n_anomaly": n_anom,
        "n_drift": n_drift,
        "tp": tp,
        "fa": fa,
        "detection_rate": round(tp / n_anom, 4) if n_anom else float("nan"),
        "false_alarm_rate": round(fa / n_drift, 4) if n_drift else float("nan"),
        "mean_lead_steps": round(float(np.mean(leads)), 2) if leads else float("nan"),
        "mean_lead_min": (
            round(float(np.mean(leads)) * STEP_SECONDS / 60.0, 2) if leads else float("nan")
        ),
    }

    if n_bootstrap > 0 and rng is not None:
        out["ci_detection_rate"] = bootstrap_proportion_ci(
            tp, n_anom, n=n_bootstrap, rng=rng
        ).as_dict() if n_anom else None
        out["ci_false_alarm_rate"] = bootstrap_proportion_ci(
            fa, n_drift, n=n_bootstrap, rng=rng
        ).as_dict() if n_drift else None
        out["ci_mean_lead_steps"] = bootstrap_mean_ci(
            np.array(leads, dtype=float), n=n_bootstrap, rng=rng
        ).as_dict() if leads else None

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Alert baseline — z-score detector")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/baselines"))
    parser.add_argument(
        "--n-sigma-values", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5],
        help="Z-score thresholds to sweep",
    )
    parser.add_argument("--ref-steps", type=int, default=5,
                        help="Reference window size (steps)")
    parser.add_argument("--cur-window", type=int, default=3,
                        help="Current sliding window size (steps)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    test_episodes = [
        (ep_id, meta)
        for ep_id, meta in cluster_manifest.items()
        if meta.get("split") == "test"
    ]
    print(f"Test episodes: {len(test_episodes)}")
    print(f"ref_steps={args.ref_steps}  cur_window={args.cur_window}  "
          f"n_sigma={args.n_sigma_values}")

    all_records: list[dict] = []
    results_by_sigma: dict[str, dict] = {}

    for n_sigma in args.n_sigma_values:
        print(f"\n--- n_sigma={n_sigma} ---")
        records: list[dict] = []

        for ep_id, meta in test_episodes:
            try:
                signal, labels = _load_episode(args.features_root, ep_id)
            except FileNotFoundError:
                print(f"  skip {ep_id}")
                continue

            rec = _eval_zscore(
                signal=signal, labels=labels, ep_id=ep_id,
                scenario=meta.get("scenario", ""),
                n_sigma=n_sigma, ref_steps=args.ref_steps,
                cur_window=args.cur_window,
            )
            records.append(rec)

        agg = _aggregate(records, n_bootstrap=args.n_bootstrap, seed=args.seed)
        results_by_sigma[str(n_sigma)] = agg
        all_records.extend(records)

        dr = agg["detection_rate"]
        fa = agg["false_alarm_rate"]
        lead = agg["mean_lead_min"]
        print(f"  detection={dr:.3f}  false_alarm={fa:.3f}  lead={lead} min")
        if args.n_bootstrap > 0:
            ci_dr = agg.get("ci_detection_rate") or {}
            ci_fa = agg.get("ci_false_alarm_rate") or {}
            print(f"  CI detect=[{ci_dr.get('ci_lo', float('nan')):.3f},"
                  f"{ci_dr.get('ci_hi', float('nan')):.3f}]  "
                  f"CI FA=[{ci_fa.get('ci_lo', float('nan')):.3f},"
                  f"{ci_fa.get('ci_hi', float('nan')):.3f}]")

    # Save
    summary = {
        "n_test_episodes": len(test_episodes),
        "ref_steps": args.ref_steps,
        "cur_window": args.cur_window,
        "n_bootstrap": args.n_bootstrap,
        "by_sigma": results_by_sigma,
    }
    (args.output / "alert_threshold_baseline.json").write_text(json.dumps(summary, indent=2))

    pd.DataFrame(all_records).to_csv(args.output / "alert_baseline_per_episode.csv", index=False)

    lines = [
        "# Baseline alerte — Détecteur z-score\n",
        f"Fenêtre ref : {args.ref_steps} steps  |  fenêtre courante : {args.cur_window} steps\n",
        f"{'σ':<6}  {'Detect.':<9}  {'FA drift':<9}  {'Lead (min)':<10}",
        "-" * 40,
    ]
    for n_sigma in args.n_sigma_values:
        agg = results_by_sigma[str(n_sigma)]
        lead = f"{agg['mean_lead_min']:.1f}" if not np.isnan(agg["mean_lead_min"]) else "N/A"
        lines.append(
            f"{n_sigma:<6.1f}  "
            f"{(agg['detection_rate'] or 0):<9.3f}  "
            f"{(agg['false_alarm_rate'] or 0):<9.3f}  "
            f"{lead:<10}"
        )
    lines += [
        "",
        "**Note** : ce détecteur ne produit pas de typage — pas de cluster prédit.",
        "Comparer Detection / FA avec EWAT sur les mêmes épisodes.",
    ]
    (args.output / "alert_threshold_baseline.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'alert_threshold_baseline.md'}")


if __name__ == "__main__":
    main()
