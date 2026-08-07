"""Évaluation en ligne du pipeline EWAT — simulation sur le test set.

Rejoue chaque épisode de test timestep par timestep via AlertAssembler.predict()
et mesure :
- Lead time : nombre de pas avant l'injection où l'alerte est levée
- Taux de détection précoce : % épisodes avec alerte avant injection
- Identification cluster : % TP avec cluster_id == ground truth
- Faux positifs sur drifts : % épisodes drift avec alerte levée
- Bootstrap 95% CIs sur toutes les métriques
- Matrice de confusion 10×10 (cluster prédit vs réel)
- Courbes ROC et PR (--roc-sweep)

Usage
-----
    python -m experiments.alerts.eval \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --precursor-dir experiments/precursor \\
        --features-root data/features/v3 \\
        --output experiments/alerts \\
        [--p-thresholds 0.3 0.4 0.5 0.6 0.7] \\
        [--n-bootstrap 1000] \\
        [--roc-sweep]
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
import torch

import mlflow
from ewat.alerts.assembler import AlertAssembler
from ewat.utils.bootstrap import bootstrap_mean_ci, bootstrap_proportion_ci

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

DRIFT_SCENARIOS = {
    "drift_config_change", "drift_rolling_deploy",
    "drift_scale_up", "drift_traffic_ramp",
}

STEP_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_episode(
    features_root: Path, ep_id: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ep_dir = features_root / ep_id
    signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
    adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
    labels = pd.read_parquet(ep_dir / "labels.parquet")
    return signal, adjacency, labels


def _injection_step(labels: pd.DataFrame) -> int | None:
    """First step where regime != 'normal'. None if all normal (drift episodes)."""
    non_normal = labels[labels["regime"] != "normal"]
    return int(non_normal.index[0]) if not non_normal.empty else None


def _eval_episode(
    assembler: AlertAssembler,
    signal: np.ndarray,
    adjacency: np.ndarray,
    labels: pd.DataFrame,
    ep_id: str,
    cluster_gt: int,
    scenario: str,
    p_threshold: float,
) -> dict:
    """Simulate online alerting for one episode. Returns per-episode metrics."""
    is_drift_ep = scenario in DRIFT_SCENARIOS
    injection_t = _injection_step(labels)
    t_total = signal.shape[0]

    k_min = max(1, min(assembler.k_optimal.values())) if assembler.k_optimal else 1

    # Anomaly episodes: stream up to injection_t.
    # Drift episodes: stream full episode — any alert is a false alarm.
    end_t = t_total if is_drift_ep else (
        (injection_t + 1) if injection_t is not None else t_total
    )

    first_alert_step: int | None = None
    first_alert_cluster: int | None = None

    assembler.threshold = p_threshold

    for t in range(k_min, end_t):
        alerts = assembler.predict(
            signal[:t],
            adjacency[:t],
            timestamp=float(t) * STEP_SECONDS,
            episode_id=ep_id,
        )
        if alerts:
            first_alert_step = t
            first_alert_cluster = alerts[0].cluster_id
            break

    result: dict = {
        "episode_id": ep_id,
        "scenario": scenario,
        "is_drift": is_drift_ep,
        "cluster_gt": cluster_gt,
        "injection_t": injection_t,
        "p_threshold": p_threshold,
        "first_alert_step": first_alert_step,
        "first_alert_cluster": first_alert_cluster,
        "lead_time_steps": None,
        "lead_time_min": None,
        "tp": False,
        "correct_cluster": False,
        "false_alarm": False,
    }

    if is_drift_ep:
        result["false_alarm"] = first_alert_step is not None
    else:
        if injection_t is None:
            return result
        if first_alert_step is not None and first_alert_step <= injection_t:
            lead = injection_t - first_alert_step
            result["lead_time_steps"] = lead
            result["lead_time_min"] = round(lead * STEP_SECONDS / 60.0, 2)
            result["tp"] = True
            result["correct_cluster"] = (first_alert_cluster == cluster_gt)

    return result


# ---------------------------------------------------------------------------
# Aggregation with bootstrap CIs
# ---------------------------------------------------------------------------


def _aggregate(records: list[dict], n_bootstrap: int = 0, seed: int = 42) -> dict:
    anomaly = [r for r in records if not r["is_drift"]]
    drift = [r for r in records if r["is_drift"]]

    rng = np.random.default_rng(seed) if n_bootstrap > 0 else None

    tp_count = sum(1 for r in anomaly if r["tp"])
    correct_cluster_count = sum(1 for r in anomaly if r["correct_cluster"])
    fa_count = sum(1 for r in drift if r["false_alarm"])
    lead_times = [r["lead_time_steps"] for r in anomaly if r["lead_time_steps"] is not None]

    n_anom = len(anomaly)
    n_drift = len(drift)

    detection_rate = tp_count / n_anom if n_anom else float("nan")
    correct_cluster_rate = correct_cluster_count / n_anom if n_anom else float("nan")
    false_alarm_rate = fa_count / n_drift if n_drift else float("nan")
    mean_lead_steps = float(np.mean(lead_times)) if lead_times else float("nan")
    mean_lead_min = mean_lead_steps * STEP_SECONDS / 60.0 if lead_times else float("nan")

    out: dict = {
        "n_anomaly_episodes": n_anom,
        "n_drift_episodes": n_drift,
        "detection_rate": round(detection_rate, 4) if not np.isnan(detection_rate) else None,
        "correct_cluster_rate": (
            round(correct_cluster_rate, 4) if not np.isnan(correct_cluster_rate) else None
        ),
        "false_alarm_rate": (
            round(false_alarm_rate, 4) if not np.isnan(false_alarm_rate) else None
        ),
        "mean_lead_steps": round(mean_lead_steps, 2) if not np.isnan(mean_lead_steps) else None,
        "mean_lead_min": round(mean_lead_min, 2) if not np.isnan(mean_lead_min) else None,
        "tp_count": tp_count,
        "correct_cluster_count": correct_cluster_count,
        "false_alarm_count": fa_count,
    }

    if n_bootstrap > 0 and rng is not None:
        out["ci_detection_rate"] = bootstrap_proportion_ci(
            tp_count, n_anom, n=n_bootstrap, rng=rng
        ).as_dict() if n_anom else None
        out["ci_correct_cluster_rate"] = bootstrap_proportion_ci(
            correct_cluster_count, n_anom, n=n_bootstrap, rng=rng
        ).as_dict() if n_anom else None
        out["ci_false_alarm_rate"] = bootstrap_proportion_ci(
            fa_count, n_drift, n=n_bootstrap, rng=rng
        ).as_dict() if n_drift else None
        out["ci_mean_lead_steps"] = bootstrap_mean_ci(
            np.array(lead_times, dtype=float), n=n_bootstrap, rng=rng
        ).as_dict() if lead_times else None

    return out


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def _confusion_matrix(records: list[dict], n_clusters: int) -> np.ndarray:
    """10×10 confusion matrix: rows=ground truth, cols=predicted cluster.

    Only counts TP anomaly episodes (those where an alert was fired before injection).
    """
    mat = np.zeros((n_clusters, n_clusters), dtype=int)
    for r in records:
        if r["is_drift"] or not r["tp"]:
            continue
        gt = r["cluster_gt"]
        pred = r["first_alert_cluster"]
        if pred is None:
            continue
        if 0 <= gt < n_clusters and 0 <= pred < n_clusters:
            mat[gt, pred] += 1
    return mat


# ---------------------------------------------------------------------------
# ROC / PR sweep
# ---------------------------------------------------------------------------


def _roc_pr_sweep(
    assembler: AlertAssembler,
    test_episodes: list[tuple[str, dict]],
    features_root: Path,
    thresholds: np.ndarray,
) -> list[dict]:
    """Sweep thresholds finely to compute ROC and PR points."""
    points: list[dict] = []
    for p_thr in thresholds:
        records: list[dict] = []
        for ep_id, meta in test_episodes:
            try:
                signal, adjacency, labels = _load_episode(features_root, ep_id)
            except FileNotFoundError:
                continue
            rec = _eval_episode(
                assembler=assembler,
                signal=signal, adjacency=adjacency, labels=labels,
                ep_id=ep_id, cluster_gt=int(meta["cluster"]),
                scenario=meta.get("scenario", ""), p_threshold=float(p_thr),
            )
            records.append(rec)

        anomaly = [r for r in records if not r["is_drift"]]
        drift = [r for r in records if r["is_drift"]]
        n_anom = len(anomaly)
        n_drift = len(drift)

        tp = sum(1 for r in anomaly if r["tp"])
        fp = sum(1 for r in drift if r["false_alarm"])
        fn = n_anom - tp

        recall = tp / n_anom if n_anom else 0.0
        fpr = fp / n_drift if n_drift else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        points.append({
            "threshold": float(p_thr),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        })
        print(f"  thr={p_thr:.2f}  recall={recall:.3f}  fpr={fpr:.3f}  "
              f"prec={precision:.3f}  F1={f1:.3f}")
    return points


def _plot_roc_pr(points: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        recalls = [p["recall"] for p in points]
        fprs = [p["fpr"] for p in points]
        precisions = [p["precision"] for p in points]

        # AUC via trapezoidal rule (FPR sorted ascending for ROC)
        order_roc = np.argsort(fprs)
        auc_roc = float(np.trapz(
            [recalls[i] for i in order_roc],
            [fprs[i] for i in order_roc],
        ))
        # PR: recall sorted ascending
        order_pr = np.argsort(recalls)
        auc_pr = float(np.trapz(
            [precisions[i] for i in order_pr],
            [recalls[i] for i in order_pr],
        ))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.plot([fprs[i] for i in order_roc], [recalls[i] for i in order_roc],
                marker="o", markersize=4, color="steelblue")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        for i, p in enumerate(points):
            ax.annotate(f"{p['threshold']:.2f}", (p["fpr"], p["recall"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
        ax.set_xlabel("False Positive Rate (drift episodes)")
        ax.set_ylabel("Recall (anomaly episodes)")
        ax.set_title(f"ROC Curve  (AUC={auc_roc:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        ax = axes[1]
        ax.plot([recalls[i] for i in order_pr], [precisions[i] for i in order_pr],
                marker="o", markersize=4, color="coral")
        for i, p in enumerate(points):
            ax.annotate(f"{p['threshold']:.2f}", (p["recall"], p["precision"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve  (AUC={auc_pr:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        fig.tight_layout()
        fig.savefig(output_dir / "roc_pr_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"ROC/PR figure saved → {output_dir / 'roc_pr_curve.png'}")
        return auc_roc, auc_pr
    except Exception as e:
        print(f"Warning: could not plot ROC/PR ({e})")
        return float("nan"), float("nan")


def _plot_confusion(mat: np.ndarray, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap="Blues")
        fig.colorbar(im, ax=ax)
        n = mat.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"C{i}" for i in range(n)])
        ax.set_yticklabels([f"C{i}" for i in range(n)])
        ax.set_xlabel("Predicted cluster")
        ax.set_ylabel("True cluster")
        ax.set_title("Cluster confusion matrix (TP anomaly episodes)")
        for i in range(n):
            for j in range(n):
                if mat[i, j] > 0:
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            color="white" if mat[i, j] > mat.max() / 2 else "black",
                            fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Confusion matrix saved → {output_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"Warning: could not plot confusion matrix ({e})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Alert evaluation — online simulation")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/alerts"))
    parser.add_argument(
        "--p-thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for CIs (0 = skip)")
    parser.add_argument("--roc-sweep", action="store_true",
                        help="Run fine-grained threshold sweep for ROC/PR curves")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cluster manifest not found: {manifest_path}. "
            "Run experiments/typing/train.py first."
        )
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1

    test_episodes = [
        (ep_id, meta)
        for ep_id, meta in cluster_manifest.items()
        if meta.get("split") == "test"
    ]
    print(f"Test episodes: {len(test_episodes)}  |  clusters: {n_clusters}")

    assembler = AlertAssembler.from_experiment_dirs(
        args.typing_dir, args.encoder_dir, args.precursor_dir,
        threshold=args.p_thresholds[0],
        device=device,
    )
    print(f"Assembler: {len(assembler.classifiers)} classifiers, "
          f"scaler={'yes' if assembler.scaler is not None else 'no'}")

    # -----------------------------------------------------------------------
    # Main evaluation loop over discrete thresholds
    # -----------------------------------------------------------------------
    all_records: list[dict] = []

    for p_thr in args.p_thresholds:
        print(f"\n--- threshold={p_thr} ---")
        records_thr: list[dict] = []

        for ep_id, meta in test_episodes:
            try:
                signal, adjacency, labels = _load_episode(args.features_root, ep_id)
            except FileNotFoundError:
                print(f"  skip {ep_id} (features not found)")
                continue

            rec = _eval_episode(
                assembler=assembler, signal=signal, adjacency=adjacency, labels=labels,
                ep_id=ep_id, cluster_gt=int(meta["cluster"]),
                scenario=meta.get("scenario", ""), p_threshold=p_thr,
            )
            records_thr.append(rec)

        agg = _aggregate(records_thr, n_bootstrap=args.n_bootstrap, seed=args.seed)
        dr = agg["detection_rate"] or float("nan")
        cc = agg["correct_cluster_rate"] or float("nan")
        fa = agg["false_alarm_rate"] or float("nan")
        lead = agg["mean_lead_min"]
        print(
            f"  detection={dr:.3f}  correct_cluster={cc:.3f}  "
            f"false_alarm={fa:.3f}  lead={lead} min"
        )
        if args.n_bootstrap > 0:
            ci_dr = agg.get("ci_detection_rate") or {}
            ci_fa = agg.get("ci_false_alarm_rate") or {}
            print(f"  CI detection=[{ci_dr.get('ci_lo', 'nan'):.3f}, "
                  f"{ci_dr.get('ci_hi', 'nan'):.3f}]  "
                  f"CI FA=[{ci_fa.get('ci_lo', 'nan'):.3f}, "
                  f"{ci_fa.get('ci_hi', 'nan'):.3f}]")
        all_records.extend(records_thr)

    # -----------------------------------------------------------------------
    # Confusion matrix (using all thresholds combined; use threshold=0.3 subset)
    # -----------------------------------------------------------------------
    thr_for_conf = args.p_thresholds[0]
    conf_records = [r for r in all_records if r["p_threshold"] == thr_for_conf]
    conf_mat = _confusion_matrix(conf_records, n_clusters)
    _plot_confusion(conf_mat, args.output)

    # -----------------------------------------------------------------------
    # ROC / PR sweep
    # -----------------------------------------------------------------------
    roc_data: list[dict] = []
    if args.roc_sweep:
        sweep_thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
        print(f"\n--- ROC/PR sweep ({len(sweep_thresholds)} thresholds) ---")
        roc_data = _roc_pr_sweep(assembler, test_episodes, args.features_root, sweep_thresholds)
        (args.output / "roc_pr_data.json").write_text(json.dumps(roc_data, indent=2))
        _plot_roc_pr(roc_data, args.output)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_records)
    df.to_csv(args.output / "per_episode.csv", index=False)

    results_by_threshold: dict[str, dict] = {}
    for p_thr in args.p_thresholds:
        subset = [r for r in all_records if r["p_threshold"] == p_thr]
        results_by_threshold[str(p_thr)] = _aggregate(
            subset, n_bootstrap=args.n_bootstrap, seed=args.seed
        )

    results = {
        "thresholds": args.p_thresholds,
        "n_test_episodes": len(test_episodes),
        "n_clusters": n_clusters,
        "n_bootstrap": args.n_bootstrap,
        "by_threshold": results_by_threshold,
        "confusion_matrix": {
            "threshold": thr_for_conf,
            "matrix": conf_mat.tolist(),
        },
        "roc_pr": roc_data if args.roc_sweep else [],
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2))

    # Human-readable report
    show_ci = args.n_bootstrap > 0
    lines = [
        "# Évaluation alertes — Simulation en ligne (test set)\n",
        f"Épisodes test : {len(test_episodes)}  (anomalie + drift)\n",
        f"Bootstrap CIs : {'oui (n=' + str(args.n_bootstrap) + ')' if show_ci else 'non'}\n",
        "## Métriques par seuil\n",
    ]
    hdr = f"{'Seuil':<6}  {'Detect.':<9}  {'Cluster':<9}  {'FA drift':<9}  {'Lead (min)':<10}"
    if show_ci:
        # E4 (audit 2026-06): le header était une chaîne littérale (accolades
        # non interpolées) + la colonne IC lead manquait
        hdr += f"  {'CI detect.':<17}  {'CI FA':<17}  {'CI lead (min)':<15}"
    lines += [hdr, "-" * (len(hdr) + 10)]

    for p_thr in args.p_thresholds:
        agg = results_by_threshold[str(p_thr)]
        dr = agg["detection_rate"]
        cc = agg["correct_cluster_rate"]
        fa = agg["false_alarm_rate"]
        lead = f"{agg['mean_lead_min']:.1f}" if agg["mean_lead_min"] is not None else "N/A"
        row = (
            f"{p_thr:<6.2f}  "
            f"{(dr or 0):<9.3f}  "
            f"{(cc or 0):<9.3f}  "
            f"{(fa or 0):<9.3f}  "
            f"{lead:<10}"
        )
        if show_ci:
            ci_dr = agg.get("ci_detection_rate") or {}
            ci_fa = agg.get("ci_false_alarm_rate") or {}
            ci_lead = agg.get("ci_mean_lead_steps") or {}
            lo_dr = ci_dr.get("ci_lo", float("nan"))
            hi_dr = ci_dr.get("ci_hi", float("nan"))
            lo_fa = ci_fa.get("ci_lo", float("nan"))
            hi_fa = ci_fa.get("ci_hi", float("nan"))
            to_min = STEP_SECONDS / 60.0
            lo_ld = ci_lead.get("ci_lo", float("nan")) * to_min
            hi_ld = ci_lead.get("ci_hi", float("nan")) * to_min
            row += (f"  [{lo_dr:.3f},{hi_dr:.3f}]     [{lo_fa:.3f},{hi_fa:.3f}]"
                    f"     [{lo_ld:.1f},{hi_ld:.1f}]")
        lines.append(row)

    lines += [
        "",
        f"## Matrice de confusion (seuil={thr_for_conf})",
        "```",
        "        " + "  ".join(f"C{j}" for j in range(n_clusters)),
    ]
    for i, row_vals in enumerate(conf_mat):
        lines.append(f"C{i:<5}  " + "  ".join(f"{v:>2}" for v in row_vals))
    lines.append("```")

    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("ewat_alerts_eval")
        with mlflow.start_run(run_name="alerts_eval"):
            mlflow.log_params({
                "p_thresholds": str(args.p_thresholds),
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "n_test_episodes": len(test_episodes),
            })
            for p_thr, agg in zip(args.p_thresholds, [
                results_by_threshold[str(t)] for t in args.p_thresholds
            ]):
                prefix = f"thr{p_thr:.1f}"
                mlflow.log_metrics({
                    f"{prefix}_detection_rate": agg.get("detection_rate", float("nan")),
                    f"{prefix}_cluster_correct_rate": agg.get("cluster_correct_rate", float("nan")),
                    f"{prefix}_false_alarm_rate": agg.get("false_alarm_rate", float("nan")),
                    f"{prefix}_lead_time_min": agg.get("lead_time_min_mean", float("nan")),
                })
    except Exception:
        pass


if __name__ == "__main__":
    main()
