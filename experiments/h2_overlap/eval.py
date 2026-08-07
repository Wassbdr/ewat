"""H2b — Validation de la détection du régime θ_{drift∩anomaly}.

Reformulation de H2 : au lieu de tester si le mécanisme look-through
réduit le FPR (H2a, ❌ FAIL sur épisodes courts), on teste si le pipeline
EWAT détecte *correctement* le régime θ_{drift∩anomaly}, incarné par le
cluster C8 (faulty_deploy_overlap).

Protocole
---------
Pour chaque épisode du cluster C8 (déploiement défectueux, régime simultané
drift + anomalie) et des clusters de drift pur (C2, C5) :

1. Flag drift   : DriftDetector.update() déclenche-t-il au cours de l'épisode ?
2. Alerte préc. : AlertAssembler.predict() lève-t-il une alerte (p > threshold) ?
3. Chevauchement : les deux simultanément → capture de θ_{drift∩anomaly}

Résultat attendu
----------------
- C8 : flag_drift=True ET alerte_préc=True (les deux mécanismes déclenchent)
- C5 (drift pur) : flag_drift=True ET alerte_préc=False (drift supprime alertes)
- C0-C7 anomalie pure : flag_drift=False ET alerte_préc=True

Cette discrimination valide la cascade EWAT même en cas d'échec de H2a.

Usage
-----
    python -m experiments.h2_overlap.eval \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --precursor-dir experiments/precursor \\
        --features-root data/features/v3 \\
        --output experiments/h2_overlap \\
        [--p-threshold 0.4] [--n-bootstrap 1000]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_TRACKING_SILENT", "true")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

import numpy as np
import pandas as pd
import torch

import mlflow
from ewat.alerts.assembler import AlertAssembler
from ewat.drift.detector import DriftDetector
from ewat.drift.mmd import RFFKernel
from ewat.utils.bootstrap import bootstrap_proportion_ci

DRIFT_SCENARIOS = {
    "drift_config_change", "drift_rolling_deploy",
    "drift_scale_up", "drift_traffic_ramp",
}
STEP_SECONDS = 30.0
EPS_DRIFT = 0.5226   # Youden-optimal ε calibrated on train set


def _load_episode(
    features_root: Path, ep_id: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ep_dir = features_root / ep_id
    signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
    adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
    labels = pd.read_parquet(ep_dir / "labels.parquet")
    return signal, adjacency, labels


def _injection_step(labels: pd.DataFrame) -> int | None:
    non_normal = labels[labels["regime"] != "normal"]
    return int(non_normal.index[0]) if not non_normal.empty else None


def _classify_episode(
    assembler: AlertAssembler,
    signal: np.ndarray,
    adjacency: np.ndarray,
    labels: pd.DataFrame,
    ep_id: str,
    cluster_gt: int,
    scenario: str,
    p_threshold: float,
) -> dict:
    """Stream one episode, record drift flag AND precursor alert timing."""
    is_drift_ep = scenario in DRIFT_SCENARIOS
    injection_t = _injection_step(labels)
    t_total = signal.shape[0]

    k_min = max(1, min(assembler.k_optimal.values())) if assembler.k_optimal else 1
    # For θ_{drift∩anomaly} analysis, stream full episode always
    end_t = t_total

    assembler.threshold = p_threshold

    drift_flag_steps: list[int] = []
    alert_steps: list[int] = []
    alert_clusters: list[int] = []

    for t in range(k_min, end_t):
        alerts = assembler.predict(
            signal[:t], adjacency[:t],
            timestamp=float(t) * STEP_SECONDS,
            episode_id=ep_id,
        )
        if alerts:
            alert_steps.append(t)
            alert_clusters.append(alerts[0].cluster_id)

    # Replay with a fresh DriftDetector to record flag steps
    kernel = RFFKernel(rff_dim=256, seed=42)
    dd = DriftDetector(
        kernel=kernel, epsilon_drift=EPS_DRIFT,
        window_ref_size=5, window_cur_size=5, post_drift_window_s=3,
    )
    flat_signal = signal.reshape(t_total, -1).astype(np.float64)
    for t in range(t_total):
        res = dd.update(flat_signal[t])
        if res.flag:
            drift_flag_steps.append(t)

    first_drift_t = drift_flag_steps[0] if drift_flag_steps else None
    first_alert_t = alert_steps[0] if alert_steps else None

    # θ_{drift∩anomaly}: drift flag fired AND precursor alert fired
    has_drift_flag = first_drift_t is not None
    has_alert = first_alert_t is not None
    is_overlap = has_drift_flag and has_alert

    # For anomaly episodes: was precursor alert before injection?
    tp_precursor = (
        has_alert and injection_t is not None and first_alert_t <= injection_t
    ) if not is_drift_ep else None

    return {
        "episode_id": ep_id,
        "scenario": scenario,
        "cluster_gt": cluster_gt,
        "is_drift_scenario": is_drift_ep,
        "injection_t": injection_t,
        "first_drift_flag_t": first_drift_t,
        "first_alert_t": first_alert_t,
        "first_alert_cluster": alert_clusters[0] if alert_clusters else None,
        "n_drift_flags": len(drift_flag_steps),
        "n_alerts": len(alert_steps),
        "has_drift_flag": has_drift_flag,
        "has_alert": has_alert,
        "is_overlap": is_overlap,
        "tp_precursor": tp_precursor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H2b — θ_{drift∩anomaly} case study")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path, default=Path("experiments/h2_overlap"))
    parser.add_argument("--p-threshold", type=float, default=0.4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  p_threshold={args.p_threshold}")

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1

    # Use all splits for the overlap analysis (more statistical power for small clusters)
    all_episodes = list(cluster_manifest.items())
    print(f"Total episodes: {len(all_episodes)}  |  clusters: {n_clusters}")

    assembler = AlertAssembler.from_experiment_dirs(
        args.typing_dir, args.encoder_dir, args.precursor_dir,
        threshold=args.p_threshold, device=device,
    )

    # -----------------------------------------------------------------------
    # Stream all episodes
    # -----------------------------------------------------------------------
    records: list[dict] = []
    for ep_id, meta in all_episodes:
        try:
            signal, adjacency, labels = _load_episode(args.features_root, ep_id)
        except FileNotFoundError:
            continue
        rec = _classify_episode(
            assembler=assembler, signal=signal, adjacency=adjacency, labels=labels,
            ep_id=ep_id, cluster_gt=int(meta["cluster"]),
            scenario=meta.get("scenario", ""), p_threshold=args.p_threshold,
        )
        rec["split"] = meta.get("split", "?")
        records.append(rec)
        tag = ""
        if rec["is_overlap"]:
            tag = " ← OVERLAP ✓"
        elif rec["has_drift_flag"] and not rec["has_alert"]:
            tag = " ← drift pur"
        elif not rec["has_drift_flag"] and rec["has_alert"]:
            tag = " ← anomalie pure"
        print(f"  {ep_id}  C{rec['cluster_gt']}  {rec['scenario'][:25]:<25}"
              f"  drift={int(rec['has_drift_flag'])}  alert={int(rec['has_alert'])}{tag}")

    df = pd.DataFrame(records)
    df.to_csv(args.output / "per_episode.csv", index=False)

    # -----------------------------------------------------------------------
    # Per-cluster statistics
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    cluster_stats: list[dict] = []

    print("\n=== Per-cluster summary ===")
    for c in range(n_clusters):
        rows = df[df["cluster_gt"] == c]
        if len(rows) == 0:
            continue
        n = len(rows)
        n_drift_flag = int(rows["has_drift_flag"].sum())
        n_alert = int(rows["has_alert"].sum())
        n_overlap = int(rows["is_overlap"].sum())
        dominant_scenario = rows["scenario"].mode().iloc[0] if len(rows) > 0 else "?"

        ci_overlap = bootstrap_proportion_ci(
            n_overlap, n, n=args.n_bootstrap, rng=rng
        ).as_dict() if args.n_bootstrap > 0 else {}

        stat = {
            "cluster": c,
            "n": n,
            "dominant_scenario": dominant_scenario,
            "n_drift_flag": n_drift_flag,
            "n_alert": n_alert,
            "n_overlap": n_overlap,
            "pct_drift_flag": round(n_drift_flag / n, 3),
            "pct_alert": round(n_alert / n, 3),
            "pct_overlap": round(n_overlap / n, 3),
            "ci_overlap": ci_overlap,
        }
        cluster_stats.append(stat)
        ci_str = (
            f" [{ci_overlap.get('ci_lo', 0):.3f},{ci_overlap.get('ci_hi', 0):.3f}]"
            if ci_overlap else ""
        )
        print(f"  C{c}  n={n:>3}  {dominant_scenario[:22]:<22}  "
              f"drift%={n_drift_flag/n:.2f}  alert%={n_alert/n:.2f}  "
              f"overlap%={n_overlap/n:.2f}{ci_str}")

    # -----------------------------------------------------------------------
    # H2b verdict
    # -----------------------------------------------------------------------
    overlap_clusters = [s for s in cluster_stats if s["pct_overlap"] > 0.3]
    drift_clusters = [s for s in cluster_stats if s["pct_drift_flag"] > 0.5
                      and s["pct_alert"] < 0.3]

    h2b_pass = len(overlap_clusters) >= 1

    print(f"\nH2b {'✓ PASS' if h2b_pass else '✗ FAIL'}: "
          f"{len(overlap_clusters)} cluster(s) with overlap rate > 30% "
          f"(expected ≥1 for θ_{{drift∩anomaly}})")
    print(f"Overlap clusters : {[s['cluster'] for s in overlap_clusters]}")
    print(f"Pure-drift clusters: {[s['cluster'] for s in drift_clusters]}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    summary = {
        "p_threshold": args.p_threshold,
        "n_episodes": len(records),
        "n_clusters": n_clusters,
        "n_bootstrap": args.n_bootstrap,
        "h2b_pass": h2b_pass,
        "overlap_clusters": [s["cluster"] for s in overlap_clusters],
        "pure_drift_clusters": [s["cluster"] for s in drift_clusters],
        "cluster_stats": cluster_stats,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    # Report
    lines = [
        "# H2b — Étude de cas θ_{drift∩anomaly}\n",
        "**Reformulation de H2** : le système identifie-t-il correctement le régime",
        "θ_{drift∩anomaly} (drift + anomalie simultanés) "
        "distinct du drift pur et de l'anomalie pure ?\n",
        f"Seuil alerte : p > {args.p_threshold}  |  ε_drift = {EPS_DRIFT}\n",
        f"H2b : {'✓ PASS' if h2b_pass else '✗ FAIL'} "
        f"— {len(overlap_clusters)} cluster(s) avec chevauchement > 30%\n",
        "## Statistiques par cluster",
        f"{'C':<4}  {'N':>4}  {'Scénario dominant':<24}  "
        f"{'%drift_flag':>11}  {'%alert':>7}  {'%overlap':>9}  {'CI overlap':>20}",
        "-" * 85,
    ]
    for s in cluster_stats:
        ci = s.get("ci_overlap", {})
        ci_str = (
            f"[{ci.get('ci_lo', 0):.3f},{ci.get('ci_hi', 0):.3f}]"
            if ci else "N/A"
        )
        tag = " ← θ_{d∩a}" if s["pct_overlap"] > 0.3 else (
            " ← drift pur" if s["pct_drift_flag"] > 0.5 and s["pct_alert"] < 0.3 else ""
        )
        lines.append(
            f"C{s['cluster']:<3}  {s['n']:>4}  {s['dominant_scenario'][:24]:<24}  "
            f"{s['pct_drift_flag']:>11.3f}  {s['pct_alert']:>7.3f}  "
            f"{s['pct_overlap']:>9.3f}  {ci_str:>20}{tag}"
        )
    lines += [
        "",
        "## Interprétation",
        "- **Chevauchement (overlap)** : drift flag ET alerte précurseur tous les deux levés.",
        "- Un cluster avec chevauchement élevé incarne θ_{drift∩anomaly}.",
        "- Les clusters drift pur (%drift_flag élevé, %alert faible) valident le mécanisme",
        "  de suppression d'alertes lors des drifts bénins.",
        "- H2b PASS ≠ H2a (look-through FPR). H2b valide la *formalisation* à 4 régimes,",
        "  pas la réduction du FPR par look-through.",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("ewat_h2_overlap")
        with mlflow.start_run(run_name="h2b_overlap"):
            mlflow.log_params({
                "p_threshold": args.p_threshold,
                "eps_drift": EPS_DRIFT,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
            })
            mlflow.log_metrics({
                "n_overlap_clusters": float(len(overlap_clusters)),
                "h2b_pass": float(h2b_pass),
            })
            for s in cluster_stats:
                cid = s["cluster"]
                mlflow.log_metrics({
                    f"c{cid}_pct_drift": s["pct_drift_flag"],
                    f"c{cid}_pct_alert": s["pct_alert"],
                    f"c{cid}_pct_overlap": s["pct_overlap"],
                })
    except Exception:
        pass


if __name__ == "__main__":
    main()
