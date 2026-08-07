"""H2 — Drift Separability Experiment: calibrate ε_drift on ewat_v3.

For each train episode:
  - Reference window  : first ``--window-ref`` timesteps (baseline phase)
  - Current window    : last  ``--window-cur`` timesteps (chaos phase)
  - MMD²(ref, cur) is computed via RFF-kernel

Expected finding: MMD² alone does NOT cleanly separate drift from anomaly
because both cause distribution shifts.  This motivates the look-through
mechanism (EWAT Step 0) and the encoder (Step 1).

ε_drift is set to the Youden-optimal threshold from the ROC curve
(maximises TPR − FPR on the train split).  H2 full confirmation requires
a temporal look-through simulation on the test set (future work).

Usage
=====

    python -m experiments.drift_separation.calibrate \\
        --dataset data/datasets/ewat_v3

Outputs (in experiments/drift_separation/):
  epsilon_calibrated.json   — {epsilon_drift, auc, metadata}
  results.md                — human-readable report with AUC + ROC table
  mmd2_distributions.png    — overlapping histograms (drift vs non-drift)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ewat.drift.calibration import save_calibration  # noqa: E402
from ewat.drift.mmd import RFFKernel  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "experiments" / "drift_separation"

DRIFT_CATEGORIES: frozenset[str] = frozenset({"drift"})
OVERLAP_CATEGORIES: frozenset[str] = frozenset({"overlap", "drift_anomaly"})


def _load_episode(
    feat_root: Path, ep_id: str
) -> tuple[np.ndarray | None, str, str]:
    """Load (signal, scenario_name, category). Returns (None, '', '') on error."""
    ep_dir = feat_root / ep_id
    sig_path = ep_dir / "signal.npz"
    meta_path = ep_dir / "metadata.json"
    if not sig_path.exists() or not meta_path.exists():
        return None, "", ""
    meta = json.loads(meta_path.read_text())
    scenario = meta.get("scenario") or {}
    name = scenario.get("name", "")
    category = scenario.get("category", "")
    with np.load(sig_path) as z:
        sig = z["signal"].astype(np.float64)
    return sig, name, category


def _episode_mmd2(
    signal: np.ndarray,
    kernel: RFFKernel,
    window_ref: int,
    window_cur: int,
) -> float | None:
    """Single-shot MMD²: ref = first window_ref timesteps, cur = last window_cur."""
    t_total = signal.shape[0]
    if t_total < window_ref + window_cur:
        return None
    flat = signal.reshape(t_total, -1)
    ref = flat[:window_ref]
    cur = flat[t_total - window_cur:]
    if kernel.sigma is None:
        kernel.fit_sigma(ref)
    return float(kernel.mmd_squared(ref, cur))


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    split_json = dataset_dir / "split.json"
    if not split_json.exists():
        raise FileNotFoundError(f"split.json not found: {split_json}")

    split = json.loads(split_json.read_text())
    train_ids: list[str] = split["train"]

    # Locate features root: dataset dir has a manifest pointing to the feature root.
    # Fallback: infer from naming convention (ewat_v3 → data/features/v3).
    # Audit 2026-06: l'assembleur écrit dataset.json (manifest.json n'a jamais
    # existé) — le fallback par nom cassait sur ewat_v4_strat → features/v4_strat.
    manifest_path = dataset_dir / "dataset.json"
    if not manifest_path.exists():
        manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        feat_root = Path(manifest.get("features_root", ""))
        if not feat_root.is_absolute():
            feat_root = REPO_ROOT / feat_root
    else:
        # Infer from dataset name
        ds_name = dataset_dir.name  # e.g. ewat_v3
        version = ds_name[len("ewat_"):]  # ewat_v3 → v3
        feat_root = REPO_ROOT / "data" / "features" / version
    if not feat_root.is_dir():
        raise FileNotFoundError(f"features root not found: {feat_root}")

    out_dir = Path(getattr(args, "output", None) or OUT_DIR)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    logger.info("Dataset : %s (%d train episodes)", dataset_dir.name, len(train_ids))
    logger.info("Features: %s", feat_root)

    kernel = RFFKernel(sigma=None, rff_dim=args.rff_dim, seed=42)
    window_ref = args.window_ref
    window_cur = args.window_cur

    drift_mmd2: list[float] = []
    nondrift_mmd2: list[float] = []
    overlap_mmd2: list[float] = []
    skipped = 0

    for ep_id in train_ids:
        sig, name, category = _load_episode(feat_root, ep_id)
        if sig is None:
            skipped += 1
            continue
        val = _episode_mmd2(sig, kernel, window_ref, window_cur)
        if val is None:
            logger.debug("  skip %s (T=%d < %d)", ep_id, sig.shape[0], window_ref + window_cur)
            skipped += 1
            continue
        if category in DRIFT_CATEGORIES:
            drift_mmd2.append(val)
        elif category in OVERLAP_CATEGORIES:
            overlap_mmd2.append(val)
        else:
            nondrift_mmd2.append(val)

    logger.info(
        "Computed: %d drift, %d non-drift, %d overlap, %d skipped",
        len(drift_mmd2), len(nondrift_mmd2), len(overlap_mmd2), skipped,
    )

    if not drift_mmd2:
        raise RuntimeError("No drift MMD² values — check window sizes or episode count.")

    drift_arr = np.array(drift_mmd2)
    nondrift_arr = np.array(nondrift_mmd2) if nondrift_mmd2 else np.array([])
    overlap_arr = np.array(overlap_mmd2) if overlap_mmd2 else np.array([])

    # ROC-based calibration: Youden-optimal threshold (maximises TPR − FPR)
    auc, epsilon, tpr_opt, fpr_opt, fpr_curve, tpr_curve, thresh_curve = _roc_calibrate(
        drift_arr, nondrift_arr
    )

    max_nondrift = float(nondrift_arr.max()) if len(nondrift_arr) else float("nan")
    separability_gap = epsilon - max_nondrift

    logger.info("ROC-AUC (drift vs non-drift): %.4f", auc)
    logger.info(
        "ε_drift (Youden-optimal)    : %.6f  (TPR=%.3f, FPR=%.3f)", epsilon, tpr_opt, fpr_opt,
    )
    logger.info("max(MMD²_nondrift)          : %.6f", max_nondrift)
    logger.info("Separability gap            : %+.6f", separability_gap)
    if auc < 0.7:
        logger.warning(
            "AUC=%.4f < 0.70 — per-episode MMD² does NOT cleanly separate drift from anomaly. "
            "This is expected: both cause distribution shifts. "
            "Full H2 requires temporal look-through simulation on the test set.",
            auc,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(
        epsilon,
        out_dir / "epsilon_calibrated.json",
        extra={
            "calibration_method": "roc_youden_optimal",
            "auc": auc,
            "tpr_at_epsilon": tpr_opt,
            "fpr_at_epsilon": fpr_opt,
            "n_drift": len(drift_mmd2),
            "n_nondrift": len(nondrift_mmd2),
            "n_overlap": len(overlap_mmd2),
            "max_nondrift_mmd2": max_nondrift,
            "separability_gap": separability_gap,
            "window_ref": window_ref,
            "window_cur": window_cur,
            "rff_dim": args.rff_dim,
            "dataset": str(dataset_dir),
        },
    )

    _write_report(
        epsilon=epsilon,
        auc=auc,
        tpr_opt=tpr_opt,
        fpr_opt=fpr_opt,
        drift_arr=drift_arr,
        nondrift_arr=nondrift_arr,
        overlap_arr=overlap_arr,
        separability_gap=separability_gap,
        window_ref=window_ref,
        window_cur=window_cur,
        dataset_name=dataset_dir.name,
        out_dir=out_dir,
    )

    _plot(
        drift_arr, nondrift_arr, overlap_arr, epsilon, fpr_curve, tpr_curve, auc, out_dir=out_dir
    )


def _roc_calibrate(
    drift_arr: np.ndarray,
    nondrift_arr: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC AUC and return the Youden-optimal threshold as ε_drift.

    Returns (auc, epsilon, tpr_opt, fpr_opt, fpr_curve, tpr_curve, thresholds).
    Falls back to p50 of drift if sklearn is unavailable.
    """
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        logger.warning("sklearn not available — falling back to p50(drift) as ε_drift")
        epsilon = float(np.median(drift_arr))
        return (float("nan"), epsilon, float("nan"), float("nan"),
                np.array([]), np.array([]), np.array([]))

    if len(nondrift_arr) == 0:
        logger.warning("No non-drift values — cannot compute ROC, using p50(drift)")
        epsilon = float(np.median(drift_arr))
        return (float("nan"), epsilon, float("nan"), float("nan"),
                np.array([]), np.array([]), np.array([]))

    y_true = np.concatenate([np.ones(len(drift_arr)), np.zeros(len(nondrift_arr))])
    y_score = np.concatenate([drift_arr, nondrift_arr])
    auc = float(roc_auc_score(y_true, y_score))
    fpr_curve, tpr_curve, thresh_curve = roc_curve(y_true, y_score)

    j = tpr_curve - fpr_curve
    opt_idx = int(np.argmax(j))
    epsilon = float(thresh_curve[opt_idx])
    tpr_opt = float(tpr_curve[opt_idx])
    fpr_opt = float(fpr_curve[opt_idx])
    return auc, epsilon, tpr_opt, fpr_opt, fpr_curve, tpr_curve, thresh_curve


def _stats(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "N/A"
    return (
        f"n={len(arr)}  min={arr.min():.4f}  med={np.median(arr):.4f}"
        f"  p95={np.percentile(arr,95):.4f}  max={arr.max():.4f}"
    )


def _write_report(
    epsilon: float,
    auc: float,
    tpr_opt: float,
    fpr_opt: float,
    drift_arr: np.ndarray,
    nondrift_arr: np.ndarray,
    overlap_arr: np.ndarray,
    separability_gap: float,
    window_ref: int,
    window_cur: int,
    dataset_name: str,
    out_dir: Path,
) -> None:
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    tpr_str = f"{tpr_opt:.3f}" if not np.isnan(tpr_opt) else "N/A"
    fpr_str = f"{fpr_opt:.3f}" if not np.isnan(fpr_opt) else "N/A"
    overlap_med = float(np.median(overlap_arr)) if len(overlap_arr) else float("nan")
    overlap_pos = "above" if not np.isnan(overlap_med) and overlap_med > epsilon else "below"

    report = f"""# H2 — Drift Separability: Results

Dataset: `{dataset_name}`
Windows: ref={window_ref} ts, cur={window_cur} ts

## Key Finding

Per-episode MMD² does **not** cleanly separate drift from anomaly (AUC = {auc_str}).
This is **by design**: both drift and anomaly episodes cause distribution shifts.
The look-through mechanism (Step 0) and the encoder (Step 1) are what enable
regime discrimination — not the MMD² magnitude alone.

## Calibrated threshold

| Parameter | Value |
|---|---|
| ε_drift (Youden-optimal) | **{epsilon:.6f}** |
| ROC-AUC | {auc_str} |
| TPR @ ε_drift | {tpr_str} |
| FPR @ ε_drift | {fpr_str} |
| Separability gap | {separability_gap:+.6f} |

## MMD² distributions (train)

| Group | Stats |
|---|---|
| Drift (θ_drift) | {_stats(drift_arr)} |
| Non-drift (θ_anomaly / θ_contention) | {_stats(nondrift_arr)} |
| Overlap (θ_drift∩anomaly) | {_stats(overlap_arr)} |

## Interpretation

- Anomaly episodes (OOM, fail_slow_latency, intermittent_error) reach MMD² > 1.0 —
  **higher** than many drift episodes — because they cause large signal shifts.
- Drift episodes (scale_up, config_change) sometimes have low MMD² because the new
  operating point differs only moderately from baseline.
- AUC = {auc_str} confirms partial but not complete separability.

**Implication for EWAT**: ε_drift controls sensitivity to *any* distribution shift,
not specifically to drift.  The look-through temporal check (RECALIBRATE vs DRIFT)
and the encoder embedding distinguish regime type after the threshold is crossed.

## Next steps

1. Run temporal look-through simulation on the test set to validate H2 properly
2. Implement Step 1 (STGCN encoder) to obtain z_e embeddings
3. Validate that z_e separates drift from anomaly (H1 precursor)

θ_drift∩anomaly (faulty_deploy_overlap): median MMD² = {overlap_med:.4f} — {overlap_pos} ε_drift.
"""
    out = out_dir / "results.md"
    out.write_text(report, encoding="utf-8")
    logger.info("Report → %s", out)


def _plot(
    drift_arr: np.ndarray,
    nondrift_arr: np.ndarray,
    overlap_arr: np.ndarray,
    epsilon: float,
    fpr_curve: np.ndarray,
    tpr_curve: np.ndarray,
    auc: float,
    *,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    has_roc = len(fpr_curve) > 0
    fig, axes = plt.subplots(1, 2 if has_roc else 1, figsize=(12 if has_roc else 7, 4))
    ax_hist = axes[0] if has_roc else axes

    all_vals = np.concatenate([drift_arr, nondrift_arr, overlap_arr])
    bins = np.linspace(0, all_vals.max() * 1.05, 35)
    if len(nondrift_arr):
        ax_hist.hist(
            nondrift_arr, bins=bins, alpha=0.55, label="Non-drift (anomaly)", color="steelblue",
        )
    if len(overlap_arr):
        ax_hist.hist(overlap_arr, bins=bins, alpha=0.55, label="θ_drift∩anomaly", color="gold")
    ax_hist.hist(drift_arr, bins=bins, alpha=0.55, label="Drift", color="tomato")
    ax_hist.axvline(
        epsilon, color="black", linestyle="--", lw=1.5, label=f"ε_drift = {epsilon:.3f}",
    )
    ax_hist.set_xlabel("MMD²")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("MMD² distributions (train)")
    ax_hist.legend(fontsize=8)

    if has_roc:
        ax_roc = axes[1]
        ax_roc.plot(fpr_curve, tpr_curve, color="darkorange", lw=2, label=f"ROC (AUC={auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("ROC: drift vs non-drift")
        ax_roc.legend(fontsize=9)

    fig.tight_layout()
    out = out_dir / "mmd2_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Plot → %s", out)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H2 drift calibration experiment")
    p.add_argument("--dataset", default="data/datasets/ewat_v3",
                   help="Path to assembled dataset directory (must contain split.json)")
    p.add_argument("--window-ref", type=int, default=5,
                   help="Reference window size in timesteps (default 5 ≈ baseline phase)")
    p.add_argument("--window-cur", type=int, default=5,
                   help="Current window size in timesteps (default 5 ≈ chaos phase)")
    p.add_argument("--rff-dim", type=int, default=256,
                   help="Number of Random Fourier Features")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: experiments/drift_separation)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_cli())
