"""H2 bis — Validation de la séparabilité drift/anomalie dans l'espace d'embeddings.

Hypothèse : MMD²(z_ref, z_cur) dans l'espace SiameseTyper sépare mieux drift et anomalie
que MMD²(signal_ref, signal_cur) dans l'espace brut 17-D (résultat de H2 FAIL).

Protocole : même que h2_lookthrough mais la statistique de test est calculée sur
les embeddings z = typer.embed(S, G) ∈ ℝ^{d_proj} plutôt que sur S(t) aplati.

Compare :
- Embedding look-through : DriftDetector sur z_t avec ε calibré par Youden sur embeddings
- Baseline : seuil simple MMD²(z) ≥ ε sur une fenêtre glissante

H2 bis PASS si FPR_lt < FPR_baseline (Student unilatéral, p < 0.05).

Usage
-----
    python -m experiments.h2_embeddings.eval \\
        --features-root data/features/v3 \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --output experiments/h2_embeddings \\
        [--epsilon None]  # None → calibré sur les données train
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ewat.drift.detector import DriftDetector
from ewat.drift.mmd import RFFKernel
from ewat.encoder.stgcn import STGCNEncoder
from ewat.typing.siamese import SiameseTyper

DRIFT_SCENARIOS = {
    "drift_config_change", "drift_rolling_deploy",
    "drift_scale_up", "drift_traffic_ramp",
}

STEP_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_typer(typing_dir: Path, encoder_dir: Path, device: torch.device) -> SiameseTyper:
    enc_ckpt = torch.load(
        encoder_dir / "checkpoints" / "best_encoder.pt",
        map_location="cpu", weights_only=False,
    )
    encoder = STGCNEncoder(d_feat=17, n_nodes=6, d_hidden=64, d_embed=64)
    encoder.load_state_dict(enc_ckpt["encoder_state"])

    typer_ckpt = torch.load(
        typing_dir / "checkpoints" / "best_siamese.pt",
        map_location="cpu", weights_only=False,
    )
    typer = SiameseTyper(encoder, d_proj=32)
    typer.load_state_dict(typer_ckpt["typer_state"])
    return typer.to(device).eval()


def _load_scaler(encoder_dir: Path) -> StandardScaler | None:
    scaler_path = encoder_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _embed_episode(
    typer: SiameseTyper,
    signal: np.ndarray,
    adjacency: np.ndarray,
    scaler: StandardScaler | None,
    device: torch.device,
) -> np.ndarray:
    """Embed each timestep using a sliding window of size 1. Returns (T, d_proj)."""
    signal = signal.astype(np.float32)
    if scaler is not None:
        t_len, n_nodes, d = signal.shape
        flat = signal.reshape(-1, d)
        flat = np.where(np.isnan(flat), 0.0, flat)
        flat = scaler.transform(flat).astype(np.float32)
        signal = flat.reshape(t_len, n_nodes, d)
    else:
        signal = np.nan_to_num(signal, nan=0.0)
    adjacency = np.nan_to_num(adjacency.astype(np.float32), nan=0.0)

    t_len = signal.shape[0]
    embeddings = []
    for t in range(t_len):
        sig_t = torch.from_numpy(signal[t:t+1]).unsqueeze(0).to(device)   # (1, 1, N, 17)
        adj_t = torch.from_numpy(adjacency[t:t+1]).unsqueeze(0).to(device)
        z = typer.embed(sig_t, adj_t).cpu().numpy()[0]   # (d_proj,)
        embeddings.append(z)
    return np.stack(embeddings)  # (T, d_proj)


# ---------------------------------------------------------------------------
# Calibration — find ε_emb via Youden on train set
# ---------------------------------------------------------------------------


def _calibrate_epsilon(
    cluster_manifest: dict,
    features_root: Path,
    typer: SiameseTyper,
    scaler: StandardScaler | None,
    device: torch.device,
    window_ref: int = 5,
    window_cur: int = 5,
    rff_dim: int = 256,
    seed: int = 42,
    n_thresholds: int = 20,
) -> float:
    """Youden-optimal ε on train set embeddings."""
    kernel = RFFKernel(rff_dim=rff_dim, seed=seed)

    drift_mmd2: list[float] = []
    anomaly_mmd2: list[float] = []

    for ep_id, meta in cluster_manifest.items():
        if meta.get("split") != "train":
            continue
        ep_dir = features_root / ep_id
        if not ep_dir.exists():
            continue
        signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
        z = _embed_episode(typer, signal, adjacency, scaler, device)  # (T, d_proj)

        if z.shape[0] < window_ref + window_cur:
            continue
        kernel.fit_sigma(z[:window_ref])

        is_drift = meta.get("scenario", "") in DRIFT_SCENARIOS
        # Compute max MMD² over all windows
        for t in range(window_ref, z.shape[0] - window_cur + 1):
            ref = z[t - window_ref: t]
            cur = z[t: t + window_cur]
            mmd2 = float(kernel.mmd_squared(ref, cur))
            if is_drift:
                drift_mmd2.append(mmd2)
            else:
                anomaly_mmd2.append(mmd2)

    all_scores = np.array(drift_mmd2 + anomaly_mmd2)
    all_labels = np.array([1] * len(drift_mmd2) + [0] * len(anomaly_mmd2))
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    best_eps, best_j = thresholds[0], -1.0
    for eps in thresholds:
        preds = (all_scores >= eps).astype(int)
        tp = int(((preds == 1) & (all_labels == 1)).sum())
        tn = int(((preds == 0) & (all_labels == 0)).sum())
        fp = int(((preds == 1) & (all_labels == 0)).sum())
        fn = int(((preds == 0) & (all_labels == 1)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_eps = j, eps

    print(f"  ε calibration: {len(drift_mmd2)} drift / {len(anomaly_mmd2)} anomaly windows"
          f"  → ε_emb={best_eps:.4f} (Youden J={best_j:.3f})")
    return float(best_eps)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def _baseline_drift(
    z: np.ndarray,
    injection_t: int | None,
    epsilon: float,
    window_ref: int,
    window_cur: int,
    seed: int = 42,
) -> bool:
    """True if MMD²(z) ≥ ε in any sliding window at or after injection."""
    kernel = RFFKernel(rff_dim=256, seed=seed)
    if z.shape[0] < window_ref:
        return False
    kernel.fit_sigma(z[:window_ref])

    for t in range(window_ref, z.shape[0] - window_cur + 1):
        ref = z[t - window_ref: t]
        cur = z[t: t + window_cur]
        mmd2 = float(kernel.mmd_squared(ref, cur))
        if mmd2 >= epsilon:
            if injection_t is None or t >= injection_t:
                return True
    return False


def _lookthrough_drift(
    z: np.ndarray,
    injection_t: int | None,
    epsilon: float,
    window_ref: int,
    window_cur: int,
    post_window: int,
    seed: int = 42,
) -> bool:
    """True if DriftDetector (look-through) flags DRIFT at or after injection."""
    kernel = RFFKernel(rff_dim=256, seed=seed)
    if z.shape[0] < window_ref:
        return False
    kernel.fit_sigma(z[:window_ref])

    detector = DriftDetector(
        kernel=kernel,
        epsilon_drift=epsilon,
        window_ref_size=window_ref,
        window_cur_size=window_cur,
        post_drift_window_s=post_window,
    )
    detector.load_reference(z[:window_ref])

    for t in range(z.shape[0]):
        result = detector.update(z[t].astype(np.float64))
        if result.flag:
            if injection_t is None or t >= injection_t:
                return True
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 bis — look-through sur embeddings")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--output", type=Path, default=Path("experiments/h2_embeddings"))
    parser.add_argument("--epsilon", type=float, default=None,
                        help="ε pour MMD²(z). None → calibré par Youden sur train.")
    parser.add_argument("--window-ref", type=int, default=5)
    parser.add_argument("--window-cur", type=int, default=5)
    parser.add_argument("--post-window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cluster manifest not found: {manifest_path}")
    cluster_manifest: dict[str, dict] = json.loads(manifest_path.read_text())

    typer = _load_typer(args.typing_dir, args.encoder_dir, device)
    scaler = _load_scaler(args.encoder_dir)
    print(f"Typer loaded. scaler={'yes' if scaler else 'no'}")

    # Calibrate ε if not provided
    if args.epsilon is None:
        print("\n[calibration — train set]")
        epsilon = _calibrate_epsilon(
            cluster_manifest, args.features_root, typer, scaler, device,
            window_ref=args.window_ref, window_cur=args.window_cur, seed=args.seed,
        )
    else:
        epsilon = args.epsilon
    print(f"ε_emb = {epsilon:.4f}")

    # Evaluate on test set
    test_episodes = [
        (ep_id, meta)
        for ep_id, meta in cluster_manifest.items()
        if meta.get("split") == "test"
    ]
    print(f"\nTest episodes: {len(test_episodes)}")

    records: list[dict] = []

    for ep_id, meta in test_episodes:
        ep_dir = args.features_root / ep_id
        if not ep_dir.exists():
            print(f"  skip {ep_id} (features not found)")
            continue

        signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
        labels = pd.read_parquet(ep_dir / "labels.parquet")

        z = _embed_episode(typer, signal, adjacency, scaler, device)

        scenario = meta.get("scenario", "")
        is_drift_ep = scenario in DRIFT_SCENARIOS

        non_normal = labels[labels["regime"] != "normal"]
        injection_t: int | None = int(non_normal.index[0]) if not non_normal.empty else None

        drift_lt = _lookthrough_drift(
            z, injection_t, epsilon, args.window_ref, args.window_cur, args.post_window, args.seed,
        )
        drift_bl = _baseline_drift(
            z, injection_t, epsilon, args.window_ref, args.window_cur, args.seed,
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

    drift_eps = df[df["is_drift"]]
    anomaly_eps = df[~df["is_drift"]]

    tpr_lt = float(drift_eps["drift_lookthrough"].mean()) if len(drift_eps) else float("nan")
    tpr_bl = float(drift_eps["drift_baseline"].mean()) if len(drift_eps) else float("nan")
    fpr_lt = float(anomaly_eps["drift_lookthrough"].mean()) if len(anomaly_eps) else float("nan")
    fpr_bl = float(anomaly_eps["drift_baseline"].mean()) if len(anomaly_eps) else float("nan")

    h2_pass = False
    p_value = float("nan")
    if len(anomaly_eps) >= 2:
        lt_vals = anomaly_eps["drift_lookthrough"].astype(float).values
        bl_vals = anomaly_eps["drift_baseline"].astype(float).values
        result = stats.ttest_rel(lt_vals, bl_vals, alternative="less")
        p_value = float(result.pvalue)
        h2_pass = bool(fpr_lt < fpr_bl and p_value < 0.05)

    print(f"\n--- H2 bis Summary (ε={epsilon:.4f}) ---")
    print(f"TPR (drift)    : lt={tpr_lt:.3f}  baseline={tpr_bl:.3f}")
    print(f"FPR (anomaly)  : lt={fpr_lt:.3f}  baseline={fpr_bl:.3f}")
    print(f"p-value (paired t, one-sided): {p_value:.4f}")
    print(f"H2 bis {'✓ PASS' if h2_pass else '✗ FAIL'}")

    summary = {
        "epsilon": epsilon,
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
        "# H2 bis — Look-through sur embeddings STGCN (test set)\n",
        f"ε_emb = {epsilon:.4f}  |  W_ref={args.window_ref}  W_cur={args.window_cur}  "
        f"W_post={args.post_window}\n",
        f"H2 bis : {'✓ PASS' if h2_pass else '✗ FAIL'}"
        f"  (p={p_value:.4f}, seuil 0.05)\n",
        f"{'':12} {'Look-through':>14} {'Baseline':>10}",
        f"{'TPR (drift)':12} {tpr_lt:>14.3f} {tpr_bl:>10.3f}",
        f"{'FPR (anomaly)':12} {fpr_lt:>14.3f} {fpr_bl:>10.3f}",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"Report: {args.output / 'results.md'}")


if __name__ == "__main__":
    main()
