"""EWAT — Feature imputation: v1 → v2.

Reads ``data/features/v1/`` and writes ``data/features/v2/`` with targeted
NaN imputation applied to signal.npz.  All other artefacts are copied verbatim.

Imputation strategy per feature (see docs/formalisation.md §Signal):

  error_rate_http  [dim 3]  83% NaN — leave as NaN (no Istio, mask in model)
  latency_p99      [dim 2]  50% NaN — forward-fill per service per episode,
                                       then backward-fill for leading NaNs
  trace_depth      [dim 9]  51% NaN — episode median per service; fallback to
                                       train-set median for (service, scenario)
  fan_out          [dim 10] 51% NaN — same
  disk_io          [dim 5]  31% NaN — episode median per service; fallback to
                                       train-set median for (service, scenario)

The train-set medians are computed from the stratified split
(``data/datasets/ewat_v1_strat/split.json``).  Features are never imputed
using val/test statistics to avoid data leakage.

Usage
=====

    python -m scripts.impute_features \\
        --features-root data/features/v1 \\
        --split-json    data/datasets/ewat_v1_strat/split.json \\
        --output        data/features/v2
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# Feature dimension indices in S(t) ∈ ℝ^{T×N×17}
_FEAT = {
    "latency_p99":     2,
    "error_rate_http": 3,
    "disk_io":         5,
    "trace_depth":     9,
    "fan_out":        10,
}
# Dims to impute with episode median + train-set fallback
_MEDIAN_DIMS = [_FEAT["disk_io"], _FEAT["trace_depth"], _FEAT["fan_out"]]

# Dims to ffill/bfill along time axis per service.
# All of these have sparse (4-6%) edge-window NaN — ffill is appropriate.
# NOTE: run this only on v1p+ (span-patched) features.
#   - latency_p99: structural 50% NaN for cart/load-gen in v1, ~17% after latency patch
#   - error_rate_http: 83% NaN in v1, ~0% after error rate patch
#   - span features: 4% edge NaN (empty windows at episode boundaries)
#   - log features: 5-6% edge NaN (same cause)
_FFILL_DIMS = [
    _FEAT["latency_p99"],        # M[2]  — spans fallback fills cart/load-gen
    _FEAT["error_rate_http"],    # M[3]  — spans fallback + edge residual
    7,   # span_dur_med
    8,   # abnormal_span_rate
    11,  # retry_rate
    12,  # latency_cv
    13,  # log_error_rate
    14,  # log_warn_rate
    15,  # semantic_anomaly
    16,  # lexical_entropy
]
_SKIP_DIMS: list[int] = []


def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill along the time axis (axis 0) for a 1-D or 2-D array."""
    out = arr.copy()
    if out.ndim == 1:
        out = out[:, np.newaxis]
    T, _ = out.shape
    # forward fill
    for t in range(1, T):
        mask = np.isnan(out[t])
        out[t, mask] = out[t - 1, mask]
    # backward fill (catches leading NaNs)
    for t in range(T - 2, -1, -1):
        mask = np.isnan(out[t])
        out[t, mask] = out[t + 1, mask]
    return out.squeeze()


def _episode_median(signal: np.ndarray, dim: int) -> np.ndarray:
    """Return per-service median of signal[:, :, dim], shape (N,). NaN where all-NaN."""
    feat = signal[:, :, dim]  # (T, N)
    return np.nanmedian(feat, axis=0).astype(np.float32)  # (N,)


def _build_train_medians(
    features_root: Path,
    train_ids: list[str],
    services_n: int,
) -> dict[tuple[str, int, int], float]:
    """Build train-set fallback medians keyed by (scenario, service_idx, dim)."""
    # Accumulate values: scenario → dim → service_idx → list of values
    from collections import defaultdict

    bucket: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for ep_id in train_ids:
        ep_dir = features_root / ep_id
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        scenario = (meta.get("scenario") or {}).get("name", "")
        with np.load(ep_dir / "signal.npz") as z:
            sig = z["signal"].astype(np.float64)  # (T, N, 17)
        for dim in _MEDIAN_DIMS:
            feat = sig[:, :, dim]  # (T, N)
            for s_idx in range(feat.shape[1]):
                vals = feat[:, s_idx]
                valid = vals[~np.isnan(vals)]
                bucket[scenario][dim][s_idx].extend(valid.tolist())

    result: dict[tuple[str, int, int], float] = {}
    for scenario, dim_map in bucket.items():
        for dim, svc_map in dim_map.items():
            for s_idx, vals in svc_map.items():
                if vals:
                    result[(scenario, s_idx, dim)] = float(np.median(vals))
    return result


def _impute_signal(
    signal: np.ndarray,
    scenario: str,
    train_medians: dict[tuple[str, int, int], float],
    global_medians: dict[tuple[int, int], float],
) -> np.ndarray:
    """Return a copy of signal with imputation applied."""
    out = signal.copy()
    T, N, _ = out.shape

    # --- forward-fill / backward-fill dims (latency_p99) ---
    for dim in _FFILL_DIMS:
        col = out[:, :, dim]  # (T, N)
        filled = _ffill_bfill(col)
        out[:, :, dim] = filled

    # --- per-service episode median with train-set fallback ---
    for dim in _MEDIAN_DIMS:
        ep_med = _episode_median(out, dim)  # (N,) — NaN where all-NaN
        for s_idx in range(N):
            col = out[:, s_idx, dim]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue
            fill_val = ep_med[s_idx]
            if np.isnan(fill_val):
                # fallback to train-set median for (scenario, service, dim)
                fill_val = train_medians.get((scenario, s_idx, dim), float("nan"))
            if np.isnan(fill_val):
                # last resort: global train median
                fill_val = global_medians.get((s_idx, dim), float("nan"))
            out[nan_mask, s_idx, dim] = fill_val

    # _SKIP_DIMS (error_rate_http) are left as-is
    return out


def _build_global_medians(
    features_root: Path,
    train_ids: list[str],
) -> dict[tuple[int, int], float]:
    """Global median across all train episodes, keyed by (service_idx, dim)."""
    from collections import defaultdict

    bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    for ep_id in train_ids:
        ep_dir = features_root / ep_id
        if not (ep_dir / "signal.npz").exists():
            continue
        with np.load(ep_dir / "signal.npz") as z:
            sig = z["signal"].astype(np.float64)
        for dim in _MEDIAN_DIMS + _FFILL_DIMS:
            feat = sig[:, :, dim]
            for s_idx in range(feat.shape[1]):
                vals = feat[:, s_idx]
                valid = vals[~np.isnan(vals)]
                bucket[(s_idx, dim)].extend(valid.tolist())
    return {k: float(np.median(v)) for k, v in bucket.items() if v}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    features_root = Path(args.features_root)
    if not features_root.is_absolute():
        features_root = REPO_ROOT / features_root
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    split_json = Path(args.split_json)
    if not split_json.is_absolute():
        split_json = REPO_ROOT / split_json

    logger.info("impute_features: %s → %s", features_root, output_root)

    split = json.loads(split_json.read_text())
    train_ids: list[str] = split["train"]
    logger.info("building train-set medians from %d train episodes …", len(train_ids))

    # Detect N from first episode
    first_ep = next(features_root.iterdir())
    with np.load(first_ep / "signal.npz") as z:
        services_n = z["signal"].shape[1]

    train_medians = _build_train_medians(features_root, train_ids, services_n)
    global_medians = _build_global_medians(features_root, train_ids)
    logger.info("train medians computed: %d entries", len(train_medians))

    output_root.mkdir(parents=True, exist_ok=True)

    ep_dirs = sorted(p for p in features_root.iterdir() if p.is_dir())
    n_imputed = 0
    for ep_dir in ep_dirs:
        dst = output_root / ep_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ep_dir, dst)  # copy everything first

        sig_path = dst / "signal.npz"
        if not sig_path.exists():
            continue

        meta_path = dst / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        scenario = (meta.get("scenario") or {}).get("name", "")

        with np.load(sig_path) as z:
            sig = z["signal"].astype(np.float32)

        imputed = _impute_signal(sig, scenario, train_medians, global_medians)

        # Write back with same keys
        with np.load(sig_path) as z:
            other_arrays = {k: v for k, v in z.items() if k != "signal"}
        np.savez_compressed(sig_path, signal=imputed, **other_arrays)
        n_imputed += 1

    logger.info("imputed %d episodes → %s", n_imputed, output_root)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EWAT Feature imputation: v1 → v2")
    p.add_argument("--features-root", default="data/features/v1")
    p.add_argument("--split-json", default="data/datasets/ewat_v1_strat/split.json",
                   help="split.json from stratified assembly (train IDs used for median estimation)")
    p.add_argument("--output", default="data/features/v2")
    return p.parse_args()


if __name__ == "__main__":
    main()
