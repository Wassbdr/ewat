"""Patch error_rate_http (signal dim 3) in existing feature episodes.

Reads an existing feature root and fills ``error_rate_http`` from Jaeger span
status codes for timesteps where Prometheus gave NaN.  All other features are
left unchanged.

Sources used (first non-NaN wins per timestep × service):
  1. Existing Prometheus value (kept as-is when non-NaN)
  2. Server-side HTTP spans  (http.status_code / http.response.status_code)
  3. Client-side gRPC spans  (rpc.grpc.status_code → attributed to callee)

This script is idempotent: existing non-NaN values are never overwritten.

Usage
=====

    # In-place (modifies data/features/v1/)
    python -m scripts.patch_error_rate \\
        --features-root data/features/v1 \\
        --raw-root      data/raw

    # Write to a new root (copy + patch)
    python -m scripts.patch_error_rate \\
        --features-root data/features/v1 \\
        --raw-root      data/raw \\
        --output        data/features/v1p
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from telemetry.extractors.traces_file import SpanErrorRateIndex  # noqa: E402

logger = logging.getLogger(__name__)

_ERR_HTTP_DIM = 3  # absolute index in the 17-dim signal


def _load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _grpc_callee_map(services: list[str]) -> dict[str, str]:
    """Map normalised gRPC service names → canonical service names."""
    mapping: dict[str, str] = {}
    for svc in services:
        mapping[svc.replace("-", "").replace("_", "")] = svc
    for svc in services:
        if "catalog" in svc:
            mapping["productcatalog"] = svc
        if "review" in svc:
            mapping["productreview"] = svc
    return {k: v for k, v in mapping.items() if v}


def patch_episode(
    feat_dir: Path,
    raw_ep_dir: Path,
    out_sig_path: Path,
    trace_window_s: float,
) -> tuple[int, int]:
    """Patch signal.npz for one episode.

    Returns (n_timestep_service_slots_filled, n_already_nonnan).
    """
    sig_path = feat_dir / "signal.npz"
    jaeger_path = raw_ep_dir / "jaeger_spans.json.gz"

    if not sig_path.exists() or not jaeger_path.exists():
        return 0, 0

    with np.load(sig_path) as z:
        sig = z["signal"].astype(np.float32)
        other = {k: v for k, v in z.items() if k != "signal"}

    T, N, _ = sig.shape
    meta = json.loads((feat_dir / "metadata.json").read_text()) if \
        (feat_dir / "metadata.json").exists() else {}
    services = json.loads((feat_dir / "services.json").read_text()) if \
        (feat_dir / "services.json").exists() else []
    aliases: dict[str, str] = (meta.get("config", {}) or {}).get("aliases", {}) or {}

    bounds = meta.get("boundaries", {}) or {}
    t_start = float(bounds.get("baseline_start", 0.0))
    t_end = float(bounds.get("recovery_end", 0.0))
    if T <= 1 or t_end <= t_start:
        return 0, 0
    grid_step = (t_end - t_start) / (T - 1)

    jaeger_dump = _load_json_gz(jaeger_path)
    callee_map = _grpc_callee_map(services)
    idx = SpanErrorRateIndex(
        jaeger_dump,
        canonical_services=services,
        aliases=dict(aliases),
        grpc_callee_map=callee_map,
    )

    n_filled = 0
    for i in range(T):
        ts = t_start + i * grid_step
        nan_mask = np.isnan(sig[i, :, _ERR_HTTP_DIM])
        if not nan_mask.any():
            continue
        err_rates = idx.error_rate_for_window(ts - trace_window_s, ts)
        for s_idx, svc_name in enumerate(services):
            if nan_mask[s_idx]:
                val = err_rates.get(svc_name, float("nan"))
                if not np.isnan(val):
                    sig[i, s_idx, _ERR_HTTP_DIM] = val
                    n_filled += 1

    np.savez_compressed(out_sig_path, signal=sig, **other)
    return n_filled, int(np.sum(~np.isnan(sig[:, :, _ERR_HTTP_DIM])))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    feat_root = Path(args.features_root)
    raw_root = Path(args.raw_root)
    out_root = Path(args.output) if args.output else feat_root
    for p in (feat_root, raw_root, out_root):
        if not p.is_absolute():
            p = REPO_ROOT / p

    feat_root = REPO_ROOT / feat_root if not feat_root.is_absolute() else feat_root
    raw_root = REPO_ROOT / raw_root if not raw_root.is_absolute() else raw_root
    out_root = REPO_ROOT / out_root if not out_root.is_absolute() else out_root

    inplace = feat_root.resolve() == out_root.resolve()
    logger.info("patch_error_rate: %s → %s (%s)",
                feat_root, out_root, "in-place" if inplace else "copy")

    if not inplace:
        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.copytree(feat_root, out_root)
        logger.info("Copied feature root to %s", out_root)

    ep_dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
    total_filled = 0
    n_ok = n_skip = 0

    for ep_dir in ep_dirs:
        ep_id = ep_dir.name
        raw_ep = raw_root / ep_id
        if not raw_ep.is_dir():
            logger.warning("  skip %s (no matching raw dir)", ep_id)
            n_skip += 1
            continue
        # When inplace, feat_dir == ep_dir (already copied above if not inplace)
        feat_dir = feat_root / ep_id
        try:
            filled, _ = patch_episode(
                feat_dir=feat_dir,
                raw_ep_dir=raw_ep,
                out_sig_path=ep_dir / "signal.npz",
                trace_window_s=args.trace_window_s,
            )
            total_filled += filled
            n_ok += 1
            if filled > 0:
                logger.info("  %s  filled %d slots", ep_id, filled)
        except Exception:
            logger.exception("  FAILED %s", ep_id)

    # Report final NaN rate on a sample
    logger.info("Done: %d patched, %d skipped — total slots filled: %d",
                n_ok, n_skip, total_filled)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch error_rate_http from Jaeger spans")
    p.add_argument("--features-root", default="data/features/v1")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--output", default=None,
                   help="Output root (default: in-place modification of --features-root)")
    p.add_argument("--trace-window-s", type=float, default=120.0,
                   help="Span window in seconds (must match build_features --trace-window-s)")
    return p.parse_args()


if __name__ == "__main__":
    main()
