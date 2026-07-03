"""Patch latency_p99 (signal dim 2) in existing feature episodes.

Fills ``latency_p99`` from Jaeger span duration P99 for timesteps where
Prometheus gave NaN (services without HTTP histogram metrics).

Only services with direct Jaeger spans benefit (cart, load-generator in the
OTel Demo).  Services without direct spans (ad, product-catalog,
recommendation) remain NaN — they require OTel SDK instrumentation.

Idempotent: existing non-NaN values are never overwritten.

Usage
=====

    # In-place
    python -m scripts.patch_latency_p99 \\
        --features-root data/features/v1p \\
        --raw-root      data/raw

    # Write to new root
    python -m scripts.patch_latency_p99 \\
        --features-root data/features/v1p \\
        --raw-root      data/raw \\
        --output        data/features/v1pp
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

from telemetry.extractors.traces_file import SpanLatencyIndex  # noqa: E402

logger = logging.getLogger(__name__)

_LAT_P99_DIM = 2  # absolute index in the 17-dim signal


def _load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def patch_episode(
    feat_dir: Path,
    raw_ep_dir: Path,
    out_sig_path: Path,
    trace_window_s: float,
) -> tuple[int, int]:
    """Patch signal.npz for one episode. Returns (slots_filled, still_nan)."""
    sig_path = feat_dir / "signal.npz"
    jaeger_path = raw_ep_dir / "jaeger_spans.json.gz"
    if not sig_path.exists() or not jaeger_path.exists():
        return 0, 0

    with np.load(sig_path) as z:
        sig = z["signal"].astype(np.float32)
        other = {k: v for k, v in z.items() if k != "signal"}

    T, N, _ = sig.shape
    meta = json.loads((feat_dir / "metadata.json").read_text()) \
        if (feat_dir / "metadata.json").exists() else {}
    services = json.loads((feat_dir / "services.json").read_text()) \
        if (feat_dir / "services.json").exists() else []
    aliases: dict[str, str] = (meta.get("config", {}) or {}).get("aliases", {}) or {}

    bounds = meta.get("boundaries", {}) or {}
    t_start = float(bounds.get("baseline_start", 0.0))
    t_end = float(bounds.get("recovery_end", 0.0))
    if T <= 1 or t_end <= t_start:
        return 0, 0
    grid_step = (t_end - t_start) / (T - 1)

    jaeger_dump = _load_json_gz(jaeger_path)
    idx = SpanLatencyIndex(jaeger_dump, canonical_services=services, aliases=dict(aliases))

    n_filled = 0
    for i in range(T):
        ts = t_start + i * grid_step
        nan_mask = np.isnan(sig[i, :, _LAT_P99_DIM])
        if not nan_mask.any():
            continue
        p99s = idx.p99_for_window(ts - trace_window_s, ts)
        for s_idx, svc_name in enumerate(services):
            if nan_mask[s_idx]:
                val = p99s.get(svc_name, float("nan"))
                if not np.isnan(val):
                    sig[i, s_idx, _LAT_P99_DIM] = val
                    n_filled += 1

    n_still_nan = int(np.isnan(sig[:, :, _LAT_P99_DIM]).sum())
    np.savez_compressed(out_sig_path, signal=sig, **other)
    return n_filled, n_still_nan


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    feat_root = Path(args.features_root)
    raw_root = Path(args.raw_root)
    out_root = Path(args.output) if args.output else feat_root
    for attr, p in [("feat_root", feat_root), ("raw_root", raw_root), ("out_root", out_root)]:
        if not p.is_absolute():
            p = REPO_ROOT / p
        locals()[attr]  # re-bind absolute path
    feat_root = REPO_ROOT / feat_root if not feat_root.is_absolute() else feat_root
    raw_root  = REPO_ROOT / raw_root  if not raw_root.is_absolute()  else raw_root
    out_root  = REPO_ROOT / out_root  if not out_root.is_absolute()  else out_root

    inplace = feat_root.resolve() == out_root.resolve()
    logger.info("patch_latency_p99: %s → %s (%s)",
                feat_root, out_root, "in-place" if inplace else "copy")

    if not inplace:
        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.copytree(feat_root, out_root)
        logger.info("Copied %s → %s", feat_root, out_root)

    ep_dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
    total_filled = total_still = 0
    n_ok = n_skip = 0

    for ep_dir in ep_dirs:
        ep_id = ep_dir.name
        raw_ep = raw_root / ep_id
        if not raw_ep.is_dir():
            n_skip += 1
            continue
        feat_dir = feat_root / ep_id
        try:
            filled, still = patch_episode(
                feat_dir=feat_dir,
                raw_ep_dir=raw_ep,
                out_sig_path=ep_dir / "signal.npz",
                trace_window_s=args.trace_window_s,
            )
            total_filled += filled
            total_still += still
            n_ok += 1
            if filled > 0:
                logger.info("  %s  filled %d slots", ep_id, filled)
        except Exception:
            logger.exception("  FAILED %s", ep_id)

    logger.info("Done: %d patched, %d skipped — filled %d slots, %d still NaN",
                n_ok, n_skip, total_filled, total_still)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch latency_p99 from Jaeger span durations")
    p.add_argument("--features-root", default="data/features/v1p")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--output", default=None)
    p.add_argument("--trace-window-s", type=float, default=120.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
