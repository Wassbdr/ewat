"""EWAT dataset — minimal standalone loader.

Self-contained : depends only on ``numpy``, ``pandas`` (with ``pyarrow``).
No dependency on the EWAT source tree — this file is copied into the release
so that downstream users can load the data without cloning the repo.

Each episode lives under ``data/<episode_id>/`` and contains:

- ``signal.npz``       — key ``signal``,       float32 ``(T, N, F)``  (F=18, schema v5.1)
- ``signal_mask.npz``  — key ``missing_mask``, bool   ``(T, N, F)``  (True = imputed)
- ``adjacency.npz``    — key ``adjacency``,    float32 ``(T, N, N, 3)`` (volume, latency, error)
- ``labels.parquet``   — per-timestep labels (regime, scenario, intensity_t, ...)
- ``metadata.json``    — scenario, boundaries, feature names, canonical services

The split assignment (train/val/test) is in ``split.json`` at the release root.

Examples
--------
>>> from release_load_ewat import load_episode, iter_split
>>> sig, mask, adj, labels, services = load_episode("data/episode_cpu_starvation_000")
>>> for ep_id, sig, mask, adj, labels, services in iter_split(".", "test"):
...     ...
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_episode(episode_dir: str | Path):
    """Load one episode's tensors and labels.

    Parameters
    ----------
    episode_dir : str or Path
        Path to a ``data/<episode_id>/`` directory.

    Returns
    -------
    signal : np.ndarray, shape (T, N, F), float32
        Imputed multi-modal signal S(t). F=18 (metrics[0:10] | traces[10:14] | logs[14:18]).
    mask : np.ndarray, shape (T, N, F), bool
        Missingness mask (True where the value was imputed).
    adjacency : np.ndarray, shape (T, N, N, 3), float32
        Dynamic service graph G(t) edge weights (volume, latency_med, error_rate).
    labels : pandas.DataFrame, T rows
        Per-timestep labels (regime, scenario, intensity_t, held_out_flag, ...).
    services : list of str
        Canonical service names, index-aligned with axis N of ``signal``/``adjacency``.
    """
    d = Path(episode_dir)
    with np.load(d / "signal.npz") as z:
        signal = z["signal"]
    with np.load(d / "signal_mask.npz") as z:
        mask = z["missing_mask"]
    with np.load(d / "adjacency.npz") as z:
        adjacency = z["adjacency"]
    labels = pd.read_parquet(d / "labels.parquet")
    services = json.loads((d / "services.json").read_text())
    return signal, mask, adjacency, labels, services


def load_split(release_root: str | Path) -> dict[str, list[str]]:
    """Return the split → list[episode_id] mapping from ``split.json``."""
    return json.loads((Path(release_root) / "split.json").read_text())


def iter_split(release_root: str | Path, split: str):
    """Iterate episodes of a given split.

    Yields ``(episode_id, signal, mask, adjacency, labels, services)`` tuples.
    """
    root = Path(release_root)
    ids = load_split(root)[split]
    for ep_id in ids:
        ep_dir = root / "data" / ep_id
        if not ep_dir.exists():
            continue
        yield (ep_id, *load_episode(ep_dir))


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    counts = {k: len(v) for k, v in load_split(root).items()}
    print("splits:", counts)
    first = next(iter_split(root, "test"), None)
    if first is not None:
        ep_id, sig, mask, adj, lab, svc = first
        print(f"sample {ep_id}: signal={sig.shape} adj={adj.shape} "
              f"services={len(svc)} labels_rows={len(lab)}")
