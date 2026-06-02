"""EWAT v5 — force les épisodes held-out en test-only dans un dataset assemblé.

`assemble_dataset.py` (split temporel/stratifié) ignore le flag `held_out_flag`.
Pour v5, les scénarios held-out (3 chaos held_* + bugs F) doivent être
**exclusivement en test**. Ce script ré-écrit `split.json` en déplaçant tout
épisode dont les labels portent `held_out_flag=True` vers test (et en le retirant
de train/val). Idempotent.

Usage :
    python scripts/enforce_heldout_v5.py --dataset data/datasets/ewat_v5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _is_heldout(ep_dir: Path) -> bool:
    lab = ep_dir / "labels.parquet"
    if not lab.exists():
        return False
    try:
        return bool(pd.read_parquet(lab, columns=["held_out_flag"])["held_out_flag"].any())
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 held-out → test-only enforcement")
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()

    split_path = args.dataset / "split.json"
    split = json.loads(split_path.read_text())
    eps_dir = args.dataset / "episodes"

    moved = []
    test = set(split.get("test", []))
    for bucket in ("train", "val"):
        kept = []
        for eid in split.get(bucket, []):
            if _is_heldout(eps_dir / eid):
                test.add(eid); moved.append((bucket, eid))
            else:
                kept.append(eid)
        split[bucket] = kept
    split["test"] = sorted(test)

    split_path.write_text(json.dumps(split, indent=2))
    print(f"held-out déplacés vers test : {len(moved)}")
    for b, e in moved:
        print(f"  {b} → test : {e}")
    print(f"split final : train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
