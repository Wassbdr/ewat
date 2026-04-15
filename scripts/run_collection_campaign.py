"""Run collection in scenario chunks and optionally merge outputs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from scripts.collect_labeled import collect_once
from scripts.merge_collection_runs import merge_runs


def _chunked(seq: list[str], chunk_size: int) -> list[list[str]]:
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run collection by scenario campaigns")
    parser.add_argument("--config", default="configs/collection.yaml")
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--endpoint-mode", choices=["cluster", "local-portforward"], default="cluster")
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = _cli()
    cfg = OmegaConf.load(args.config)
    scenarios = list(cfg.collection.scenarios)
    groups = _chunked(scenarios, args.chunk_size)
    campaign_runs: list[Path] = []

    for i, chunk in enumerate(groups):
        cfg.collection.scenarios = chunk
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            OmegaConf.save(config=cfg, f=tmp.name)
        run_dir = collect_once(
            config_path=tmp_path,
            base_config_path=Path(args.base_config),
            dry_run=args.dry_run,
            endpoint_mode=args.endpoint_mode,
        )
        campaign_runs.append(run_dir)
        print(json.dumps({"campaign_chunk": i, "scenarios": chunk, "run_dir": str(run_dir)}))

    if args.merge_output_dir:
        merged = merge_runs(
            run_dirs=campaign_runs,
            output_dir=Path(args.merge_output_dir),
        )
        print(json.dumps({"merged_run_dir": str(merged)}))


if __name__ == "__main__":
    main()
