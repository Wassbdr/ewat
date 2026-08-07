"""Hydra entry point for precursor training.

Delegates to ``experiments.precursor.train.run`` so behaviour is identical to
the legacy argparse CLI. See ``configs/experiment/precursor.yaml`` for the
default config and override syntax.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.precursor.train import run

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs")


def cfg_to_args(cfg: DictConfig) -> argparse.Namespace:
    raw = OmegaConf.to_container(cfg, resolve=True)
    return argparse.Namespace(
        typing_dir=Path(raw["typing_dir"]),
        features_root=Path(raw["features_root"]),
        output=Path(raw["output"]),
        k_values=[int(k) for k in raw["k_values"]],
        reg_c=float(raw["reg_c"]),
        max_iter=int(raw["max_iter"]),
        n_bootstrap=int(raw["n_bootstrap"]),
        seed=int(raw["seed"]),
    )


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="experiment/precursor")
def hydra_main(cfg: DictConfig) -> None:
    print("Resolved Hydra config:")
    print(OmegaConf.to_yaml(cfg))
    args = cfg_to_args(cfg)
    run(args)


if __name__ == "__main__":
    hydra_main()
