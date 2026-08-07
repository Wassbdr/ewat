"""Hydra entry point for encoder pre-training.

Composes a Hydra DictConfig from ``configs/experiment/encoder.yaml`` (or any
override the user passes) and delegates to the same ``run(args)`` function
used by the legacy argparse entry point. This means:

* No behavioural drift between ``train.py`` (argparse) and this file.
* Hydra composition / multirun works out of the box.
* Any future migration of more knobs is additive — just expand the YAML.

Usage
-----
    # default run
    python -m experiments.encoder.train_hydra \
        dataset=data/datasets/ewat_v3 \
        features_root=data/features/v3

    # override hyperparameters
    python -m experiments.encoder.train_hydra epochs=20 lr=5e-4

    # multirun (Hydra sweep)
    python -m experiments.encoder.train_hydra -m seed=42,1,2,3,4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.encoder.train import run

# Resolve repo-root-relative path to the configs folder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs")


def cfg_to_args(cfg: DictConfig) -> argparse.Namespace:
    """Translate a Hydra DictConfig into the Namespace expected by ``run``.

    Coerces the path-typed fields back to ``Path`` so the downstream training
    code (which assumes ``args.dataset / "split.json"``, etc.) keeps working.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    return argparse.Namespace(
        dataset=Path(raw["dataset"]),
        features_root=Path(raw["features_root"]),
        output=Path(raw["output"]),
        epochs=int(raw["epochs"]),
        lr=float(raw["lr"]),
        batch_size=int(raw["batch_size"]),
        patience=int(raw["patience"]),
        d_hidden=int(raw["d_hidden"]),
        d_embed=int(raw["d_embed"]),
        seed=int(raw["seed"]),
    )


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="experiment/encoder")
def hydra_main(cfg: DictConfig) -> None:
    print("Resolved Hydra config:")
    print(OmegaConf.to_yaml(cfg))
    args = cfg_to_args(cfg)
    run(args)


if __name__ == "__main__":
    hydra_main()
