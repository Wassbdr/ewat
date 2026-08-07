"""Hydra entry point for siamese typing training.

Delegates to ``experiments.typing.train.run`` so behaviour is identical to
the legacy argparse CLI. See ``configs/experiment/typing.yaml`` for the
default config and override syntax.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.typing.train import run

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs")


def cfg_to_args(cfg: DictConfig) -> argparse.Namespace:
    raw = OmegaConf.to_container(cfg, resolve=True)
    return argparse.Namespace(
        dataset=Path(raw["dataset"]),
        features_root=Path(raw["features_root"]),
        encoder_checkpoint=Path(raw["encoder_checkpoint"]),
        output=Path(raw["output"]),
        epochs=int(raw["epochs"]),
        lr=float(raw["lr"]),
        batch_size=int(raw["batch_size"]),
        patience=int(raw["patience"]),
        d_proj=int(raw["d_proj"]),
        margin=float(raw["margin"]),
        n_neg_per_anchor=int(raw["n_neg_per_anchor"]),
        freeze_encoder=bool(raw["freeze_encoder"]),
        k_range_max=int(raw["k_range_max"]),
        n_gap_refs=int(raw["n_gap_refs"]),
        n_shap_bg=int(raw["n_shap_bg"]),
        n_bootstrap=int(raw["n_bootstrap"]),
        seed=int(raw["seed"]),
        eval_only=bool(raw["eval_only"]),
        mining=str(raw.get("mining", "random")),
        mining_warmup_epochs=int(raw.get("mining_warmup_epochs", 3)),
        mining_pool_size=int(raw.get("mining_pool_size", 0)),
    )


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="experiment/typing")
def hydra_main(cfg: DictConfig) -> None:
    print("Resolved Hydra config:")
    print(OmegaConf.to_yaml(cfg))
    args = cfg_to_args(cfg)
    run(args)


if __name__ == "__main__":
    hydra_main()
