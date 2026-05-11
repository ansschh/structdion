"""
Config registry for AnshAdaDion optimizer experiments.

Usage:
    python -m torchtitan.train \
        --module ortho_matrix.ansh_adadion \
        --config llama3_160m_ansh_adadion \
        --training.steps 2000
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from ..common.base_container import BaseHybridOptimizersContainer
from ..common.model_configs import model_registry_160m, model_registry_320m
from ..common.training_configs import (
    base_lr_scheduler_config,
    base_metrics_config,
    base_training_config,
    debug_trainer_base,
    llama3_160m_adamw,
    llama3_320m_adamw,
)


# ======================================================================
# AnshAdaDion optimizer container
# ======================================================================

class AnshAdaDionContainer(BaseHybridOptimizersContainer):
    """AnshAdaDion (anchor regularization) + AdamW scalar optimizer container."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseHybridOptimizersContainer.Config):
        name: str = "AnshAdaDion"
        mu: float = 0.95
        rank_fraction: float = 0.25
        anchor_lambda: float = 0.1
        anchor_rho: float = 0.99
        tau_hi: float = 0.8
        tau_lo: float = 0.3
        refresh_period: int = 100

    @staticmethod
    def _create_optimizer(
        config: "AnshAdaDionContainer.Config",
        param_groups: list[dict],
        mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        from .ansh_adadion import AdaDion as AnshAdaDion

        return AnshAdaDion(
            param_groups,
            lr=config.lr,
            mu=config.mu,
            rank_fraction=config.rank_fraction,
            anchor_lambda=config.anchor_lambda,
            anchor_rho=config.anchor_rho,
            tau_hi=config.tau_hi,
            tau_lo=config.tau_lo,
            weight_decay=config.weight_decay,
            refresh_period=config.refresh_period,
        )


# ======================================================================
# 160M configs
# ======================================================================

def llama3_160m_ansh_adadion() -> Trainer.Config:
    """LLaMA3 160M with AnshAdaDion (anchor regularization) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_160m(),
        optimizer=AnshAdaDionContainer.Config(
            lr=0.02,
            mu=0.95,
            rank_fraction=0.25,
            anchor_lambda=0.1,
            anchor_rho=0.99,
            weight_decay=0.0,
            scalar_lr=3e-4,
            scalar_weight_decay=0.01,
        ),
        lr_scheduler=base_lr_scheduler_config(),
        training=base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=base_metrics_config(),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=True,
        ),
        validator=Validator.Config(
            freq=100,
            steps=20,
        ),
    )


# ======================================================================
# 320M configs
# ======================================================================

def llama3_320m_ansh_adadion() -> Trainer.Config:
    """LLaMA3 320M with AnshAdaDion (anchor regularization) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=AnshAdaDionContainer.Config(
            lr=0.02,
            mu=0.95,
            rank_fraction=0.25,
            anchor_lambda=0.1,
            anchor_rho=0.99,
            weight_decay=0.0,
            scalar_lr=3e-4,
            scalar_weight_decay=0.01,
        ),
        lr_scheduler=base_lr_scheduler_config(),
        training=base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=base_metrics_config(),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=True,
        ),
        validator=Validator.Config(
            freq=100,
            steps=20,
        ),
    )


# ======================================================================
# Debug config
# ======================================================================

def llama3_debug_ansh_adadion() -> Trainer.Config:
    """Tiny debug model with AnshAdaDion — for local validation."""
    config = debug_trainer_base()
    config.optimizer = AnshAdaDionContainer.Config(
        lr=0.02,
        mu=0.95,
        rank_fraction=0.25,
        anchor_lambda=0.1,
        anchor_rho=0.99,
        weight_decay=0.0,
        scalar_lr=3e-4,
        scalar_weight_decay=0.01,
    )
    return config
