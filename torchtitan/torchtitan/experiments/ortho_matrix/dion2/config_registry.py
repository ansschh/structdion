"""
Config registry for Dion2 optimizer experiments.

Usage:
    python -m torchtitan.train \
        --module ortho_matrix.dion2 \
        --config llama3_160m_dion2 \
        --training.steps 2000
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
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
# Dion2 optimizer container
# ======================================================================

class Dion2Container(BaseHybridOptimizersContainer):
    """Dion2 (fraction selection + NS) + AdamW scalar optimizer container."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseHybridOptimizersContainer.Config):
        name: str = "Dion2"
        fraction: float = 0.25
        ef_decay: float = 0.95

    @staticmethod
    def _create_optimizer(
        config: "Dion2Container.Config",
        param_groups: list[dict],
        mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        from dion import Dion2

        return Dion2(
            param_groups,
            distributed_mesh=mesh,
            lr=config.lr,
            fraction=config.fraction,
            ef_decay=config.ef_decay,
            weight_decay=config.weight_decay,
        )


# ======================================================================
# 160M configs
# ======================================================================

def llama3_160m_dion2() -> Trainer.Config:
    """LLaMA3 160M with Dion2 (fraction selection + NS) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_160m(),
        optimizer=Dion2Container.Config(
            lr=0.02,
            fraction=0.25,
            ef_decay=0.95,
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

def llama3_320m_dion2() -> Trainer.Config:
    """LLaMA3 320M with Dion2 (fraction selection + NS) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=Dion2Container.Config(
            lr=0.012,
            fraction=0.5,
            ef_decay=0.95,
            weight_decay=0.1,
            scalar_lr=0.012,
            scalar_beta1=0.95,
            scalar_beta2=0.95,
            scalar_eps=1e-8,
            scalar_weight_decay=0.1,
            output_head_lr_scaling=False,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=610,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
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

def llama3_debug_dion2() -> Trainer.Config:
    """Tiny debug model with Dion2 — for local validation."""
    config = debug_trainer_base()
    config.optimizer = Dion2Container.Config(
        lr=0.02,
        fraction=0.25,
        ef_decay=0.95,
        weight_decay=0.0,
        scalar_lr=3e-4,
        scalar_weight_decay=0.01,
    )
    return config
