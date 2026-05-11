"""
Config registry for AdaDion optimizer experiments.

Usage:
    python -m torchtitan.train \
        --module ortho_matrix.ada_dion \
        --config llama3_320m_adadion \
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
# AdaDion optimizer container
# ======================================================================

class AdaDionContainer(BaseHybridOptimizersContainer):
    """AdaDion (adaptive rank) + AdamW scalar optimizer container."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseHybridOptimizersContainer.Config):
        name: str = "AdaDion"
        mu: float = 0.95
        init_rank: int = 64
        erank_ema_beta: float = 0.9
        rank_scale: float = 1.5
        rank_min: int = 16
        rank_max: int = 256
        rank_quantize: int = 8
        rank_step_up: int = 16
        rank_step_down: int = 8
        use_quality_control: bool = True
        aerr_ema_beta: float = 0.95
        aerr_target: float = 0.08
        aerr_up_margin: float = 0.15
        aerr_down_margin: float = 0.15

    @staticmethod
    def _create_optimizer(
        config: "AdaDionContainer.Config",
        param_groups: list[dict],
        mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        from .adadion import AdaDion

        return AdaDion(
            param_groups,
            lr=config.lr,
            mu=config.mu,
            init_rank=config.init_rank,
            erank_ema_beta=config.erank_ema_beta,
            rank_scale=config.rank_scale,
            rank_min=config.rank_min,
            rank_max=config.rank_max,
            rank_quantize=config.rank_quantize,
            rank_step_up=config.rank_step_up,
            rank_step_down=config.rank_step_down,
            use_quality_control=config.use_quality_control,
            aerr_ema_beta=config.aerr_ema_beta,
            aerr_target=config.aerr_target,
            aerr_up_margin=config.aerr_up_margin,
            aerr_down_margin=config.aerr_down_margin,
            weight_decay=config.weight_decay,
            device_mesh=mesh,
        )


# ======================================================================
# 160M configs
# ======================================================================

def llama3_160m_adadion() -> Trainer.Config:
    """LLaMA3 160M with AdaDion (adaptive rank) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_160m(),
        optimizer=AdaDionContainer.Config(
            lr=0.02,
            mu=0.95,
            init_rank=64,

            erank_ema_beta=0.9,
            rank_scale=1.5,
            rank_min=16,
            rank_max=256,
            rank_quantize=8,

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

def llama3_320m_adadion() -> Trainer.Config:
    """LLaMA3 320M with AdaDion (adaptive rank) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=AdaDionContainer.Config(
            lr=0.016,
            mu=0.95,
            init_rank=64,

            erank_ema_beta=0.9,
            rank_scale=1.5,
            rank_min=16,
            rank_max=256,
            rank_quantize=8,

            weight_decay=0.1,
            scalar_lr=0.008,
            scalar_weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=610,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
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

def llama3_debug_adadion() -> Trainer.Config:
    """Tiny debug model with AdaDion — for local validation."""
    config = debug_trainer_base()
    config.optimizer = AdaDionContainer.Config(
        lr=0.02,
        mu=0.95,
        init_rank=16,
        erank_ema_beta=0.9,
        rank_scale=1.5,
        rank_min=8,
        rank_max=64,
        rank_quantize=8,

        weight_decay=0.0,
        scalar_lr=3e-4,
        scalar_weight_decay=0.01,
    )
    return config
