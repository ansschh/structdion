"""
Config registry for Muon optimizer experiments.

Usage:
    python -m torchtitan.train \
        --module ortho_matrix.muon \
        --config llama3_160m_muon \
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
# Muon optimizer container
# ======================================================================

class MuonContainer(BaseHybridOptimizersContainer):
    """Muon (Newton-Schulz) + AdamW scalar optimizer container."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseHybridOptimizersContainer.Config):
        name: str = "Muon"
        mu: float = 0.95
        nesterov: bool = True
        adjust_lr: str = "spectral_norm"

    @staticmethod
    def _create_optimizer(
        config: "MuonContainer.Config",
        param_groups: list[dict],
        mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        from dion import Muon

        adjust_lr = config.adjust_lr if config.adjust_lr != "none" else None
        return Muon(
            param_groups,
            distributed_mesh=mesh,
            lr=config.lr,
            mu=config.mu,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
            adjust_lr=adjust_lr,
        )


# ======================================================================
# 160M configs
# ======================================================================

def llama3_160m_muon() -> Trainer.Config:
    """LLaMA3 160M with Muon (Newton-Schulz) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_160m(),
        optimizer=MuonContainer.Config(
            lr=0.01,
            mu=0.95,
            weight_decay=0.1,
            scalar_lr=0.01,
            scalar_weight_decay=0.1,
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

def llama3_320m_muon() -> Trainer.Config:
    """LLaMA3 320M with Muon (Newton-Schulz) + AdamW scalar."""
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=MuonContainer.Config(
            lr=0.008,
            mu=0.95,
            nesterov=True,
            adjust_lr="rms_norm",
            weight_decay=0.1,
            scalar_lr=0.008,
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


# --- Muon ablation configs ---

def _muon_320m_ablation(**optimizer_overrides) -> Trainer.Config:
    """Helper: llama3_320m_muon with optimizer overrides."""
    base = dict(
        lr=0.008,
        mu=0.95,
        weight_decay=0.1,
        scalar_lr=0.008,
        scalar_beta1=0.95,
        scalar_beta2=0.95,
        scalar_eps=1e-8,
        scalar_weight_decay=0.1,
    )
    base.update(optimizer_overrides)
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=MuonContainer.Config(**base),
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


def llama3_320m_muon_combined() -> Trainer.Config:
    """Muon 320M: nesterov + rms_norm + no output head scaling."""
    return _muon_320m_ablation(
        nesterov=True, adjust_lr="rms_norm", output_head_lr_scaling=False,
    )


def llama3_320m_muon_combined_head_scale() -> Trainer.Config:
    """Muon 320M: nesterov + rms_norm + output head scaling."""
    return _muon_320m_ablation(
        nesterov=True, adjust_lr="rms_norm", output_head_lr_scaling=True,
    )


def llama3_320m_muon_nesterov() -> Trainer.Config:
    """Muon 320M: nesterov=True only."""
    return _muon_320m_ablation(nesterov=True)


def llama3_320m_muon_rms() -> Trainer.Config:
    """Muon 320M: adjust_lr=rms_norm only."""
    return _muon_320m_ablation(adjust_lr="rms_norm")


def llama3_320m_muon_no_head_scale() -> Trainer.Config:
    """Muon 320M: output_head_lr_scaling=False only."""
    return _muon_320m_ablation(output_head_lr_scaling=False)


# ======================================================================
# Debug config
# ======================================================================

def llama3_debug_muon() -> Trainer.Config:
    """Tiny debug model with Muon — for local validation."""
    config = debug_trainer_base()
    config.optimizer = MuonContainer.Config(
        lr=0.02,
        mu=0.95,
        weight_decay=0.0,
        scalar_lr=3e-4,
        scalar_weight_decay=0.01,
    )
    return config
