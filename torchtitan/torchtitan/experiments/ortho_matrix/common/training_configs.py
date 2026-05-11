"""
Shared training, scheduling, and metrics configs for ortho_matrix experiments.
"""
from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    TrainingConfig,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    FeedForward,
    GQAttention,
    RoPE,
)
from torchtitan.models.llama3 import (
    Llama3Model,
    Llama3TransformerBlock,
    parallelize_llama,
)
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .model_configs import model_registry_160m, model_registry_320m


# ======================================================================
# Common training and scheduling configs
# ======================================================================

def base_training_config(steps: int = 10000) -> TrainingConfig:
    return TrainingConfig(
        local_batch_size=32,
        seq_len=2048,
        steps=steps,
        dtype="bfloat16",
        max_norm=1.0,
    )


def base_lr_scheduler_config() -> LRSchedulersContainer.Config:
    return LRSchedulersContainer.Config(
        warmup_steps=200,
        decay_type="cosine",
        min_lr_factor=0.0,
    )


def base_metrics_config() -> MetricsProcessor.Config:
    return MetricsProcessor.Config(
        log_freq=10,
        enable_tensorboard=True,
        enable_wandb=True,
    )


# ======================================================================
# Debug model spec
# ======================================================================

def debug_model_spec() -> ModelSpec:
    """Create a tiny debug ModelSpec for local validation."""
    return ModelSpec(
        name="llama3",
        flavor="debugmodel",
        model=Llama3Model.Config(
            dim=256,
            n_layers=6,
            vocab_size=2048,
            layer=Llama3TransformerBlock.Config(
                feed_forward=FeedForward.Config(
                    hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
                ),
                attention=GQAttention.Config(
                    n_heads=16,
                    attn_backend="sdpa",
                    rope_backend="complex",
                ),
            ),
            rope=RoPE.Config(
                dim=256 // 16,
                max_seq_len=131072,
                theta=500000,
                backend="complex",
                scaling="llama",
            ),
        ),
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def debug_trainer_base() -> Trainer.Config:
    """Base debug Trainer.Config without optimizer set."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=debug_model_spec(),
        optimizer=OptimizersContainer.Config(),  # placeholder, override in caller
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
    )


# ======================================================================
# AdamW baseline configs
# ======================================================================

def llama3_160m_adamw() -> Trainer.Config:
    """LLaMA3 160M with AdamW optimizer (baseline)."""
    return Trainer.Config(
        model_spec=model_registry_160m(),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
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


def llama3_320m_adamw() -> Trainer.Config:
    """LLaMA3 320M with AdamW optimizer (baseline)."""
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=0.008,
            beta1=0.95,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
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
