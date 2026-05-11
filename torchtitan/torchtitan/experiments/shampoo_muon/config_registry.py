"""
Config registry for shampoo_muon experiments.

Paper-faithful Muon (SVD) via Distributed Shampoo for replicating Table 1
of 2602.09314v1 ("Shampoo : Muon :: Adam : Signum").

Usage:
    torchrun --nproc_per_node=N -m torchtitan.train \
        --module shampoo_muon \
        --config llama3_320m_muon_svd \
        --training.steps 6104
"""
from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
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
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from .shampoo_container import ShampooMuonContainer


# ======================================================================
# LLaMA3 320M model config
# ======================================================================
# ~324M params: dim=768, n_layers=18, n_heads=12, MHA (n_kv_heads=12)

_LLAMA3_320M = Llama3Model.Config(
    dim=768,
    n_layers=18,
    vocab_size=128256,
    layer=Llama3TransformerBlock.Config(
        feed_forward=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(768, multiple_of=256),
        ),
        attention=GQAttention.Config(
            n_heads=12,
            attn_backend="sdpa",
            rope_backend="complex",
        ),
    ),
    rope=RoPE.Config(
        dim=768 // 12,
        max_seq_len=8192,
        theta=500000,
        backend="complex",
        scaling="none",
    ),
)


def _model_registry_320m() -> ModelSpec:
    return ModelSpec(
        name="llama3",
        flavor="320M",
        model=_LLAMA3_320M,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def _base_training_config(steps: int = 6104) -> TrainingConfig:
    return TrainingConfig(
        local_batch_size=32,  # B=256 global / 8 GPUs = 32 per GPU
        seq_len=2048,
        steps=steps,
        dtype="bfloat16",
        max_norm=1.0,
    )


def _base_metrics_config() -> MetricsProcessor.Config:
    return MetricsProcessor.Config(
        log_freq=10,
        enable_tensorboard=True,
        enable_wandb=True,
    )


# ======================================================================
# Muon (SVD) — paper Table 11, 320M, 1x, B=256
# ======================================================================

def llama3_320m_muon_svd() -> Trainer.Config:
    """LLaMA3 320M with Muon (SVD orthogonalization + Adam grafting)."""
    return Trainer.Config(
        model_spec=_model_registry_320m(),
        optimizer=ShampooMuonContainer.Config(
            name="ShampooMuon",
            lr=0.016,
            beta1=0.95,
            weight_decay=0.1,
            orthogonalization="svd",
            graft_beta2=0.95,
            graft_eps=1e-8,
            scalar_lr=0.008,
            scalar_beta1=0.95,
            scalar_beta2=0.95,
            scalar_eps=1e-8,
            scalar_weight_decay=0.1,
            norm_weight_decay=0.0,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=610,   # 10% warmup
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=_base_training_config(steps=6104),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=_base_metrics_config(),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=True,
        ),
        compile=CompileConfig(enable=True),
        validator=Validator.Config(
            freq=100,
            steps=20,
        ),
    )


# ======================================================================
# Muon (Newton-Schulz) — same hyperparams, NS orthogonalization
# ======================================================================

def llama3_320m_muon_ns() -> Trainer.Config:
    """LLaMA3 320M with Muon (Newton-Schulz orthogonalization + Adam grafting)."""
    return Trainer.Config(
        model_spec=_model_registry_320m(),
        optimizer=ShampooMuonContainer.Config(
            name="ShampooMuon",
            lr=0.016,
            beta1=0.95,
            weight_decay=0.1,
            orthogonalization="newton_schulz",
            ns_steps=5,
            graft_beta2=0.95,
            graft_eps=1e-8,
            scalar_lr=0.008,
            scalar_beta1=0.95,
            scalar_beta2=0.95,
            scalar_eps=1e-8,
            scalar_weight_decay=0.1,
            norm_weight_decay=0.0,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=610,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=_base_training_config(steps=6104),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=_base_metrics_config(),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=True,
        ),
        compile=CompileConfig(enable=True),
        validator=Validator.Config(
            freq=100,
            steps=20,
        ),
    )


# ======================================================================
# AdamW baseline (same hyperparams for fair comparison)
# ======================================================================

def llama3_320m_adamw() -> Trainer.Config:
    """LLaMA3 320M with AdamW optimizer (baseline)."""
    return Trainer.Config(
        model_spec=_model_registry_320m(),
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
        training=_base_training_config(steps=6104),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=_base_metrics_config(),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=True,
        ),
        validator=Validator.Config(
            freq=100,
            steps=20,
        ),
    )
