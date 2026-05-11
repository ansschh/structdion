"""
Shared LLaMA3 model configurations for ortho_matrix experiments.
"""
from __future__ import annotations

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
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


# ======================================================================
# LLaMA3 160M model config
# ======================================================================
# ~160M params: dim=768, n_layers=12, n_heads=12, n_kv_heads=4
# hidden_dim ~= 2048 (via compute_ffn_hidden_dim with multiple_of=256)

_LLAMA3_160M = Llama3Model.Config(
    dim=768,
    n_layers=12,
    vocab_size=128256,
    layer=Llama3TransformerBlock.Config(
        feed_forward=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(768, multiple_of=256),
        ),
        attention=GQAttention.Config(
            n_heads=12,
            n_kv_heads=4,
            attn_backend="sdpa",
            rope_backend="complex",
        ),
    ),
    rope=RoPE.Config(
        dim=768 // 12,  # head_dim = dim // n_heads
        max_seq_len=8192,
        theta=500000,
        backend="complex",
        scaling="none",
    ),
)


def model_registry_160m() -> ModelSpec:
    """Create a ModelSpec for LLaMA3 160M."""
    return ModelSpec(
        name="llama3",
        flavor="160M",
        model=_LLAMA3_160M,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


# ======================================================================
# LLaMA3 320M model config
# ======================================================================
# ~324M params: dim=768, n_layers=18, n_heads=12, MHA (n_kv_heads=12)
# hidden_dim ~= 2048 (via compute_ffn_hidden_dim with multiple_of=256)

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
        dim=768 // 12,  # head_dim = dim // n_heads
        max_seq_len=8192,
        theta=500000,
        backend="complex",
        scaling="none",
    ),
)


def model_registry_320m() -> ModelSpec:
    """Create a ModelSpec for LLaMA3 320M."""
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
