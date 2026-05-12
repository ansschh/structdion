"""
LLaMA-style transformer models at three scales for the StructDion paper.

All three models share the same Attention / SwiGLUFFN / Block / RoPE
implementation; only L, d, n_heads, hidden_dim differ. We use MHA (no GQA)
at every scale to eliminate the MHA-vs-GQA confound. Sequence length is
configurable; defaults to 1024 for compute reasons.

Approximate parameter counts (vocab_size=50257):

    LLaMA320M:  d=768,  L=18, H=12, hidden=2048   ~320M  (Tatsu config)
    LLaMA700M:  d=1536, L=22, H=24, hidden=4096   ~700M
    LLaMA1B:    d=2048, L=18, H=32, hidden=5504   ~1.0B
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks (shared by all scales)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_seq: int, theta: float = 500000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim). cos/sin: (T, head_dim/2)
    T = x.shape[-2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, rope_cos, rope_sin), apply_rope(k, rope_cos, rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, C))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden: int):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, hidden)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# Generic LLaMA model
# =============================================================================

class LLaMA(nn.Module):
    """Parametric LLaMA-style model. Shared base class for the three scales."""

    def __init__(self, vocab_size: int = 50257, dim: int = 768, n_layers: int = 18,
                 n_heads: int = 12, hidden_dim: int = 2048, max_seq: int = 2048):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.hidden_dim = hidden_dim
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([Block(dim, n_heads, hidden_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        rope_cos, rope_sin = precompute_rope(self.head_dim, max_seq)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.norm(x))


# =============================================================================
# Scale specs
# =============================================================================

# (dim, n_layers, n_heads, hidden_dim, target param count for label)
SCALE_SPECS = {
    "340m":       dict(dim=768,  n_layers=18, n_heads=12, hidden_dim=2048),  # ~320M, d_h=64
    "340m_dh128": dict(dim=768,  n_layers=18, n_heads=6,  hidden_dim=2048),  # same params, d_h=128
    "700m":       dict(dim=1536, n_layers=22, n_heads=24, hidden_dim=4096),  # ~700M, d_h=64
    "1b":         dict(dim=2048, n_layers=18, n_heads=32, hidden_dim=5504),  # ~1.0B, d_h=64
}


def build_model(scale: str, max_seq: int = 2048, vocab_size: int = 50257) -> nn.Module:
    """Construct a model from a scale label ('340m', '700m', or '1b')."""
    scale = scale.lower()
    if scale not in SCALE_SPECS:
        raise ValueError(f"Unknown scale '{scale}'. Pick one of {list(SCALE_SPECS)}")
    spec = SCALE_SPECS[scale]
    return LLaMA(vocab_size=vocab_size, max_seq=max_seq, **spec)


def scale_dims(scale: str) -> Tuple[int, int, int, int]:
    """Return (d, d_h, ffn_hidden, n_layers) for the named scale."""
    spec = SCALE_SPECS[scale.lower()]
    d = spec["dim"]
    n_heads = spec["n_heads"]
    return d, d // n_heads, spec["hidden_dim"], spec["n_layers"]


# Backward-compatibility alias: keep LLaMA320M as the 340M-spec class so
# legacy imports in train_320m.py still resolve.
class LLaMA320M(LLaMA):
    def __init__(self, vocab_size=50257, max_seq=2048):
        super().__init__(vocab_size=vocab_size, max_seq=max_seq, **SCALE_SPECS["340m"])


if __name__ == "__main__":
    # Tiny sanity check: build each scale and print parameter count.
    for scale in ("340m", "700m", "1b"):
        m = build_model(scale, max_seq=1024)
        n = sum(p.numel() for p in set(m.parameters()))
        print(f"{scale}: {n/1e6:.1f}M parameters")
