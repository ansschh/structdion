"""
Standalone training script for LLaMA 320M polar factor experiments.

Uses Tatsu's AdaDion as base, with our PolarAdaDion extension.
Does not depend on torchtitan's training pipeline (avoids import issues).

Usage:
    # Single GPU test
    python train_320m.py --variant colnorm --steps 500

    # Multi-GPU (FSDP)
    torchrun --nproc_per_node=8 train_320m.py --variant polar --steps 2000

    # Full sweep
    torchrun --nproc_per_node=8 train_320m.py --variant all --steps 2000
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion
from torchtitan.experiments.ortho_matrix.ada_dion.polar_adadion import PolarAdaDion


# =========================================================================
# LLaMA 320M model (standalone, no pipeline parallel dependency)
# =========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim, max_seq, theta=500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
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
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, n_heads, hidden):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, hidden)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ln2(x))
        return x


class LLaMA320M(nn.Module):
    """LLaMA 320M matching Tatsu's config: dim=768, 18 layers, 12 heads."""

    def __init__(self, vocab_size=50257, dim=768, n_layers=18, n_heads=12,
                 hidden_dim=2048, max_seq=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([Block(dim, n_heads, hidden_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        rope_cos, rope_sin = precompute_rope(dim // n_heads, max_seq)
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


# =========================================================================
# Data
# =========================================================================

def get_c4_dataloader(batch_size=32, seq_len=2048, num_workers=2):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_path = "/workspace/data/c4_tokens.pt"
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, weights_only=True)
    else:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        all_tokens = []
        for i, example in enumerate(ds):
            if i >= 50000:
                break
            ids = tokenizer.encode(example["text"], add_special_tokens=False)
            all_tokens.extend(ids)
        tokens = torch.tensor(all_tokens, dtype=torch.int32)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(tokens, cache_path)

    n_seq = (len(tokens) - 1) // seq_len
    tokens = tokens[:n_seq * seq_len + 1]

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, data, sl):
            self.data = data
            self.sl = sl
        def __len__(self):
            return (len(self.data) - 1) // self.sl
        def __getitem__(self, i):
            s = i * self.sl
            return self.data[s:s+self.sl].long(), self.data[s+1:s+self.sl+1].long()

    ds = TokenDataset(tokens, seq_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True, drop_last=True)


# =========================================================================
# Optimizer factory
# =========================================================================

def create_optimizer(variant: str, model: nn.Module, lr: float = 0.012,
                     mesh=None) -> Tuple:
    """Create optimizer variant. Returns (optimizer, description)."""
    matrix_params = []
    scalar_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    param_groups = [{"params": matrix_params}]
    if scalar_params:
        param_groups.append({
            "params": scalar_params,
            "algorithm": "adamw",
            "lr": 0.012,
            "weight_decay": 0.1,
            "betas": (0.95, 0.95),
            "eps": 1e-8,
        })

    shared = dict(
        lr=lr, weight_decay=0.1, rank_fraction_max=1.0,
        adaptive_rank=True, init_rank_fraction=0.5,
        erank_ema_beta=0.5, rank_scale=2.0,
        rank_min=16, rank_quantize=8,
    )
    if mesh is not None:
        shared["outer_shard_mesh"] = mesh

    if variant == "colnorm":
        opt = AdaDion(param_groups, **shared)
        desc = "AdaDion (ColNorm, tau=0)"
    elif variant == "polar":
        opt = PolarAdaDion(param_groups, tau=1.0, **shared)
        desc = "PolarAdaDion (tau=1.0)"
    elif variant == "interp_025":
        opt = PolarAdaDion(param_groups, tau=0.25, **shared)
        desc = "PolarAdaDion (tau=0.25)"
    elif variant == "interp_050":
        opt = PolarAdaDion(param_groups, tau=0.5, **shared)
        desc = "PolarAdaDion (tau=0.50)"
    elif variant == "interp_075":
        opt = PolarAdaDion(param_groups, tau=0.75, **shared)
        desc = "PolarAdaDion (tau=0.75)"
    elif variant == "qr":
        # QR Q-factor for comparison (original Orth-Dion)
        # Use PolarAdaDion with a flag to force QR
        opt = PolarAdaDion(param_groups, tau=-1.0, **shared)  # tau<0 signals QR mode
        desc = "Orth-Dion (QR Q-factor)"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return opt, desc


# =========================================================================
# Training
# =========================================================================

def get_lr(step, max_steps, peak_lr, warmup=610):
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=20):
    model.eval()
    total_loss, total_tokens = 0, 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    avg = total_loss / max(total_tokens, 1)
    return {"val_loss": avg, "val_ppl": math.exp(min(avg, 20))}


def train_variant(variant, max_steps=2000, batch_size=8, seq_len=2048,
                  lr=0.012, device="cuda", output_dir="results/320m"):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    torch.manual_seed(42)
    model = LLaMA320M(max_seq=seq_len).to(device)
    n_params = sum(p.numel() for p in set(model.parameters()))

    if rank == 0:
        print(f"LLaMA 320M: {n_params/1e6:.1f}M params")

    # FSDP wrap if multi-GPU
    mesh = None
    if world_size > 1:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (world_size,))
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                      mixed_precision=MixedPrecision(
                          param_dtype=torch.bfloat16,
                          reduce_dtype=torch.bfloat16,
                      ))

    opt, desc = create_optimizer(variant, model, lr=lr, mesh=mesh)

    if rank == 0:
        print(f"Variant: {desc}")
        print(f"Steps: {max_steps}, batch={batch_size}, seq={seq_len}")

    dataloader = get_c4_dataloader(batch_size=batch_size, seq_len=seq_len)
    train_iter = iter(dataloader)
    start_time = time.time()

    step_log = []
    val_log = []

    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        lr_now = get_lr(step, max_steps, lr, warmup=610)
        for group in opt.param_groups:
            group["lr"] = lr_now

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if rank == 0:
            step_log.append({"step": step, "loss": loss.item(), "lr": lr_now})
            if step % 100 == 0:
                tok_s = (step + 1) * batch_size * seq_len / (time.time() - start_time)
                print(f"step {step:5d} | loss {loss.item():.4f} | "
                      f"ppl {math.exp(min(loss.item(), 20)):8.2f} | "
                      f"lr {lr_now:.6f} | {tok_s:.0f} tok/s")

            if step > 0 and step % 500 == 0:
                val = evaluate(model, dataloader, device)
                val["step"] = step
                val_log.append(val)
                print(f"  EVAL: val_loss={val['val_loss']:.4f} ppl={val['val_ppl']:.2f}")

    if rank == 0:
        final = evaluate(model, dataloader, device)
        final["step"] = max_steps
        val_log.append(final)

        out_dir = os.path.join(output_dir, variant)
        os.makedirs(out_dir, exist_ok=True)
        result = {
            "variant": variant, "desc": desc, "lr": lr,
            "final_val_loss": final["val_loss"], "final_val_ppl": final["val_ppl"],
            "total_time_s": time.time() - start_time,
            "n_params": n_params, "max_steps": max_steps,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(out_dir, "step_log.json"), "w") as f:
            json.dump(step_log, f)
        with open(os.path.join(out_dir, "val_log.json"), "w") as f:
            json.dump(val_log, f)
        print(f"\nDone: {desc} | val_loss={final['val_loss']:.4f}")

    if world_size > 1:
        dist.destroy_process_group()
    return result if rank == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="colnorm",
                        choices=["colnorm", "polar", "interp_025", "interp_050",
                                 "interp_075", "qr", "all"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--output_dir", default="results/320m")
    args = parser.parse_args()

    if args.variant == "all":
        for v in ["colnorm", "polar", "interp_025", "interp_050", "interp_075", "qr"]:
            train_variant(v, max_steps=args.steps, batch_size=args.batch_size,
                         seq_len=args.seq_len, lr=args.lr, output_dir=args.output_dir)
    else:
        train_variant(args.variant, max_steps=args.steps, batch_size=args.batch_size,
                     seq_len=args.seq_len, lr=args.lr, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
