#!/usr/bin/env python
"""
Structural rank-allocation training launcher.

Supports three scales and five optimizers, with FSDP for multi-GPU.
Profiles:
  struct            r_FFN=d,    r_qko=4*d_h,    r_v=d_h
  uniform_comm      r_l = R for every matrix layer, R chosen so
                    B(uniform_R) per layer equals B(struct) per layer
  inverted_matched  cyclic-shifted role assignment, FFN-bumped to match B*
  adamw             non-spectral baseline (AdamW on every parameter)
  muon              full-rank spectral baseline (Newton-Schulz orthogonalisation
                    on matrix params, AdamW on scalars)

Scales: 340m, 700m, 1b (see models.py).

Launch single-GPU:
  python structural_law_entry.py --profile struct --scale 340m --seed 0

Launch multi-GPU via torchrun (FSDP):
  torchrun --nproc_per_node=4 structural_law_entry.py \
      --profile struct --scale 1b --seed 0

Spectral-capture and qbuf logging are always on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torch._dynamo
# dion uses @torch.compile inside orthogonalize/Newton-Schulz. On some
# torch + triton + driver combos the guard check raises
# "OpaqueUnaryFn_sqrt is not defined". Suppress and fall back to eager.
torch._dynamo.config.suppress_errors = True
try:
    torch._dynamo.config.cache_size_limit = 64
except Exception:
    pass

import dion.dion as dd  # noqa: F401
from dion import Dion

# Optional Muon and Dion2 imports (only needed for those profiles).
try:
    from dion import Muon
except Exception:
    Muon = None  # type: ignore
try:
    from dion import Dion2
except Exception:
    Dion2 = None  # type: ignore

# Make sibling imports work.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from models import build_model, scale_dims, Block, SCALE_SPECS  # noqa: E402
from train_320m import get_c4_dataloader, get_lr, evaluate  # noqa: E402


# =============================================================================
# Role classification
# =============================================================================

def classify(name: str) -> str:
    n = name.lower()
    if any(tag in n for tag in ("attn.wq", "attn.wk", "attn.wo")):
        return "qko"
    if "attn.wv" in n:
        return "v"
    if any(tag in n for tag in ("ffn.w1", "ffn.w2", "ffn.w3")):
        return "ffn"
    if "tok_emb" in n or "lm_head" in n or "embed" in n:
        return "embed"
    return "other"


# =============================================================================
# Budget arithmetic and profile resolution
# =============================================================================

def per_layer_budget(r: dict, d: int, ffn_hidden: int) -> int:
    a_attn = 2 * d
    a_ffn = ffn_hidden + d
    return 3 * r["qko"] * a_attn + r["v"] * a_attn + 3 * r["ffn"] * a_ffn


def uniform_matched_rank(d: int, d_h: int, ffn_hidden: int) -> int:
    struct = {"ffn": d, "qko": 4 * d_h, "v": d_h}
    b_struct = per_layer_budget(struct, d, ffn_hidden)
    per_rank = 4 * (2 * d) + 3 * (ffn_hidden + d)
    return round(b_struct / per_rank)


def inverted_matched_ranks(d: int, d_h: int, ffn_hidden: int, max_rank: int) -> dict:
    """Cyclic-shift the role-to-rank assignment, then bump FFN so the
    total budget equals B(struct). Concretely: start from (qko=d_h, v=d,
    FFN=4*d_h), cap v at max_rank, and add rank to FFN until B matches."""
    base_qko = d_h
    base_v = min(d, max_rank)
    base_ffn = 4 * d_h
    a_attn = 2 * d
    a_ffn = ffn_hidden + d
    b_base = 3 * base_qko * a_attn + base_v * a_attn + 3 * base_ffn * a_ffn
    b_struct = per_layer_budget({"ffn": d, "qko": 4 * d_h, "v": d_h}, d, ffn_hidden)
    gap = max(b_struct - b_base, 0)
    extra_ffn = gap // (3 * a_ffn)
    ffn = min(base_ffn + int(extra_ffn), max_rank)
    return {"qko": base_qko, "v": base_v, "ffn": ffn}


def resolve_profile(name: str, d: int, d_h: int, ffn_hidden: int, max_rank: int) -> dict:
    if name == "struct":
        ranks = {"ffn": d, "qko": 4 * d_h, "v": d_h}
    elif name == "uniform_comm":
        R = uniform_matched_rank(d, d_h, ffn_hidden)
        ranks = {"ffn": R, "qko": R, "v": R}
    elif name == "inverted_matched":
        return inverted_matched_ranks(d, d_h, ffn_hidden, max_rank)
    elif name in ("adamw", "muon"):
        # No rank assignment; matrix optimizer handles all params at full rank.
        return {"ffn": max_rank, "qko": max_rank, "v": max_rank}
    else:
        raise ValueError(f"unknown profile {name}")
    return {k: min(v, max_rank) for k, v in ranks.items()}


# =============================================================================
# Spectral capture logging
# =============================================================================

@torch.no_grad()
def _accumulate_grads(model, dl_iter, n_batches: int, device: str):
    model.zero_grad()
    seen = 0
    for _ in range(n_batches):
        try:
            x, y = next(dl_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
        seen += 1


def log_capture(model, role_params: dict, dl, n_batches: int, device: str, step: int) -> dict:
    dl_iter = iter(dl)
    _accumulate_grads(model, dl_iter, n_batches, device)
    record = {"step": step, "roles": {}}
    for role in ("qko", "v", "ffn"):
        per_param = []
        for name, p in role_params[role]:
            if p.grad is None:
                continue
            G = p.grad.detach().float()
            try:
                s = torch.linalg.svdvals(G)
            except Exception:
                continue
            s_cpu = s.cpu()
            per_param.append({
                "name": name,
                "shape": list(G.shape),
                "sigma": s_cpu.tolist(),
                "cumsum_kf": s_cpu.cumsum(0).tolist(),
                "cumsum_f": (s_cpu ** 2).cumsum(0).tolist(),
            })
        record["roles"][role] = per_param
    model.zero_grad()
    return record


# =============================================================================
# qbuf verification
# =============================================================================

def verify_qbuf(optimizers: list, requested_rank: dict) -> dict:
    """Walk optimizer states and confirm deployed rank matches requested.
    Supports both AdaDion (state['Qbuf'], state['r']) and pure Dion
    (state['Q'], inferred rank from shape). Adam / Muon are skipped."""
    out = {"mismatches": [], "ok": []}
    for opt in optimizers:
        for group in opt.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            role = group.get("role")
            req = requested_rank.get(role) if role else None
            for p in group["params"]:
                state = opt.state.get(p, {})
                buf = state.get("Qbuf", state.get("Q"))
                if buf is None or not isinstance(buf, Tensor):
                    continue
                q_rank = state.get("r")
                if q_rank is None:
                    q_rank = buf.shape[-1]
                rec = {
                    "role": role, "requested": req,
                    "deployed": int(q_rank),
                    "qbuf_shape": list(buf.shape),
                }
                if req is None:
                    continue
                if rec["deployed"] != req:
                    out["mismatches"].append(rec)
                else:
                    out["ok"].append(rec)
    return out


# =============================================================================
# Optimizer construction
# =============================================================================

def _gather_scalars(role_params: dict) -> list:
    """Collect embedding/other params into a deduplicated list."""
    scalars = []
    seen = set()
    for role in ("embed", "other"):
        for _, p in role_params[role]:
            if id(p) not in seen:
                scalars.append(p)
                seen.add(id(p))
    return scalars


def build_optimizers(
    profile: str, role_params: dict, target: dict, lr: float, max_rank: int,
    mesh=None,
) -> List[torch.optim.Optimizer]:
    """Construct the right optimizer for each profile.

    Spectral profiles (struct, uniform_comm, inverted_matched) use pure
    microsoft/dion's Dion with per-group rank_fraction. The dion2 profile
    uses dion.Dion2. Muon uses dion.Muon (full-rank Newton-Schulz).
    AdamW uses torch.optim.AdamW on every parameter.
    """
    scalars = _gather_scalars(role_params)

    if profile == "adamw":
        all_matrix = [p for role in ("qko", "v", "ffn")
                      for _, p in role_params[role]]
        dedup, seen = [], set()
        for p in all_matrix + scalars:
            if id(p) not in seen:
                dedup.append(p)
                seen.add(id(p))
        return [torch.optim.AdamW(
            dedup, lr=lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8,
        )]

    if profile == "muon":
        if Muon is None:
            raise RuntimeError("dion.Muon not importable")
        all_matrix = [p for role in ("qko", "v", "ffn")
                      for _, p in role_params[role]]
        groups = [{"params": all_matrix}]
        if scalars:
            groups.append({"params": scalars, "algorithm": "adamw",
                           "lr": 0.012, "weight_decay": 0.1})
        return [Muon(groups, lr=lr, weight_decay=0.1, flatten=True)]

    if profile == "dion2":
        if Dion2 is None:
            raise RuntimeError("dion.Dion2 not importable")
        groups = []
        for role in ("qko", "v", "ffn"):
            if not role_params[role]:
                continue
            params = [p for _, p in role_params[role]]
            rf = target[role] / max_rank
            groups.append({"params": params, "role": role,
                           "fraction": rf})
        if scalars:
            groups.append({"params": scalars, "algorithm": "adamw",
                           "lr": 0.012, "weight_decay": 0.1})
        return [Dion2(
            groups, lr=lr, weight_decay=0.1, flatten=True,
            outer_shard_mesh=mesh,
        )]

    # Spectral profiles: struct, uniform_comm, inverted_matched -> pure Dion
    groups = []
    for role in ("qko", "v", "ffn"):
        if not role_params[role]:
            continue
        params = [p for _, p in role_params[role]]
        rf = target[role] / max_rank
        groups.append({"params": params, "role": role,
                       "rank_fraction": rf})
    if scalars:
        groups.append({"params": scalars, "algorithm": "adamw",
                       "lr": 0.012, "weight_decay": 0.1,
                       "betas": (0.95, 0.95), "eps": 1e-8})

    return [Dion(
        groups, lr=lr, weight_decay=0.1,
        rank_fraction=target.get("qko", 64) / max_rank,
        outer_shard_mesh=mesh,
    )]


# =============================================================================
# Distributed setup
# =============================================================================

def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    choices=["struct", "uniform_comm", "inverted_matched",
                             "adamw", "muon", "dion2"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="autocast dtype; bf16 is ~2-4x faster on H100/H200")
    ap.add_argument("--scale", default="340m", choices=list(SCALE_SPECS.keys()))
    ap.add_argument("--steps", type=int, default=None,
                    help="if unset, use T/P schedule: 340M=830000, 700M=1300000, 1B=750000")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/struct_law")
    ap.add_argument("--log_spectra_every", type=int, default=2000)
    ap.add_argument("--spectra_batches", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=610)
    args = ap.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = f"cuda:{local_rank}" if world_size > 1 else "cuda"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Default T/P=10 step counts at batch_size=4, seq_len=1024:
    #   340M: 3.4B tokens = 830,000 steps
    #   700M: 7.0B tokens = 1,700,000 steps  (T/P=10, but ~30h single-GPU H200)
    #   1B:  10.0B tokens = 2,440,000 steps
    # To fit a 2-day wall on the cluster we cut T/P for the bigger scales:
    #   340M: T/P=10 (3.4B tokens), 830k steps
    #   700M: T/P=5  (3.5B tokens), 860k steps
    #   1B:   T/P=3  (3B tokens),   730k steps
    DEFAULT_STEPS = {"340m": 830000, "700m": 860000, "1b": 730000}
    if args.steps is None:
        args.steps = DEFAULT_STEPS[args.scale]

    if is_main:
        print(f"[dist] rank={rank} world={world_size} local_rank={local_rank} device={device}")
        print(f"[scale {args.scale}] steps={args.steps}")

    # Build model
    model = build_model(args.scale, max_seq=args.seq_len)
    d, d_h, ffn_hidden, n_layers = scale_dims(args.scale)
    max_rank = min(d, ffn_hidden)
    model = model.to(device)

    n_params = sum(p.numel() for p in set(model.parameters()))
    if is_main:
        print(f"[arch] d={d} d_h={d_h} ffn_hidden={ffn_hidden} L={n_layers} "
              f"max_rank={max_rank} params={n_params/1e6:.1f}M")

    # FSDP wrap if distributed
    mesh = None
    if world_size > 1:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (world_size,))
        wrap_policy = lambda m, recurse, **kw: isinstance(m, Block) if not recurse else True
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )

    # Resolve profile
    target = resolve_profile(args.profile, d, d_h, ffn_hidden, max_rank)
    b_actual = per_layer_budget(target, d, ffn_hidden)
    b_struct = per_layer_budget({"ffn": d, "qko": 4 * d_h, "v": d_h}, d, ffn_hidden)
    if is_main:
        print(f"[profile {args.profile}] ranks={target}  "
              f"B/B_struct={b_actual / b_struct:.3f}")

    # Classify params
    role_params = {"qko": [], "v": [], "ffn": [], "embed": [], "other": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        role = classify(name)
        if role in ("qko", "v", "ffn") and p.ndim != 2:
            role = "other"
        role_params[role].append((name, p))
    if is_main:
        for role in ("qko", "v", "ffn", "embed", "other"):
            print(f"[role {role:6s}] {len(role_params[role])} params")

    optimizers = build_optimizers(
        args.profile, role_params, target, args.lr, max_rank, mesh=mesh,
    )
    if is_main:
        print(f"[opt] {[type(o).__name__ for o in optimizers]}")

    # Data
    dl = get_c4_dataloader(batch_size=args.batch_size, seq_len=args.seq_len)
    train_iter = iter(dl)

    step_log: List[dict] = []
    val_log: List[dict] = []
    capture_log: List[dict] = []

    qbuf_pre = verify_qbuf(optimizers, target)
    if is_main:
        print(f"[qbuf_pre] ok={len(qbuf_pre['ok'])} mismatches={len(qbuf_pre['mismatches'])}")

    start = time.time()
    for step in range(args.steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(dl)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        lr_now = get_lr(step, args.steps, args.lr, warmup=args.warmup)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = lr_now

        autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        with torch.autocast(device_type="cuda", dtype=autocast_dtype,
                            enabled=(args.dtype != "fp32")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        step_log.append({"step": step, "loss": float(loss.item()), "lr": lr_now})

        if is_main and step % 100 == 0:
            tok_s = (step + 1) * args.batch_size * args.seq_len * world_size / (time.time() - start)
            print(f"step {step:7d} | loss {loss.item():.4f} | "
                  f"ppl {math.exp(min(loss.item(), 20)):8.2f} | "
                  f"lr {lr_now:.6f} | {tok_s:.0f} tok/s")

        if is_main and step > 0 and step % args.log_spectra_every == 0:
            rec = log_capture(model, role_params, dl, args.spectra_batches, device, step)
            capture_log.append(rec)
            print(f"  [capture step {step}] roles: "
                  f"qko={len(rec['roles'].get('qko', []))} "
                  f"v={len(rec['roles'].get('v', []))} "
                  f"ffn={len(rec['roles'].get('ffn', []))}")

        if step > 0 and step % args.eval_every == 0:
            val = evaluate(model, dl, device)
            if is_main:
                val["step"] = step
                val_log.append(val)
                print(f"  EVAL step {step}: val_loss={val['val_loss']:.4f}")

    final = evaluate(model, dl, device)
    if is_main:
        final["step"] = args.steps
        val_log.append(final)

    qbuf_post = verify_qbuf(optimizers, target)

    if is_main:
        tag = f"{args.scale}_{args.profile}_seed{args.seed}_lr{args.lr}"
        out = os.path.join(args.output_dir, tag)
        os.makedirs(out, exist_ok=True)
        result = {
            "tag": tag,
            "scale": args.scale, "profile": args.profile,
            "target_ranks": target,
            "budget_per_layer": b_actual, "budget_struct_per_layer": b_struct,
            "budget_ratio_vs_struct": b_actual / b_struct,
            "arch": {"d": d, "d_h": d_h, "ffn_hidden": ffn_hidden,
                     "n_layers": n_layers, "max_rank": max_rank},
            "world_size": world_size, "lr": args.lr, "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "final_val_loss": final["val_loss"],
            "final_val_ppl": final.get("val_ppl", float("nan")),
            "time_s": time.time() - start,
            "qbuf_pre_ok": len(qbuf_pre["ok"]),
            "qbuf_pre_mismatches": len(qbuf_pre["mismatches"]),
            "qbuf_post_ok": len(qbuf_post["ok"]),
            "qbuf_post_mismatches": len(qbuf_post["mismatches"]),
        }
        json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=2)
        json.dump(step_log, open(os.path.join(out, "step_log.json"), "w"))
        json.dump(val_log, open(os.path.join(out, "val_log.json"), "w"))
        json.dump(capture_log, open(os.path.join(out, "capture_log.json"), "w"))
        json.dump(qbuf_pre, open(os.path.join(out, "qbuf_pre.json"), "w"), indent=2)
        json.dump(qbuf_post, open(os.path.join(out, "qbuf_post.json"), "w"), indent=2)
        print(f"\n[done] tag={tag} final_val_loss={final['val_loss']:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
