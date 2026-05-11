#!/usr/bin/env python
"""
Structural rank allocation experiment runner for LLaMA320M.

Profiles:
  struct        r_FFN = d,    r_qko = 4 * d_h,    r_v = d_h
  uniform_comm  r_l = R for every matrix layer, with R chosen so that
                B(uniform_R) = B(struct) (matched per-step communication)
  inverted      r_FFN = d_h,  r_qko = d/4,        r_v = d
                (natural budget, NOT matched to struct; reported as ratio)

Per-role rank is achieved by routing matrix parameters into three
separate AdaDion optimizers (qko, v, ffn) each with adaptive_rank=False
and a fixed init_rank_fraction. Scalars and embeddings ride on AdamW.

We log Ky-Fan and Frobenius capture spectra at fixed intervals via
batch SVD of the per-parameter gradient signal.

Run e.g.
  python structural_law_entry.py --profile struct --seed 0 --steps 5000
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

import dion.dion as dd  # noqa: F401  (kept in case other patches are needed later)

from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion
from torchtitan.experiments.ortho_matrix.ada_dion.train_320m import (
    LLaMA320M, get_c4_dataloader, get_lr, evaluate,
)


# =============================================================================
# Role classification
# =============================================================================

def classify(name: str) -> str:
    """Map a parameter name to one of qko, v, ffn, embed, other."""
    n = name.lower()
    if any(tag in n for tag in ('attn.wq', 'attn.wk', 'attn.wo',
                                'q_proj', 'k_proj', 'o_proj')):
        return 'qko'
    if 'attn.wv' in n or 'v_proj' in n:
        return 'v'
    if any(tag in n for tag in ('ffn.w1', 'ffn.w2', 'ffn.w3',
                                'feed_forward', 'mlp', 'gate_proj',
                                'up_proj', 'down_proj')):
        return 'ffn'
    if 'tok_emb' in n or 'lm_head' in n or 'embed' in n:
        return 'embed'
    return 'other'


# =============================================================================
# Budget arithmetic
# =============================================================================

def per_layer_budget(r: dict, d: int, ffn_hidden: int) -> int:
    """Communication cost per transformer layer for rank dict
    r = {'ffn', 'qko', 'v'}, in units of rank * (m + n)."""
    a_attn = d + d            # attn matrices are d x d
    a_ffn = ffn_hidden + d    # ffn matrices are hidden x d (and d x hidden)
    return 3 * r['qko'] * a_attn + r['v'] * a_attn + 3 * r['ffn'] * a_ffn


def uniform_matched_rank(d: int, d_h: int, ffn_hidden: int) -> int:
    """Uniform rank R such that B(uniform_R) per layer equals
    B(struct) per layer."""
    struct = {'ffn': d, 'qko': 4 * d_h, 'v': d_h}
    b_struct = per_layer_budget(struct, d, ffn_hidden)
    # B(uniform_R) per layer = R * (4 * a_attn + 3 * a_ffn)
    per_rank = 4 * (d + d) + 3 * (ffn_hidden + d)
    return round(b_struct / per_rank)


def resolve_profile(name: str, d: int, d_h: int, ffn_hidden: int, max_rank: int) -> dict:
    if name == 'struct':
        ranks = {'ffn': d, 'qko': 4 * d_h, 'v': d_h}
    elif name == 'uniform_comm':
        R = uniform_matched_rank(d, d_h, ffn_hidden)
        ranks = {'ffn': R, 'qko': R, 'v': R}
    elif name == 'inverted':
        ranks = {'ffn': d_h, 'qko': max(d // 4, 1), 'v': d}
    else:
        raise ValueError(f'unknown profile {name}')
    # Cap each role's rank at max_rank (= min(m, n) for the role's largest matrix)
    return {k: min(v, max_rank) for k, v in ranks.items()}


# =============================================================================
# Capture spectra logging
# =============================================================================

@torch.no_grad()
def _accumulate_grads(model, dl_iter, n_batches: int, device: str):
    """Run n_batches forward-backward passes, accumulating gradients."""
    model.zero_grad()
    total_loss = 0.0
    seen = 0
    for _ in range(n_batches):
        try:
            x, y = next(dl_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        # Re-enable grad inside this no_grad scope for backward
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
        total_loss += float(loss.item())
        seen += 1
    return (total_loss / seen) if seen else float('nan')


def log_capture(model, role_params: dict, dl, n_batches: int, device: str, step: int) -> dict:
    """Compute per-parameter SVD of the gradient on a fresh n_batches mini-batch
    and record both Ky-Fan partial sums and Frobenius partial sums per role."""
    dl_iter = iter(dl)
    _accumulate_grads(model, dl_iter, n_batches, device)
    record = {'step': step, 'roles': {}}
    for role in ('qko', 'v', 'ffn'):
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
                'name': name,
                'shape': list(G.shape),
                'sigma': s_cpu.tolist(),
                'cumsum_kf': s_cpu.cumsum(0).tolist(),
                'cumsum_f': (s_cpu ** 2).cumsum(0).tolist(),
            })
        record['roles'][role] = per_param
    model.zero_grad()
    return record


# =============================================================================
# qbuf verification (effective rank vs requested)
# =============================================================================

def verify_qbuf(optimizers: list, requested_rank: dict, role_by_optid: dict) -> dict:
    """Read each optimizer's internal Q buffers and confirm that the
    deployed rank matches the requested rank. Returns mismatches."""
    out = {'mismatches': [], 'ok': []}
    for opt in optimizers:
        if not isinstance(opt, AdaDion):
            continue
        role = role_by_optid.get(id(opt))
        req = requested_rank.get(role)
        for group in opt.param_groups:
            for p in group['params']:
                state = opt.state.get(p, {})
                Q = state.get('Q', None)
                if Q is None:
                    continue
                q_rank = state.get('r', None)
                if q_rank is None and isinstance(Q, Tensor):
                    q_rank = Q.shape[-1]
                rec = {
                    'role': role,
                    'requested': req,
                    'deployed': int(q_rank) if q_rank is not None else None,
                    'qbuf_shape': list(Q.shape) if isinstance(Q, Tensor) else None,
                }
                if rec['deployed'] is None or req is None:
                    continue
                if rec['deployed'] != req:
                    out['mismatches'].append(rec)
                else:
                    out['ok'].append(rec)
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', required=True,
                    choices=['struct', 'uniform_comm', 'inverted'])
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--seq_len', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=0.012)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='results/struct_law')
    ap.add_argument('--log_spectra_every', type=int, default=500)
    ap.add_argument('--spectra_batches', type=int, default=4)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=610)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda'

    model = LLaMA320M(max_seq=args.seq_len).to(device)
    d = 768
    d_h = 64
    ffn_hidden = 2048
    max_rank = min(d, ffn_hidden)  # 768

    target = resolve_profile(args.profile, d, d_h, ffn_hidden, max_rank)
    b_actual = per_layer_budget(target, d, ffn_hidden)
    b_struct = per_layer_budget({'ffn': d, 'qko': 4 * d_h, 'v': d_h}, d, ffn_hidden)
    print(f'[arch] d={d}, d_h={d_h}, ffn_hidden={ffn_hidden}, max_rank={max_rank}')
    print(f'[profile {args.profile}] target ranks: {target}')
    print(f'[profile {args.profile}] per-layer B = {b_actual} '
          f'(B/B_struct = {b_actual / b_struct:.3f})')

    # Classify parameters
    role_params = {'qko': [], 'v': [], 'ffn': [], 'embed': [], 'other': []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        role = classify(name)
        # Only 2D params go to matrix optimizers
        if role in ('qko', 'v', 'ffn') and p.ndim != 2:
            role = 'other'
        role_params[role].append((name, p))

    for role in ('qko', 'v', 'ffn', 'embed', 'other'):
        print(f'[role {role:6s}] {len(role_params[role])} params')

    # Build optimizers: one AdaDion per matrix role, AdamW for the rest
    optimizers = []
    role_by_optid = {}
    for role in ('qko', 'v', 'ffn'):
        if not role_params[role]:
            continue
        params = [p for _, p in role_params[role]]
        target_rank = target[role]
        rank_fraction = target_rank / max_rank
        opt = AdaDion(
            [{'params': params}],
            lr=args.lr, weight_decay=0.1,
            rank_fraction_max=rank_fraction,
            adaptive_rank=False,
            init_rank_fraction=rank_fraction,
            erank_ema_beta=0.5, rank_scale=2.0, rank_min=16, rank_quantize=8,
        )
        optimizers.append(opt)
        role_by_optid[id(opt)] = role
        print(f'[opt {role}] AdaDion rank_fraction={rank_fraction:.4f} '
              f'(target_rank={target_rank}, max_rank={max_rank})')

    # AdamW group: embed/other and all non-2D params
    scalar_params = []
    seen = set()
    for role in ('embed', 'other'):
        for _, p in role_params[role]:
            if id(p) not in seen:
                scalar_params.append(p)
                seen.add(id(p))
    if scalar_params:
        adam = torch.optim.AdamW(
            scalar_params, lr=0.012, weight_decay=0.1,
            betas=(0.95, 0.95), eps=1e-8,
        )
        optimizers.append(adam)
        print(f'[opt adam] AdamW on {len(scalar_params)} scalar/embed params')

    # Data
    dl = get_c4_dataloader(batch_size=args.batch_size, seq_len=args.seq_len)
    train_iter = iter(dl)

    # Logging buffers
    step_log = []
    val_log = []
    capture_log = []

    # Pre-train spectral capture and qbuf verification
    qbuf_pre = verify_qbuf(optimizers, target, role_by_optid)
    print(f'[qbuf_pre] ok={len(qbuf_pre["ok"])} mismatches={len(qbuf_pre["mismatches"])}')

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
                g['lr'] = lr_now

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        step_log.append({'step': step, 'loss': float(loss.item()), 'lr': lr_now})

        if step % 100 == 0:
            tok_s = (step + 1) * args.batch_size * args.seq_len / (time.time() - start)
            print(f'step {step:5d} | loss {loss.item():.4f} | '
                  f'ppl {math.exp(min(loss.item(), 20)):8.2f} | '
                  f'lr {lr_now:.6f} | {tok_s:.0f} tok/s')

        # Spectral capture
        if step > 0 and step % args.log_spectra_every == 0:
            rec = log_capture(model, role_params, dl, args.spectra_batches, device, step)
            capture_log.append(rec)
            n_qko = len(rec['roles'].get('qko', []))
            n_v = len(rec['roles'].get('v', []))
            n_ffn = len(rec['roles'].get('ffn', []))
            print(f'  [capture step {step}] svd counts: '
                  f'qko={n_qko} v={n_v} ffn={n_ffn}')

        # Eval
        if step > 0 and step % args.eval_every == 0:
            val = evaluate(model, dl, device)
            val['step'] = step
            val_log.append(val)
            print(f'  EVAL step {step}: val_loss={val["val_loss"]:.4f} '
                  f'ppl={val.get("val_ppl", float("nan")):.2f}')

    # Final
    final = evaluate(model, dl, device)
    final['step'] = args.steps
    val_log.append(final)
    qbuf_post = verify_qbuf(optimizers, target, role_by_optid)

    tag = f'{args.profile}_seed{args.seed}_lr{args.lr}'
    out = os.path.join(args.output_dir, tag)
    os.makedirs(out, exist_ok=True)

    result = {
        'tag': tag,
        'profile': args.profile,
        'target_ranks': target,
        'budget_per_layer': b_actual,
        'budget_struct_per_layer': b_struct,
        'budget_ratio_vs_struct': b_actual / b_struct,
        'arch': {'d': d, 'd_h': d_h, 'ffn_hidden': ffn_hidden,
                 'max_rank': max_rank, 'n_layers': 18},
        'lr': args.lr,
        'seed': args.seed,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'final_val_loss': final['val_loss'],
        'final_val_ppl': final.get('val_ppl', float('nan')),
        'time_s': time.time() - start,
        'qbuf_pre_ok': len(qbuf_pre['ok']),
        'qbuf_pre_mismatches': len(qbuf_pre['mismatches']),
        'qbuf_post_ok': len(qbuf_post['ok']),
        'qbuf_post_mismatches': len(qbuf_post['mismatches']),
    }
    json.dump(result, open(os.path.join(out, 'result.json'), 'w'), indent=2)
    json.dump(step_log, open(os.path.join(out, 'step_log.json'), 'w'))
    json.dump(val_log, open(os.path.join(out, 'val_log.json'), 'w'))
    json.dump(capture_log, open(os.path.join(out, 'capture_log.json'), 'w'))
    json.dump(qbuf_pre, open(os.path.join(out, 'qbuf_pre.json'), 'w'), indent=2)
    json.dump(qbuf_post, open(os.path.join(out, 'qbuf_post.json'), 'w'), indent=2)
    print(f'\n[done] tag={tag} final_val_loss={final["val_loss"]:.4f}')


if __name__ == '__main__':
    main()
