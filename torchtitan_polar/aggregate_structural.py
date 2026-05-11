#!/usr/bin/env python
"""
Aggregate structural-law results into the paper tables and figures.

Inputs : results_root with subdirs <profile>_seed<i>_lr<x>/result.json,
         val_log.json, capture_log.json.
Outputs (written to results_root):
  table_main.txt         Headline matched-comm validation-loss table.
  table_budgets.json     Per-profile B(r) and ratio vs struct.
  fig_capture.pdf        Per-role marginal-capture curves (Theorem 1 diagnostic).
  fig_trajectory.pdf     Val loss vs step, one curve per profile, CI shaded.
  summary.json           Per-profile mean, std, 95 percent CI, paired delta.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_runs(root: Path):
    runs = []
    for sub in sorted(root.iterdir()):
        rp = sub / 'result.json'
        vp = sub / 'val_log.json'
        cp = sub / 'capture_log.json'
        if not rp.exists():
            continue
        try:
            result = json.load(open(rp))
        except Exception:
            continue
        val_log = json.load(open(vp)) if vp.exists() else []
        capture_log = json.load(open(cp)) if cp.exists() else []
        runs.append({'dir': sub.name, 'result': result,
                     'val_log': val_log, 'capture_log': capture_log})
    return runs


def _paired_bootstrap_ci(deltas: list, n_resamples: int = 10000, alpha: float = 0.05):
    a = np.asarray(deltas, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(0)
    means = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.integers(0, a.size, a.size)
        means[k] = a[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(a.mean()), lo, hi


def _aggregate(runs):
    by_profile = defaultdict(list)
    for r in runs:
        p = r['result'].get('profile')
        if p is None:
            continue
        by_profile[p].append(r)
    return by_profile


def table_main(by_profile, out_path: Path):
    """Headline matched-communication validation-loss table."""
    profiles = ['uniform_comm', 'struct', 'inverted']
    rows = []
    for profile in profiles:
        items = by_profile.get(profile, [])
        if not items:
            continue
        losses = [r['result']['final_val_loss'] for r in items]
        seeds = sorted(int(r['result']['seed']) for r in items)
        b_ratio = items[0]['result']['budget_ratio_vs_struct']
        rows.append({
            'profile': profile,
            'n_seeds': len(losses),
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            'seeds': seeds,
            'losses': losses,
            'budget_ratio': b_ratio,
        })

    # Paired deltas vs uniform_comm
    delta_struct_vs_uniform = []
    delta_inverted_vs_uniform = []
    uniform_by_seed = {r['result']['seed']: r['result']['final_val_loss']
                       for r in by_profile.get('uniform_comm', [])}
    struct_by_seed = {r['result']['seed']: r['result']['final_val_loss']
                      for r in by_profile.get('struct', [])}
    inverted_by_seed = {r['result']['seed']: r['result']['final_val_loss']
                        for r in by_profile.get('inverted', [])}
    for seed, u_loss in uniform_by_seed.items():
        if seed in struct_by_seed:
            delta_struct_vs_uniform.append(struct_by_seed[seed] - u_loss)
        if seed in inverted_by_seed:
            delta_inverted_vs_uniform.append(inverted_by_seed[seed] - u_loss)

    mean_s, lo_s, hi_s = _paired_bootstrap_ci(delta_struct_vs_uniform)
    mean_i, lo_i, hi_i = _paired_bootstrap_ci(delta_inverted_vs_uniform)

    text = []
    text.append('Structural law: matched-communication validation loss')
    text.append('=' * 60)
    text.append(f'{"profile":15s} {"n":>3s} {"mean":>8s} {"std":>8s} '
                f'{"B/B*":>6s} {"losses":<30s}')
    for row in rows:
        text.append(f'{row["profile"]:15s} {row["n_seeds"]:>3d} '
                    f'{row["mean"]:8.4f} {row["std"]:8.4f} '
                    f'{row["budget_ratio"]:6.3f} {row["losses"]}')
    text.append('')
    text.append('Paired deltas (vs uniform_comm at matched seed):')
    text.append(f'  struct   - uniform :  mean={mean_s:+.4f} '
                f'  95%CI=[{lo_s:+.4f}, {hi_s:+.4f}]  n={len(delta_struct_vs_uniform)}')
    text.append(f'  inverted - uniform :  mean={mean_i:+.4f} '
                f'  95%CI=[{lo_i:+.4f}, {hi_i:+.4f}]  n={len(delta_inverted_vs_uniform)}')
    text.append('')
    text.append('Interpretation:')
    text.append('  struct - uniform should be NEGATIVE (struct wins). ')
    text.append('  inverted - uniform should be POSITIVE if inverted is a true falsifier.')
    text.append('  If inverted has B/B* != 1.0, the comparison is NOT matched-comm.')

    out_path.write_text('\n'.join(text), encoding='utf-8')
    print(f'wrote {out_path}')
    return {
        'rows': rows,
        'delta_struct_vs_uniform': {'mean': mean_s, 'ci_lo': lo_s, 'ci_hi': hi_s,
                                    'values': delta_struct_vs_uniform},
        'delta_inverted_vs_uniform': {'mean': mean_i, 'ci_lo': lo_i, 'ci_hi': hi_i,
                                      'values': delta_inverted_vs_uniform},
    }


def fig_capture(by_profile, out_dir: Path):
    """Figure: Ky-Fan and Frobenius per-role marginal capture curves
    averaged over seeds of the STRUCT profile at the latest captured step.
    These are the Theorem 1 diagnostic curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'matplotlib unavailable: {e}; skipping figure')
        return None

    runs = by_profile.get('struct', [])
    if not runs:
        print('no struct runs; skipping capture figure')
        return None

    # For each run, take the last capture record; for each role, average per-param
    # cumulative-capture curves across params and seeds. Plot Ky-Fan and Frobenius
    # side by side, layer-family overlaid.
    roles = ('qko', 'v', 'ffn')
    role_colors = {'qko': '#1f77b4', 'v': '#d62728', 'ffn': '#2ca02c'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax_idx, geom in enumerate(('cumsum_kf', 'cumsum_f')):
        ax = axes[ax_idx]
        for role in roles:
            curves = []
            for run in runs:
                cap = run['capture_log']
                if not cap:
                    continue
                last = cap[-1]
                per = last.get('roles', {}).get(role, [])
                for entry in per:
                    c = np.asarray(entry[geom], dtype=float)
                    if c.size == 0:
                        continue
                    # Normalize x-axis to relative rank r / r_max
                    x = np.arange(1, c.size + 1) / c.size
                    # Normalize y to fractional capture
                    y = c / max(c.max(), 1e-12)
                    curves.append((x, y))
            if not curves:
                continue
            # Resample all curves to a common grid
            grid = np.linspace(0, 1, 64)
            ys = []
            for x, y in curves:
                yi = np.interp(grid, x, y)
                ys.append(yi)
            ys = np.stack(ys)
            mean = ys.mean(axis=0)
            lo = np.percentile(ys, 2.5, axis=0)
            hi = np.percentile(ys, 97.5, axis=0)
            ax.plot(grid, mean, color=role_colors[role], label=role, linewidth=2)
            ax.fill_between(grid, lo, hi, color=role_colors[role], alpha=0.18)
        ax.set_xlabel('relative rank r / r_max')
        ax.set_ylabel('cumulative capture / max')
        ax.set_title('Ky-Fan' if geom == 'cumsum_kf' else 'Frobenius')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')

    fig.tight_layout()
    p_pdf = out_dir / 'fig_capture.pdf'
    p_png = out_dir / 'fig_capture.png'
    fig.savefig(p_pdf, bbox_inches='tight')
    fig.savefig(p_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {p_pdf} and {p_png}')
    return str(p_pdf)


def fig_trajectory(by_profile, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'matplotlib unavailable: {e}; skipping trajectory figure')
        return None

    profile_colors = {'uniform_comm': '#888888', 'struct': '#1f77b4',
                      'inverted': '#d62728'}
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.0))
    for profile, runs in by_profile.items():
        steps_by_seed = []
        losses_by_seed = []
        for run in runs:
            vl = run.get('val_log', [])
            if not vl:
                continue
            xs = [v['step'] for v in vl if 'val_loss' in v]
            ys = [v['val_loss'] for v in vl if 'val_loss' in v]
            if not xs:
                continue
            steps_by_seed.append(np.asarray(xs))
            losses_by_seed.append(np.asarray(ys))
        if not steps_by_seed:
            continue
        # Pad to common grid (longest steps array)
        common_steps = max(steps_by_seed, key=len)
        ys_stack = []
        for xs, ys in zip(steps_by_seed, losses_by_seed):
            if len(xs) == len(common_steps) and np.allclose(xs, common_steps):
                ys_stack.append(ys)
            else:
                yi = np.interp(common_steps, xs, ys)
                ys_stack.append(yi)
        ys_stack = np.stack(ys_stack)
        mean = ys_stack.mean(axis=0)
        ax.plot(common_steps, mean, color=profile_colors.get(profile, '#444'),
                label=profile, linewidth=2)
        if ys_stack.shape[0] > 1:
            lo = np.percentile(ys_stack, 2.5, axis=0)
            hi = np.percentile(ys_stack, 97.5, axis=0)
            ax.fill_between(common_steps, lo, hi,
                            color=profile_colors.get(profile, '#444'), alpha=0.18)
    ax.set_xlabel('training step')
    ax.set_ylabel('validation loss')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    p_pdf = out_dir / 'fig_trajectory.pdf'
    p_png = out_dir / 'fig_trajectory.png'
    fig.savefig(p_pdf, bbox_inches='tight')
    fig.savefig(p_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {p_pdf} and {p_png}')
    return str(p_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_root', type=str)
    args = ap.parse_args()
    root = Path(args.results_root)
    runs = _read_runs(root)
    if not runs:
        print(f'no runs found in {root}')
        return
    by_profile = _aggregate(runs)
    summary = table_main(by_profile, root / 'table_main.txt')
    capture_path = fig_capture(by_profile, root)
    traj_path = fig_trajectory(by_profile, root)

    out = {
        'n_runs': len(runs),
        'profiles': {p: len(runs) for p, runs in by_profile.items()},
        'summary': summary,
        'fig_capture': capture_path,
        'fig_trajectory': traj_path,
    }
    json.dump(out, open(root / 'summary.json', 'w'), indent=2)
    print(f'wrote {root / "summary.json"}')


if __name__ == '__main__':
    main()
