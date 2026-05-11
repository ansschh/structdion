# Structural rank-allocation experiments

Driver code for the HiLD 2026 paper. One launcher, three profiles
(`uniform_comm`, `struct`, `inverted`), three seeds each, with per-layer
Ky-Fan and Frobenius spectral capture logged and qbuf retro-verification
recorded.

## Files

| File | Purpose |
|---|---|
| `structural_law_entry.py` | The training script (single run). |
| `run_structural_law.sh` | Sweep driver over 3 profiles x 3 seeds. |
| `aggregate_structural.py` | Reads results, writes table + figures. |

## Profile definitions

For LLaMA320M (`d = 768`, `d_h = 64`, `ffn_hidden = 2048`):

| Profile | r_FFN | r_qko | r_v | per-layer B(r) | B / B_struct |
|---|---:|---:|---:|---:|---:|
| `struct` | 768 | 256 | 64 | 7,766,016 | 1.000 |
| `uniform_comm` | 532 | 532 | 532 | 7,765,824 | 1.000 |
| `inverted` (literal) | 64 | 192 | 768 | 2,605,056 | 0.335 |

The `inverted` profile is NOT matched-communication; it is the natural
"role-inverted" profile and ships with a smaller budget. We report its
budget ratio in `result.json`. It is included as a role-assignment
falsifier: if the structural advantage came from "more rank on FFN"
alone, an inverted profile that gives less rank to FFN should lose at
its own budget; if the advantage comes from rank shuffling that happens
to give attention more capacity, inverted should win.

The matched-communication falsifier is the comparison `struct` vs
`uniform_comm`; both have the same per-step communication budget.

## One run

```bash
cd torchtitan_polar
python structural_law_entry.py \
    --profile struct \
    --seed 0 \
    --steps 5000 \
    --lr 0.012 \
    --output_dir results/struct_law
```

Outputs in `results/struct_law/struct_seed0_lr0.012/`:

* `result.json` headline metrics and metadata
* `step_log.json` per-step loss / lr
* `val_log.json` evaluations
* `capture_log.json` Ky-Fan + Frobenius per-param SVD at intervals
* `qbuf_pre.json` and `qbuf_post.json` rank verification

## Smoke test (500 steps)

```bash
cd torchtitan_polar
STEPS=500 bash run_structural_law.sh results/struct_law_smoke
```

Confirms the harness produces the expected JSONs and that `qbuf` matches
requested rank.

## Full sweep (3.4B tokens, 3 profiles, 3 seeds)

```bash
cd torchtitan_polar
bash run_structural_law.sh results/struct_law
```

At batch_size 4, seq_len 1024, 3.4B tokens is ~830k steps. On 50 sustained
H100s, one run is roughly 25 H100-hours; 9 runs is ~225 H100-hours, well
under the 1-day budget.

## Aggregate to paper artifacts

```bash
python aggregate_structural.py results/struct_law
```

Writes:

* `table_main.txt` headline matched-comm validation-loss table with paired
  deltas and 95% bootstrap CI
* `fig_capture.pdf` Ky-Fan and Frobenius per-role marginal capture
  (Theorem 1 diagnostic)
* `fig_trajectory.pdf` validation-loss trajectory per profile, CI shaded
* `summary.json` everything in machine-readable form

These are the inputs for the `\NUM{...}` and `\FIGBOX{...}` placeholders
in `HiLD_style_2026/main.tex` and `HiLD_style_2026/appendix.tex`.

## Notes

* Per-role rank is enforced by three separate AdaDion optimizer
  instances with `adaptive_rank=False`. AdamW handles scalars and
  embeddings.
* The training loop is otherwise identical to `train_fair.py`, so any
  baseline runs from earlier in this campaign remain directly
  comparable.
* If you need a deviation from the default LR (0.012) for a particular
  profile, pass `--lr` on the launcher; the sweep script also reads
  `LR` from the environment.
