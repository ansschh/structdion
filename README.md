# StructDion: Architecture-Conditioned Rank Allocation

Driver code for the HiLD 2026 paper on rank allocation in low-rank spectral
optimizers (Muon, Dion, AdaDion, PowerSGD, GaLore).

## TL;DR

```bash
# 1. clone in scratch (NOT home; home is too small)
cd /resnick/scratch/$USER
git clone https://github.com/ansschh/structdion.git
cd structdion

# 2. install env (login node, no GPU needed)
bash setup.sh

# 3. smoke test (one GPU, 30 to 45 min)
bash smoke.sh

# 4. full sweep (sbatch array of 9 GPUs, ~24h)
bash full.sh

# 5. aggregate paper-ready table + figures
bash aggregate.sh
```

## What this runs

Three rank-profile configurations on LLaMA320M, three seeds each:

| Profile | r_FFN | r_qko | r_v | B / B_struct |
|---|---:|---:|---:|---:|
| `struct` (headline) | 768 | 256 | 64 | 1.000 |
| `uniform_comm` (matched-comm baseline) | 532 | 532 | 532 | 1.000 |
| `inverted` (role falsifier) | 64 | 192 | 768 | 0.335 |

Each run logs per-layer Ky-Fan and Frobenius capture spectra, qbuf
verification, and validation loss trajectories.

## Files

| File | Purpose |
|---|---|
| `setup.sh` | One-shot env install (login node) |
| `smoke.sh` | 500-step interactive smoke via `srun` |
| `full.sh` | sbatch array for the 9-run full sweep |
| `aggregate.sh` | Build paper-ready table and figures |
| `torchtitan_polar/structural_law_entry.py` | Single-run training script |
| `torchtitan_polar/run_structural_law.sh` | Sweep driver (called by full.sh) |
| `torchtitan_polar/aggregate_structural.py` | Aggregation script |
| `torchtitan_polar/STRUCTURAL_LAW_README.md` | Detailed protocol notes |

## Overriding defaults

SLURM partition / time / GPU resource are overridable via env vars:

```bash
PARTITION=expansion TIME=12:00:00 bash full.sh
```

Step count, batch, sequence length, LR are also overridable:

```bash
STEPS=5000 BATCH_SIZE=8 bash smoke.sh
```

## Cleaning up old stuff

The cluster home dir on Caltech HPC is small. Move stale checkouts into
scratch:

```bash
# from your home dir, kill anything you do not need
cd ~
rm -rf ada-dion              # the old dion-only checkout
rm -rf torchtitan            # any stale torchtitan
rm -rf .cache                # pip cache, will be recreated
# keep .ssh, .bashrc, .conda (if you use conda); inspect first
du -sh ~/.* ~/* 2>/dev/null | sort -hr | head
```

For this project, EVERYTHING goes in `/resnick/scratch/$USER/structdion/`.
