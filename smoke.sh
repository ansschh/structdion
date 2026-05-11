#!/usr/bin/env bash
# Smoke test: 500 steps per profile, on one GPU, via srun.
#
# Usage:
#   bash smoke.sh
#
# Override SLURM partition/account/time with env vars:
#   PARTITION=gpu ACCOUNT=resnick TIME=00:30:00 bash smoke.sh

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-00:45:00}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
EXTRA="${EXTRA:-}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

cmd=(srun
    --partition="$PARTITION"
    --gres="$GRES"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME"
    --pty bash -lc "
        cd '$REPO_DIR'
        source venv/bin/activate
        nvidia-smi
        cd torchtitan_polar
        STEPS=500 LOG_SPECTRA_EVERY=200 EVAL_EVERY=200 \
            bash run_structural_law.sh '$REPO_DIR/results/smoke'
        cd '$REPO_DIR'
        python torchtitan_polar/aggregate_structural.py results/smoke
        echo
        echo '======================================================='
        echo 'Smoke aggregation written to results/smoke/'
        cat results/smoke/table_main.txt
        echo '======================================================='
    "
)

if [[ -n "$EXTRA" ]]; then
    cmd+=("$EXTRA")
fi

echo "[smoke] launching: ${cmd[*]}"
"${cmd[@]}"
