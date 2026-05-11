#!/usr/bin/env bash
# Smoke test: 500 steps per profile, on one GPU, via srun.
#
# Usage:
#   bash smoke.sh
#   GRES=gpu:a100:1 bash smoke.sh         # ask for an A100 specifically
#   GRES=gpu:h100:1 PARTITION=gpu bash smoke.sh
#
# Default GRES requests any GPU; the Caltech cluster may give you a P100,
# which is too old for cu121 torch. If you get a P100 the launcher will
# print a clear error and abort.

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-00:45:00}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

cmd=(srun
    --partition="$PARTITION"
    --gres="$GRES"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME"
    --pty bash -lc "
        set -e
        cd '$REPO_DIR'
        source venv/bin/activate
        echo '--- GPU allocated ---'
        nvidia-smi
        GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        case \"\$GPU_NAME\" in
            *P100*|*K80*|*M40*|*M60*)
                echo
                echo '[abort] Got an old GPU (\$GPU_NAME). cu121 torch needs >= Volta.'
                echo '       Resubmit with: GRES=gpu:a100:1 bash smoke.sh'
                echo '       or:            GRES=gpu:v100:1 bash smoke.sh'
                exit 2
                ;;
        esac
        cd torchtitan_polar
        STEPS=500 LOG_SPECTRA_EVERY=200 EVAL_EVERY=200 \
            bash run_structural_law.sh '$REPO_DIR/results/smoke'
        cd '$REPO_DIR'
        python torchtitan_polar/aggregate_structural.py results/smoke
        echo
        echo '======================================================='
        echo 'Smoke aggregation written to results/smoke/'
        echo '======================================================='
        cat results/smoke/table_main.txt
    "
)

echo "[smoke] launching: ${cmd[*]}"
"${cmd[@]}"
