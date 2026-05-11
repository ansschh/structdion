#!/usr/bin/env bash
# Smoke test for the multi-scale pipeline.
#
# Three stages, all on one srun-allocated H200:
#   1) 340M:  5 profiles x 1 seed x 200 steps (verifies new profile code)
#   2) 700M:  1 profile  x 1 seed x 100 steps (verifies 700M model builds)
#   3) (optional, slow) 1B:  1 profile x 1 seed x 100 steps single-GPU
#
# Each smoke run writes to a FRESH timestamped results dir under
# results/smoke_<timestamp>/ so we never re-skip stale results.
#
# Override GRES / PARTITION / etc via env vars as before.

set -euo pipefail

PARTITION="${PARTITION:-beta}"
TIME="${TIME:-01:00:00}"
GRES="${GRES:-gpu:nvidia_h200:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
SKIP_1B="${SKIP_1B:-0}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SMOKE_ROOT="$REPO_DIR/results/smoke_$(date +%Y%m%d_%H%M%S)"

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
        source env_setup.sh
        echo '--- GPU allocated ---'
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        case \"\$GPU_NAME\" in
            *P100*|*K80*|*M40*|*M60*)
                echo '[abort] Pascal-class GPU; cu121 torch needs >= Volta.'
                exit 2
                ;;
        esac

        echo
        echo '================================================================'
        echo 'Smoke stage 1/3: 340M, all 5 profiles, 1 seed each, 200 steps'
        echo '================================================================'
        SCALE=340m STEPS=200 SEEDS='0' \
            PROFILES='uniform_comm struct inverted_matched adamw muon' \
            LOG_SPECTRA_EVERY=200 EVAL_EVERY=200 \
            bash torchtitan_polar/run_structural_law.sh '$SMOKE_ROOT/340m'

        echo
        echo '================================================================'
        echo 'Smoke stage 2/3: 700M, struct profile, 1 seed, 100 steps'
        echo '================================================================'
        SCALE=700m STEPS=100 SEEDS='0' \
            PROFILES='struct' \
            LOG_SPECTRA_EVERY=100 EVAL_EVERY=100 \
            bash torchtitan_polar/run_structural_law.sh '$SMOKE_ROOT/700m'

        if [[ '$SKIP_1B' != '1' ]]; then
            echo
            echo '================================================================'
            echo 'Smoke stage 3/3: 1B, struct profile, 1 seed, 100 steps (single GPU)'
            echo '================================================================'
            SCALE=1b STEPS=100 SEEDS='0' \
                PROFILES='struct' \
                LOG_SPECTRA_EVERY=100 EVAL_EVERY=100 \
                bash torchtitan_polar/run_structural_law.sh '$SMOKE_ROOT/1b'
        fi

        echo
        echo '================================================================'
        echo 'Aggregating 340M smoke'
        echo '================================================================'
        python torchtitan_polar/aggregate_structural.py '$SMOKE_ROOT/340m'
        cat '$SMOKE_ROOT/340m/table_main.txt' || true
    "
)

echo "[smoke] launching: ${cmd[*]}"
"${cmd[@]}"
