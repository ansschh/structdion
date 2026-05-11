#!/bin/bash
# Reproduce Paper Table 2: AdamW vs Muon at B=64 (no gradient accumulation)
# 160M LLaMA3 on C4, 3.2B tokens, 3 seeds
# Order: for each seed, run AdamW then Muon before moving to next seed.
#
# Waits for all 8 GPUs to be free before starting.
#
# Usage:
#   bash scripts/run_table2_b64.sh
#   DRY_RUN=1 bash scripts/run_table2_b64.sh

set -e

# Activate venv if torchrun is not on PATH
if ! command -v torchrun &>/dev/null; then
    if [ -f "./venv/bin/activate" ]; then
        source ./venv/bin/activate
    else
        echo "ERROR: torchrun not found. Activate your virtual environment first."
        exit 1
    fi
fi

# ============================================================
# Configuration
# ============================================================
NGPU=8
BATCH_SIZE=64
LOCAL_BATCH_SIZE=8   # 64 / 8 GPUs = 8, no gradient accumulation
SEQ_LEN=2048
TOKEN_BUDGET=3200000000
NUM_SEEDS=3
DRY_RUN=${DRY_RUN:-0}

TOKENS_PER_STEP=$((BATCH_SIZE * SEQ_LEN))
STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))

# Directories
LOG_DIR="./logs/paper_table2/B${BATCH_SIZE}"
OUTPUT_DIR="./outputs/paper_table2/B${BATCH_SIZE}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Paper Table 2 — B=${BATCH_SIZE} (no gradient accumulation)"
echo "============================================================"
echo "  GPUs:             $NGPU"
echo "  Batch size:       $BATCH_SIZE (local: $LOCAL_BATCH_SIZE, no accum)"
echo "  Seq length:       $SEQ_LEN"
echo "  Token budget:     $TOKEN_BUDGET"
echo "  Training steps:   $STEPS"
echo "  Seeds:            0..$((NUM_SEEDS - 1))"
echo "  Optimizers:       adamw, muon (per seed)"
echo "  Log dir:          $LOG_DIR"
echo "============================================================"

# ============================================================
# GPU monitoring — wait until all 8 GPUs are free
# ============================================================
GPU_MEM_THRESHOLD_MB=1000  # Consider GPU "free" if < 1GB used

wait_for_gpus() {
    echo ""
    echo "Waiting for all $NGPU GPUs to be free (< ${GPU_MEM_THRESHOLD_MB}MB used each)..."
    while true; do
        # Get memory used (MiB) for each GPU, one per line
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        all_free=true
        gpu_idx=0
        while IFS= read -r mem; do
            mem=$(echo "$mem" | tr -d ' ')
            if [ "$mem" -ge "$GPU_MEM_THRESHOLD_MB" ]; then
                all_free=false
                break
            fi
            gpu_idx=$((gpu_idx + 1))
        done <<< "$mem_used"

        if [ "$gpu_idx" -lt "$NGPU" ] && [ "$all_free" = false ]; then
            # Not enough GPUs or some are busy
            printf "\r  GPU %d using %s MB — checking again in 30s...    " "$gpu_idx" "$mem"
            sleep 30
        elif [ "$all_free" = true ] && [ "$gpu_idx" -ge "$NGPU" ]; then
            echo ""
            echo "All $NGPU GPUs are free. Starting experiments!"
            echo ""
            return 0
        else
            printf "\r  Waiting... checked %d GPUs — retrying in 30s...    " "$gpu_idx"
            sleep 30
        fi
    done
}

if [ "$DRY_RUN" -eq 0 ]; then
    wait_for_gpus
fi

# ============================================================
# Run experiments: seed-major order (adamw, muon per seed)
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OPTIMIZERS=(adamw muon)
TOTAL=$((NUM_SEEDS * ${#OPTIMIZERS[@]}))
DONE=0
FAILED=0

for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    for opt in "${OPTIMIZERS[@]}"; do
        RUN_NAME="${opt}_seed${seed}"
        DONE_MARKER="$LOG_DIR/${RUN_NAME}.done"
        LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
        DUMP_FOLDER="$OUTPUT_DIR/${RUN_NAME}"

        # Resume support: skip completed runs
        if [ -f "$DONE_MARKER" ]; then
            echo "[SKIP] $RUN_NAME (already completed)"
            DONE=$((DONE + 1))
            continue
        fi

        echo ""
        echo "--- [$((DONE + 1))/$TOTAL] $RUN_NAME ---"

        CMD=(
            torchrun
            --nproc_per_node="$NGPU"
            --rdzv_backend=c10d
            --rdzv_endpoint="localhost:0"
            --local-ranks-filter 0
            --role rank
            --tee 3
            -m torchtitan.train
            --module ada_dion
            --config "llama3_160m_${opt}"
            --training.steps "$STEPS"
            --training.global-batch-size "$BATCH_SIZE"
            --training.local-batch-size "$LOCAL_BATCH_SIZE"
            --dataloader.dataset c4
            --dump-folder "$DUMP_FOLDER"
            --debug.seed "$seed"
            --validator.enable
            --validator.freq 500
            --validator.steps 20
            --checkpoint.interval 2000
            --checkpoint.last-save-model-only
            --metrics.enable-tensorboard
            --metrics.no-enable-wandb
            --parallelism.data-parallel-shard-degree "$NGPU"
            --parallelism.data-parallel-replicate-degree 1
        )

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[DRY RUN] ${CMD[*]}"
            DONE=$((DONE + 1))
            continue
        fi

        mkdir -p "$DUMP_FOLDER"

        if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
            touch "$DONE_MARKER"
            echo "[DONE] $RUN_NAME"
        else
            echo "[FAIL] $RUN_NAME (exit code: $?)"
            FAILED=$((FAILED + 1))
        fi

        DONE=$((DONE + 1))
    done

    # Print interim comparison after each seed
    if [ "$DRY_RUN" -eq 0 ]; then
        echo ""
        echo "=== Seed $seed complete ==="
        for opt in "${OPTIMIZERS[@]}"; do
            log="$LOG_DIR/${opt}_seed${seed}.log"
            if [ -f "$log" ]; then
                last_val=$(grep -oP 'validate step:\s*\d+\s+loss:\s*[\d.]+' "$log" | tail -1)
                echo "  ${opt}: ${last_val:-N/A}"
            fi
        done
        echo "=========================="
    fi
done

echo ""
echo "============================================================"
echo "  Completed: $((DONE - FAILED))/$TOTAL   Failed: $FAILED"
echo "============================================================"
echo ""
echo "To generate the results table:"
echo "  python scripts/parse_paper_table2.py --batch-size $BATCH_SIZE"