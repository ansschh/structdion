#!/bin/bash
# Reproduce Paper Table 2: AdamW vs Muon validation loss
# 160M LLaMA3 on C4, 3.2B tokens, batch size 256, 10 seeds
#
# Usage:
#   bash scripts/run_paper_table2.sh              # Run all
#   bash scripts/run_paper_table2.sh adamw        # Run only AdamW
#   bash scripts/run_paper_table2.sh muon         # Run only Muon
#   NGPU=4 bash scripts/run_paper_table2.sh       # Override GPU count
#   BATCH_SIZE=64 bash scripts/run_paper_table2.sh  # Override batch size
#   DRY_RUN=1 bash scripts/run_paper_table2.sh    # Print commands without running

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
NGPU=${NGPU:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
SEQ_LEN=2048
TOKEN_BUDGET=3200000000
NUM_SEEDS=${NUM_SEEDS:-10}
DRY_RUN=${DRY_RUN:-0}

# Local batch size per GPU (kept small to avoid OOM with large vocab)
# Gradient accumulation handles the rest: accum_steps = BATCH_SIZE / (LOCAL_BATCH_SIZE * NGPU)
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-8}

# Compute training steps
TOKENS_PER_STEP=$((BATCH_SIZE * SEQ_LEN))
STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
ACCUM_STEPS=$((BATCH_SIZE / (LOCAL_BATCH_SIZE * NGPU)))

# Validate
if [ $((LOCAL_BATCH_SIZE * NGPU * ACCUM_STEPS)) -ne "$BATCH_SIZE" ]; then
    echo "ERROR: BATCH_SIZE=$BATCH_SIZE not achievable with LOCAL_BATCH_SIZE=$LOCAL_BATCH_SIZE, NGPU=$NGPU"
    echo "  Need BATCH_SIZE = LOCAL_BATCH_SIZE * NGPU * accum_steps"
    exit 1
fi

# Directories
LOG_DIR="./logs/paper_table2/B${BATCH_SIZE}"
OUTPUT_DIR="./outputs/paper_table2/B${BATCH_SIZE}"
mkdir -p "$LOG_DIR"

# Filter optimizer if specified
FILTER_OPT="${1:-}"
OPTIMIZERS=(adamw muon)
if [ -n "$FILTER_OPT" ]; then
    OPTIMIZERS=("$FILTER_OPT")
fi

echo "============================================================"
echo "  Paper Table 2 Reproduction"
echo "============================================================"
echo "  GPUs:             $NGPU"
echo "  Batch size:       $BATCH_SIZE (local: $LOCAL_BATCH_SIZE, accum: $ACCUM_STEPS)"
echo "  Seq length:       $SEQ_LEN"
echo "  Token budget:     $TOKEN_BUDGET"
echo "  Training steps:   $STEPS"
echo "  Seeds:            0..$((NUM_SEEDS - 1))"
echo "  Optimizers:       ${OPTIMIZERS[*]}"
echo "  Log dir:          $LOG_DIR"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============================================================
# Run experiments
# ============================================================
TOTAL=$((${#OPTIMIZERS[@]} * NUM_SEEDS))
DONE=0
FAILED=0

for opt in "${OPTIMIZERS[@]}"; do
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
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
done

echo ""
echo "============================================================"
echo "  Completed: $((DONE - FAILED))/$TOTAL   Failed: $FAILED"
echo "============================================================"
echo ""
echo "To generate the results table:"
echo "  python scripts/parse_paper_table2.py --batch-size $BATCH_SIZE"
