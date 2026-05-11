#!/bin/bash
# Shampoo Muon Replication: Table 1 from 2602.09314v1
# Muon (SVD) on LLaMA3 320M, B=256, 3.2B tokens
#
# NOTE: Must run inside the 'torchtitan' Docker container.
# From host:
#   docker exec -it torchtitan bash -c 'cd /home/torchtitan && bash env/dgx/run_320m_shampoo_muon_20260312.sh'
#
# Inside container:
#   cd /home/torchtitan
#   STEPS=10 bash env/dgx/run_320m_shampoo_muon_20260312.sh          # Test run
#   bash env/dgx/run_320m_shampoo_muon_20260312.sh                    # Full run
#   bash env/dgx/run_320m_shampoo_muon_20260312.sh muon_ns            # Newton-Schulz variant
#   DRY_RUN=1 bash env/dgx/run_320m_shampoo_muon_20260312.sh          # Dry run

set -e

# Activate venv
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

if ! command -v torchrun &>/dev/null; then
    echo "ERROR: torchrun not found. Are you inside the torchtitan Docker container?"
    echo "  docker exec -it torchtitan bash"
    echo "  source /home/torchtitan/venv/bin/activate"
    exit 1
fi

# ============================================================
# Configuration
# ============================================================
NGPU=${NGPU:-8}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
SEQ_LEN=2048
TOKEN_BUDGET=3200000000
TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * SEQ_LEN))
MAX_STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))

STEPS=${STEPS:-$MAX_STEPS}
TIME_LIMIT=${TIME_LIMIT:--1}  # No time limit by default
DRY_RUN=${DRY_RUN:-0}
SEED=${SEED:-0}

# Triton compatibility
export CC=gcc
export CXX=g++

# WandB
export WANDB_TEAM="${WANDB_TEAM:-llm_jp_pp}"
export WANDB_PROJECT="${WANDB_PROJECT:-distributed_muon_v4}"

# Memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Directories
DATE_TAG="20260312"
LOG_DIR="./logs/320m_shampoo_muon_${DATE_TAG}"
mkdir -p "$LOG_DIR"

# Optimizer filter
FILTER_OPT="${1:-}"
OPTIMIZERS=(muon_svd)
if [ -n "$FILTER_OPT" ]; then
    OPTIMIZERS=("$FILTER_OPT")
fi

echo "============================================================"
echo "  Shampoo Muon Replication: 2602.09314v1 Table 1"
echo "  LLaMA3 320M, B=256, 3.2B tokens"
echo "============================================================"
echo "  GPUs:             $NGPU"
echo "  Batch size:       $GLOBAL_BATCH_SIZE (local: $LOCAL_BATCH_SIZE)"
echo "  Seq length:       $SEQ_LEN"
echo "  Max steps:        $STEPS"
echo "  Time limit:       ${TIME_LIMIT}s"
echo "  Seed:             $SEED"
echo "  Optimizers:       ${OPTIMIZERS[*]}"
echo "  WandB:            $WANDB_TEAM/$WANDB_PROJECT"
echo "  Log dir:          $LOG_DIR"
echo "============================================================"

# ============================================================
# Run experiments
# ============================================================
TOTAL=${#OPTIMIZERS[@]}
DONE=0
FAILED=0

for opt in "${OPTIMIZERS[@]}"; do
    RUN_NAME="320m_shampoo_${opt}_B${GLOBAL_BATCH_SIZE}_${DATE_TAG}"
    LOG_FILE="$LOG_DIR/${opt}.log"

    echo ""
    echo "--- [$((DONE + 1))/$TOTAL] $opt ($RUN_NAME) ---"

    export WANDB_RUN_NAME="$RUN_NAME"

    CMD=(
        torchrun
        --nproc_per_node="$NGPU"
        --rdzv_backend=c10d
        --rdzv_endpoint="localhost:0"
        --local-ranks-filter 0
        --role rank
        --tee 3
        -m torchtitan.train
        --module shampoo_muon
        --config "llama3_320m_${opt}"
        --training.steps "$STEPS"
        --training.time-limit-s "$TIME_LIMIT"
        --training.local-batch-size "$LOCAL_BATCH_SIZE"
        --training.global-batch-size "$GLOBAL_BATCH_SIZE"
        --dataloader.dataset c4_local
        --hf-assets-path ./tokenizers/llama-3.1-8b
        --dump-folder "$LOG_DIR/outputs/${opt}"
        --debug.seed "$SEED"
        --validator.enable
        --validator.freq 100
        --validator.steps 20
        --validator.dataloader.dataset c4_local_validation
        --compile.enable
        --metrics.enable-wandb
        --metrics.enable-tensorboard
        --parallelism.data-parallel-shard-degree "$NGPU"
        --parallelism.data-parallel-replicate-degree 1
    )

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY RUN] ${CMD[*]}"
        DONE=$((DONE + 1))
        continue
    fi

    mkdir -p "$LOG_DIR/outputs/${opt}"

    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
        echo "[DONE] $opt"
    else
        echo "[FAIL] $opt (exit code: $?)"
        FAILED=$((FAILED + 1))
    fi

    DONE=$((DONE + 1))
done

echo ""
echo "============================================================"
echo "  Completed: $((DONE - FAILED))/$TOTAL   Failed: $FAILED"
echo "============================================================"
