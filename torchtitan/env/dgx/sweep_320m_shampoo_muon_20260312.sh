#!/bin/bash
# Coarse Muon (SVD) Hyperparameter Sweep: Table 9 from 2602.09314v1
# LLaMA3 320M, B=256, 2000 steps (directional signal)
#
# Sweep grid (Table 9, 320M 1x B=256):
#   lr:   {0.008, 0.016, 0.032, 0.064}   (2x increments)
#   β₁:   {0.8, 0.9, 0.95, 0.975}        (coarse 2x(1-β₁) increments)
#   scalar_lr = lr / 2                    (paper convention)
#
# Total: 16 runs × ~35 min ≈ 9 hours
#
# Usage:
#   docker exec -it torchtitan bash -c 'cd /home/torchtitan && bash env/dgx/sweep_320m_shampoo_muon_20260312.sh'
#   STEPS=100 bash env/dgx/sweep_320m_shampoo_muon_20260312.sh        # Quick test
#   DRY_RUN=1 bash env/dgx/sweep_320m_shampoo_muon_20260312.sh        # Print commands only

set -e

# Activate venv
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

if ! command -v torchrun &>/dev/null; then
    echo "ERROR: torchrun not found. Are you inside the torchtitan Docker container?"
    exit 1
fi

# ============================================================
# Configuration
# ============================================================
NGPU=${NGPU:-8}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
SEQ_LEN=2048
STEPS=${STEPS:-2000}
TIME_LIMIT=${TIME_LIMIT:--1}
DRY_RUN=${DRY_RUN:-0}
SEED=${SEED:-0}

# Triton compatibility
export CC=gcc
export CXX=g++

# WandB
export WANDB_TEAM="${WANDB_TEAM:-llm_jp_pp}"
export WANDB_PROJECT="${WANDB_PROJECT:-shampoo_muon_v1}"

# Memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Directories
DATE_TAG="20260312"
LOG_DIR="./logs/sweep_shampoo_muon_${DATE_TAG}"
mkdir -p "$LOG_DIR"

# ============================================================
# Sweep grid (Table 9: Muon 320M 1x B=256)
# ============================================================
LR_VALUES=(0.008 0.016 0.032 0.064)
BETA1_VALUES=(0.8 0.9 0.95 0.975)

# Build list of (lr, beta1) pairs
declare -a SWEEP_LR
declare -a SWEEP_BETA1
for lr in "${LR_VALUES[@]}"; do
    for b1 in "${BETA1_VALUES[@]}"; do
        SWEEP_LR+=("$lr")
        SWEEP_BETA1+=("$b1")
    done
done

TOTAL=${#SWEEP_LR[@]}

echo "============================================================"
echo "  Shampoo Muon (SVD) Coarse Sweep: 2602.09314v1 Table 9"
echo "  LLaMA3 320M, B=256, ${STEPS} steps"
echo "============================================================"
echo "  GPUs:             $NGPU"
echo "  Batch size:       $GLOBAL_BATCH_SIZE (local: $LOCAL_BATCH_SIZE)"
echo "  Seq length:       $SEQ_LEN"
echo "  Steps:            $STEPS"
echo "  Seed:             $SEED"
echo "  LR values:        ${LR_VALUES[*]}"
echo "  Beta1 values:     ${BETA1_VALUES[*]}"
echo "  Total runs:       $TOTAL"
echo "  WandB:            $WANDB_TEAM/$WANDB_PROJECT"
echo "  Log dir:          $LOG_DIR"
echo "============================================================"

# ============================================================
# Run sweep
# ============================================================
DONE=0
FAILED=0

for i in $(seq 0 $((TOTAL - 1))); do
    lr="${SWEEP_LR[$i]}"
    b1="${SWEEP_BETA1[$i]}"
    # scalar_lr = lr / 2 (paper convention)
    scalar_lr=$(python3 -c "print(${lr} / 2)")

    RUN_NAME="sweep_muon_svd_lr${lr}_b1${b1}_B${GLOBAL_BATCH_SIZE}_${DATE_TAG}"
    LOG_FILE="$LOG_DIR/lr${lr}_b1${b1}.log"

    echo ""
    echo "--- [$((DONE + 1))/$TOTAL] lr=$lr  β₁=$b1  scalar_lr=$scalar_lr ($RUN_NAME) ---"

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
        --config llama3_320m_muon_svd
        --training.steps "$STEPS"
        --training.time-limit-s "$TIME_LIMIT"
        --training.local-batch-size "$LOCAL_BATCH_SIZE"
        --training.global-batch-size "$GLOBAL_BATCH_SIZE"
        --optimizer.lr "$lr"
        --optimizer.beta1 "$b1"
        --optimizer.scalar-lr "$scalar_lr"
        --optimizer.scalar-beta1 "$b1"
        --dataloader.dataset c4_local
        --hf-assets-path ./tokenizers/llama-3.1-8b
        --dump-folder "$LOG_DIR/outputs/lr${lr}_b1${b1}"
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

    mkdir -p "$LOG_DIR/outputs/lr${lr}_b1${b1}"

    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
        echo "[DONE] lr=$lr β₁=$b1"
    else
        echo "[FAIL] lr=$lr β₁=$b1 (exit code: $?)"
        FAILED=$((FAILED + 1))
    fi

    DONE=$((DONE + 1))
done

echo ""
echo "============================================================"
echo "  Sweep complete: $((DONE - FAILED))/$TOTAL succeeded   Failed: $FAILED"
echo "============================================================"

# Print summary of validation losses from logs
echo ""
echo "--- Validation Loss Summary ---"
for i in $(seq 0 $((TOTAL - 1))); do
    lr="${SWEEP_LR[$i]}"
    b1="${SWEEP_BETA1[$i]}"
    LOG_FILE="$LOG_DIR/lr${lr}_b1${b1}.log"
    if [ -f "$LOG_FILE" ]; then
        LAST_VAL=$(grep "validate step:" "$LOG_FILE" | tail -1 | grep -oP 'loss:\s+\K[0-9.]+' || echo "N/A")
        LAST_PPL=$(grep "validate step:" "$LOG_FILE" | tail -1 | grep -oP 'ppl:\s+\K[0-9.]+' || echo "N/A")
        printf "  lr=%-6s β₁=%-6s  val_loss=%-10s  ppl=%-10s\n" "$lr" "$b1" "$LAST_VAL" "$LAST_PPL"
    fi
done
