#!/bin/bash
# Unified training script for torchtitan experiments.
#
# Usage:
#   bash env/dgx/run.sh [OPTIONS] [EXTRA_TORCHTITAN_ARGS...]
#
# Required:
#   -c CONFIG       Config name. Available configs:
#                     llama3_320m_adamw, llama3_320m_muon, llama3_320m_dion,
#                     llama3_320m_dion2, llama3_320m_adadion
#                   Or pass multiple: -c "llama3_320m_adamw llama3_320m_muon"
#
# Optional:
#   -m MODULE       Experiment module (default: auto-detected from config)
#                   Modules: ortho_matrix.muon, ortho_matrix.dion,
#                            ortho_matrix.dion2, ortho_matrix.ada_dion
#   -g NGPU         Number of GPUs (default: 8)
#   -b GBS          Global batch size (default: 256)
#   -l LBS          Local batch size (default: GBS/NGPU)
#   -s STEPS        Training steps (default: compute from 3.2B token budget)
#   -t TIME_LIMIT   Time limit in seconds (default: -1, no limit)
#   -n RUN_NAME     Custom run name prefix (default: derived from config)
#   -d LOG_DIR      Log directory (default: ./logs/<run_name>)
#   --seed SEED     Random seed (default: 0)
#   --dry-run       Print commands without running
#   --no-wandb      Disable WandB logging
#   --no-compile    Disable torch.compile
#   --no-validate   Disable validation
#
# Extra args after -- are passed directly to torchtitan (e.g. --optimizer.lr 0.008)
#
# Examples:
#   # Single AdamW run (module auto-detected)
#   bash env/dgx/run.sh -c llama3_320m_adamw
#
#   # Multiple optimizers sequentially
#   bash env/dgx/run.sh -c "llama3_320m_adamw llama3_320m_muon"
#
#   # Quick test (10 steps)
#   bash env/dgx/run.sh -c llama3_320m_adamw -s 10
#
#   # Custom LR override
#   bash env/dgx/run.sh -c llama3_320m_muon -- --optimizer.lr 0.006
#
#   # Sweep: loop over configs/args yourself
#   for lr in 0.004 0.006 0.008; do
#     bash env/dgx/run.sh -c llama3_320m_dion \
#       -n "sweep_lr${lr}" -- --optimizer.lr $lr
#   done
#
#   # Dry run
#   bash env/dgx/run.sh -c llama3_320m_adamw --dry-run
#
#   # 4 GPUs, custom batch size
#   bash env/dgx/run.sh -c llama3_320m_adamw -g 4 -b 256 -l 16

set -e

# ============================================================
# Parse arguments
# ============================================================
MODULE=""
CONFIGS=()
NGPU=${NGPU:-8}
GLOBAL_BATCH_SIZE=""
LOCAL_BATCH_SIZE=""
STEPS=""
TIME_LIMIT=${TIME_LIMIT:--1}
RUN_NAME_PREFIX=""
LOG_DIR=""
SEED=${SEED:-0}
DRY_RUN=0
WANDB_ENABLED=1
COMPILE_ENABLED=1
VALIDATE_ENABLED=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m) MODULE="$2"; shift 2 ;;
        -c) read -ra CONFIGS <<< "$2"; shift 2 ;;
        -g) NGPU="$2"; shift 2 ;;
        -b) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        -l) LOCAL_BATCH_SIZE="$2"; shift 2 ;;
        -s) STEPS="$2"; shift 2 ;;
        -t) TIME_LIMIT="$2"; shift 2 ;;
        -n) RUN_NAME_PREFIX="$2"; shift 2 ;;
        -d) LOG_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-wandb) WANDB_ENABLED=0; shift ;;
        --no-compile) COMPILE_ENABLED=0; shift ;;
        --no-validate) VALIDATE_ENABLED=0; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required args
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "ERROR: -c CONFIG is required"
    echo "  Available configs:"
    echo "    llama3_320m_adamw    (AdamW baseline)"
    echo "    llama3_320m_muon     (Muon)"
    echo "    llama3_320m_dion     (Dion)"
    echo "    llama3_320m_dion2    (Dion2)"
    echo "    llama3_320m_adadion  (AdaDion)"
    exit 1
fi

# Auto-detect module from config if not specified
detect_module() {
    local config="$1"
    case "$config" in
        *muon*)    echo "ortho_matrix.muon" ;;
        *dion2*)   echo "ortho_matrix.dion2" ;;
        *dion*)    echo "ortho_matrix.dion" ;;
        *adadion*) echo "ortho_matrix.ada_dion" ;;
        *adamw*)   echo "ortho_matrix.muon" ;;
        *)
            echo "ERROR: Cannot auto-detect module for config '$config'." >&2
            echo "  Please specify -m MODULE explicitly." >&2
            exit 1
            ;;
    esac
}

# ============================================================
# Defaults
# ============================================================
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / NGPU))}
SEQ_LEN=2048
TOKEN_BUDGET=3200000000
TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * SEQ_LEN))
MAX_STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
STEPS=${STEPS:-$MAX_STEPS}
DATE_TAG=$(date +%Y%m%d_%H%M)

# ============================================================
# Environment setup
# ============================================================

# Activate venv
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

if ! command -v torchrun &>/dev/null; then
    echo "ERROR: torchrun not found. Are you inside the torchtitan Docker container?"
    exit 1
fi

# Triton compatibility
export CC=gcc
export CXX=g++

# WandB
if [ $WANDB_ENABLED -eq 1 ]; then
    export WANDB_TEAM="${WANDB_TEAM:-llm_jp_pp}"
    if [ -z "$WANDB_PROJECT" ]; then
        echo ""
        echo "WandB project not set. Please specify:"
        read -rp "  WANDB_PROJECT: " WANDB_PROJECT
        if [ -z "$WANDB_PROJECT" ]; then
            echo "ERROR: WANDB_PROJECT is required (or use --no-wandb)"
            exit 1
        fi
    fi
    export WANDB_PROJECT
fi

# Memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# HF assets (tokenizer)
TOKENIZER_PATH="./assets/hf/Llama-3.1-8B"

# ============================================================
# Print summary
# ============================================================
echo "============================================================"
echo "  torchtitan training"
echo "============================================================"
echo "  Module:           ${MODULE:-auto-detect}"
echo "  Configs:          ${CONFIGS[*]}"
echo "  GPUs:             $NGPU"
echo "  Batch size:       $GLOBAL_BATCH_SIZE (local: $LOCAL_BATCH_SIZE)"
echo "  Seq length:       $SEQ_LEN"
echo "  Steps:            $STEPS"
echo "  Time limit:       ${TIME_LIMIT}s"
echo "  Seed:             $SEED"
echo "  WandB:            $([ $WANDB_ENABLED -eq 1 ] && echo "$WANDB_TEAM/$WANDB_PROJECT" || echo "disabled")"
echo "  Compile:          $([ $COMPILE_ENABLED -eq 1 ] && echo "yes" || echo "no")"
echo "  Validate:         $([ $VALIDATE_ENABLED -eq 1 ] && echo "yes" || echo "no")"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Extra args:       ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

# ============================================================
# Run experiments
# ============================================================
TOTAL=${#CONFIGS[@]}
DONE=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    # Derive short name from config (strip "llama3_320m_" prefix if present)
    SHORT_NAME="${config#llama3_320m_}"
    SHORT_NAME="${SHORT_NAME#llama3_160m_}"

    if [ -n "$RUN_NAME_PREFIX" ]; then
        RUN_NAME="${RUN_NAME_PREFIX}_${SHORT_NAME}_${DATE_TAG}"
    else
        RUN_NAME="${SHORT_NAME}_${DATE_TAG}"
    fi

    if [ -n "$LOG_DIR" ]; then
        RUN_LOG_DIR="$LOG_DIR"
    else
        RUN_LOG_DIR="./logs/${RUN_NAME}"
    fi
    LOG_FILE="$RUN_LOG_DIR/${SHORT_NAME}.log"

    echo ""
    echo "--- [$((DONE + 1))/$TOTAL] $config ($RUN_NAME) ---"

    # Determine module for this config
    if [ -n "$MODULE" ]; then
        RUN_MODULE="$MODULE"
    else
        RUN_MODULE=$(detect_module "$config")
    fi

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
        --module "$RUN_MODULE"
        --config "$config"
        --training.steps "$STEPS"
        --training.time-limit-s "$TIME_LIMIT"
        --training.local-batch-size "$LOCAL_BATCH_SIZE"
        --training.global-batch-size "$GLOBAL_BATCH_SIZE"
        --dataloader.dataset c4
        --hf-assets-path "$TOKENIZER_PATH"
        --dump-folder "$RUN_LOG_DIR/outputs/${SHORT_NAME}"
        --debug.seed "$SEED"
        --parallelism.data-parallel-shard-degree "$NGPU"
        --parallelism.data-parallel-replicate-degree 1
    )

    # Optional flags
    if [ $VALIDATE_ENABLED -eq 1 ]; then
        CMD+=(
            --validator.enable
            --validator.freq 100
            --validator.steps 20
            --validator.dataloader.dataset c4_validation
        )
    fi
    if [ $COMPILE_ENABLED -eq 1 ]; then
        CMD+=(--compile.enable)
    fi
    if [ $WANDB_ENABLED -eq 1 ]; then
        CMD+=(--metrics.enable-wandb)
    fi
    CMD+=(--metrics.enable-tensorboard)

    # Append any extra torchtitan args
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY RUN] ${CMD[*]}"
        DONE=$((DONE + 1))
        continue
    fi

    mkdir -p "$RUN_LOG_DIR/outputs/${SHORT_NAME}"

    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
        echo "[DONE] $config"
    else
        echo "[FAIL] $config (exit code: $?)"
        FAILED=$((FAILED + 1))
    fi

    DONE=$((DONE + 1))
done

echo ""
echo "============================================================"
echo "  Completed: $((DONE - FAILED))/$TOTAL   Failed: $FAILED"
echo "============================================================"
