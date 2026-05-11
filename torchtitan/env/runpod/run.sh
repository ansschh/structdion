#!/bin/bash
# Unified training script for torchtitan experiments on RunPod.
# Runs torchrun directly (no PBS/qsub) on a single multi-GPU node.
#
# Usage:
#   bash env/runpod/run.sh [OPTIONS] [EXTRA_TORCHTITAN_ARGS...]
#
# Required:
#   -m MODULE       Experiment module (ortho_matrix.muon, ortho_matrix.dion,
#                   ortho_matrix.ada_dion_v2, ada_dion, shampoo_muon, muon_grafted)
#   -c CONFIG       Config name (e.g. llama3_320m_adamw, llama3_320m_adadion_v2)
#                   Or pass multiple: -c "llama3_320m_adamw llama3_320m_muon"
#
# Optional:
#   -g NGPU         Number of GPUs (default: all available)
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
#   bash env/runpod/run.sh -m ortho_matrix.ada_dion_v2 -c llama3_320m_adadion_v2
#
#   bash env/runpod/run.sh -m ortho_matrix.ada_dion_v2 -c llama3_320m_adadion_v2 \
#     -n "sweep_lr012" -- --optimizer.lr 0.012

set -e

# ============================================================
# Parse arguments
# ============================================================
MODULE=""
CONFIGS=()
NGPU=${NGPU:-""}
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
if [ -z "$MODULE" ]; then
    echo "ERROR: -m MODULE is required"
    exit 1
fi
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "ERROR: -c CONFIG is required"
    exit 1
fi

# ============================================================
# Defaults
# ============================================================
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_DIR}"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f "${REPO_DIR}/venv/bin/activate" ]; then
    source "${REPO_DIR}/venv/bin/activate"
fi

# Auto-detect GPU count if not specified
if [ -z "$NGPU" ]; then
    NGPU=$(python -c 'import torch; print(torch.cuda.device_count())')
fi

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / NGPU))}
SEQ_LEN=2048
TOKEN_BUDGET=3200000000
TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * SEQ_LEN))
MAX_STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
STEPS=${STEPS:-$MAX_STEPS}
DATE_TAG=$(date +%Y%m%d_%H%M)

# Tokenizer
HF_ASSETS_PATH="${HF_ASSETS_PATH:-./assets/hf/Llama-3.1-8B}"

# WandB
WANDB_TEAM_NAME="${WANDB_TEAM:-}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-distributed_muon_v4}"

# ============================================================
# Print summary
# ============================================================
echo "============================================================"
echo "  torchtitan training (RunPod)"
echo "============================================================"
echo "  Module:           $MODULE"
echo "  Configs:          ${CONFIGS[*]}"
echo "  GPUs:             $NGPU"
echo "  Batch size:       $GLOBAL_BATCH_SIZE (local: $LOCAL_BATCH_SIZE)"
echo "  Seq length:       $SEQ_LEN"
echo "  Steps:            $STEPS"
echo "  Time limit:       ${TIME_LIMIT}s"
echo "  Seed:             $SEED"
echo "  WandB:            $([ $WANDB_ENABLED -eq 1 ] && echo "${WANDB_TEAM_NAME:-(no team)}/${WANDB_PROJECT_NAME}" || echo "disabled")"
echo "  Compile:          $([ $COMPILE_ENABLED -eq 1 ] && echo "yes" || echo "no")"
echo "  Validate:         $([ $VALIDATE_ENABLED -eq 1 ] && echo "yes" || echo "no")"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Extra args:       ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

# ============================================================
# Run training
# ============================================================
for config in "${CONFIGS[@]}"; do
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

    DUMP_DIR="./outputs/${RUN_NAME}/${SHORT_NAME}"
    mkdir -p "$RUN_LOG_DIR" "$DUMP_DIR"

    # Build torchrun arguments
    TRAIN_ARGS=(
        "--module ${MODULE}"
        "--config ${config}"
        "--hf-assets-path ${HF_ASSETS_PATH}"
        "--training.steps ${STEPS}"
        "--training.time-limit-s ${TIME_LIMIT}"
        "--training.global-batch-size ${GLOBAL_BATCH_SIZE}"
        "--training.local-batch-size ${LOCAL_BATCH_SIZE}"
        "--dataloader.dataset c4"
        "--dump-folder ${DUMP_DIR}"
        "--debug.seed ${SEED}"
        "--parallelism.data-parallel-shard-degree ${NGPU}"
        "--parallelism.data-parallel-replicate-degree 1"
        "--metrics.enable-tensorboard"
    )

    if [ $VALIDATE_ENABLED -eq 1 ]; then
        TRAIN_ARGS+=(
            "--validator.enable"
            "--validator.freq 100"
            "--validator.steps 20"
            "--validator.dataloader.dataset c4_validation"
        )
    fi
    if [ $COMPILE_ENABLED -eq 1 ]; then
        TRAIN_ARGS+=("--compile.enable")
    fi
    if [ $WANDB_ENABLED -eq 1 ]; then
        TRAIN_ARGS+=("--metrics.enable-wandb")
    fi

    # Append extra args
    for arg in "${EXTRA_ARGS[@]}"; do
        TRAIN_ARGS+=("$arg")
    done

    TRAIN_ARGS_STR="${TRAIN_ARGS[*]}"

    # Export WandB env
    export WANDB_TEAM="${WANDB_TEAM_NAME}"
    export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
    export WANDB_RUN_NAME="${RUN_NAME}"

    # Environment
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCHDYNAMO_CACHE_SIZE_LIMIT=64

    echo ""
    echo "--- Running: $config ($RUN_NAME) ---"
    echo "  Log: $RUN_LOG_DIR"
    echo "  Dump: $DUMP_DIR"
    if [ $WANDB_ENABLED -eq 1 ]; then
        echo "  WandB run: $RUN_NAME"
    fi

    CMD="torchrun \
        --nproc_per_node=${NGPU} \
        --rdzv_backend c10d \
        --rdzv_endpoint=localhost:0 \
        --local-ranks-filter 0 \
        --role rank --tee 3 \
        -m torchtitan.train \
        ${TRAIN_ARGS_STR}"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY RUN] $CMD"
    else
        echo "$CMD"
        eval "$CMD" 2>&1 | tee "${RUN_LOG_DIR}/${SHORT_NAME}.log"
    fi

    echo "--- ${config}: DONE ---"
done

echo ""
echo "============================================================"
echo "  All runs complete"
echo "============================================================"
