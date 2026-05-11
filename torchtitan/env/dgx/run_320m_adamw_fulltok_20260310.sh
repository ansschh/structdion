#!/bin/bash
# AdamW replication with full Llama-3.1-8B tokenizer
# Previous run used test tokenizer (~2K vocab) → artificially low loss
# This run uses the full 128K tokenizer to replicate paper Table 2 (val loss ~3.3)
#
# NOTE: Must run inside the 'torchtitan' Docker container.
# From host:
#   docker exec -it torchtitan bash -c 'cd /home/torchtitan && bash env/dgx/run_320m_adamw_fulltok_20260310.sh'
#
# Debug (10 steps):
#   docker exec -it torchtitan bash -c 'cd /home/torchtitan && STEPS=10 TIME_LIMIT=-1 bash env/dgx/run_320m_adamw_fulltok_20260310.sh'
#
# Dry run:
#   DRY_RUN=1 bash env/dgx/run_320m_adamw_fulltok_20260310.sh

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
GLOBAL_BATCH_SIZE=$((LOCAL_BATCH_SIZE * NGPU))  # 256 with defaults
SEQ_LEN=2048
TOKEN_BUDGET=3200000000
TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * SEQ_LEN))
MAX_STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))

STEPS=${STEPS:-$MAX_STEPS}
TIME_LIMIT=${TIME_LIMIT:-900}  # 15 minutes default
DRY_RUN=${DRY_RUN:-0}
SEED=${SEED:-0}

# Full Llama-3.1-8B tokenizer (128K vocab)
TOKENIZER_PATH=${TOKENIZER_PATH:-"./tokenizers/llama-3.1-8b"}

# Triton compatibility
export CC=gcc
export CXX=g++

# WandB
export WANDB_TEAM="${WANDB_TEAM:-llm_jp_pp}"
export WANDB_PROJECT="${WANDB_PROJECT:-distributed_muon_v2}"

# Memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Directories
DATE_TAG="20260310"
LOG_DIR="./logs/320m_adamw_fulltok_${DATE_TAG}"
mkdir -p "$LOG_DIR"

# Verify tokenizer exists
if [ ! -f "${TOKENIZER_PATH}/tokenizer.json" ]; then
    echo "ERROR: Full tokenizer not found at ${TOKENIZER_PATH}/tokenizer.json"
    echo "Download it first:"
    echo "  huggingface-cli download meta-llama/Llama-3.1-8B --include 'tokenizer*' --local-dir ${TOKENIZER_PATH}"
    exit 1
fi

echo "============================================================"
echo "  AdamW Replication: LLaMA3 320M (Full Tokenizer)"
echo "============================================================"
echo "  GPUs:             $NGPU"
echo "  Batch size:       $GLOBAL_BATCH_SIZE (local: $LOCAL_BATCH_SIZE)"
echo "  Seq length:       $SEQ_LEN"
echo "  Max steps:        $STEPS"
echo "  Time limit:       ${TIME_LIMIT}s"
echo "  Seed:             $SEED"
echo "  Tokenizer:        $TOKENIZER_PATH"
echo "  WandB:            $WANDB_TEAM/$WANDB_PROJECT"
echo "  Log dir:          $LOG_DIR"
echo "============================================================"

RUN_NAME="320m_adamw_fulltok_B${GLOBAL_BATCH_SIZE}_${DATE_TAG}"
LOG_FILE="$LOG_DIR/adamw.log"

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
    --module ada_dion
    --config "llama3_320m_adamw"
    --training.steps "$STEPS"
    --training.time-limit-s "$TIME_LIMIT"
    --training.local-batch-size "$LOCAL_BATCH_SIZE"
    --training.global-batch-size "$GLOBAL_BATCH_SIZE"
    --dataloader.dataset c4_local
    --hf-assets-path "$TOKENIZER_PATH"
    --dump-folder "$LOG_DIR/outputs/adamw"
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
    exit 0
fi

mkdir -p "$LOG_DIR/outputs/adamw"

echo ""
echo "--- Running AdamW ($RUN_NAME) ---"

if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
    echo "[DONE] adamw"
else
    echo "[FAIL] adamw (exit code: $?)"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Completed: AdamW with full tokenizer"
echo "============================================================"
