#!/usr/bin/env bash
# Multi-GPU pod launcher for the d_h=128 perturbation. Detects GPU count
# on the pod and launches one cell per GPU in parallel, starting from
# INSTANCE_IDX=$START_IDX.
#
# Usage:
#   START_IDX=0 bash <(curl -sL https://raw.githubusercontent.com/ansschh/structdion/main/runpod_multi.sh)
#
# Examples:
#   8-GPU pod: START_IDX=0  -> runs cells 0..7 in parallel on GPUs 0..7
#   1-GPU pod: START_IDX=8  -> runs cell 8 only on GPU 0
#   2 pods, each 4 GPUs:
#     pod1: START_IDX=0     -> cells 0..3
#     pod2: START_IDX=4     -> cells 4..7
#     pod1 (after done): START_IDX=8 -> cell 8

set -euo pipefail

START_IDX="${START_IDX:?set START_IDX=0..8 before running}"

# Detect GPU count
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [[ "$N_GPUS" -lt 1 ]]; then
    echo "ERROR: no GPUs detected"; exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "============================================================"
echo "Multi-GPU launcher: $N_GPUS x $GPU_NAME"
echo "Starting at INSTANCE_IDX=$START_IDX"
echo "============================================================"

# Sanity: skip cells beyond 8
N_CELLS=$N_GPUS
if [[ $((START_IDX + N_GPUS - 1)) -gt 8 ]]; then
    N_CELLS=$((9 - START_IDX))
    echo "Note: only $N_CELLS cells in range; some GPUs will idle"
fi

# ===== Setup (once for the pod) =====
cd /workspace 2>/dev/null || cd ~

if [[ ! -d structdion ]]; then
    echo "[setup] git clone"
    git clone https://github.com/ansschh/structdion.git
fi
cd structdion

# Pip-install deps if missing
if ! python -c "import torch, dion.dion, transformers" 2>/dev/null; then
    echo "[setup] installing python deps (takes ~3 min)"
    python -m pip install --upgrade pip
    python -m pip install "dion @ git+https://github.com/microsoft/dion.git" --no-deps
    if ! python -c "import torch" 2>/dev/null; then
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi
    python -m pip install -e torchtitan
    python -m pip install datasets tokenizers transformers safetensors tqdm matplotlib numpy certifi
fi

# Env (SSL + cache redirects)
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
REPO=$(pwd)
export TRITON_CACHE_DIR=$REPO/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$REPO/.cache/inductor
export TORCH_HOME=$REPO/.cache/torch
export HF_HOME=$REPO/.cache/huggingface
export C4_CACHE=$REPO/data/c4_tokens.pt
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_HOME data results/dh128
export TORCHDYNAMO_SUPPRESS_ERRORS=1

# Pre-tokenize C4 once (~3 min)
if [[ ! -f "$C4_CACHE" ]]; then
    echo "[prep] tokenizing C4 (~3 min)"
    python - <<'PY'
import os, torch
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
all_tokens = []
for i, ex in enumerate(ds):
    if i >= 50000: break
    all_tokens.extend(tok.encode(ex["text"], add_special_tokens=False))
torch.save(torch.tensor(all_tokens, dtype=torch.int32), os.environ["C4_CACHE"])
print(f"[ok] cache at {os.environ['C4_CACHE']}: {len(all_tokens)} tokens")
PY
fi

# ===== Launch one cell per GPU in parallel =====
PROFILES=(uniform_comm uniform_comm uniform_comm \
          struct struct struct \
          inverted_matched inverted_matched inverted_matched)
SEEDS=(0 1 2 0 1 2 0 1 2)

PIDS=()
for OFFSET in $(seq 0 $((N_CELLS - 1))); do
    IDX=$((START_IDX + OFFSET))
    P=${PROFILES[$IDX]}; S=${SEEDS[$IDX]}
    GPU=$OFFSET
    LOG=results/dh128/cell_${IDX}_${P}_s${S}.log
    echo "[launch] GPU $GPU  IDX=$IDX  profile=$P  seed=$S  -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU \
    python -u torchtitan_polar/structural_law_entry.py \
        --profile "$P" --scale 340m_dh128 --seed "$S" \
        --steps 200000 --lr 0.012 --dtype bf16 \
        --output_dir results/dh128 \
        --log_spectra_every 2000 --eval_every 1000 \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo
echo "============================================================"
echo "All $N_CELLS cells launched. Waiting for completion..."
echo "Watch progress: tail -f results/dh128/cell_*.log"
echo "============================================================"

# Wait for all
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=$((FAILED + 1))
    fi
done

echo "============================================================"
echo "DONE. $((N_CELLS - FAILED))/$N_CELLS cells succeeded."
echo "Results in results/dh128/"
echo "Send back:"
echo "  runpodctl send results/dh128       (then 'runpodctl receive <code>' on target)"
echo "  OR  scp -r results/dh128/* target:/path/"
echo "============================================================"
