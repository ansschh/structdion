#!/usr/bin/env bash
# One-shot setup-and-run for the d_h=128 perturbation on a RunPod pod.
#
# Usage on a fresh RunPod A100-80GB or H100-80GB community pod:
#
#   INSTANCE_IDX=0 bash <(curl -sL https://raw.githubusercontent.com/ansschh/structdion/main/runpod_oneshot.sh)
#
# Change INSTANCE_IDX (0..8) for each pod:
#   pod 1: 0  uniform_comm seed 0
#   pod 2: 1  uniform_comm seed 1
#   pod 3: 2  uniform_comm seed 2
#   pod 4: 3  struct seed 0
#   pod 5: 4  struct seed 1
#   pod 6: 5  struct seed 2
#   pod 7: 6  inverted_matched seed 0
#   pod 8: 7  inverted_matched seed 1
#   pod 9: 8  inverted_matched seed 2
#
# At the end the result JSON lives under
#   /workspace/structdion/results/dh128/340m_dh128_<P>_seed<S>_lr0.012/
# Copy back to Caltech with runpodctl or scp.

set -euo pipefail

IDX="${INSTANCE_IDX:?set INSTANCE_IDX=0..8 before running}"

echo "============================================================"
echo "RunPod d_h=128 oneshot. INSTANCE_IDX=$IDX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

cd /workspace

# Clone repo (idempotent)
if [[ ! -d structdion ]]; then
    git clone https://github.com/ansschh/structdion.git
fi
cd structdion

# Setup env (RunPod's pytorch image already has torch installed but pip-installing
# our pinned deps is fast and safe).
if ! python -c "import torch, dion.dion, transformers" 2>/dev/null; then
    echo "[setup] installing python deps"
    python -m pip install --upgrade pip
    python -m pip install "dion @ git+https://github.com/microsoft/dion.git" --no-deps
    if ! python -c "import torch" 2>/dev/null; then
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi
    python -m pip install -e torchtitan
    python -m pip install datasets tokenizers transformers safetensors tqdm matplotlib numpy certifi
fi

# Configure env (SSL bundle, cache redirection)
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
export TRITON_CACHE_DIR=/workspace/structdion/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/structdion/.cache/inductor
export TORCH_HOME=/workspace/structdion/.cache/torch
export HF_HOME=/workspace/structdion/.cache/huggingface
export C4_CACHE=/workspace/structdion/data/c4_tokens.pt
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_HOME data
export TORCHDYNAMO_SUPPRESS_ERRORS=1

# Pre-tokenize C4 (once on this pod)
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
    if (i+1) % 5000 == 0:
        print(f"  tokenized {i+1}/50000 docs ({len(all_tokens)} tokens)")
torch.save(torch.tensor(all_tokens, dtype=torch.int32),
           os.environ["C4_CACHE"])
print(f"[ok] cache at {os.environ['C4_CACHE']}: {len(all_tokens)} tokens")
PY
fi

# Map index -> (profile, seed)
PROFILES=(uniform_comm uniform_comm uniform_comm \
          struct struct struct \
          inverted_matched inverted_matched inverted_matched)
SEEDS=(0 1 2 0 1 2 0 1 2)
P=${PROFILES[$IDX]}; S=${SEEDS[$IDX]}

echo "============================================================"
echo "TRAINING: scale=340m_dh128  profile=$P  seed=$S  T/P~2.4"
echo "  expected wall: 5h on H100, 7-8h on A100-80, 12h on RTX 4090"
echo "============================================================"

mkdir -p results/dh128
python -u torchtitan_polar/structural_law_entry.py \
    --profile "$P" \
    --scale 340m_dh128 \
    --seed "$S" \
    --steps 200000 \
    --lr 0.012 \
    --dtype bf16 \
    --output_dir results/dh128 \
    --log_spectra_every 2000 \
    --eval_every 1000

echo "============================================================"
echo "DONE: results/dh128/340m_dh128_${P}_seed${S}_lr0.012/"
echo "Copy back: runpodctl send results/dh128 (then 'runpodctl receive <code>' on target)"
echo "Or: scp -r root@<this-pod-ip>:/workspace/structdion/results/dh128/* dest:/path/"
echo "============================================================"
