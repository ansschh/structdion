#!/usr/bin/env bash
# Centralised env exports for StructDion on Caltech HPC.
# Source this AFTER activating the venv. It redirects every disk-heavy
# cache (triton, torch, huggingface, pip) from $HOME (small quota) to
# the scratch filesystem.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"

# Triton kernel cache (torch.compile / Muon hit this hard)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_DIR/.cache/triton}"

# Inductor / dynamo caches
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$REPO_DIR/.cache/inductor}"
export TORCH_HOME="${TORCH_HOME:-$REPO_DIR/.cache/torch}"

# Huggingface caches (tokenizer, datasets, hub)
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# pip cache (in case we install anything during runs)
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_DIR/.cache/pip}"

# SSL bundle (Caltech login/compute nodes lack a CA chain for hf.co)
if command -v python >/dev/null 2>&1; then
    CERTIFI_PATH="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
    if [[ -n "$CERTIFI_PATH" ]]; then
        export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERTIFI_PATH}"
        export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERTIFI_PATH}"
        export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$CERTIFI_PATH}"
    fi
fi

# C4 token cache (used by train_320m.get_c4_dataloader)
export C4_CACHE="${C4_CACHE:-$REPO_DIR/data/c4_tokens.pt}"

mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME" \
         "$HF_HOME" "$PIP_CACHE_DIR" 2>/dev/null || true
