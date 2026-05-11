#!/usr/bin/env bash
# Setup script for torchtitan on RunPod instances.
# RunPod already provides a GPU-enabled environment, so no Docker needed.
# This script installs system deps, creates a venv, and installs all packages.
#
# Usage:
#   cd <torchtitan-repo-root>
#   bash env/runpod/setup.sh
#
# After setup:
#   source venv/bin/activate

set -euo pipefail

# ---------- Configuration ----------
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_VERSION="3.12"

echo "==> Repo root: ${REPO_DIR}"
cd "${REPO_DIR}"

# ---------- System packages ----------
echo "==> Installing system packages ..."
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
    vim \
    tmux \
    htop \
    build-essential \
    python3-pip \
    ripgrep \
    fd-find \
    tree \
    unzip \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    > /dev/null 2>&1

# Install Python 3.12 if not already available
if ! command -v "python${PYTHON_VERSION}" &> /dev/null; then
    echo "==> Installing Python ${PYTHON_VERSION} ..."
    add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-venv" \
        "python${PYTHON_VERSION}-dev" \
        > /dev/null 2>&1
else
    echo "==> Python ${PYTHON_VERSION} already installed"
fi

echo "  system packages OK"

# ---------- Virtual environment ----------
echo "==> Creating venv and installing packages ..."

"python${PYTHON_VERSION}" -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel -q

# Detect CUDA version for PyTorch index URL
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
case "${CUDA_VERSION}" in
    12.8*) CUDA_TAG="cu128" ;;
    12.6*) CUDA_TAG="cu126" ;;
    12.4*) CUDA_TAG="cu124" ;;
    12.1*) CUDA_TAG="cu121" ;;
    11.8*) CUDA_TAG="cu118" ;;
    *)     CUDA_TAG="cu126" ; echo "  Warning: could not detect CUDA version (got '${CUDA_VERSION}'), defaulting to ${CUDA_TAG}" ;;
esac

# cu128 (Blackwell / RTX 5090) requires nightly builds
if [ "${CUDA_TAG}" = "cu128" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/nightly/${CUDA_TAG}"
    echo "==> Installing PyTorch nightly with ${CUDA_TAG} (Blackwell GPU support) ..."
    pip install --force-reinstall torch torchvision torchaudio --index-url "${TORCH_INDEX}" -q
else
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
    echo "==> Installing PyTorch with ${CUDA_TAG} ..."
    pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}" -q
fi

# Install torchtitan in editable mode (pulls in all deps from pyproject.toml)
pip install -e . -q

# Install requirements.txt explicitly as well (overlaps, but ensures everything)
pip install -r requirements.txt -q

# Install dion (microsoft/dion) for ada_dion experiment
pip install git+https://github.com/microsoft/dion.git -q

echo "  venv + packages OK"
echo "  Python: $(python --version)"
TORCH_VER=$(python -c 'import torch; print(torch.__version__)')
CUDA_AVAIL=$(python -c 'import torch; print(torch.cuda.is_available())')
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "  PyTorch: ${TORCH_VER}"
echo "  CUDA available: ${CUDA_AVAIL}"
echo "  GPU count: ${GPU_COUNT}"

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo " Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo " Quick test (debug model, single GPU):"
echo "   NGPU=1 ./run_train.sh"
echo ""
echo " Full test (debug model, all GPUs):"
echo "   ./run_train.sh"
echo ""
