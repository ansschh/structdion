#!/usr/bin/env bash
# Setup script for torchtitan on DGX systems.
# Creates a Docker container with all dependencies and a ready-to-use venv.
#
# Usage:
#   cd <torchtitan-repo-root>
#   bash env/dgx/setup.sh
#
# After setup, attach to the container:
#   docker exec -it torchtitan bash
#   source /home/torchtitan/venv/bin/activate

set -euo pipefail

# ---------- Configuration ----------
CONTAINER_NAME="torchtitan"
IMAGE="nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04"
MEMORY="128g"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"   # torchtitan repo root
PYTHON_VERSION="3.12"

echo "==> Repo root: ${REPO_DIR}"

# ---------- Pull base image ----------
echo "==> Pulling base image ${IMAGE} ..."
docker pull "${IMAGE}"

# ---------- Stop old container if exists ----------
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "==> Removing existing container '${CONTAINER_NAME}' ..."
    docker rm -f "${CONTAINER_NAME}"
fi

# ---------- Create container ----------
echo "==> Creating container '${CONTAINER_NAME}' ..."
docker run -dit \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --memory="${MEMORY}" \
    --shm-size=64g \
    -v "${REPO_DIR}":/home/torchtitan \
    -w /home/torchtitan \
    "${IMAGE}" \
    bash

# ---------- In-container setup ----------
echo "==> Installing system packages inside container ..."
docker exec "${CONTAINER_NAME}" bash -c '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    > /dev/null 2>&1

# Install Python 3.12
add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
apt-get update -qq
apt-get install -y -qq \
    python'"${PYTHON_VERSION}"' \
    python'"${PYTHON_VERSION}"'-venv \
    python'"${PYTHON_VERSION}"'-dev \
    > /dev/null 2>&1

echo "  system packages OK"
'

echo "==> Creating venv and installing packages ..."
docker exec "${CONTAINER_NAME}" bash -c '
set -euo pipefail

cd /home/torchtitan

# Create virtual environment
python'"${PYTHON_VERSION}"' -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel -q

# Install PyTorch 2.10.0 stable with CUDA 12.6
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -q

# Install torchtitan in editable mode (pulls in all deps from pyproject.toml)
pip install -e . -q

# Install requirements.txt explicitly as well (overlaps, but ensures everything)
pip install -r requirements.txt -q

# Install dion (microsoft/dion) for ada_dion experiment
pip install git+https://github.com/microsoft/dion.git -q

echo "  venv + packages OK"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c "import torch; print(torch.__version__)")"
echo "  CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "  GPU count: $(python -c "import torch; print(torch.cuda.device_count())")"
'

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo " To use the container:"
echo "   docker exec -it ${CONTAINER_NAME} bash"
echo "   source /home/torchtitan/venv/bin/activate"
echo ""
echo " Quick test (debug model, single GPU):"
echo "   NGPU=1 ./run_train.sh"
echo ""
echo " Full test (debug model, all GPUs):"
echo "   ./run_train.sh"
echo ""
