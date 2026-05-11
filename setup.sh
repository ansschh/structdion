#!/usr/bin/env bash
# One-shot environment setup for the StructDion experiments on Caltech HPC.
# Run on the LOGIN node (no GPU needed for install).
#
# Usage:
#   bash setup.sh
#
# What this does:
#   1. Loads python/3.11.6
#   2. Creates a venv at ./venv (in the current working dir)
#   3. pip installs torch (cu121), torchtitan (Pro-Place fork, Tatzhiro/replication),
#      microsoft dion, and utility deps
#   4. Verifies that the four critical imports work

set -euo pipefail

echo "============================================================"
echo "StructDion environment setup"
echo "============================================================"

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
echo "[info] repo dir: $REPO_DIR"

# Load python module
module load python/3.11.6-gcc-13.2.0 || {
    echo "[warn] could not module-load python/3.11.6; trying python3 directly"
}

# Create venv
if [[ ! -d venv ]]; then
    echo "[1/6] Creating venv at $REPO_DIR/venv"
    python -m venv venv
else
    echo "[1/6] venv already exists, reusing"
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "[2/6] Upgrading pip"
pip install --upgrade pip

echo "[3/6] Installing torch + torchvision (cu121)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] Cloning + installing torchtitan (Pro-Place fork, Tatzhiro/replication)"
if [[ ! -d torchtitan ]]; then
    git clone --branch Tatzhiro/replication --depth 1 \
        https://github.com/Pro-Place/torchtitan.git
fi
pip install -e torchtitan

echo "[5/6] Installing microsoft dion"
pip install git+https://github.com/microsoft/dion.git

echo "    + utility deps"
pip install datasets tokenizers safetensors tqdm matplotlib numpy

echo "[6/6] Verifying imports"
python - <<'PY'
import torch
print(f"  torch        {torch.__version__}  cuda_available={torch.cuda.is_available()}")
from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion
print("  AdaDion       OK")
from torchtitan.experiments.ortho_matrix.ada_dion.train_320m import LLaMA320M, get_c4_dataloader, get_lr, evaluate
print("  train_320m    OK (LLaMA320M, get_c4_dataloader, get_lr, evaluate)")
import dion.dion as dd
print("  dion.dion     OK")
PY

echo
echo "============================================================"
echo "Setup complete."
echo
echo "Next: grab a GPU node and run the smoke test."
echo "  bash smoke.sh          # 500 steps, 3 profiles, fits in 30 min"
echo "  bash full.sh           # 3 profiles x 3 seeds, full 3.4B-token sweep"
echo "============================================================"
