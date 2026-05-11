#!/usr/bin/env bash
# One-shot environment setup for the StructDion experiments on Caltech HPC.
# Run on the LOGIN node (no GPU needed for install).
#
# Usage:
#   bash setup.sh           # full setup
#   FORCE=1 bash setup.sh   # nuke existing venv and start clean

set -euo pipefail

echo "============================================================"
echo "StructDion environment setup"
echo "============================================================"

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
echo "[info] repo dir: $REPO_DIR"

# ---------------------------------------------------------------------------
# Pick a Python 3.10+ module. Caltech modules have spack hash suffixes,
# so we resolve the exact name from `module avail`.
# ---------------------------------------------------------------------------
echo "[step] resolving Python 3.10+ module"
PY_MODULE=""
for candidate in 3.11.6-gcc-13.2.0 3.11.6-gcc-11.3.1 3.10.12-gcc-13.2.0 3.10.12-gcc-11.3.1; do
    line="$(module avail "python/$candidate" 2>&1 | grep -oE "python/$candidate[^ ]*" | head -1 || true)"
    if [[ -n "$line" ]]; then
        PY_MODULE="$line"
        break
    fi
done

if [[ -z "$PY_MODULE" ]]; then
    echo "[error] could not find a python/3.10+ module via 'module avail'"
    echo "        run 'module avail python' manually and pass the name as PY_MODULE=..."
    if [[ -n "${PY_MODULE_OVERRIDE:-}" ]]; then
        PY_MODULE="$PY_MODULE_OVERRIDE"
    else
        exit 1
    fi
fi
echo "[ok] using module: $PY_MODULE"
module load "$PY_MODULE"

PY_BIN="$(command -v python3)"
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
echo "[ok] python: $PY_BIN  version: $PY_VER"

PY_MAJOR_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor:02d}")')"
if [[ "$PY_MAJOR_MINOR" -lt 310 ]]; then
    echo "[error] resolved python is $PY_VER, need >= 3.10"
    exit 1
fi

# ---------------------------------------------------------------------------
# venv: rebuild if FORCE=1 OR if the existing venv uses an older Python
# ---------------------------------------------------------------------------
NEED_REBUILD=0
if [[ "${FORCE:-0}" == "1" ]]; then
    NEED_REBUILD=1
elif [[ -d venv ]]; then
    EXIST_PY="$(./venv/bin/python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor:02d}")' 2>/dev/null || echo 0)"
    if [[ "$EXIST_PY" -lt 310 ]]; then
        echo "[step] existing venv uses Python <3.10; rebuilding"
        NEED_REBUILD=1
    fi
fi

if [[ "$NEED_REBUILD" == "1" ]]; then
    rm -rf venv
fi

if [[ ! -d venv ]]; then
    echo "[1/6] Creating venv at $REPO_DIR/venv with $PY_BIN"
    python3 -m venv venv
else
    echo "[1/6] venv exists, reusing"
fi
# shellcheck disable=SC1091
source venv/bin/activate

# Sanity: confirm venv python is 3.10+
VENV_PY="$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor:02d}")')"
if [[ "$VENV_PY" -lt 310 ]]; then
    echo "[error] venv python is $(python --version), still <3.10; deleting and re-running"
    deactivate || true
    rm -rf venv
    python3 -m venv venv
    source venv/bin/activate
fi
echo "[ok] venv python: $(python --version)"

# ---------------------------------------------------------------------------
# pip + torch + bundled torchtitan + microsoft dion + utility deps
# ---------------------------------------------------------------------------
echo "[2/6] Upgrading pip"
pip install --upgrade pip

echo "[3/6] Installing torch + torchvision (cu121)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] Installing bundled torchtitan"
if [[ ! -d torchtitan ]]; then
    echo "[error] torchtitan/ directory not found in repo"
    exit 1
fi
pip install -e torchtitan

echo "[5/6] Installing microsoft dion"
pip install "dion @ git+https://github.com/microsoft/dion.git"

echo "    + utility deps"
pip install datasets tokenizers safetensors tqdm matplotlib numpy

echo "[6/6] Verifying imports"
python - <<PY
import sys, os
sys.path.insert(0, os.path.join("$REPO_DIR", "torchtitan_polar"))
import torch
print(f"  torch        {torch.__version__}  cuda_available={torch.cuda.is_available()}")
from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion
print("  AdaDion       OK")
from train_320m import LLaMA320M, get_c4_dataloader, get_lr, evaluate
print("  train_320m    OK (LLaMA320M, get_c4_dataloader, get_lr, evaluate)")
import dion.dion as dd
print("  dion.dion     OK")
PY

echo
echo "============================================================"
echo "Setup complete."
echo
echo "Activate venv for future shells:"
echo "  module load $PY_MODULE"
echo "  source $REPO_DIR/venv/bin/activate"
echo
echo "Next:"
echo "  bash smoke.sh          # 500 steps, 3 profiles, fits in 30 min"
echo "  bash full.sh           # 3 profiles x 3 seeds, full 3.4B-token sweep"
echo "============================================================"
