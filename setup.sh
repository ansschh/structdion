#!/usr/bin/env bash
# One-shot environment setup for the StructDion experiments on Caltech HPC.
# Run on the LOGIN node.
#
# Idempotent: if torch / torchtitan / dion already import OK and torch is
# cu121-built (matching the cluster's CUDA 12.x driver), setup is a no-op.
#
# Usage:
#   bash setup.sh             # smart: skip steps already done
#   FORCE=1 bash setup.sh     # nuke the venv and reinstall from scratch
#   FORCE_TORCH=1 bash setup.sh   # force-reinstall torch with cu121

set -euo pipefail

echo "============================================================"
echo "StructDion environment setup"
echo "============================================================"

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
echo "[info] repo dir: $REPO_DIR"

# --------------------------------------------------------------------
# Pick a Python 3.10+ module (spack hash suffix).
# --------------------------------------------------------------------
echo "[step] resolving Python 3.10+ module"
PY_MODULE=""
for candidate in 3.11.6-gcc-13.2.0 3.11.6-gcc-11.3.1 3.10.12-gcc-13.2.0 3.10.12-gcc-11.3.1; do
    line="$(module avail "python/$candidate" 2>&1 | grep -oE "python/$candidate[^ ]*" | head -1 || true)"
    if [[ -n "$line" ]]; then
        PY_MODULE="$line"
        break
    fi
done
if [[ -z "$PY_MODULE" && -n "${PY_MODULE_OVERRIDE:-}" ]]; then
    PY_MODULE="$PY_MODULE_OVERRIDE"
fi
if [[ -z "$PY_MODULE" ]]; then
    echo "[error] no python/3.10+ module found"
    exit 1
fi
echo "[ok] using module: $PY_MODULE"
module load "$PY_MODULE"

# --------------------------------------------------------------------
# venv: build only if needed
# --------------------------------------------------------------------
NEED_REBUILD=0
if [[ "${FORCE:-0}" == "1" ]]; then
    NEED_REBUILD=1
elif [[ -d venv ]]; then
    EXIST_PY="$(./venv/bin/python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor:02d}")' 2>/dev/null || echo 0)"
    if [[ "$EXIST_PY" -lt 310 ]]; then NEED_REBUILD=1; fi
fi
if [[ "$NEED_REBUILD" == "1" ]]; then rm -rf venv; fi
if [[ ! -d venv ]]; then
    echo "[step] creating venv at $REPO_DIR/venv"
    python3 -m venv venv
else
    echo "[ok] venv exists"
fi
# shellcheck disable=SC1091
source venv/bin/activate
# Redirect caches to scratch BEFORE any install (home is small).
if [[ -f env_setup.sh ]]; then
    source env_setup.sh
fi
pip install --upgrade pip >/dev/null

# --------------------------------------------------------------------
# Step 1: Install dion WITHOUT deps so it doesn't drag in cu13 torch
# --------------------------------------------------------------------
check_import() { python -c "$1" >/dev/null 2>&1; }

if check_import "import dion.dion"; then
    echo "[skip] dion already installed"
else
    echo "[install] microsoft/dion (--no-deps so torch is not touched)"
    pip install --no-deps "dion @ git+https://github.com/microsoft/dion.git"
fi

# --------------------------------------------------------------------
# Step 2: torch with cu121 (matches Caltech HPC CUDA 12.x driver)
# Forces a downgrade if torch is currently cu130 (incompatible with driver)
# --------------------------------------------------------------------
TORCH_OK=0
if check_import "import torch"; then
    TV="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
    case "$TV" in
        *cu121*|*cu122*|*cu123*|*cu124*) TORCH_OK=1 ;;
    esac
fi
if [[ "${FORCE_TORCH:-0}" == "1" ]]; then TORCH_OK=0; fi

if [[ "$TORCH_OK" == "1" ]]; then
    echo "[ok] torch $TV (cu12-built)"
else
    if check_import "import torch"; then
        BAD="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
        echo "[step] torch is $BAD (incompatible with CUDA 12.x driver); reinstalling cu121"
        pip uninstall -y torch torchvision triton >/dev/null 2>&1 || true
    else
        echo "[step] installing torch cu121"
    fi
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"
fi

# --------------------------------------------------------------------
# Step 3: bundled torchtitan
# --------------------------------------------------------------------
if check_import "from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion"; then
    echo "[skip] torchtitan already installed"
else
    if [[ ! -d torchtitan ]]; then
        echo "[error] torchtitan/ not in repo"; exit 1
    fi
    echo "[install] bundled torchtitan (editable)"
    pip install -e torchtitan
fi

# --------------------------------------------------------------------
# Step 4: utility deps
# --------------------------------------------------------------------
if ! check_import "import datasets, tokenizers, transformers, safetensors, matplotlib, numpy, tqdm"; then
    echo "[install] utility deps"
    pip install datasets tokenizers transformers safetensors tqdm matplotlib numpy
else
    echo "[skip] utility deps already installed"
fi

# --------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------
echo "[verify] imports"
python - <<PY
import sys, os
sys.path.insert(0, os.path.join("$REPO_DIR", "torchtitan_polar"))
import torch
print(f"  torch        {torch.__version__}  cuda_built={torch.version.cuda}")
from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion
print("  AdaDion       OK")
from train_320m import LLaMA320M, get_c4_dataloader, get_lr, evaluate
print("  train_320m    OK")
import dion.dion as dd
print("  dion.dion     OK")
PY

echo
echo "============================================================"
echo "Setup complete."
echo
echo "For future shells:"
echo "  module load $PY_MODULE"
echo "  source $REPO_DIR/venv/bin/activate"
echo
echo "Next, IN ORDER:"
echo "  1) bash prepare_data.sh                                # tokenize C4 once on the login node"
echo "  2) GRES=gpu:nvidia_h200:1 PARTITION=beta bash smoke.sh # 500-step smoke on H200"
echo "  3) GRES=gpu:nvidia_h200:1 PARTITION=beta bash full.sh  # full 3-profile x 3-seed sweep"
echo "============================================================"
