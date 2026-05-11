#!/usr/bin/env bash
# One-shot environment setup for the StructDion experiments on Caltech HPC.
# Run on the LOGIN node (no GPU needed for install).
#
# Idempotent: if torch / torchtitan / dion already import OK, setup is a no-op.
#
# Usage:
#   bash setup.sh            # smart: skip steps already done
#   FORCE=1 bash setup.sh    # nuke the venv and reinstall from scratch
#   RECHECK=1 bash setup.sh  # always run the import-verification step

set -euo pipefail

echo "============================================================"
echo "StructDion environment setup"
echo "============================================================"

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
echo "[info] repo dir: $REPO_DIR"

# ---------------------------------------------------------------------------
# Pick a Python 3.10+ module (spack hash suffix on Caltech HPC).
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
if [[ -z "$PY_MODULE" && -n "${PY_MODULE_OVERRIDE:-}" ]]; then
    PY_MODULE="$PY_MODULE_OVERRIDE"
fi
if [[ -z "$PY_MODULE" ]]; then
    echo "[error] no python/3.10+ module found. set PY_MODULE_OVERRIDE=..."
    exit 1
fi
echo "[ok] using module: $PY_MODULE"
module load "$PY_MODULE"

# ---------------------------------------------------------------------------
# venv: build only if needed
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
    echo "[step] creating venv at $REPO_DIR/venv"
    python3 -m venv venv
else
    echo "[ok] venv exists"
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "[ok] venv python: $(python --version)"

# ---------------------------------------------------------------------------
# Idempotent install: skip each step if the import already works.
# We do dion FIRST because it pulls torch>=2.7.1 (cu130) and would otherwise
# clobber a cu121 install. If dion is already installed, this step is a no-op.
# ---------------------------------------------------------------------------

check_import() {
    # Returns 0 if the import works, 1 otherwise
    python -c "$1" >/dev/null 2>&1
}

# Step 1: dion (also brings torch and triton transitively)
if check_import "import dion.dion"; then
    echo "[skip] dion already installed"
else
    echo "[install] microsoft/dion (transitively installs torch)"
    pip install --upgrade pip
    pip install "dion @ git+https://github.com/microsoft/dion.git"
fi

# Step 2: torch sanity (don't reinstall; just confirm it's there)
if check_import "import torch"; then
    TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
    echo "[ok] torch $TORCH_VER"
else
    echo "[install] torch (fallback; dion install should have brought it)"
    pip install torch torchvision
fi

# Step 3: bundled torchtitan
if check_import "from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion"; then
    echo "[skip] torchtitan already installed"
else
    if [[ ! -d torchtitan ]]; then
        echo "[error] torchtitan/ not in repo. run 'git pull' or re-clone."
        exit 1
    fi
    echo "[install] bundled torchtitan (editable)"
    pip install -e torchtitan
fi

# Step 4: utility deps
if ! check_import "import datasets, tokenizers, safetensors, matplotlib, numpy, tqdm"; then
    echo "[install] utility deps"
    pip install datasets tokenizers safetensors tqdm matplotlib numpy
else
    echo "[skip] utility deps already installed"
fi

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
echo "[verify] imports"
python - <<PY
import sys, os
sys.path.insert(0, os.path.join("$REPO_DIR", "torchtitan_polar"))
import torch
print(f"  torch        {torch.__version__}  cuda_available={torch.cuda.is_available()}")
from torchtitan.experiments.ortho_matrix.ada_dion.adadion import AdaDion
print("  AdaDion       OK")
from train_320m import LLaMA320M, get_c4_dataloader, get_lr, evaluate
print("  train_320m    OK")
import dion.dion as dd
print("  dion.dion     OK")
PY

echo
echo "============================================================"
echo "Setup complete (idempotent: rerunning is a no-op)."
echo
echo "For future shells:"
echo "  module load $PY_MODULE"
echo "  source $REPO_DIR/venv/bin/activate"
echo
echo "Next:"
echo "  bash smoke.sh"
echo "  bash full.sh"
echo "============================================================"
