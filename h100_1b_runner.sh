#!/usr/bin/env bash
# Helper runner for a single 1B job on a specific (profile, seed) cell.
# Intended to be invoked from sbatch:
#   sbatch ... h100_1b_runner.sh <profile> <seed>
#
# Hardcoded to scale=1b, T/P=2 (500k steps), lr=0.012; output goes to
# results/struct_law_full. Reuse for any GRES (H100 or H200), it doesn't
# care which GPU it lands on as long as cu121 torch is happy.

set -euo pipefail

PROFILE="${1:?missing profile arg}"
SEED="${2:?missing seed arg}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Activate env
PY_MODULE="${PY_MODULE_OVERRIDE:-}"
if [[ -z "$PY_MODULE" ]]; then
    for cand in 3.11.6-gcc-13.2.0 3.11.6-gcc-11.3.1 3.10.12-gcc-13.2.0; do
        line="$(module avail "python/$cand" 2>&1 | grep -oE "python/$cand[^ ]*" | head -1 || true)"
        [[ -n "$line" ]] && PY_MODULE="$line" && break
    done
fi
if [[ -n "$PY_MODULE" ]]; then
    module load "$PY_MODULE"
fi

# shellcheck disable=SC1091
source venv/bin/activate
# shellcheck disable=SC1091
source env_setup.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u torchtitan_polar/structural_law_entry.py \
    --profile "$PROFILE" --scale 1b --seed "$SEED" \
    --steps 730000 --lr 0.012 --dtype bf16 \
    --output_dir results/struct_law_full \
    --log_spectra_every 2000 --eval_every 1000
