#!/usr/bin/env bash
# Single-pod cloud launcher for the 340M d_h=128 perturbation experiment.
# Run this on each cloud pod with a different INSTANCE_IDX (0..8).
#
# Cell mapping (9 cells total = 3 profiles x 3 seeds at 340m_dh128):
#   IDX  profile           seed
#    0   uniform_comm        0
#    1   uniform_comm        1
#    2   uniform_comm        2
#    3   struct              0
#    4   struct              1
#    5   struct              2
#    6   inverted_matched    0
#    7   inverted_matched    1
#    8   inverted_matched    2
#
# Usage on a fresh pod (RunPod / Vast.ai / Lambda):
#   git clone https://github.com/ansschh/structdion.git
#   cd structdion
#   bash setup.sh
#   bash prepare_data.sh
#   INSTANCE_IDX=0 bash cloud_dh128.sh    # change IDX per pod, 0..8

set -euo pipefail

IDX="${INSTANCE_IDX:-${1:-}}"
if [[ -z "$IDX" || "$IDX" -lt 0 || "$IDX" -gt 8 ]]; then
    echo "ERROR: set INSTANCE_IDX=0..8 (or pass as arg)"; exit 1
fi

PROFILES=(uniform_comm uniform_comm uniform_comm \
          struct struct struct \
          inverted_matched inverted_matched inverted_matched)
SEEDS=(0 1 2 0 1 2 0 1 2)

P=${PROFILES[$IDX]}
S=${SEEDS[$IDX]}

echo "============================================================"
echo "Cloud d_h=128 cell  IDX=$IDX  profile=$P  seed=$S"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
source venv/bin/activate
source env_setup.sh

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
echo "DONE: $P seed $S written to results/dh128/340m_dh128_${P}_seed${S}_lr0.012/"
echo "scp this dir back to your Caltech results/struct_law_full to aggregate."
echo "============================================================"
