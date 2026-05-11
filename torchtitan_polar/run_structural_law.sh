#!/usr/bin/env bash
# Sequential driver: run a fixed list of (scale, profile, seed) cells.
# Honored env vars:
#   STEPS              steps per run (default: 500 for smoke)
#   LOG_SPECTRA_EVERY  default 100
#   EVAL_EVERY         default 100
#   BATCH_SIZE         default 4
#   SEQ_LEN            default 1024
#   LR                 default 0.012
#   SCALE              default 340m
#   PROFILES           default "uniform_comm struct inverted_matched"
#   SEEDS              default "0 1 2"
#   NPROC              default 1; if >1 we use torchrun
#
# Usage:
#   bash run_structural_law.sh [results_root]

set -euo pipefail

RESULTS_ROOT="${1:-results/struct_law}"
mkdir -p "$RESULTS_ROOT"

PROFILES_STR="${PROFILES:-uniform_comm struct inverted_matched}"
SEEDS_STR="${SEEDS:-0 1 2}"
SCALE="${SCALE:-340m}"

STEPS="${STEPS:-500}"
LOG_SPECTRA_EVERY="${LOG_SPECTRA_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-1024}"
LR="${LR:-0.012}"
NPROC="${NPROC:-1}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export C4_CACHE="${C4_CACHE:-$REPO_DIR/data/c4_tokens.pt}"

read -r -a PROFILES <<< "$PROFILES_STR"
read -r -a SEEDS <<< "$SEEDS_STR"

echo "================================================================="
echo "Structural law sequential driver"
echo "  results root        : $RESULTS_ROOT"
echo "  scale               : $SCALE"
echo "  profiles            : ${PROFILES[*]}"
echo "  seeds               : ${SEEDS[*]}"
echo "  steps               : $STEPS"
echo "  nproc per run       : $NPROC"
echo "================================================================="

cd "$REPO_DIR"

for PROFILE in "${PROFILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="${SCALE}_${PROFILE}_seed${SEED}_lr${LR}"
        OUT="$RESULTS_ROOT/$TAG"
        if [[ -f "$OUT/result.json" ]]; then
            echo "[skip] $TAG already has result.json"
            continue
        fi
        echo
        echo "================================================================="
        echo "LAUNCH: scale=$SCALE profile=$PROFILE seed=$SEED nproc=$NPROC"
        echo "================================================================="
        if [[ "$NPROC" -gt 1 ]]; then
            torchrun --standalone --nproc_per_node="$NPROC" \
                torchtitan_polar/structural_law_entry.py \
                --profile "$PROFILE" \
                --scale "$SCALE" \
                --seed "$SEED" \
                --lr "$LR" \
                --steps "$STEPS" \
                --batch_size "$BATCH_SIZE" \
                --seq_len "$SEQ_LEN" \
                --output_dir "$RESULTS_ROOT" \
                --log_spectra_every "$LOG_SPECTRA_EVERY" \
                --eval_every "$EVAL_EVERY"
        else
            python -u torchtitan_polar/structural_law_entry.py \
                --profile "$PROFILE" \
                --scale "$SCALE" \
                --seed "$SEED" \
                --lr "$LR" \
                --steps "$STEPS" \
                --batch_size "$BATCH_SIZE" \
                --seq_len "$SEQ_LEN" \
                --output_dir "$RESULTS_ROOT" \
                --log_spectra_every "$LOG_SPECTRA_EVERY" \
                --eval_every "$EVAL_EVERY"
        fi
    done
done

echo
echo "================================================================="
echo "Driver complete. Aggregate with:"
echo "  python torchtitan_polar/aggregate_structural.py $RESULTS_ROOT"
echo "================================================================="
