#!/usr/bin/env bash
# Structural rank-allocation experiment driver for the HiLD 2026 paper.
#
# Runs three profiles (struct, uniform_comm, inverted) at three seeds each
# on LLaMA320M. With 3.4B tokens per run (T/P = 10) this should fit comfortably
# in a 1-day window on 50 sustained H100s.
#
# Usage:
#   bash run_structural_law.sh [results_root]
#
# Defaults to results/struct_law if not given.

set -euo pipefail

RESULTS_ROOT="${1:-results/struct_law}"
mkdir -p "$RESULTS_ROOT"

PROFILES=(uniform_comm struct inverted)
SEEDS=(0 1 2)

# 3.4B tokens at batch_size=4, seq_len=1024 needs ~830k steps.
# We use the established AdaDion 320M settings: bs=4, seq=1024.
# Override STEPS via env if you want a smoke (e.g. STEPS=500).
STEPS="${STEPS:-830000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-1024}"
LR="${LR:-0.012}"
LOG_SPECTRA_EVERY="${LOG_SPECTRA_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"

echo "================================================================="
echo "Structural law experiments"
echo "  results root        : $RESULTS_ROOT"
echo "  steps               : $STEPS  (override with STEPS=N)"
echo "  batch_size, seq_len : $BATCH_SIZE, $SEQ_LEN"
echo "  lr                  : $LR"
echo "  spectra logged every: $LOG_SPECTRA_EVERY"
echo "  eval every          : $EVAL_EVERY"
echo "  profiles            : ${PROFILES[*]}"
echo "  seeds               : ${SEEDS[*]}"
echo "================================================================="

for PROFILE in "${PROFILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="${PROFILE}_seed${SEED}_lr${LR}"
        OUT="$RESULTS_ROOT/$TAG"
        if [[ -f "$OUT/result.json" ]]; then
            echo "[skip] $TAG already has result.json"
            continue
        fi
        echo
        echo "================================================================="
        echo "LAUNCH: profile=$PROFILE seed=$SEED"
        echo "================================================================="
        python -u structural_law_entry.py \
            --profile "$PROFILE" \
            --seed "$SEED" \
            --lr "$LR" \
            --steps "$STEPS" \
            --batch_size "$BATCH_SIZE" \
            --seq_len "$SEQ_LEN" \
            --output_dir "$RESULTS_ROOT" \
            --log_spectra_every "$LOG_SPECTRA_EVERY" \
            --eval_every "$EVAL_EVERY" \
            2>&1 | tee -a "$RESULTS_ROOT/${TAG}.log"
    done
done

echo
echo "================================================================="
echo "All runs complete."
echo "Aggregate with: python aggregate_structural.py $RESULTS_ROOT"
echo "================================================================="
