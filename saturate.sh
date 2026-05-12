#!/usr/bin/env bash
# Submit the FULL multi-scale sweep to SLURM, saturating Caltech beta H200s.
#
# Submits THREE sbatch arrays:
#   340M: 15 tasks  (5 profiles x 3 seeds), 1 GPU each, T/P=10 (3.4B tokens)
#   700M: 15 tasks  (5 profiles x 3 seeds), 1 GPU each, T/P=5  (3.5B tokens)
#   1B:    15 tasks (5 profiles x 3 seeds), 4 GPUs (FSDP) each, T/P=3 (3B tokens)
#
# Total: 45 jobs. Each lands on Caltech beta as H200 nodes free up.
#
# Usage:
#   bash saturate.sh                                         # default beta partition
#   PARTITION=beta TIME=24:00:00 bash saturate.sh
#   SCALES="340m 700m" bash saturate.sh                      # subset only

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/slurm_logs"
RESULTS_DIR="$REPO_DIR/results/struct_law_full"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

PARTITION="${PARTITION:-beta}"
GRES_TYPE="${GRES_TYPE:-nvidia_h200}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
SEEDS=(0 1 2)
PROFILES=(uniform_comm struct inverted_matched adamw muon)
SCALES_STR="${SCALES:-340m 700m 1b}"
read -r -a SCALES <<< "$SCALES_STR"

# Per-scale resource plan. With FAST=1 (default for the 6-hour deadline mode):
#   340M:  1 GPU,         T/P ~2.4,  200k steps, ~4h
#   700M:  4 GPUs FSDP,   T/P ~1.1,  200k steps, ~4h
#   1B:    4 GPUs FSDP,   T/P ~0.8,  200k steps, ~5 to 6h
# With FAST=0 you get the full T/P=10 / T/P=5 / T/P=3 budget instead.
if [[ "${FAST:-1}" == "1" ]]; then
    declare -A SCALE_GPUS=(  [340m]=1         [700m]=4         [1b]=4 )
    declare -A SCALE_TIME=(  [340m]=06:00:00  [700m]=06:00:00  [1b]=06:30:00 )
    declare -A SCALE_STEPS=( [340m]=200000    [700m]=200000    [1b]=200000 )
else
    declare -A SCALE_GPUS=(  [340m]=1         [700m]=1         [1b]=4 )
    declare -A SCALE_TIME=(  [340m]=12:00:00  [700m]=24:00:00  [1b]=12:00:00 )
    declare -A SCALE_STEPS=( [340m]=830000    [700m]=860000    [1b]=730000 )
fi

echo "============================================================"
echo "StructDion full sweep"
echo "  partition      : $PARTITION"
echo "  gres type      : $GRES_TYPE"
echo "  profiles       : ${PROFILES[*]}"
echo "  seeds          : ${SEEDS[*]}"
echo "  scales         : ${SCALES[*]}"
echo "  results        : $RESULTS_DIR"
echo "  slurm logs     : $LOG_DIR"
echo "============================================================"

submit_scale() {
    local scale="$1"
    local gpus="${SCALE_GPUS[$scale]}"
    local timelimit="${SCALE_TIME[$scale]}"
    local steps="${SCALE_STEPS[$scale]}"
    local n_tasks=$(( ${#PROFILES[@]} * ${#SEEDS[@]} ))
    local last=$(( n_tasks - 1 ))
    local script="$LOG_DIR/array_${scale}.sh"

    {
        echo "#!/usr/bin/env bash"
        echo "#SBATCH --job-name=structdion_${scale}"
        echo "#SBATCH --partition=$PARTITION"
        echo "#SBATCH --gres=gpu:${GRES_TYPE}:${gpus}"
        echo "#SBATCH --cpus-per-task=${CPUS}"
        echo "#SBATCH --mem=${MEM}"
        echo "#SBATCH --time=${timelimit}"
        echo "#SBATCH --output=${LOG_DIR}/structdion_${scale}_%A_%a.out"
        echo "#SBATCH --error=${LOG_DIR}/structdion_${scale}_%A_%a.err"
        echo "#SBATCH --array=0-${last}"
        echo ""
        echo "set -euo pipefail"
        echo ""
        echo "PROFILES=( ${PROFILES[*]} )"
        echo "SEEDS=( ${SEEDS[*]} )"
        echo "IDX=\$SLURM_ARRAY_TASK_ID"
        echo "N_SEEDS=\${#SEEDS[@]}"
        echo "P=\$(( IDX / N_SEEDS ))"
        echo "S=\$(( IDX % N_SEEDS ))"
        echo "PROFILE=\${PROFILES[\$P]}"
        echo "SEED=\${SEEDS[\$S]}"
        echo ""
        echo "cd \"$REPO_DIR\""
        echo "source venv/bin/activate"
        echo "source env_setup.sh"
        echo "echo \"--- GPU(s) allocated for $scale profile=\$PROFILE seed=\$SEED ---\""
        echo "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        echo ""
        if [[ "$gpus" -gt 1 ]]; then
            echo "torchrun --standalone --nproc_per_node=$gpus \\"
            echo "    torchtitan_polar/structural_law_entry.py \\"
        else
            echo "python -u torchtitan_polar/structural_law_entry.py \\"
        fi
        echo "    --profile \"\$PROFILE\" \\"
        echo "    --scale \"$scale\" \\"
        echo "    --seed \"\$SEED\" \\"
        echo "    --steps $steps \\"
        echo "    --lr 0.012 \\"
        echo "    --dtype bf16 \\"
        echo "    --output_dir \"$RESULTS_DIR\" \\"
        echo "    --log_spectra_every 2000 \\"
        echo "    --eval_every 1000"
    } > "$script"
    chmod +x "$script"

    local jobid
    jobid="$(sbatch "$script" | awk '{print $NF}')"
    echo "[submit] scale=$scale  array $jobid  ($n_tasks tasks, ${gpus} GPU each, time=$timelimit)"
}

for scale in "${SCALES[@]}"; do
    if [[ -z "${SCALE_GPUS[$scale]:-}" ]]; then
        echo "[error] unknown scale '$scale'"; exit 1
    fi
    submit_scale "$scale"
done

echo
echo "============================================================"
echo "All arrays submitted. Watch progress with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOG_DIR/structdion_*_*.out"
echo
echo "After all jobs finish:"
echo "  python torchtitan_polar/aggregate_structural.py $RESULTS_DIR"
echo "============================================================"
