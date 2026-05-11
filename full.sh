#!/usr/bin/env bash
# Full sweep: 3 profiles x 3 seeds, 3.4B tokens each, submitted as a single
# array job. Each task gets one GPU.
#
# Usage:
#   bash full.sh                  # submits the sbatch array
#   bash full.sh --watch          # submits then tails the array progress
#
# Override SLURM settings via env vars:
#   PARTITION=gpu  TIME=24:00:00  GRES=gpu:1  bash full.sh
#
# After all tasks finish, aggregate with:
#   bash aggregate.sh

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-24:00:00}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
STEPS="${STEPS:-830000}"   # 3.4B tokens at bs=4, seq=1024
LOG_SPECTRA_EVERY="${LOG_SPECTRA_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/slurm_logs"
RESULTS_DIR="$REPO_DIR/results/struct_law"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Build the array task list: (profile, seed) pairs
# Array index 0..8 maps to:
#   0,1,2: uniform_comm seeds 0,1,2
#   3,4,5: struct       seeds 0,1,2
#   6,7,8: inverted     seeds 0,1,2
PROFILES=(uniform_comm uniform_comm uniform_comm struct struct struct inverted inverted inverted)
SEEDS=(0 1 2 0 1 2 0 1 2)

cat > "$LOG_DIR/array_dispatch.sh" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=structdion
#SBATCH --partition=$PARTITION
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/structdion_%A_%a.out
#SBATCH --error=$LOG_DIR/structdion_%A_%a.err
#SBATCH --array=0-8

set -euo pipefail
PROFILES=(${PROFILES[*]})
SEEDS=(${SEEDS[*]})
IDX=\$SLURM_ARRAY_TASK_ID
PROFILE=\${PROFILES[\$IDX]}
SEED=\${SEEDS[\$IDX]}

cd "$REPO_DIR"
source venv/bin/activate
nvidia-smi

cd torchtitan_polar
python -u structural_law_entry.py \
    --profile "\$PROFILE" \
    --seed "\$SEED" \
    --steps $STEPS \
    --lr 0.012 \
    --output_dir "$RESULTS_DIR" \
    --log_spectra_every $LOG_SPECTRA_EVERY \
    --eval_every $EVAL_EVERY
EOF

chmod +x "$LOG_DIR/array_dispatch.sh"
echo "[full] sbatch script written to $LOG_DIR/array_dispatch.sh"
JOBID="$(sbatch "$LOG_DIR/array_dispatch.sh" | awk '{print $NF}')"
echo "[full] submitted job array: $JOBID"
echo "[full] watch with: squeue -j $JOBID"
echo "[full] logs: $LOG_DIR/structdion_${JOBID}_*.out"
echo
echo "[full] after all 9 tasks finish, aggregate:"
echo "  bash aggregate.sh"

if [[ "${1:-}" == "--watch" ]]; then
    echo "[full] watching..."
    watch -n 30 "squeue -j $JOBID"
fi
