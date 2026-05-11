#!/usr/bin/env bash
# Full sweep: 3 profiles x 3 seeds, submitted as a SLURM array.
#
# Usage:
#   bash full.sh                          # default GRES, default partition
#   GRES=gpu:a100:1 bash full.sh          # request A100 specifically
#   PARTITION=gpu TIME=24:00:00 bash full.sh
#
# After tasks finish:
#   bash aggregate.sh

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-24:00:00}"
GRES="${GRES:-gpu:a100:1}"       # default to A100; override if your cluster uses a different label
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
STEPS="${STEPS:-830000}"
LOG_SPECTRA_EVERY="${LOG_SPECTRA_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/slurm_logs"
RESULTS_DIR="$REPO_DIR/results/struct_law"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

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
echo '--- GPU allocated for task \$IDX ---'
nvidia-smi
GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
case "\$GPU_NAME" in
    *P100*|*K80*|*M40*|*M60*)
        echo "[abort] Got an old GPU (\$GPU_NAME); cu121 torch needs >= Volta."
        exit 2
        ;;
esac

cd torchtitan_polar
python -u structural_law_entry.py \\
    --profile "\$PROFILE" \\
    --seed "\$SEED" \\
    --steps $STEPS \\
    --lr 0.012 \\
    --output_dir "$RESULTS_DIR" \\
    --log_spectra_every $LOG_SPECTRA_EVERY \\
    --eval_every $EVAL_EVERY
EOF

chmod +x "$LOG_DIR/array_dispatch.sh"
echo "[full] sbatch script: $LOG_DIR/array_dispatch.sh"
JOBID="$(sbatch "$LOG_DIR/array_dispatch.sh" | awk '{print $NF}')"
echo "[full] submitted job array: $JOBID"
echo "[full] watch: squeue -j $JOBID"
echo "[full] logs : $LOG_DIR/structdion_${JOBID}_*.out"
echo
echo "[full] after all 9 tasks finish:"
echo "  bash aggregate.sh"
