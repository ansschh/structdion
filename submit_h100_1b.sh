#!/usr/bin/env bash
# Submit the 10 trailing 1B (profile, seed) cells to the H100 nodes on
# the gpu partition. Use this when the H200 pool is saturated and the
# remaining 1B tasks have been canceled or never started.
#
# Usage:
#   bash submit_h100_1b.sh
#
# By default this submits all of:
#   struct s2; inverted_matched s0,s1,s2; adamw s0,s1,s2; muon s0,s1,s2
# These map to indices 5..14 in the saturate.sh array. If you want a
# subset, pass index numbers as args:
#   bash submit_h100_1b.sh 6 7 8                # only inverted_matched seeds

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$REPO_DIR/slurm_logs"
mkdir -p "$LOGDIR"

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:h100:1}"
TIME="${TIME:-26:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"

PROFILES=(uniform_comm uniform_comm uniform_comm struct struct struct \
          inverted_matched inverted_matched inverted_matched \
          adamw adamw adamw muon muon muon)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)

# Default index list: 5..14 (the typical trailing-1B set).
if [[ $# -eq 0 ]]; then
    set -- 5 6 7 8 9 10 11 12 13 14
fi

echo "============================================================"
echo "Submitting 1B cells to $PARTITION ($GRES)"
echo "  helper script: $REPO_DIR/h100_1b_runner.sh"
echo "  log dir      : $LOGDIR"
echo "  indices      : $*"
echo "============================================================"

for IDX in "$@"; do
    if [[ -z "${PROFILES[$IDX]+x}" ]]; then
        echo "[skip] index $IDX out of range"; continue
    fi
    P="${PROFILES[$IDX]}"
    S="${SEEDS[$IDX]}"

    JOBID="$(sbatch --partition="$PARTITION" --gres="$GRES" \
                    --time="$TIME" \
                    --cpus-per-task="$CPUS" --mem="$MEM" \
                    --job-name="structdion_1b_h100_${P}_s${S}" \
                    --output="$LOGDIR/h100_1b_${P}_s${S}_%j.out" \
                    --error="$LOGDIR/h100_1b_${P}_s${S}_%j.err" \
                    "$REPO_DIR/h100_1b_runner.sh" "$P" "$S" \
                    | awk '{print $NF}')"
    echo "[submit] idx=$IDX profile=$P seed=$S  -> job $JOBID"
done

echo
echo "Watch:  squeue -u \$USER -o \"%i %P %j %b %T %r %M\""
