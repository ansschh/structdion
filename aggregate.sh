#!/usr/bin/env bash
# Aggregate results from the full sweep into paper-ready table + figures.
#
# Usage:
#   bash aggregate.sh [results_root]
#
# Defaults to results/struct_law.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${1:-$REPO_DIR/results/struct_law}"

cd "$REPO_DIR"
source venv/bin/activate

python torchtitan_polar/aggregate_structural.py "$RESULTS_DIR"

echo
echo "==================================================================="
echo "Aggregation complete: $RESULTS_DIR"
echo "  table_main.txt       paired-seed validation-loss table"
echo "  fig_capture.pdf      Theorem 1 diagnostic (Ky-Fan + Frobenius)"
echo "  fig_trajectory.pdf   per-profile val-loss trajectories"
echo "  summary.json         machine-readable"
echo "==================================================================="
cat "$RESULTS_DIR/table_main.txt"
