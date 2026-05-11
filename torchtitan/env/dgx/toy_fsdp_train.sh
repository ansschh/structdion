#!/usr/bin/env bash
# Toy FSDP training script to verify torchtitan setup inside the DGX container.
#
# Usage (inside the container):
#   source /home/torchtitan/venv/bin/activate
#   bash env/dgx/toy_fsdp_train.sh
#
# This runs the built-in llama3 debugmodel (tiny 6-layer, dim=256) on the
# bundled c4_test dataset for 10 steps with FSDP across all available GPUs.

set -euo pipefail

cd /home/torchtitan

echo "==> Verifying environment ..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "==> Running toy FSDP training with ${NGPU} GPUs ..."
echo "    Model: llama3 debugmodel (6 layers, dim=256, vocab=2048)"
echo "    Dataset: c4_test (bundled)"
echo "    Steps: 10"
echo ""

# Use torchtitan's standard training entry point with the debugmodel config.
# FSDP is enabled by default (data_parallel_shard_degree=-1).
NGPU="${NGPU}" MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
    --training.steps 10

echo ""
echo "==> Toy FSDP training completed successfully!"
