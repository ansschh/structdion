#!/bin/bash
# Muon + Adam Grafting hyperparameter sweep — one param at a time
# Baseline: lr=0.006, scalar_lr=0.008, graft_beta2=0.95, weight_decay=0.1
# WandB: llm_jp_pp/grafted_muon_v1

set -e

NGPU=8
STEPS=6104
MODULE=muon_grafted
CONFIG=llama3_320m_muon_grafted
DATASET=c4_local
VAL_DATASET=c4_local_validation
TOKENIZER=./tokenizers/llama-3.1-8b
WANDB_TEAM=llm_jp_pp
WANDB_PROJECT=grafted_muon_v1
DATE=$(date +%Y%m%d_%H%M)

# Baseline values
BASE_LR=0.006
BASE_SLR=0.008
BASE_GB2=0.95
BASE_WD=0.1

run_experiment() {
    local lr=$1
    local scalar_lr=$2
    local graft_beta2=$3
    local weight_decay=$4

    # Build run name
    local run_name="muon_grafted_lr${lr}_slr${scalar_lr}_gb2${graft_beta2}_wd${weight_decay}_${DATE}"
    local log_dir="./logs/${run_name}"

    echo "=========================================="
    echo "  Running: ${run_name}"
    echo "  lr=${lr} scalar_lr=${scalar_lr} graft_beta2=${graft_beta2} weight_decay=${weight_decay}"
    echo "=========================================="

    mkdir -p ${log_dir}/outputs

    WANDB_TEAM=${WANDB_TEAM} \
    WANDB_PROJECT=${WANDB_PROJECT} \
    WANDB_RUN_NAME=${run_name} \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    CC=gcc CXX=g++ \
    NGPU=${NGPU} torchrun \
        --nproc_per_node=${NGPU} \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:0" \
        --local-ranks-filter 0 \
        --role rank \
        --tee 3 \
        -m torchtitan.train \
        --module ${MODULE} \
        --config ${CONFIG} \
        --training.steps ${STEPS} \
        --training.local-batch-size 32 \
        --training.global-batch-size 256 \
        --dataloader.dataset ${DATASET} \
        --hf-assets-path ${TOKENIZER} \
        --dump-folder ${log_dir}/outputs \
        --optimizer.lr ${lr} \
        --optimizer.scalar_lr ${scalar_lr} \
        --optimizer.graft_beta2 ${graft_beta2} \
        --optimizer.weight_decay ${weight_decay} \
        --validator.enable \
        --validator.freq 100 \
        --validator.steps 20 \
        --validator.dataloader.dataset ${VAL_DATASET} \
        --compile.enable \
        --metrics.enable-wandb \
        --metrics.enable-tensorboard \
        --parallelism.data-parallel-shard-degree ${NGPU} \
        --parallelism.data-parallel-replicate-degree 1 \
        2>&1 | tee ${log_dir}/train.log

    echo "  Done: ${run_name}"
    echo ""
}

# Baseline already done: lr=0.006, slr=0.008, gb2=0.95, wd=0.1

echo "=== Individual parameter sweep (baseline: lr=${BASE_LR}, slr=${BASE_SLR}, gb2=${BASE_GB2}, wd=${BASE_WD}) ==="
echo ""

# 1. Sweep scalar_lr (baseline lr, gb2, wd)
echo "--- Sweep: scalar_lr ---"
for slr in 0.012 0.016 0.020; do
    run_experiment ${BASE_LR} ${slr} ${BASE_GB2} ${BASE_WD}
done

# 2. Sweep graft_beta2 (baseline lr, slr, wd)
echo "--- Sweep: graft_beta2 ---"
for gb2 in 0.99; do
    run_experiment ${BASE_LR} ${BASE_SLR} ${gb2} ${BASE_WD}
done

# 3. Sweep weight_decay (baseline lr, slr, gb2)
echo "--- Sweep: weight_decay ---"
for wd in 0.05; do
    run_experiment ${BASE_LR} ${BASE_SLR} ${BASE_GB2} ${wd}
done

echo "Sweep complete! (4 runs)"
