#\!/bin/bash
# Wait for GPUs to be free, then launch the sweep
LOG="/home/torchtitan/logs/sweep_shampoo_muon_20260312/wait.log"
mkdir -p /home/torchtitan/logs/sweep_shampoo_muon_20260312

echo "[$(date)] Waiting for GPUs to become available..." | tee -a "$LOG"

while true; do
    # Check if any torchrun process is running
    if \! pgrep -f "torchrun.*torchtitan.train" > /dev/null 2>&1; then
        # Double check GPU utilization
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk "{s+=\$1} END {print s}")
        if [ "$GPU_UTIL" -lt 50 ]; then
            echo "[$(date)] GPUs are free (total util: ${GPU_UTIL}%). Launching sweep." | tee -a "$LOG"
            break
        fi
    fi
    echo "[$(date)] GPUs still busy. Checking again in 5 minutes..." | tee -a "$LOG"
    sleep 300
done

cd /home/torchtitan
exec bash env/dgx/sweep_320m_shampoo_muon_20260312.sh
