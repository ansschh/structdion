"""Parse phase 1 experiment logs and create visual comparison."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

LOG_DIR = "/home/torchtitan/logs/phase1"
OUTPUT = "/home/torchtitan/logs/phase1/phase1_comparison.png"

OPTIMIZERS = ["adamw", "muon", "dion", "dion2"]
COLORS = {"adamw": "#1f77b4", "muon": "#ff7f0e", "dion": "#d62728", "dion2": "#2ca02c"}
LABELS = {"adamw": "AdamW", "muon": "Muon", "dion": "Dion", "dion2": "Dion2"}

# Regex to parse log lines (strip ANSI codes first)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:\s*(\d+)\s+loss:\s*([\d.]+)\s+grad_norm:\s*([\d.]+)\s+memory:\s*([\d.]+)GiB.*?tps:\s*([\d,]+)\s+tflops:\s*([\d.]+)\s+mfu:\s*([\d.]+)%")


def parse_log(path):
    steps, losses, tps_list, mfu_list = [], [], [], []
    seen = set()
    with open(path) as f:
        for line in f:
            clean = ANSI_RE.sub("", line)
            m = STEP_RE.search(clean)
            if m:
                step = int(m.group(1))
                if step in seen:
                    continue  # skip duplicate rank logs
                seen.add(step)
                steps.append(step)
                losses.append(float(m.group(2)))
                tps_list.append(int(m.group(5).replace(",", "")))
                mfu_list.append(float(m.group(7)))
    return steps, losses, tps_list, mfu_list


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 1: Optimizer Comparison on LLaMA3 160M (2 GPUs, 500 steps)", fontsize=14, fontweight="bold")

all_data = {}
for opt in OPTIMIZERS:
    log_path = os.path.join(LOG_DIR, f"{opt}.log")
    if os.path.exists(log_path):
        all_data[opt] = parse_log(log_path)

# Plot 1: Training Loss
ax = axes[0, 0]
for opt in OPTIMIZERS:
    if opt in all_data:
        steps, losses, _, _ = all_data[opt]
        ax.plot(steps, losses, label=LABELS[opt], color=COLORS[opt], linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Training Loss (log scale)
ax = axes[0, 1]
for opt in OPTIMIZERS:
    if opt in all_data:
        steps, losses, _, _ = all_data[opt]
        ax.semilogy(steps, losses, label=LABELS[opt], color=COLORS[opt], linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Training Loss (Log Scale)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Tokens per Second (throughput)
ax = axes[1, 0]
for opt in OPTIMIZERS:
    if opt in all_data:
        steps, _, tps, _ = all_data[opt]
        ax.plot(steps, tps, label=LABELS[opt], color=COLORS[opt], linewidth=1.5, alpha=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Tokens/sec")
ax.set_title("Throughput (Tokens per Second)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Summary bar chart — final loss + avg throughput
ax = axes[1, 1]
x_pos = range(len(OPTIMIZERS))
final_losses = []
avg_tps = []
opt_labels = []
for opt in OPTIMIZERS:
    if opt in all_data:
        steps, losses, tps, mfu = all_data[opt]
        final_losses.append(losses[-1])
        avg_tps.append(sum(tps[1:]) / len(tps[1:]))  # skip first step (warmup)
        opt_labels.append(LABELS[opt])

ax2 = ax.twinx()
bars1 = ax.bar([x - 0.2 for x in x_pos[:len(opt_labels)]], final_losses, 0.35, label="Final Loss", color="#4a90d9", alpha=0.8)
bars2 = ax2.bar([x + 0.2 for x in x_pos[:len(opt_labels)]], [t / 1000 for t in avg_tps], 0.35, label="Avg TPS (k)", color="#e8834a", alpha=0.8)
ax.set_ylabel("Final Loss", color="#4a90d9")
ax2.set_ylabel("Avg Tokens/sec (k)", color="#e8834a")
ax.set_xticks(x_pos[:len(opt_labels)])
ax.set_xticklabels(opt_labels)
ax.set_title("Final Loss & Avg Throughput")

# Add value labels on bars
for bar, val in zip(bars1, final_losses):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color="#4a90d9")
for bar, val in zip(bars2, [t/1000 for t in avg_tps]):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f"{val:.1f}k", ha='center', va='bottom', fontsize=9, color="#e8834a")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT}")

# Print summary table
print("\n" + "=" * 70)
print(f"{'Optimizer':<12} {'Final Loss':>12} {'Avg TPS':>12} {'Avg MFU':>10}")
print("=" * 70)
for opt in OPTIMIZERS:
    if opt in all_data:
        steps, losses, tps, mfu = all_data[opt]
        avg_t = sum(tps[1:]) / len(tps[1:])
        avg_m = sum(mfu[1:]) / len(mfu[1:])
        print(f"{LABELS[opt]:<12} {losses[-1]:>12.4f} {avg_t:>12,.0f} {avg_m:>9.2f}%")
print("=" * 70)
