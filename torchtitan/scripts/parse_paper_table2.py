"""Parse Paper Table 2 experiment logs and generate a publication-quality table.

Reads validation loss from experiment logs, computes mean +/- std across seeds,
and produces:
  1. A matplotlib table image (matching the paper's Table 2 visual style)
  2. LaTeX table source
  3. CSV file with raw per-seed data

Usage:
    python scripts/parse_paper_table2.py
    python scripts/parse_paper_table2.py --batch-size 64
    python scripts/parse_paper_table2.py --batch-size 256 --batch-size 64
    python scripts/parse_paper_table2.py --log-dir ./logs/paper_table2
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import Table


# ============================================================
# Log parsing
# ============================================================

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
VALIDATE_RE = re.compile(
    r"validate step:\s*(\d+)\s+loss:\s*([\d.]+)"
)

OPTIMIZERS = ["adamw", "muon"]
LABELS = {"adamw": "AdamW", "muon": "Muon"}


def parse_final_val_loss(log_path: str) -> float | None:
    """Extract the last validation loss from a training log file."""
    last_loss = None
    last_step = -1
    with open(log_path) as f:
        for line in f:
            clean = ANSI_RE.sub("", line)
            m = VALIDATE_RE.search(clean)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                if step >= last_step:
                    last_step = step
                    last_loss = loss
    return last_loss


def collect_results(log_dir: str, num_seeds: int = 10) -> dict[str, list[float]]:
    """Collect final validation losses for all optimizers and seeds."""
    results: dict[str, list[float]] = {}
    for opt in OPTIMIZERS:
        losses = []
        for seed in range(num_seeds):
            log_path = os.path.join(log_dir, f"{opt}_seed{seed}.log")
            if not os.path.exists(log_path):
                continue
            loss = parse_final_val_loss(log_path)
            if loss is not None:
                losses.append(loss)
            else:
                print(f"  WARNING: No validation loss found in {log_path}")
        if losses:
            results[opt] = losses
    return results


# ============================================================
# Output: text table
# ============================================================

def print_text_table(all_results: dict[int, dict[str, list[float]]]):
    batch_sizes = sorted(all_results.keys())
    header_cols = "".join(f"{'B=' + str(b):>22}" for b in batch_sizes)
    print(f"\n{'Optimizer':<12}{header_cols}")
    print("=" * (12 + 22 * len(batch_sizes)))
    for opt in OPTIMIZERS:
        label = LABELS.get(opt, opt)
        row = f"{label:<12}"
        for b in batch_sizes:
            losses = all_results[b].get(opt, [])
            if losses:
                mean = np.mean(losses)
                std = np.std(losses, ddof=1) if len(losses) > 1 else 0.0
                row += f"{mean:.4f} +/- {std:.4f}".rjust(22)
            else:
                row += "N/A".rjust(22)
        print(row)
    print("=" * (12 + 22 * len(batch_sizes)))


def print_per_seed(all_results: dict[int, dict[str, list[float]]]):
    for b, results in sorted(all_results.items()):
        print(f"\n--- Per-seed breakdown (B={b}) ---")
        for opt in OPTIMIZERS:
            losses = results.get(opt, [])
            if losses:
                label = LABELS.get(opt, opt)
                vals = ", ".join(f"{l:.4f}" for l in losses)
                print(f"  {label}: [{vals}]  (n={len(losses)})")


# ============================================================
# Output: LaTeX
# ============================================================

def print_latex_table(all_results: dict[int, dict[str, list[float]]]):
    batch_sizes = sorted(all_results.keys())
    cols = "l" + "c" * len(batch_sizes)
    header = " & ".join([r"\textsc{Optimizer}"] +
                        [f"$B = {b}$" for b in batch_sizes])

    print(f"\n% LaTeX table")
    print(r"\begin{tabular}{" + cols + "}")
    print(r"\toprule")
    print(header + r" \\")
    print(r"\midrule")
    for opt in OPTIMIZERS:
        label = LABELS.get(opt, opt).upper()
        cells = [f"\\textsc{{{label}}}"]
        for b in batch_sizes:
            losses = all_results[b].get(opt, [])
            if losses:
                mean = np.mean(losses)
                std = np.std(losses, ddof=1) if len(losses) > 1 else 0.0
                cells.append(f"${mean:.4f} \\pm {std:.4f}$")
            else:
                cells.append("---")
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ============================================================
# Output: CSV
# ============================================================

def save_csv(all_results: dict[int, dict[str, list[float]]], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "batch_size", "seed", "val_loss"])
        for b, results in sorted(all_results.items()):
            for opt in OPTIMIZERS:
                for i, loss in enumerate(results.get(opt, [])):
                    writer.writerow([opt, b, i, f"{loss:.6f}"])
    print(f"\nCSV saved to {output_path}")


# ============================================================
# Output: matplotlib table image
# ============================================================

# Color scheme matching the paper's Table 2
ROW_COLORS = {
    "adamw": "#dce6f1",   # light blue
    "muon":  "#fef3cd",   # light yellow
}
HEADER_COLOR = "#4472c4"
HEADER_TEXT_COLOR = "white"


def render_table_image(
    all_results: dict[int, dict[str, list[float]]],
    output_path: str,
):
    batch_sizes = sorted(all_results.keys())
    n_cols = 1 + len(batch_sizes)  # optimizer + one col per batch size
    n_rows = len(OPTIMIZERS)

    fig_width = 2.5 + 2.5 * len(batch_sizes)
    fig_height = 1.0 + 0.5 * n_rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 1.5)
    ax.axis("off")

    # Title
    fig.text(
        0.5, 0.95,
        "Table 2. Validation loss for the 160M Llama 3 model "
        "trained for 3.2B tokens (10 seeds) over the C4 dataset.",
        ha="center", va="top", fontsize=9, style="italic",
        wrap=True,
    )

    col_widths = [1.8] + [2.5] * len(batch_sizes)
    col_starts = [0.0]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    def draw_cell(row, col, text, bg_color="white", text_color="black",
                  fontsize=10, fontweight="normal", ha="center"):
        x = col_starts[col]
        w = col_widths[col]
        y = n_rows - row  # flip so row 0 is at top
        rect = plt.Rectangle(
            (x, y), w, 0.5, facecolor=bg_color,
            edgecolor="#999999", linewidth=0.5,
        )
        ax.add_patch(rect)
        x_text = x + w / 2 if ha == "center" else x + 0.15
        ax.text(
            x_text, y + 0.25, text,
            ha=ha, va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color,
            fontfamily="sans-serif",
        )

    # Header row
    draw_cell(-1, 0, "Optimizer", bg_color=HEADER_COLOR,
              text_color=HEADER_TEXT_COLOR, fontweight="bold", fontsize=10)
    for i, b in enumerate(batch_sizes):
        header = f"Validation Loss\nB = {b}"
        draw_cell(-1, i + 1, header, bg_color=HEADER_COLOR,
                  text_color=HEADER_TEXT_COLOR, fontweight="bold", fontsize=9)

    # Data rows
    for row_idx, opt in enumerate(OPTIMIZERS):
        label = LABELS.get(opt, opt)
        bg = ROW_COLORS.get(opt, "white")
        draw_cell(row_idx, 0, label, bg_color=bg, fontweight="bold",
                  fontsize=10, ha="left")
        for col_idx, b in enumerate(batch_sizes):
            losses = all_results[b].get(opt, [])
            if losses:
                mean = np.mean(losses)
                std = np.std(losses, ddof=1) if len(losses) > 1 else 0.0
                text = f"{mean:.4f} \u00b1 {std:.4f}"
            else:
                text = "N/A"
            draw_cell(row_idx, col_idx + 1, text, bg_color=bg, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\nTable image saved to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse Paper Table 2 experiment logs and generate results table."
    )
    parser.add_argument(
        "--log-dir", default="./logs/paper_table2",
        help="Base log directory (default: ./logs/paper_table2)",
    )
    parser.add_argument(
        "--batch-size", type=int, action="append", default=None,
        help="Batch size(s) to include. Can be specified multiple times. "
             "Default: auto-detect from log directory.",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=10,
        help="Number of seeds (default: 10)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for table image and CSV (default: same as log-dir)",
    )
    args = parser.parse_args()

    # Auto-detect batch sizes from subdirectories like B256, B64
    if args.batch_size is None:
        batch_sizes = []
        if os.path.isdir(args.log_dir):
            for name in sorted(os.listdir(args.log_dir)):
                if name.startswith("B") and os.path.isdir(
                    os.path.join(args.log_dir, name)
                ):
                    try:
                        batch_sizes.append(int(name[1:]))
                    except ValueError:
                        pass
        if not batch_sizes:
            print(f"ERROR: No B<size> subdirectories found in {args.log_dir}")
            print("Run experiments first with: bash scripts/run_paper_table2.sh")
            sys.exit(1)
    else:
        batch_sizes = args.batch_size

    output_dir = args.output_dir or args.log_dir

    # Collect results for each batch size
    all_results: dict[int, dict[str, list[float]]] = {}
    for b in batch_sizes:
        log_subdir = os.path.join(args.log_dir, f"B{b}")
        if not os.path.isdir(log_subdir):
            print(f"WARNING: Log directory not found: {log_subdir}")
            continue
        print(f"Parsing logs from {log_subdir} ...")
        results = collect_results(log_subdir, num_seeds=args.num_seeds)
        if results:
            all_results[b] = results

    if not all_results:
        print("ERROR: No results found. Run experiments first.")
        sys.exit(1)

    # Print results
    print_text_table(all_results)
    print_per_seed(all_results)
    print_latex_table(all_results)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "table2_results.csv")
    save_csv(all_results, csv_path)

    img_path = os.path.join(output_dir, "table2_results.png")
    render_table_image(all_results, img_path)


if __name__ == "__main__":
    main()
