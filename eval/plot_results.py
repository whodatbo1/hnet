#!/usr/bin/env python3
"""
Plot evaluation results produced by run_benchmarks.py.

Generates:
  1. A summary table (models × benchmarks) printed to stdout and saved as CSV
  2. Per-benchmark bar charts comparing all models
  3. A combined overview figure with all benchmarks as subplots

Usage:
    python eval/plot_results.py --results-dir eval/results/train/mhb/XXS
    python eval/plot_results.py --results-dir eval/results/train/mhb/XXS \\
        --output-dir plots/XXS --strip-prefix hnet_1stage_XXS_
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive — safe for cluster / no display
import matplotlib.pyplot as plt


# Primary metric per benchmark (matches HNet paper Table 2 conventions)
TASK_ORDER = [
    "lambada_openai",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "openbookqa",
]

# (metric_key, stderr_key, display_label)
BENCHMARK_METRICS = {
    "lambada_openai": ("acc,none",      "acc_stderr,none",      "LAMBADA (acc)"),
    "hellaswag":      ("acc_norm,none", "acc_norm_stderr,none", "HellaSwag (acc_norm)"),
    "piqa":           ("acc,none",      "acc_stderr,none",      "PIQA (acc)"),
    "arc_easy":       ("acc,none",      "acc_stderr,none",      "ARC-Easy (acc)"),
    "arc_challenge":  ("acc_norm,none", "acc_norm_stderr,none", "ARC-Challenge (acc_norm)"),
    "winogrande":     ("acc,none",      "acc_stderr,none",      "WinoGrande (acc)"),
    "openbookqa":     ("acc_norm,none", "acc_norm_stderr,none", "OpenBookQA (acc_norm)"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_result_files(results_dir: Path) -> dict[str, Path]:
    """Return {model_name: path_to_json}, keeping the latest file per model."""
    model_files: dict[str, Path] = {}
    for json_path in sorted(results_dir.glob("**/*.json")):
        model_name = json_path.parent.name
        # JSON filenames contain a timestamp — lexicographic max = latest
        if model_name not in model_files or json_path.name > model_files[model_name].name:
            model_files[model_name] = json_path
    return model_files


def extract_metric(task_results: dict, task: str):
    """Return (value, stderr) for the primary metric of a task, or (None, None)."""
    if task not in task_results:
        return None, None
    metric_key, stderr_key, _ = BENCHMARK_METRICS[task]
    val = task_results[task].get(metric_key)
    err = task_results[task].get(stderr_key)
    return val, err


def build_rows(model_data: dict) -> list:
    """Return sorted list of (model_name, {task: (val, err)})."""
    rows = []
    for model_name, results in model_data.items():
        task_results = results.get("results", {})
        metrics = {task: extract_metric(task_results, task) for task in TASK_ORDER}
        rows.append((model_name, metrics))
    rows.sort(key=lambda x: x[0])
    return rows


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def shorten(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    return name.strip("_- ") or name


def auto_prefix(names: list[str]) -> str:
    """Strip the longest common underscore-delimited prefix."""
    if len(names) < 2:
        return ""
    common = os.path.commonprefix(names)
    # Trim to the last underscore so we don't cut mid-word
    if "_" in common:
        common = common.rsplit("_", 1)[0] + "_"
    return common


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def print_table(rows: list, prefix: str = "") -> None:
    col_w = 32
    val_w = 16
    header = f"{'Model':<{col_w}}" + "".join(
        BENCHMARK_METRICS[t][2][:val_w].ljust(val_w) for t in TASK_ORDER
    )
    print(header)
    print("-" * len(header))
    for model_name, metrics in rows:
        label = shorten(model_name, prefix)[:col_w - 1]
        row = f"{label:<{col_w}}"
        for task in TASK_ORDER:
            val, err = metrics[task]
            if val is None:
                cell = "—"
            elif err is not None:
                cell = f"{val*100:.1f}±{err*100:.1f}"
            else:
                cell = f"{val*100:.1f}"
            row += cell.ljust(val_w)
        print(row)


def save_csv(rows: list, path: Path) -> None:
    with open(path, "w") as f:
        header = ["model"] + [BENCHMARK_METRICS[t][2] for t in TASK_ORDER]
        f.write(",".join(header) + "\n")
        for model_name, metrics in rows:
            cells = [model_name]
            for task in TASK_ORDER:
                val, err = metrics[task]
                if val is None:
                    cells.append("")
                elif err is not None:
                    cells.append(f"{val*100:.2f}±{err*100:.2f}")
                else:
                    cells.append(f"{val*100:.2f}")
            f.write(",".join(cells) + "\n")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_ax(ax, model_labels, vals, errs, title):
    x = np.arange(len(model_labels))
    colors = plt.cm.tab10.colors
    bars = ax.bar(x, vals, yerr=errs, capsize=4, width=0.6,
                  color=[colors[i % len(colors)] for i in range(len(model_labels))],
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(errs) if any(errs) else 0) + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9)
    upper = max(vals) * 1.2 + 5 if any(vals) else 100
    ax.set_ylim(0, min(100, upper))
    ax.set_ylabel("Score (%)", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def plot_per_benchmark(rows: list, output_dir: Path, prefix: str = "") -> None:
    model_labels = [shorten(r[0], prefix) for r in rows]
    for task in TASK_ORDER:
        _, _, label = BENCHMARK_METRICS[task]
        vals = [(m[task][0] or 0) * 100 for _, m in rows]
        errs = [(m[task][1] or 0) * 100 for _, m in rows]
        if not any(vals):
            continue
        fig, ax = plt.subplots(figsize=(max(5, len(model_labels) * 1.3), 4))
        _bar_ax(ax, model_labels, vals, errs, label)
        fig.tight_layout()
        out = output_dir / f"{task}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out}")


def plot_combined(rows: list, output_dir: Path, prefix: str = "") -> None:
    model_labels = [shorten(r[0], prefix) for r in rows]
    ncols = 4
    nrows = (len(TASK_ORDER) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * max(4, len(model_labels) * 0.9),
                                      nrows * 3.8))
    axes = axes.flatten()
    for i, task in enumerate(TASK_ORDER):
        _, _, label = BENCHMARK_METRICS[task]
        vals = [(m[task][0] or 0) * 100 for _, m in rows]
        errs = [(m[task][1] or 0) * 100 for _, m in rows]
        _bar_ax(axes[i], model_labels, vals, errs, label)
    for j in range(len(TASK_ORDER), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Benchmark Comparison", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = output_dir / "all_benchmarks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarise and plot HNet benchmark evaluation results"
    )
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing per-model result subdirectories")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save plots and CSV (default: <results-dir>/plots)")
    parser.add_argument("--strip-prefix", type=str, default=None,
                        help="Model name prefix to strip from labels "
                             "(auto-detected if omitted)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else results_dir / "plots"

    print(f"Scanning {results_dir} ...")
    model_files = find_result_files(results_dir)
    if not model_files:
        print("No JSON result files found.")
        return

    names = sorted(model_files)
    print(f"Found {len(names)} model(s): {', '.join(names)}\n")

    model_data = {name: json.loads(model_files[name].read_text()) for name in names}
    rows = build_rows(model_data)

    prefix = args.strip_prefix if args.strip_prefix is not None else auto_prefix(names)
    if prefix:
        print(f"Stripping common prefix: '{prefix}'\n")

    # Table
    print_table(rows, prefix)
    print()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    save_csv(rows, csv_path)
    print(f"Saved CSV: {csv_path}\n")

    print("Per-benchmark charts:")
    plot_per_benchmark(rows, output_dir, prefix)

    print("Combined overview:")
    plot_combined(rows, output_dir, prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
