"""Combine lm-eval result JSONs (eval/results/hf) into a comparison table.

Outputs a plain-text table to stdout and optionally a LaTeX table.

Usage:
    python combine_hf_results.py [--dir results/hf] [--latex combined.tex] [--metric auto|acc|acc_norm]
"""

import argparse
import glob
import json
import os
import re

# Metric used per task in "auto" mode (mirrors common reporting conventions:
# length-normalized accuracy where available, plain accuracy otherwise).
PREFERRED_METRIC = {
    "lambada_openai": "acc,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "arc_easy": "acc,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
    "openbookqa": "acc_norm,none",
}

TASK_LABELS = {
    "lambada_openai": "LAMBADA",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "arc_easy": "ARC-e",
    "arc_challenge": "ARC-c",
    "winogrande": "WinoGrande",
    "openbookqa": "OpenBookQA",
}


def model_name_from_file(path, data):
    """Derive a short model label from the filename (fallback: hf_path)."""
    base = os.path.basename(path)
    # Strip suffix like _iter_0008719_hf_paper_20260610_124224.json
    m = re.match(r"(.+?)_iter_\d+_hf.*\.json$", base)
    if m:
        return m.group(1)
    hf_path = data.get("hf_path", "")
    return os.path.basename(hf_path.rstrip("/")) or base


def pick_metric(task_results, task, mode):
    if mode == "auto":
        key = PREFERRED_METRIC.get(task, "acc,none")
    else:
        key = f"{mode},none"
    if key not in task_results:
        key = "acc,none"
    value = task_results.get(key)
    stderr = task_results.get(key.replace(",none", "_stderr,none"))
    return key.split(",")[0], value, stderr


def collect(result_dir, metric_mode):
    rows = {}  # model -> {task: (metric_name, value, stderr)}
    tasks = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        if "results" not in data:
            continue
        model = model_name_from_file(path, data)
        rows[model] = {}
        for task, task_results in data["results"].items():
            if task not in tasks:
                tasks.append(task)
            rows[model][task] = pick_metric(task_results, task, metric_mode)
    return rows, tasks


def fmt(value, scale100=True):
    if value is None:
        return "--"
    return f"{value * 100:.1f}" if scale100 else f"{value:.3f}"


def print_text_table(rows, tasks):
    labels = [TASK_LABELS.get(t, t) for t in tasks]
    col_w = max(len(m) for m in rows) + 2
    header = "Model".ljust(col_w) + "".join(l.rjust(12) for l in labels) + "Avg".rjust(12)
    print(header)
    print("-" * len(header))
    for model, task_vals in rows.items():
        vals = [task_vals.get(t, (None, None, None))[1] for t in tasks]
        avg = sum(v for v in vals if v is not None) / max(1, sum(v is not None for v in vals))
        line = model.ljust(col_w) + "".join(fmt(v).rjust(12) for v in vals) + fmt(avg).rjust(12)
        print(line)


def latex_table(rows, tasks, metric_mode):
    labels = [TASK_LABELS.get(t, t) for t in tasks]
    # Note which metric each column uses
    metric_notes = []
    for t in tasks:
        names = {rows[m][t][0] for m in rows if t in rows[m]}
        metric_notes.append("/".join(sorted(names)))

    n_models = len(rows)
    # Bold the best value per task
    best = {}
    for i, t in enumerate(tasks):
        vals = {m: rows[m][t][1] for m in rows if t in rows[m] and rows[m][t][1] is not None}
        if vals:
            best[t] = max(vals, key=vals.get)
    avgs = {}
    for m in rows:
        vs = [rows[m][t][1] for t in tasks if t in rows[m] and rows[m][t][1] is not None]
        avgs[m] = sum(vs) / len(vs) if vs else None
    best_avg = max((m for m in avgs if avgs[m] is not None), key=lambda m: avgs[m], default=None)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}")
    lines.append(r"\toprule")
    lines.append("Model & " + " & ".join(labels) + r" & Avg \\")
    lines.append(r"\midrule")
    for model, task_vals in rows.items():
        cells = []
        for t in tasks:
            v = task_vals.get(t, (None, None, None))[1]
            cell = fmt(v)
            if best.get(t) == model:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        avg_cell = fmt(avgs[model])
        if model == best_avg:
            avg_cell = r"\textbf{" + avg_cell + "}"
        model_tex = model.replace("_", r"\_")
        lines.append(model_tex + " & " + " & ".join(cells) + " & " + avg_cell + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    metric_desc = (
        "acc\\_norm for HellaSwag/PIQA/ARC-c/OpenBookQA, acc otherwise"
        if metric_mode == "auto"
        else metric_mode.replace("_", r"\_")
    )
    lines.append(
        r"\caption{Zero-shot downstream evaluation (\%; " + metric_desc + r"). "
        r"Best result per task in bold.}"
    )
    lines.append(r"\label{tab:hf-eval-results}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "results", "hf"))
    parser.add_argument("--latex", default=None, help="Path to write a LaTeX table")
    parser.add_argument("--metric", default="auto", choices=["auto", "acc", "acc_norm"])
    args = parser.parse_args()

    rows, tasks = collect(args.dir, args.metric)
    if not rows:
        raise SystemExit(f"No result JSONs found in {args.dir}")

    print_text_table(rows, tasks)

    if args.latex:
        tex = latex_table(rows, tasks, args.metric)
        with open(args.latex, "w") as f:
            f.write(tex + "\n")
        print(f"\nLaTeX table written to {args.latex}")


if __name__ == "__main__":
    main()
