"""Aggregate EuroEval result jsonl files into one model x dataset table.

Scans a results directory (recursively) for euroeval_*.jsonl files written by
eval/run_euroeval.py and produces a wide table: one row per (language, dataset),
one column per model, showing the task's primary metric +- its standard error.

Primary metric per task (chance level in parentheses, for reading the table):
    sentiment-classification      test_mcc          (0)
    linguistic-acceptability      test_mcc          (0)
    knowledge                     test_mcc          (0)
    common-sense-reasoning        test_mcc          (0)
    named-entity-recognition      test_micro_f1_no_misc
    reading-comprehension         test_f1
    summarization                 test_chr_f3pp

MCC is preferred over accuracy/macro-F1 for all classification-style tasks
because it is invariant to label imbalance (majority-class predictions score 0).
A trailing '!' marks results with failed instances (unparseable generations).

Usage:
    python eval/aggregate_euroeval.py                          # default dir
    python eval/aggregate_euroeval.py --results-dir eval/results/euroeval/S
    python eval/aggregate_euroeval.py --csv out.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PRIMARY_METRIC = {
    "sentiment-classification": "test_mcc",
    "linguistic-acceptability": "test_mcc",
    "knowledge": "test_mcc",
    "common-sense-reasoning": "test_mcc",
    "named-entity-recognition": "test_micro_f1_no_misc",
    "reading-comprehension": "test_f1",
    "summarization": "test_chr_f3pp",
}


def load_records(results_dir: Path):
    """Latest record per (model, dataset): files scanned in mtime order, later
    lines/files overwrite earlier ones."""
    records = {}
    files = sorted(results_dir.rglob("euroeval_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"No euroeval_*.jsonl files under {results_dir}")
    for f in files:
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[(r["model"], r["dataset"])] = r
    return records


def cell(record):
    task = record.get("task")
    metric = PRIMARY_METRIC.get(task)
    total = record.get("results", {}).get("total", {})
    if metric is None or metric not in total:
        # fall back to the first non-SE metric
        candidates = [k for k in total if not k.endswith("_se") and k != "num_failed_instances"]
        if not candidates:
            return "-"
        metric = candidates[0]
    value = total[metric]
    se = total.get(f"{metric}_se")
    failed = total.get("num_failed_instances", 0) or 0
    out = f"{value:.1f}"
    if se is not None:
        out += f"±{se:.1f}"
    if failed:
        out += "!"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="eval/results/euroeval")
    ap.add_argument("--csv", default=None, help="Also write the table as CSV to this path.")
    args = ap.parse_args()

    records = load_records(Path(args.results_dir))

    models = sorted({m for m, _ in records})
    # (language, task, dataset) rows, sorted for stable grouping
    def row_key(r):
        langs = ",".join(sorted(l if isinstance(l, str) else l.get("code", "?") for l in r.get("languages", [])))
        return (langs, r.get("task", ""), r["dataset"])

    rows = sorted({row_key(r) for r in records.values()})

    table = []
    for langs, task, dataset in rows:
        metric = PRIMARY_METRIC.get(task, "?")
        line = [langs, dataset, task, metric]
        for m in models:
            r = records.get((m, dataset))
            line.append(cell(r) if r and row_key(r) == (langs, task, dataset) else "-")
        table.append(line)

    header = ["lang", "dataset", "task", "metric"] + models
    widths = [max(len(str(row[i])) for row in [header] + table) for i in range(len(header))]

    def fmt(row):
        return "  ".join(str(v).ljust(w) for v, w in zip(row, widths))

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print(fmt(row))
    print("\nMCC/F1 scales: 0 = chance for MCC tasks. '!' = had unparseable/failed instances.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(table)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
