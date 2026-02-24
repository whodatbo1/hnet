#!/usr/bin/env python3
"""
Run zero-shot benchmarks on HNet models using lm-evaluation-harness.

Reproduces Table 2 from the HNet paper (arxiv 2507.07955):
  LAMBADA (acc), HellaSwag (acc_norm), PIQA (acc), ARC-easy (acc),
  ARC-challenge (acc_norm), WinoGrande (acc), OpenBookQA (acc_norm)

Usage:
    # Run all benchmarks on pretrained model
    python eval/run_benchmarks.py \
        --model-path checkpoints/hnet_2stage_XL/model.pt \
        --config-path configs/hnet_2stage_XL.json

    # Run a single task
    python eval/run_benchmarks.py \
        --model-path checkpoints/hnet_2stage_XL/model.pt \
        --config-path configs/hnet_2stage_XL.json \
        --tasks hellaswag

    # Custom batch size and output
    python eval/run_benchmarks.py \
        --model-path checkpoints/hnet_2stage_XL/model.pt \
        --config-path configs/hnet_2stage_XL.json \
        --batch-size 4 \
        --output-dir eval/results
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add project root to path so we can import generate.py and hnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lm_eval
from lm_eval_wrapper import HNetLM


# Paper benchmarks: task_name -> metric_key
PAPER_BENCHMARKS = {
    "lambada_openai": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "openbookqa": "acc_norm",
}


def run_eval(args):
    print(f"Loading model from {args.model_path}...")
    model = HNetLM(
        model_path=args.model_path,
        config_path=args.config_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if args.tasks:
        tasks = args.tasks.split(",")
    else:
        tasks = list(PAPER_BENCHMARKS.keys())

    print(f"Running tasks: {', '.join(tasks)}")
    print(f"Batch size: {args.batch_size}, Max length: {args.max_length}")
    print("-" * 60)

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        log_samples=args.log_samples,
    )

    # Print results table
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Task':<20} {'Metric':<12} {'Score':>8}")
    print("-" * 60)

    scores = []
    for task_name in tasks:
        if task_name not in results["results"]:
            print(f"{task_name:<20} {'N/A':<12} {'N/A':>8}")
            continue

        task_results = results["results"][task_name]
        metric_key = PAPER_BENCHMARKS.get(task_name, "acc")

        # lm-eval stores metrics with comma-separated aliases
        score = None
        for key, val in task_results.items():
            if metric_key in key and "stderr" not in key:
                score = val
                break

        if score is not None:
            print(f"{task_name:<20} {metric_key:<12} {score * 100:>7.1f}%")
            scores.append(score)
        else:
            print(f"{task_name:<20} {metric_key:<12} {'N/A':>8}")

    if scores:
        avg = sum(scores) / len(scores)
        print("-" * 60)
        print(f"{'Average':<20} {'':>12} {avg * 100:>7.1f}%")
    print("=" * 60)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(args.config_path))[0]
        output_path = os.path.join(
            args.output_dir, f"{model_name}_{timestamp}.json"
        )

        output = {
            "model_path": args.model_path,
            "config_path": args.config_path,
            "tasks": tasks,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "timestamp": timestamp,
            "results": results["results"],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot benchmarks on HNet models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to model config (.json file)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks (default: all paper benchmarks)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Max sequence length (default: 8192)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Directory to save results JSON (default: eval/results)",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Log individual samples (increases output size)",
    )

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
