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


def freeze_pad_dims(model):
    """Zero out all pad_dimension parameters to measure their contribution."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "pad_dimension") and module.pad_dimension is not None:
            d_pad = module.pad_dimension.shape[0]
            module.pad_dimension.data.zero_()
            module.pad_dimension.requires_grad_(False)
            count += 1
            print(f"  Froze pad_dimension at {name} ({d_pad} dims)")
    return count


def zero_original_dims(model):
    """Register hooks to zero the first d_pad original dims (control experiment)."""
    import torch

    count = 0
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, "pad_dimension") and module.pad_dimension is not None:
            d_pad = module.pad_dimension.shape[0]
            # Hook the submodule that receives the padded input
            target = module.main_network if module.is_innermost else module.encoder

            def make_hook(n_dims):
                def hook(mod, args):
                    hidden_states = args[0]
                    hidden_states = hidden_states.clone()
                    hidden_states[..., :n_dims] = 0.0
                    return (hidden_states,) + args[1:]
                return hook

            h = target.register_forward_pre_hook(make_hook(d_pad))
            hooks.append(h)
            count += 1
            print(f"  Zeroing first {d_pad} original dims at {name} (hook on {'main_network' if module.is_innermost else 'encoder'})")
    return count


class _FakeReq:
    """Minimal stand-in for lm_eval.api.instance.Instance for ad-hoc generation."""
    def __init__(self, args):
        self.args = args


def _extract_ll(resp):
    """Pull the loglikelihood scalar out of an lm-eval per-request response."""
    # Typical shapes: [(ll, is_greedy)] or (ll, is_greedy) or [ll]
    if isinstance(resp, (list, tuple)) and resp:
        first = resp[0]
        if isinstance(first, (list, tuple)) and first:
            return float(first[0])
        return float(first)
    return None


def _print_examples(results, tasks, n, hnet_lm):
    samples = results.get("samples") or {}
    if not samples:
        print("\n(No samples were logged — cannot show examples.)")
        return

    print("\n" + "=" * 70)
    print(f"EXAMPLES (first {n} per task)")
    print("=" * 70)

    for task in tasks:
        task_samples = samples.get(task)
        if not task_samples:
            print(f"\n[{task}] no samples logged")
            continue

        print(f"\n{'-' * 70}\n[{task}]  showing {min(n, len(task_samples))} of {len(task_samples)}\n{'-' * 70}")
        for i, s in enumerate(task_samples[:n]):
            _print_one_sample(i, s, hnet_lm)


def _print_one_sample(idx, sample, hnet_lm, ctx_max_chars=1000):
    args = sample.get("arguments") or []
    resps = sample.get("filtered_resps") or sample.get("resps") or []
    target = sample.get("target", "")

    if not args:
        print(f"\n#{idx}: (no arguments recorded)")
        return

    contexts = [a[0] if isinstance(a, (list, tuple)) and len(a) > 0 else "" for a in args]
    continuations = [a[1] if isinstance(a, (list, tuple)) and len(a) > 1 else "" for a in args]
    lls = [_extract_ll(r) for r in resps]

    def _trim(s):
        return s if len(s) <= ctx_max_chars else "…" + s[-ctx_max_chars:]

    if len(continuations) > 1:
        valid_lls = [(j, ll) for j, ll in enumerate(lls) if ll is not None]
        pred = max(valid_lls, key=lambda x: x[1])[0] if valid_lls else None
        contexts_all_same = all(c == contexts[0] for c in contexts)

        if contexts_all_same:
            # Standard MC: shared context, different continuations (hellaswag, piqa, arc_*, openbookqa).
            print(f"\n#{idx} Context:\n{_trim(contexts[0])}")
            print("\nChoices (loglikelihood):")
            for j, cont in enumerate(continuations):
                marker = "→" if j == pred else " "
                ll = lls[j] if j < len(lls) else None
                ll_str = f"{ll:+.3f}" if ll is not None else "   n/a"
                print(f"  {marker} [{j}] ll={ll_str}  {cont!r}")
        else:
            # Winogrande-style: each choice substitutes a candidate into the context;
            # continuation is (usually) shared. Show per-choice contexts.
            shared_cont = continuations[0] if all(c == continuations[0] for c in continuations) else None
            if shared_cont is not None:
                print(f"\n#{idx} Shared continuation: {shared_cont!r}")
            print("\nChoices (loglikelihood):")
            for j in range(len(contexts)):
                marker = "→" if j == pred else " "
                ll = lls[j] if j < len(lls) else None
                ll_str = f"{ll:+.3f}" if ll is not None else "   n/a"
                print(f"  {marker} [{j}] ll={ll_str}")
                print(f"        context:      {_trim(contexts[j])!r}")
                if shared_cont is None:
                    print(f"        continuation: {continuations[j]!r}")
        print(f"Predicted: [{pred}]   Gold: {target!r}")
    else:
        # Single-continuation loglikelihood task (e.g. lambada): score the gold and free-generate.
        context = contexts[0]
        gold_cont = continuations[0]
        gold_ll = lls[0] if lls else None
        ll_str = f"{gold_ll:+.3f}" if gold_ll is not None else "n/a"
        print(f"\n#{idx} Context:\n{_trim(context)}")
        print(f"\nGold continuation: {gold_cont!r}  (ll={ll_str})")

        try:
            gen = hnet_lm.generate_until(
                [_FakeReq((context, {"until": ["\n"], "max_gen_toks": 64}))],
                disable_tqdm=True,
            )[0]
            print(f"Model generated (greedy, until '\\n' or 64 tok): {gen!r}")
        except Exception:
            # Print full traceback so the underlying bug is visible,
            # then continue with the next example.
            import traceback
            print("Model generation failed:")
            traceback.print_exc()


def run_eval(args):
    print(f"Loading model from {args.model_path}...")
    model = HNetLM(
        model_path=args.model_path,
        config_path=args.config_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if args.freeze_pad_dims:
        print("Freezing pad dimensions (zeroing out)...")
        n = freeze_pad_dims(model.model)
        if n == 0:
            print("  No pad_dimension found in this model.")
        else:
            print(f"  Froze {n} pad_dimension parameter(s).")

    if args.zero_original_dims:
        print("Zeroing original dims (control experiment)...")
        n = zero_original_dims(model.model)
        if n == 0:
            print("  No pad_dimension found in this model.")
        else:
            print(f"  Hooked {n} stage(s) to zero original dims.")

    if args.tasks:
        tasks = args.tasks.split(",")
    else:
        tasks = list(PAPER_BENCHMARKS.keys())

    print(f"Running tasks: {', '.join(tasks)}")
    print(f"Batch size: {args.batch_size}, Max length: {args.max_length}")
    print("-" * 60)

    log_samples = args.log_samples or args.show_examples > 0
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        log_samples=log_samples,
    )

    if args.show_examples > 0:
        _print_examples(results, tasks, args.show_examples, model)

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
    parser.add_argument(
        "--freeze-pad-dims",
        action="store_true",
        help="Zero out pad_dimension parameters to measure their contribution",
    )
    parser.add_argument(
        "--zero-original-dims",
        action="store_true",
        help="Zero the first d_pad original dims as a control experiment",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help=(
            "If > 0, print the first N examples of each task: question/context, "
            "per-choice loglikelihoods (MC) or gold continuation + greedy generation "
            "(loglikelihood tasks like lambada). Forces log_samples=True."
        ),
    )

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
