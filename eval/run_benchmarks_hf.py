"""Run the same H-Net benchmark suite (paper / OLMES / EU20) against a
HuggingFace LlamaForCausalLM checkpoint.

Mirrors run_benchmarks.py but swaps the H-Net `HNetLM(TemplateLM)` wrapper
for lm-evaluation-harness's built-in `model="hf"` adapter. This lets you
evaluate a Megatron-trained LLaMA (exported via
`Megatron-LM-Snellius/4_eval/export_to_hf.py`) without writing a custom
TemplateLM subclass.

Quick start:

    python run_benchmarks_hf.py \\
        --hf-path /scratch-shared/$USER/Megatron-LM/exports/iter_0000500_hf \\
        --paper --batch-size 8

    python run_benchmarks_hf.py --hf-path <...> --olmes --formulation both
    python run_benchmarks_hf.py --hf-path <...> --eu20  --eu20-langs EN,NL

Caveats:
- The OLMES/EU20 base YAMLs include `acc_bytes` in their metric list. For
  a BPE model that metric is still well-defined (loglik / utf-8 byte length)
  and equals OLMES's `acc_per_char` on ASCII. The printer uses `acc_norm`
  as the headline CF metric instead — `acc_bytes` is still reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import lm_eval
from lm_eval.tasks import TaskManager

# Reuse all the run_benchmarks helpers so we don't duplicate the
# benchmark tables or the per-mode printers.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_benchmarks import (  # noqa: E402
    PAPER_BENCHMARKS, OLMES_BENCHMARKS, EU20_BENCHMARKS, EU20_LANGUAGES,
    _print_paper_table, _print_olmes_table, _print_eu20_table,
)


def _build_model_args(args) -> str:
    """Build the lm_eval `model_args` string for the HF backend."""
    pieces = [
        f"pretrained={args.hf_path}",
        f"dtype={args.dtype}",
    ]
    if args.trust_remote_code:
        pieces.append("trust_remote_code=True")
    if args.max_length:
        pieces.append(f"max_length={args.max_length}")
    if args.parallelize:
        pieces.append("parallelize=True")
    return ",".join(pieces)


def _select_tasks(args):
    olmes_mode, eu20_mode = args.olmes, args.eu20
    if sum([olmes_mode, eu20_mode, args.paper]) > 1:
        raise SystemExit("Pass at most one of --olmes / --eu20 / --paper.")

    if args.tasks:
        return args.tasks.split(",")
    if olmes_mode:
        tasks = []
        for ids in OLMES_BENCHMARKS.values():
            if args.formulation in ("cf", "both"):
                tasks.append(ids["cf"])
            if args.formulation in ("mcf", "both"):
                tasks.append(ids["mcf"])
        return tasks
    if eu20_mode:
        langs = args.eu20_langs.split(",") if args.eu20_langs else EU20_LANGUAGES
        tasks = []
        for lang in langs:
            ll = lang.lower()
            for formulations in EU20_BENCHMARKS.values():
                for form, template in formulations.items():
                    if form == "cf" and args.formulation == "mcf":
                        continue
                    if form == "mcf" and args.formulation == "cf":
                        continue
                    tasks.append(template.format(lang=ll))
        return tasks
    return list(PAPER_BENCHMARKS.keys())


def _default_fewshot(args):
    if args.num_fewshot is not None:
        return args.num_fewshot
    if args.olmes:
        return 5
    if args.eu20:
        return None
    return 0


def _task_manager(args):
    include_paths = []
    if args.olmes:
        if not os.path.isdir(args.olmes_tasks_dir):
            raise SystemExit(f"OLMES tasks dir not found: {args.olmes_tasks_dir}")
        include_paths.append(args.olmes_tasks_dir)
    if args.eu20:
        if not os.path.isdir(args.eu20_tasks_dir):
            raise SystemExit(f"EU20 tasks dir not found: {args.eu20_tasks_dir}")
        include_paths.append(args.eu20_tasks_dir)
    if not include_paths:
        return None
    return TaskManager(
        include_path=include_paths if len(include_paths) > 1 else include_paths[0],
        include_defaults=True,
    )


def run_eval(args):
    tasks = _select_tasks(args)
    num_fewshot = _default_fewshot(args)
    task_manager = _task_manager(args)
    model_args = _build_model_args(args)

    print(f"HF path:  {args.hf_path}")
    print(f"Tasks:    {', '.join(tasks)}")
    print(f"Fewshot:  {num_fewshot}")
    print(f"Backend:  hf  ({model_args})")
    print("-" * 60)

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=args.batch_size,
        log_samples=args.log_samples,
        task_manager=task_manager,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    olmes_summary = eu20_summary = None
    if args.olmes:
        olmes_summary = _print_olmes_table(results, args.formulation)
    elif args.eu20:
        langs = args.eu20_langs.split(",") if args.eu20_langs else EU20_LANGUAGES
        eu20_summary = _print_eu20_table(results, langs, args.formulation)
    else:
        _print_paper_table(results, tasks)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(os.path.normpath(args.hf_path))
        suffix = "olmes" if args.olmes else "eu20" if args.eu20 else "paper"
        path = os.path.join(args.output_dir, f"{model_name}_{suffix}_{ts}.json")
        payload = {
            "hf_path": args.hf_path,
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "batch_size": args.batch_size,
            "timestamp": ts,
            "mode": suffix,
            "formulation": args.formulation if (args.olmes or args.eu20) else None,
            "olmes_summary": olmes_summary,
            "eu20_summary": eu20_summary,
            "results": results["results"],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nResults saved to {path}")

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-path", required=True,
                    help="Directory containing the exported HF checkpoint")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch-size", default="auto",
                    help='Pass an int or "auto" / "auto:N" for lm-eval-harness auto batch')
    ap.add_argument("--max-length", type=int, default=None,
                    help="Override max sequence length (default: model config)")
    ap.add_argument("--parallelize", action="store_true",
                    help="Spread the model across multiple GPUs (accelerate)")
    ap.add_argument("--trust-remote-code", action="store_true")

    mode = ap.add_argument_group("mode")
    mode.add_argument("--paper", action="store_true",
                      help="Run the H-Net paper benchmark suite (default if no mode given)")
    mode.add_argument("--olmes", action="store_true",
                      help="Run the OLMES suite")
    mode.add_argument("--eu20", action="store_true",
                      help="Run the EU20 suite")

    ap.add_argument("--formulation", choices=["cf", "mcf", "both"], default="both",
                    help="OLMES/EU20 formulation to run")
    ap.add_argument("--num-fewshot", type=int, default=None,
                    help="Override fewshot count (default: 5 for olmes, 0 for paper)")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated explicit task list (overrides --paper/--olmes/--eu20)")
    ap.add_argument("--eu20-langs", default=None,
                    help=f"Comma-separated lang codes (default: {','.join(EU20_LANGUAGES)})")

    ap.add_argument("--olmes-tasks-dir", default=os.path.join(_HERE, "olmes_tasks"))
    ap.add_argument("--eu20-tasks-dir",  default=os.path.join(_HERE, "eu20_tasks"))

    ap.add_argument("--log-samples", action="store_true")
    ap.add_argument("--output-dir", default=os.path.join(_HERE, "results", "hf"))

    args = ap.parse_args()

    if not (args.paper or args.olmes or args.eu20 or args.tasks):
        args.paper = True

    run_eval(args)


if __name__ == "__main__":
    main()
