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
from lm_eval.tasks import TaskManager
from lm_eval_wrapper import HNetLM


# Paper benchmarks: task_name -> metric_key (HNet paper Table 2 reproduction).
PAPER_BENCHMARKS = {
    "lambada_openai": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "openbookqa": "acc_norm",
}

# OLMES (arxiv 2406.08446): per logical task, run both cloze (CF) and
# multiple-choice (MCF) formulations and report max. Headline metric for CF
# under HNet's byte-level tokenizer is `acc_bytes` (acc / byte-length, equals
# OLMES `acc_per_char` for ASCII text); for MCF it's just `acc`.
OLMES_BENCHMARKS = {
    "arc_challenge": {"cf": "olmes_arc_challenge_cf", "mcf": "olmes_arc_challenge_mcf"},
    "arc_easy":      {"cf": "olmes_arc_easy_cf",      "mcf": "olmes_arc_easy_mcf"},
    "boolq":         {"cf": "olmes_boolq_cf",         "mcf": "olmes_boolq_mcf"},
    "csqa":          {"cf": "olmes_csqa_cf",          "mcf": "olmes_csqa_mcf"},
    "hellaswag":     {"cf": "olmes_hellaswag_cf",     "mcf": "olmes_hellaswag_mcf"},
    "openbookqa":    {"cf": "olmes_openbookqa_cf",    "mcf": "olmes_openbookqa_mcf"},
    "piqa":          {"cf": "olmes_piqa_cf",          "mcf": "olmes_piqa_mcf"},
    "siqa":          {"cf": "olmes_siqa_cf",          "mcf": "olmes_siqa_mcf"},
    "winogrande":    {"cf": "olmes_winogrande_cf",    "mcf": "olmes_winogrande_mcf"},
}
OLMES_CF_METRICS = ["acc", "acc_norm", "acc_bytes"]
OLMES_HEADLINE_CF_METRIC = "acc_bytes"

# EU20 (Thellmann et al. 2410.08928) — translated benchmarks for European languages.
# Each entry maps logical_benchmark -> {formulation -> task_template}.
# `{lang}` is filled in with the lowercased language code at expansion time.
# GSM8K is generative (single formulation); TruthfulQA mc2 is CF-only because
# it scores sum-of-correct-prob over a variable-length set.
EU20_LANGUAGES = ["EN", "NL", "FI", "BG"]
EU20_BENCHMARKS = {
    "arc_challenge":   {"cf": "eu20_arc_challenge_{lang}_cf",
                        "mcf": "eu20_arc_challenge_{lang}_mcf"},
    "hellaswag":       {"cf": "eu20_hellaswag_{lang}_cf",
                        "mcf": "eu20_hellaswag_{lang}_mcf"},
    "truthfulqa_mc1":  {"cf": "eu20_truthfulqa_mc1_{lang}_cf",
                        "mcf": "eu20_truthfulqa_mc1_{lang}_mcf"},
    "truthfulqa_mc2":  {"cf": "eu20_truthfulqa_mc2_{lang}_cf"},
    "gsm8k":           {"gen": "eu20_gsm8k_{lang}"},
}


def _lookup_metric(task_results, metric):
    """lm-eval stores metric keys with ',<filter>' suffixes; match by prefix."""
    for key, val in task_results.items():
        if "stderr" in key:
            continue
        if key == metric or key.startswith(f"{metric},"):
            return val
    return None


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


def _fmt_ll(ll, cont):
    """Format ll as raw sum and per-byte (matches `acc_bytes` normalization)."""
    if ll is None:
        return "ll=   n/a"
    n_bytes = len(cont.encode("utf-8")) if cont else 0
    if n_bytes == 0:
        return f"ll={ll:+.3f}"
    return f"ll={ll:+.3f} ({ll / n_bytes:+.3f}/byte)"


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
                print(f"  {marker} [{j}] {_fmt_ll(ll, cont)}  {cont!r}")
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
                cont = shared_cont if shared_cont is not None else continuations[j]
                print(f"  {marker} [{j}] {_fmt_ll(ll, cont)}")
                print(f"        context:      {_trim(contexts[j])!r}")
                if shared_cont is None:
                    print(f"        continuation: {continuations[j]!r}")
        print(f"Predicted: [{pred}]   Gold: {target!r}")
    else:
        # Single-continuation loglikelihood task (e.g. lambada): score the gold and free-generate.
        context = contexts[0]
        gold_cont = continuations[0]
        gold_ll = lls[0] if lls else None
        print(f"\n#{idx} Context:\n{_trim(context)}")
        print(f"\nGold continuation: {gold_cont!r}  ({_fmt_ll(gold_ll, gold_cont)})")

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

    olmes_mode = args.olmes
    eu20_mode = args.eu20
    mode_count = sum([olmes_mode, eu20_mode, args.paper])
    if mode_count > 1:
        raise SystemExit("Pass at most one of --olmes / --eu20 / --paper.")

    num_fewshot = args.num_fewshot
    if num_fewshot is None:
        if olmes_mode:
            num_fewshot = 5
        elif eu20_mode:
            num_fewshot = None  # let each yaml decide (mc1/mc2 are 0-shot, gsm8k is 5-shot, ARC/HellaSwag use first_n from train)
        else:
            num_fewshot = 0

    # Build TaskManager with whichever include_paths are needed.
    include_paths = []
    if olmes_mode:
        if not os.path.isdir(args.olmes_tasks_dir):
            raise SystemExit(f"OLMES tasks dir not found: {args.olmes_tasks_dir}")
        include_paths.append(args.olmes_tasks_dir)
    if eu20_mode:
        if not os.path.isdir(args.eu20_tasks_dir):
            raise SystemExit(f"EU20 tasks dir not found: {args.eu20_tasks_dir}")
        include_paths.append(args.eu20_tasks_dir)

    task_manager = None
    if include_paths:
        task_manager = TaskManager(
            include_path=include_paths if len(include_paths) > 1 else include_paths[0],
            include_defaults=True,
        )

    if args.tasks:
        tasks = args.tasks.split(",")
    elif olmes_mode:
        tasks = []
        for logical, ids in OLMES_BENCHMARKS.items():
            if args.formulation in ("cf", "both"):
                tasks.append(ids["cf"])
            if args.formulation in ("mcf", "both"):
                tasks.append(ids["mcf"])
    elif eu20_mode:
        langs = args.eu20_langs.split(",") if args.eu20_langs else EU20_LANGUAGES
        tasks = []
        for lang in langs:
            ll = lang.lower()
            for logical, formulations in EU20_BENCHMARKS.items():
                for form, template in formulations.items():
                    if form == "cf" and args.formulation == "mcf":
                        continue
                    if form == "mcf" and args.formulation == "cf":
                        continue
                    tasks.append(template.format(lang=ll))
    else:
        tasks = list(PAPER_BENCHMARKS.keys())

    print(f"Running tasks: {', '.join(tasks)}")
    print(f"Batch size: {args.batch_size}, Max length: {args.max_length}, "
          f"Fewshot: {num_fewshot}")
    print("-" * 60)

    log_samples = args.log_samples or args.show_examples > 0
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=args.batch_size,
        log_samples=log_samples,
        task_manager=task_manager,
    )

    if args.show_examples > 0:
        _print_examples(results, tasks, args.show_examples, model)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    olmes_summary = None
    eu20_summary = None
    if olmes_mode:
        olmes_summary = _print_olmes_table(results, args.formulation)
    elif eu20_mode:
        langs = args.eu20_langs.split(",") if args.eu20_langs else EU20_LANGUAGES
        eu20_summary = _print_eu20_table(results, langs, args.formulation)
    else:
        _print_paper_table(results, tasks)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(args.config_path))[0]
        if olmes_mode:
            suffix = "olmes"
        elif eu20_mode:
            suffix = "eu20"
        else:
            suffix = "paper"
        output_path = os.path.join(
            args.output_dir, f"{model_name}_{suffix}_{timestamp}.json"
        )

        if olmes_mode:
            mode_label = "olmes"
        elif eu20_mode:
            mode_label = "eu20"
        else:
            mode_label = "paper"

        output = {
            "model_path": args.model_path,
            "config_path": args.config_path,
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "timestamp": timestamp,
            "mode": mode_label,
            "formulation": args.formulation if (olmes_mode or eu20_mode) else None,
            "olmes_summary": olmes_summary,
            "eu20_summary": eu20_summary,
            "results": results["results"],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return results


def _print_paper_table(results, tasks):
    print(f"{'Task':<20} {'Metric':<12} {'Score':>8}")
    print("-" * 60)
    scores = []
    for task_name in tasks:
        if task_name not in results["results"]:
            print(f"{task_name:<20} {'N/A':<12} {'N/A':>8}")
            continue
        metric_key = PAPER_BENCHMARKS.get(task_name, "acc")
        score = _lookup_metric(results["results"][task_name], metric_key)
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


def _print_olmes_table(results, formulation):
    """Per logical task, print full breakdown: CF (acc, acc_norm, acc_bytes), MCF acc, max."""
    cols = (
        f"{'Task':<16}"
        f"{'CF acc':>9}"
        f"{'CF acc_n':>10}"
        f"{'CF acc_b':>10}"
        f"{'MCF acc':>10}"
        f"{'max':>9}"
        f"  source"
    )
    print(cols)
    print("-" * len(cols))

    summary = {}
    headline_scores = []
    for logical, ids in OLMES_BENCHMARKS.items():
        cf_id, mcf_id = ids["cf"], ids["mcf"]
        cf_res = results["results"].get(cf_id) if formulation in ("cf", "both") else None
        mcf_res = results["results"].get(mcf_id) if formulation in ("mcf", "both") else None

        cf_acc = _lookup_metric(cf_res, "acc") if cf_res else None
        cf_norm = _lookup_metric(cf_res, "acc_norm") if cf_res else None
        cf_bytes = _lookup_metric(cf_res, "acc_bytes") if cf_res else None
        mcf_acc = _lookup_metric(mcf_res, "acc") if mcf_res else None

        cf_headline = cf_bytes if cf_bytes is not None else cf_norm if cf_norm is not None else cf_acc
        candidates = []
        if cf_headline is not None:
            candidates.append((cf_headline, "cf"))
        if mcf_acc is not None:
            candidates.append((mcf_acc, "mcf"))
        if candidates:
            best_score, best_src = max(candidates, key=lambda x: x[0])
        else:
            best_score, best_src = None, "n/a"

        def fmt(v):
            return f"{v * 100:>8.1f}%" if v is not None else f"{'-':>9}"

        # Print row
        print(
            f"{logical:<16}"
            f"{fmt(cf_acc)}"
            f"{fmt(cf_norm).rjust(10)}"
            f"{fmt(cf_bytes).rjust(10)}"
            f"{fmt(mcf_acc).rjust(10)}"
            f"{fmt(best_score)}"
            f"  {best_src}"
        )

        summary[logical] = {
            "cf": {"acc": cf_acc, "acc_norm": cf_norm, "acc_bytes": cf_bytes},
            "mcf": {"acc": mcf_acc},
            "headline": {"score": best_score, "source": best_src},
        }
        if best_score is not None:
            headline_scores.append(best_score)

    if headline_scores:
        avg = sum(headline_scores) / len(headline_scores)
        print("-" * len(cols))
        print(f"{'Average (max)':<55}{avg * 100:>8.1f}%")
    print("=" * len(cols))
    return summary


def _print_eu20_table(results, langs, formulation):
    """Per (language, benchmark) print headline metric. CF: acc_bytes (or
    acc_norm/acc fallback); MCF: acc; GSM8K: exact_match.
    """
    benchmarks = list(EU20_BENCHMARKS.keys())

    def _cf_headline(task_id):
        r = results["results"].get(task_id) or {}
        for m in ("acc_bytes", "acc_norm", "acc"):
            v = _lookup_metric(r, m)
            if v is not None:
                return v
        return None

    def _mcf_headline(task_id):
        r = results["results"].get(task_id) or {}
        return _lookup_metric(r, "acc")

    def _gen_headline(task_id):
        r = results["results"].get(task_id) or {}
        return _lookup_metric(r, "exact_match")

    show_cf = formulation in ("cf", "both")
    show_mcf = formulation in ("mcf", "both")
    cols_per_bench = []  # (logical_name, formulation_key, getter)
    for logical in benchmarks:
        forms = EU20_BENCHMARKS[logical]
        if "gen" in forms:
            cols_per_bench.append((logical, "gen", _gen_headline))
            continue
        if show_cf and "cf" in forms:
            cols_per_bench.append((logical, "cf", _cf_headline))
        if show_mcf and "mcf" in forms:
            cols_per_bench.append((logical, "mcf", _mcf_headline))

    header = f"{'lang':<6}" + "".join(f"{f'{l[:11]}.{f}':>16}" for l, f, _ in cols_per_bench)
    print(header)
    print("-" * len(header))

    summary = {}
    for lang in langs:
        ll = lang.lower()
        summary[lang] = {}
        row = f"{lang:<6}"
        for logical, form, getter in cols_per_bench:
            template = EU20_BENCHMARKS[logical][form]
            task_id = template.format(lang=ll)
            v = getter(task_id)
            summary[lang].setdefault(logical, {})[form] = v
            row += (f"{v * 100:>15.1f}%" if v is not None else f"{'-':>16}")
        print(row)
    print("=" * len(header))
    return summary


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
    parser.add_argument(
        "--olmes",
        action="store_true",
        help=(
            "Run the OLMES-9 task suite (arxiv 2406.08446) with curated 5-shot "
            "examples and both cloze (CF) and multiple-choice (MCF) formulations. "
            "MMLU is staged separately and not included here."
        ),
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run the original HNet paper benchmarks (PAPER_BENCHMARKS), 0-shot.",
    )
    parser.add_argument(
        "--formulation",
        choices=["cf", "mcf", "both"],
        default="both",
        help="Which OLMES formulation(s) to run. Only used with --olmes.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help=(
            "In-context examples per task. Default: 5 with --olmes, 0 with --paper. "
            "Overrides any task-config default."
        ),
    )
    parser.add_argument(
        "--olmes-tasks-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "olmes_tasks"),
        help="Directory containing OLMES task YAML files (used with --olmes).",
    )
    parser.add_argument(
        "--eu20",
        action="store_true",
        help=(
            "Run the EU20 multilingual benchmark suite (Thellmann et al. 2410.08928): "
            "ARC-Challenge, HellaSwag, TruthfulQA mc1+mc2, GSM8K across "
            f"{', '.join(EU20_LANGUAGES)}. Restrict languages with --eu20-langs."
        ),
    )
    parser.add_argument(
        "--eu20-langs",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of EU20 languages to run (default: all configured). "
            f"Available: {', '.join(EU20_LANGUAGES)}."
        ),
    )
    parser.add_argument(
        "--eu20-tasks-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eu20_tasks"),
        help="Directory containing EU20 task YAML files (used with --eu20).",
    )

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
