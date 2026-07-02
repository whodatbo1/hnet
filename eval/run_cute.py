"""Run CUTE / EXECUTE character-level evaluation on H-Net.

CUTE (https://github.com/Leukas/CUTE) and EXECUTE
(https://github.com/Leukas/EXECUTE) probe a model's character-level
understanding via spelling, character insertion/deletion/substitution/swap,
word-level edits, and orthographic/semantic similarity tasks.

The upstream runners load HF `AutoModelForCausalLM` directly, which doesn't
work for H-Net (custom `HNetForCausalLM`, byte tokenizer, custom forward
output). This script bypasses the upstream runner: it loads the HF dataset
(`leukas/cute`) directly and drives H-Net via the same prefill+step pattern
used by `eval/lm_eval_wrapper.py:generate_until`.

EXECUTE is not on the HuggingFace Hub as of this writing — its language-
specific tasks live as TSVs in the upstream repo under `data/tasks/{lang}/`.
The runner's `--dataset` / `--language` flags are generic, so if EXECUTE
(or any same-schema dataset) gets published to HF later it can be plugged
in without code changes.

Each dataset row contains a fully-formatted `prompt` (task description +
4-shot examples + the query) and an `answer` string. The prompt is fed to the
model verbatim — control its formatting at dataset-build time (e.g. via
`eval/make_cute_base.py`'s JSON template config, which can bake in an answer
cue such as a trailing `Answer: "`).

We greedy-decode, stopping at a newline or a `"` (CUTE answers never contain
either). `parse_answer` then strips an echoed `Answer:` prefix and wrapping
quotes before scoring against the gold.

Beyond binary exact-match (which is harsh on small byte-level models) we
also report:
  - char_similarity = 1 - levenshtein(pred, gold) / max(|pred|, |gold|)
  - char_f1         = F1 over multisets of characters

These tolerant metrics still saturate to 1.0 on perfect output but give a
graded signal on partial-credit answers (one bad byte vs. completely wrong).

Usage:

    python eval/run_cute.py \\
        --model-path /scratch-shared/.../latest.pt \\
        --config-path configs/.../hnet.json \\
        --model-name hnet-xxs \\
        --limit 50          # smoke test

    # Single task, full eval:
    python eval/run_cute.py \\
        --model-path ... --config-path ... \\
        --tasks spell
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained
from hnet.utils.byte_tokenizer import ByteTokenizer


# CUTE groups tasks into three families. Used only for category-level
# aggregates in the summary; the runner itself iterates whatever splits
# the loaded dataset exposes.
CUTE_CATEGORIES = {
    "composition":  ("spell", "spell_inverse", "contains_char", "contains_word"),
    "similarity":   ("orth", "sem"),
    "manipulation": (
        "ins_char", "del_char", "sub_char", "swap_char",
        "ins_word", "del_word", "sub_word", "swap_word",
    ),
}


def load_dataset_with_retry(dataset: str, language: Optional[str],
                            attempts: int = 5, base_delay: float = 15.0):
    """Load a dataset, retrying on transient cache-download races.

    If `dataset` is a local directory, every `*.jsonl` in it is loaded as a
    split named after the file stem (the format emitted by
    `eval/make_cute_base.py`). Otherwise it's treated as an HF Hub id.

    When several runs share one HF cache (e.g. a SLURM job array all calling
    `load_dataset` at once), one process can clean up the `.incomplete`
    staging dir out from under another, yielding a spurious "Cannot find data
    file" OSError. The losing process just needs to retry once the winner has
    finished materializing the cache, so we back off and try again.
    """
    local = Path(dataset)
    if local.is_dir():
        data_files = {p.stem: str(p) for p in sorted(local.glob("*.jsonl"))}
        if not data_files:
            raise SystemExit(f"No *.jsonl files found in local dataset dir: {dataset}")
        return load_dataset("json", data_files=data_files)

    for attempt in range(1, attempts + 1):
        try:
            return (
                load_dataset(dataset, language)
                if language else load_dataset(dataset)
            )
        except (OSError, FileNotFoundError) as e:
            if attempt == attempts:
                raise
            delay = base_delay * attempt
            print(
                f"  load_dataset failed (attempt {attempt}/{attempts}), likely a "
                f"concurrent cache-download race; retrying in {delay:.0f}s.\n  {e}",
                flush=True,
            )
            time.sleep(delay)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def char_f1(pred: str, gold: str) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    cp, cg = Counter(pred), Counter(gold)
    common = sum((cp & cg).values())
    if common == 0:
        return 0.0
    p = common / sum(cp.values())
    r = common / sum(cg.values())
    return 2 * p * r / (p + r)


# Strips a leading "Answer:" the model may echo (with optional opening quote).
_ANSWER_PREFIX = re.compile(r'^\s*answer\s*:\s*"?\s*', re.IGNORECASE)


def parse_answer(text: str) -> str:
    """Take the first non-empty line; drop an echoed `Answer:` prefix and quotes.

    If the prompt baked in an answer cue, the model continues from inside the
    answer quote, so its output looks like ` t h e "\\n` — leading space,
    content, trailing close quote. If not, the model may also re-emit the
    `Answer: "` prefix (e.g. ` Answer: " t h e "`). We strip that prefix and any
    wrapping quote chars independently rather than requiring symmetric pairs.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _ANSWER_PREFIX.sub("", line)
        line = line.strip('"\'').strip()
        return line
    return _ANSWER_PREFIX.sub("", text.strip()).strip('"\'').strip()


@torch.no_grad()
def generate_answer(
    model,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    max_context: int,
    stop: list[str],
    device: torch.device,
) -> tuple[str, str]:
    """Greedy-decode at temperature 0 until a stop string or EOS.

    Mirrors `HNetLM.generate_until` (eval/lm_eval_wrapper.py:213-285) but
    inlined to avoid an lm-eval-harness dependency for this script.

    Returns ``(text, raw_full)`` where ``text`` is truncated at the first
    stop string and ``raw_full`` is the untruncated decode of every generated
    token (useful for --debug).
    """
    enc = tokenizer.encode([prompt], add_bos=True)[0]["input_ids"].tolist()
    if len(enc) > max_context:
        enc = [enc[0]] + enc[-(max_context - 1):]
    input_ids = torch.tensor([enc], dtype=torch.long, device=device)

    cache = model.allocate_inference_cache(
        1, len(enc) + max_new_tokens, dtype=torch.bfloat16
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.forward(
            input_ids,
            mask=torch.ones_like(input_ids, dtype=torch.bool),
            inference_params=cache,
        )
    logits = out.logits[0, -1, :]

    gen: list[int] = []
    for _ in range(max_new_tokens):
        nt = int(logits.argmax(dim=-1).item())
        if nt == tokenizer.eos_idx:
            break
        gen.append(nt)
        try:
            partial = tokenizer.decode(gen, errors="replace")
            if any(s in partial for s in stop):
                break
        except (UnicodeDecodeError, ValueError):
            pass
        step_input = torch.tensor([[nt]], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.step(step_input, cache)
        logits = out.logits[0, -1, :]

    try:
        raw_full = tokenizer.decode(gen, errors="replace")
    except (UnicodeDecodeError, ValueError):
        raw_full = ""
    text = raw_full
    for s in stop:
        if s in text:
            text = text[: text.index(s)]
            break
    return text, raw_full


def evaluate_task(
    model,
    tokenizer: ByteTokenizer,
    ds,
    task_name: str,
    max_new_tokens: int,
    max_context: int,
    limit: Optional[int],
    device: torch.device,
    debug: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    # CUTE answers contain neither newlines nor quotes, so both are safe stops
    # (the closing `"` matters when the prompt format quotes the answer).
    stop = ["\n", '"']
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in tqdm(range(n), desc=task_name, leave=False):
        ex = ds[i]
        prompt = ex["prompt"]
        gold = ex["answer"].strip()
        raw, raw_full = generate_answer(
            model, tokenizer, prompt, max_new_tokens, max_context,
            stop=stop, device=device,
        )
        pred = parse_answer(raw)
        denom = max(len(pred), len(gold), 1)
        sim = 1.0 - levenshtein(pred, gold) / denom
        row = {
            "task": task_name,
            "idx": i,
            "gold": gold,
            "pred": pred,
            "raw_output": raw,
            "exact_match": pred == gold,
            "char_similarity": sim,
            "char_f1": char_f1(pred, gold),
        }
        if debug:
            # Verbatim record: the exact prompt fed to the model and the
            # untruncated decode of every generated token, before stop-string
            # truncation (raw_output) and answer parsing (pred).
            row["prompt"] = prompt
            row["raw_output_full"] = raw_full
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "char_similarity": sum(r["char_similarity"] for r in rows) / n,
        "char_f1": sum(r["char_f1"] for r in rows) / n,
    }


def category_aggregates(per_task: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cat, names in CUTE_CATEGORIES.items():
        matched = [per_task[n] for n in names if n in per_task and per_task[n].get("n", 0) > 0]
        if not matched:
            continue
        keys = ("exact_match", "char_similarity", "char_f1")
        out[cat] = {
            "n_tasks": len(matched),
            **{k: sum(t[k] for t in matched) / len(matched) for k in keys},
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model-name", default="hnet")
    parser.add_argument("--dataset", default="leukas/cute",
                        help="HF dataset id. CUTE: leukas/cute. EXECUTE: leukas/execute "
                             "(may need --language).")
    parser.add_argument("--language", default=None,
                        help="Dataset config name (e.g. language code for EXECUTE).")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated split names. Default: all splits.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Examples per task. Useful for smoke tests.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--output-dir", default="eval/results/cute")
    parser.add_argument("--debug", action="store_true",
                        help="Also write a structured JSON file with the verbatim "
                             "prompt and untruncated model response for every example.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path}", flush=True)
    model = load_from_pretrained(args.model_path, args.config_path)
    model.eval()
    tokenizer = ByteTokenizer()

    print(f"Loading dataset {args.dataset}"
          + (f" (config={args.language})" if args.language else ""), flush=True)
    ds_all = load_dataset_with_retry(args.dataset, args.language)
    available = list(ds_all.keys())
    print(f"Available splits: {available}", flush=True)

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in tasks if t not in available]
        if missing:
            raise SystemExit(
                f"Tasks not in dataset splits: {missing}. Available: {available}"
            )
    else:
        tasks = available

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    lang_tag = f"_{args.language}" if args.language else ""
    detail_path = out_dir / f"cute_{args.model_name}{lang_tag}_{run_id}.jsonl"
    summary_path = out_dir / f"cute_{args.model_name}{lang_tag}_{run_id}_summary.json"
    debug_path = out_dir / f"cute_{args.model_name}{lang_tag}_{run_id}_debug.json"

    summary = {
        "model": args.model_name,
        "model_path": args.model_path,
        "config_path": args.config_path,
        "dataset": args.dataset,
        "language": args.language,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "run_id": run_id,
        "tasks": {},
    }

    debug_records: list[dict] = []
    print(f"\nRunning {len(tasks)} task(s) -> {detail_path}\n", flush=True)
    with detail_path.open("w") as f:
        for task in tasks:
            rows = evaluate_task(
                model, tokenizer, ds_all[task], task,
                args.max_new_tokens, args.max_context, args.limit, device,
                debug=args.debug,
            )
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if args.debug:
                debug_records.extend(rows)
            agg = aggregate(rows)
            summary["tasks"][task] = agg
            print(
                f"  {task:<20}  n={agg['n']:<4}  "
                f"EM={agg.get('exact_match', 0):.3f}  "
                f"sim={agg.get('char_similarity', 0):.3f}  "
                f"F1={agg.get('char_f1', 0):.3f}",
                flush=True,
            )

    if summary["tasks"]:
        keys = ("exact_match", "char_similarity", "char_f1")
        n = len(summary["tasks"])
        summary["macro_avg"] = {
            k: sum(t.get(k, 0) for t in summary["tasks"].values()) / n
            for k in keys
        }
        cats = category_aggregates(summary["tasks"])
        if cats:
            summary["category_avg"] = cats

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.debug:
        debug = {
            "model": args.model_name,
            "model_path": args.model_path,
            "config_path": args.config_path,
            "dataset": args.dataset,
            "language": args.language,
            "run_id": run_id,
            "responses": debug_records,
        }
        with debug_path.open("w") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)

    print(f"\nPer-example results: {detail_path}")
    print(f"Summary: {summary_path}")
    if args.debug:
        print(f"Debug responses ({len(debug_records)} examples): {debug_path}")
    if "macro_avg" in summary:
        m = summary["macro_avg"]
        print(
            f"Macro-avg over {len(summary['tasks'])} tasks:  "
            f"EM={m['exact_match']:.3f}  "
            f"sim={m['char_similarity']:.3f}  "
            f"F1={m['char_f1']:.3f}"
        )
    if "category_avg" in summary:
        for cat, v in summary["category_avg"].items():
            print(
                f"  [{cat}] EM={v['exact_match']:.3f}  "
                f"sim={v['char_similarity']:.3f}  "
                f"F1={v['char_f1']:.3f}  ({v['n_tasks']} tasks)"
            )


if __name__ == "__main__":
    main()
