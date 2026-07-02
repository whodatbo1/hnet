"""Re-format CUTE into a base-model-friendly few-shot dataset.

CUTE (`leukas/cute`) ships every example as one fully-rendered, instruction-
tuned `prompt` string: a verbose preamble, four few-shot examples, and a final
`Question:` line, all carrying ~12 spaces of triple-quoted-string indentation.
That phrasing assumes an instruction-tuned model. This script rebuilds the same
tasks with simpler, configurable phrasing aimed at base (non-instruct) models.

How it works
------------
The CUTE input structure is fixed, so *extraction* is hardcoded here (this is
the part that must be exact). For every task we split each prompt into:

    <preamble>                       (discarded; we supply our own)
    1. <example body>. Answer: <ans> (x4; the few-shot block)
    ...
    Question: <question body>

and pull the task's *variables* out of each body (e.g. spell -> {word};
orth/sem -> {target, opt1, opt2}; sub_char -> {old, new, word}). Variables are
quoted spans in the body, so we read them positionally; orth/sem get a lenient
parser because one upstream few-shot example is malformed (`"glad, "apply"`).
The few-shot block is byte-identical across all rows of a task, so it is
extracted once. Per-row ground truth comes straight from the dataset `answer`.

*Formatting* is entirely config-driven. A JSON config maps each task to a
`preamble`, an `example_format`, and a `question_format` (plus optional
`example_separator` / `section_separator`). Templates are Python str.format
strings over the task's variable names plus `{i}` (1-based example index) and,
for `example_format`, `{answer}`. Use --dump-default-config to get an editable
copy of the built-in defaults.

Output mirrors CUTE's schema (`{prompt, answer}`) as one JSONL per task, ending
at the `Question:` line (no answer cue) so it is a drop-in for run_cute.py:

    python eval/make_cute_base.py --output-dir eval/data/cute_base
    python eval/run_cute.py --dataset eval/data/cute_base \\
        --model-path ... --config-path ... --prompt-style cue
"""

import argparse
import copy
import json
import re
from pathlib import Path

from datasets import load_dataset


# Ordered variable names per task. For every task except orth/sem these map
# positionally onto the quoted spans in the example/question body.
TASK_VARS = {
    "spell":          ["word"],
    "spell_inverse":  ["spelled"],
    "contains_char":  ["char", "word"],
    "contains_word":  ["word", "sentence"],
    "orth":           ["target", "opt1", "opt2"],   # custom parser
    "sem":            ["target", "opt1", "opt2"],    # custom parser
    "ins_char":       ["add", "after", "word"],
    "ins_word":       ["add", "after", "sentence"],
    "del_char":       ["char", "word"],
    "del_word":       ["word", "sentence"],
    "sub_char":       ["old", "new", "word"],
    "sub_word":       ["old", "new", "sentence"],
    "swap_char":      ["first", "second", "word"],
    "swap_word":      ["first", "second", "sentence"],
}
CUSTOM_TASKS = {"orth", "sem"}

# Built-in base-model formats. example_format ends with `Answer: "{answer}"` so
# the quoted-answer convention matches run_cute.py's appended ` Answer: "` cue.
DEFAULT_FORMATS = {
    "spell": {
        "preamble": "Spell out each word by putting a single space between every letter.",
        "example_format": '{i}. Spell out the word "{word}". Answer: "{answer}"',
        "question_format": 'Question: Spell out the word "{word}".',
    },
    "spell_inverse": {
        "preamble": "Write each spelled-out word back as a normal word, without any spaces.",
        "example_format": '{i}. Write the word "{spelled}". Answer: "{answer}"',
        "question_format": 'Question: Write the word "{spelled}".',
    },
    "contains_char": {
        "preamble": "Answer whether the given letter appears in the given word. Answer Yes or No.",
        "example_format": '{i}. Is there a "{char}" in "{word}"? Answer: "{answer}"',
        "question_format": 'Question: Is there a "{char}" in "{word}"?',
    },
    "contains_word": {
        "preamble": "Answer whether the given word appears in the given sentence "
                    "(case insensitive). Answer Yes or No.",
        "example_format": '{i}. Is there a "{word}" in "{sentence}"? Answer: "{answer}"',
        "question_format": 'Question: Is there a "{word}" in "{sentence}"?',
    },
    "orth": {
        "preamble": "Pick which of the two candidate words is closer in spelling "
                    "(Levenshtein edit distance) to the given word.",
        "example_format": '{i}. Closer to "{target}": "{opt1}" or "{opt2}"? Answer: "{answer}"',
        "question_format": 'Question: Closer to "{target}": "{opt1}" or "{opt2}"?',
    },
    "sem": {
        "preamble": "Pick which of the two candidate words is more closely related "
                    "in meaning to the given word.",
        "example_format": '{i}. More related to "{target}": "{opt1}" or "{opt2}"? Answer: "{answer}"',
        "question_format": 'Question: More related to "{target}": "{opt1}" or "{opt2}"?',
    },
    "ins_char": {
        "preamble": "Insert the first letter immediately after every occurrence of "
                    "the second letter in the given word.",
        "example_format": '{i}. Add "{add}" after every "{after}" in "{word}". Answer: "{answer}"',
        "question_format": 'Question: Add "{add}" after every "{after}" in "{word}".',
    },
    "ins_word": {
        "preamble": "Insert the first word immediately after every occurrence of "
                    "the second word in the given sentence.",
        "example_format": '{i}. Add "{add}" after every "{after}" in "{sentence}". Answer: "{answer}"',
        "question_format": 'Question: Add "{add}" after every "{after}" in "{sentence}".',
    },
    "del_char": {
        "preamble": "Delete every occurrence of the given letter from the given word.",
        "example_format": '{i}. Delete every "{char}" in "{word}". Answer: "{answer}"',
        "question_format": 'Question: Delete every "{char}" in "{word}".',
    },
    "del_word": {
        "preamble": "Delete every occurrence of the given word from the given sentence.",
        "example_format": '{i}. Delete every "{word}" in "{sentence}". Answer: "{answer}"',
        "question_format": 'Question: Delete every "{word}" in "{sentence}".',
    },
    "sub_char": {
        "preamble": "Replace every occurrence of the first letter with the second "
                    "letter in the given word.",
        "example_format": '{i}. Substitute "{old}" with "{new}" in "{word}". Answer: "{answer}"',
        "question_format": 'Question: Substitute "{old}" with "{new}" in "{word}".',
    },
    "sub_word": {
        "preamble": "Replace every occurrence of the first word with the second "
                    "word in the given sentence.",
        "example_format": '{i}. Substitute "{old}" with "{new}" in "{sentence}". Answer: "{answer}"',
        "question_format": 'Question: Substitute "{old}" with "{new}" in "{sentence}".',
    },
    "swap_char": {
        "preamble": "Swap the positions of the two given letters in the given word.",
        "example_format": '{i}. Swap "{first}" and "{second}" in "{word}". Answer: "{answer}"',
        "question_format": 'Question: Swap "{first}" and "{second}" in "{word}".',
    },
    "swap_word": {
        "preamble": "Swap the positions of the two given words in the given sentence.",
        "example_format": '{i}. Swap "{first}" and "{second}" in "{sentence}". Answer: "{answer}"',
        "question_format": 'Question: Swap "{first}" and "{second}" in "{sentence}".',
    },
}

_QUOTED = re.compile(r'"([^"]*)"')
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


# ---------------------------------------------------------------------------
# Extraction (hardcoded against CUTE's fixed structure).
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    """Strip surrounding whitespace and a wrapping layer of quote chars."""
    return s.strip().strip('"\'').strip()


def split_examples(prompt: str) -> list[str]:
    """Return the four few-shot example strings (each `<body>. Answer: <ans>`)."""
    block = re.search(r"(\n\s*1\..*?)(?=\n\s*Question:)", prompt, re.S)
    if not block:
        return []
    parts = re.split(r"(?:^|\n)\s*\d+\.\s+", block.group(1))
    return [p.strip() for p in parts if p.strip()]


def question_body(prompt: str) -> str:
    m = re.search(r"\n\s*Question:\s*(.*)\s*\Z", prompt, re.S)
    return m.group(1).strip() if m else ""


def split_answer(example: str) -> tuple[str, str]:
    """Split `<body>. Answer: <ans>` into (body, raw_answer)."""
    parts = re.split(r"\bAnswer:\s*", example, maxsplit=1)
    body = parts[0].rstrip()
    answer = parts[1].strip() if len(parts) > 1 else ""
    return body, answer


def _extract_orth_sem(body: str) -> dict:
    """orth/sem: `... to "<target>": "<opt1>", "<opt2>".`

    One upstream few-shot example is malformed (`"happy": "glad, "apply"`), so
    we read the target as the first quoted span and recover the two options
    leniently from the remainder (quoted spans if well-formed, else word runs).
    """
    spans = _QUOTED.findall(body)
    target = spans[0].strip() if spans else ""
    # Remainder after the first quoted span (the target).
    first_close = body.find('"', body.find('"') + 1)
    rest = body[first_close + 1:] if first_close != -1 else ""
    opt_spans = _QUOTED.findall(rest)
    if len(opt_spans) >= 2:
        opt1, opt2 = opt_spans[0].strip(), opt_spans[-1].strip()
    else:
        words = _WORD.findall(rest)
        opt1, opt2 = (words[0], words[-1]) if len(words) >= 2 else ("", "")
    return {"target": target, "opt1": opt1, "opt2": opt2}


def extract_vars(task: str, body: str) -> dict | None:
    """Extract the task's variables from an example/question body.

    Returns None if the number of quoted spans doesn't match the expected
    variable count (a parse failure worth reporting).
    """
    if task in CUSTOM_TASKS:
        return _extract_orth_sem(body)
    names = TASK_VARS[task]
    spans = [s.strip() for s in _QUOTED.findall(body)]
    if len(spans) != len(names):
        return None
    return dict(zip(names, spans))


def extract_fewshot(task: str, prompt: str) -> list[tuple[dict, str]]:
    """Return [(vars, answer), ...] for the four (constant) few-shot examples."""
    out = []
    for ex in split_examples(prompt):
        body, raw_ans = split_answer(ex)
        out.append((extract_vars(task, body), _clean(raw_ans)))
    return out


# ---------------------------------------------------------------------------
# Rendering (config-driven).
# ---------------------------------------------------------------------------

def render_prompt(fmt: dict, fewshot: list[tuple[dict, str]], qvars: dict) -> str:
    ex_sep = fmt.get("example_separator", "\n")
    sec_sep = fmt.get("section_separator", "\n\n")
    examples = [
        fmt["example_format"].format(i=k, answer=ans, **vars_)
        for k, (vars_, ans) in enumerate(fewshot, 1)
    ]
    question = fmt["question_format"].format(**qvars)
    return sec_sep.join([fmt["preamble"], ex_sep.join(examples), question])


def load_config(path: str | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_FORMATS)
    if path:
        with open(path) as f:
            override = json.load(f)
        for task, spec in override.items():
            cfg.setdefault(task, {}).update(spec)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="leukas/cute",
                        help="Source HF dataset id (CUTE-schema).")
    parser.add_argument("--language", default=None,
                        help="Dataset config name (e.g. language for EXECUTE).")
    parser.add_argument("--config", default=None,
                        help="JSON of per-task formats; merged over the built-in defaults.")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated task names. Default: all available splits.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows written per task (smoke test).")
    parser.add_argument("--output-dir", default="eval/data/cute_base")
    parser.add_argument("--dump-default-config", default=None,
                        help="Write the built-in default formats to this path and exit.")
    args = parser.parse_args()

    if args.dump_default_config:
        Path(args.dump_default_config).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_default_config, "w") as f:
            json.dump(DEFAULT_FORMATS, f, indent=2, ensure_ascii=False)
        print(f"Wrote default config -> {args.dump_default_config}")
        return

    cfg = load_config(args.config)
    ds_all = (
        load_dataset(args.dataset, args.language)
        if args.language else load_dataset(args.dataset)
    )
    available = list(ds_all.keys())

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in tasks if t not in available]
        if missing:
            raise SystemExit(f"Tasks not in dataset: {missing}. Available: {available}")
    else:
        tasks = available

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.dataset}  ->  {out_dir}\n")
    for task in tasks:
        if task not in TASK_VARS:
            print(f"  [skip] {task}: no extractor defined (not a known CUTE task).")
            continue
        if task not in cfg:
            print(f"  [skip] {task}: no format in config.")
            continue
        ds = ds_all[task]
        fewshot = extract_fewshot(task, ds[0]["prompt"])

        # Validate few-shot extraction up front — a bad few-shot poisons every row.
        bad_fs = [k for k, (v, _) in enumerate(fewshot, 1) if v is None]
        if len(fewshot) != 4 or bad_fs:
            print(f"  [warn] {task}: few-shot parse issue "
                  f"(got {len(fewshot)} examples, bad={bad_fs}).")

        n = len(ds) if args.limit is None else min(args.limit, len(ds))
        path = out_dir / f"{task}.jsonl"
        n_fail = 0
        with path.open("w") as f:
            for i in range(n):
                row = ds[i]
                qvars = extract_vars(task, question_body(row["prompt"]))
                if qvars is None:
                    n_fail += 1
                    continue
                prompt = render_prompt(cfg[task], fewshot, qvars)
                f.write(json.dumps(
                    {"prompt": prompt, "answer": row["answer"]}, ensure_ascii=False
                ) + "\n")
        tag = f"  ({n_fail} rows failed to parse)" if n_fail else ""
        print(f"  {task:<15} {n - n_fail:>5} rows -> {path}{tag}")

    # Show one fully-rendered sample so the user can eyeball the new format.
    if tasks:
        sample_task = next((t for t in tasks if t in cfg and t in TASK_VARS), None)
        if sample_task:
            ds = ds_all[sample_task]
            fewshot = extract_fewshot(sample_task, ds[0]["prompt"])
            qvars = extract_vars(sample_task, question_body(ds[0]["prompt"]))
            print(f"\n--- sample rendered prompt [{sample_task}] ---")
            print(render_prompt(cfg[sample_task], fewshot, qvars))
            print(f"--- gold answer: {ds[0]['answer']!r} ---")


if __name__ == "__main__":
    main()
