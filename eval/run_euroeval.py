"""Run EuroEval against a locally-served H-Net via the LiteLLM path.

EuroEval refuses to import if `flash_attn` is installed in the same env, but
H-Net needs `flash_attn`. So the server (which loads H-Net) and the driver
(which imports `euroeval`) MUST run in separate Python environments.

Recommended workflow (two envs):

    # Terminal A — hnet env:
    python eval/hnet_openai_server.py \\
        --model-path .../latest.pt --config-path .../hnet.json --port 8765

    # Terminal B — euroeval env:
    python eval/run_euroeval.py \\
        --api-base http://localhost:8765/v1 \\
        --languages en,bg --tasks sentiment-classification

Alternatively, the driver can spawn the server itself if you pass the path to
the hnet env's interpreter via `--server-python`:

    python eval/run_euroeval.py \\
        --start-server --server-python /path/to/hnet-env/bin/python \\
        --model-path .../latest.pt --config-path .../hnet.json \\
        --languages en --tasks sentiment-classification
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_mc_choices(prompt: str, letters: list) -> Optional[list]:
    """Extract the final question's full choice texts from an EuroEval MC prompt.

    EuroEval MC prompts embed the options inline after a localized keyword,
    e.g. '... Choices: a. X b. Y c. Z d. W\\nAnswer:' ('Възможности:'/
    'Отговор:' in bg, etc.). The final line is the answer cue and is dropped;
    the options of the LAST question are anchored on the last 'a.'-style
    marker so few-shot examples (which contain their own options) are skipped.

    Returns a list of choice texts aligned with `letters`, or None if the
    prompt doesn't match the expected structure.
    """
    body = prompt.rstrip()
    if "\n" in body:
        body = body[: body.rfind("\n")]  # drop the answer cue line
    m0 = None
    for m0 in re.finditer(rf"(?:^|\s){re.escape(letters[0])}\.\s", body):
        pass  # keep the LAST match
    if m0 is None:
        return None
    spans = [(m0.start(), m0.end())]
    pos = m0.end()
    for letter in letters[1:]:
        m = re.search(rf"\s{re.escape(letter)}\.\s", body[pos:])
        if m is None:
            return None
        spans.append((pos + m.start(), pos + m.end()))
        pos += m.end()
    texts = []
    for j, (_, end) in enumerate(spans):
        stop = spans[j + 1][0] if j + 1 < len(spans) else len(body)
        texts.append(body[end:stop].strip())
    return texts if all(texts) else None


def wait_for_server(base_url: str, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/models", timeout=2.0)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(1.0)
    raise TimeoutError(f"Server at {base_url} not ready in {timeout}s (last error: {last_err}).")


def _result_to_payload(r) -> dict:
    if hasattr(r, "to_dict"):
        try:
            return r.to_dict()
        except Exception:
            pass
    if hasattr(r, "__dict__"):
        return {k: v for k, v in r.__dict__.items()}
    return {"repr": repr(r)}


def main():
    parser = argparse.ArgumentParser(
        description="Run EuroEval against H-Net via an OpenAI-compatible HTTP server."
    )
    parser.add_argument("--model-path", help="Required when --start-server is set.")
    parser.add_argument("--config-path", help="Required when --start-server is set.")
    parser.add_argument("--api-base", default=None,
                        help="Base URL of a running server, e.g. http://localhost:8765/v1. "
                             "If omitted with --start-server, derived from --host/--port.")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--start-server", action="store_true",
                        help="Spawn eval/hnet_openai_server.py as a subprocess. "
                             "Almost always requires --server-python pointing at the hnet env, "
                             "since euroeval can't coexist with flash_attn in one env.")
    parser.add_argument("--server-python", default=sys.executable,
                        help="Python interpreter used to launch the server subprocess "
                             "(default: this script's interpreter). Point this at the hnet env's "
                             "python when running the driver from a separate euroeval env.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-name", default="hnet-base")
    parser.add_argument("--max-gen-tokens", type=int, default=4096,
                        help="Server-side hard cap on generated bytes per request.")
    parser.add_argument("--max-context-length", type=int, default=8192)
    parser.add_argument("--gen-byte-scale", type=float, default=4.0,
                        help="Server-side multiplier converting client token budgets "
                             "to bytes (only used with --start-server).")
    parser.add_argument("--languages", default="en",
                        help="Comma-separated language codes (e.g. en,bg).")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated EuroEval task names.")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated EuroEval dataset names.")
    parser.add_argument("--output-dir", default="eval/results/euroeval")
    parser.add_argument("--cache-dir", default=".euroeval_cache",
                        help="EuroEval cache directory. Also where the HF "
                             "`evaluate` metric lock files live. Concurrent runs "
                             "(e.g. SLURM array tasks) MUST use distinct dirs: "
                             "evaluate omits experiment_id, so they otherwise "
                             "deadlock on a shared default_experiment-1-0.arrow "
                             "lock ('another evaluation module instance is "
                             "already using the local cache file').")
    parser.add_argument("--server-startup-timeout", type=float, default=300.0)
    parser.add_argument("--debug", action="store_true",
                        help="EuroEval debug mode: dumps every model output to "
                             "<model>-<dataset>-model-outputs.json in the CWD with full "
                             "metadata (prompts, generations, logprobs, predicted vs gold) "
                             "and logs per-sample (input, pred, gold) lines. Implies --verbose.")
    parser.add_argument("--verbose", action="store_true",
                        help="EuroEval verbose logging (auto-on when --debug is set).")
    parser.add_argument("--force", action="store_true",
                        help="Force re-evaluation even if (model, dataset) is already "
                             "in euroeval_benchmark_results.jsonl in the CWD. Without "
                             "this, EuroEval skips cached runs and produces no model "
                             "outputs (so --debug also writes nothing).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug-only: truncate every dataset's test split to the "
                             "first N samples. Monkey-patches euroeval.benchmarker."
                             "load_data; results are not leaderboard-valid.")
    parser.add_argument("--generative-type", default="base",
                        choices=["base", "instruction_tuned", "reasoning"],
                        help="Tells EuroEval whether H-Net should be treated as a base "
                             "LM (raw /v1/completions prompt) or instruction-tuned "
                             "(/v1/chat/completions with messages). H-Net is a base LM, "
                             "so the default is 'base'.")
    parser.add_argument("--cloze-scoring", dest="cloze_scoring",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="For sequence-classification and multiple-choice tasks, "
                             "skip free-form generation and instead score each candidate "
                             "answer as a continuation via /v1/loglikelihood. For MC "
                             "tasks the candidates are the FULL choice texts parsed from "
                             "the prompt (not the option letters); for classification "
                             "they are the localized label words. Sidesteps EuroEval's "
                             "first-token prefix-match which is broken for byte-level "
                             "models with multi-byte (e.g. Cyrillic) labels. Default: on.")
    parser.add_argument("--cloze-norm", default="byte", choices=["byte", "none"],
                        help="Length normalization for cloze candidate selection: 'byte' "
                             "picks the highest per-byte log-prob (removes the length "
                             "penalty of raw sums); 'none' uses the raw summed log-prob. "
                             "All scores are dumped to <output-dir>/cloze_scores/ either "
                             "way. Default: byte.")
    args = parser.parse_args()

    api_base = args.api_base or f"http://{args.host}:{args.port}/v1"
    server_proc: Optional[subprocess.Popen] = None

    try:
        if args.start_server:
            if not args.model_path or not args.config_path:
                parser.error("--model-path and --config-path are required when --start-server is set.")
            server_script = Path(__file__).resolve().parent / "hnet_openai_server.py"
            cmd = [
                args.server_python, str(server_script),
                "--model-path", args.model_path,
                "--config-path", args.config_path,
                "--host", args.host,
                "--port", str(args.port),
                "--model-name", args.model_name,
                "--max-gen-tokens", str(args.max_gen_tokens),
                "--max-context-length", str(args.max_context_length),
                "--gen-byte-scale", str(args.gen_byte_scale),
            ]
            print(f"Starting H-Net server: {' '.join(cmd)}", flush=True)
            server_proc = subprocess.Popen(cmd)
            print(f"Waiting for {api_base} to be ready...", flush=True)
            wait_for_server(api_base, timeout=args.server_startup_timeout)
            print("Server is ready.\n", flush=True)
        else:
            wait_for_server(api_base, timeout=10.0)
            print(f"Using server at {api_base}.\n", flush=True)

        from euroeval import Benchmarker
        from euroeval.enums import GenerativeType

        gtype = GenerativeType[args.generative_type.upper()]

        if gtype == GenerativeType.BASE:
            # Upstream bug: euroeval/benchmark_modules/litellm.py:881 checks
            # `response.choices[0].message.content == "{}"` on every response,
            # but the /v1/completions path returns TextChoices (with `.text`,
            # not `.message`). Add a compatibility shim so the empty-output
            # check works for both response shapes.
            from litellm.types.utils import TextChoices

            class _MessageAdapter:
                __slots__ = ("content",)
                def __init__(self, content):
                    self.content = content

            if not hasattr(TextChoices, "_euroeval_message_patch"):
                TextChoices.message = property(
                    lambda self: _MessageAdapter(getattr(self, "text", "") or "")
                )
                TextChoices._euroeval_message_patch = True
                print("[run_euroeval] Patched TextChoices.message for BASE-LM path.",
                      flush=True)

        if args.cloze_scoring:
            # Replace LiteLLMModel.generate with a continuation-cloze scorer for
            # sequence-classification and multiple-choice tasks: score each
            # candidate ANSWER as a continuation of the prompt via
            # /v1/loglikelihood (loss restricted to the answer bytes) and pick
            # the best by per-byte-normalized log-prob (--cloze-norm). For MC
            # tasks the candidates are the full choice texts parsed from the
            # prompt, and the returned sequence is the corresponding option
            # letter. Sidesteps EuroEval's first-token prefix-match (broken for
            # byte-level models with multi-byte / shared-prefix labels). Other
            # task groups (TEXT_TO_TEXT, NER, QA, SPEED) fall through to the
            # original method.
            from euroeval.benchmark_modules.litellm import LiteLLMModel
            from euroeval.enums import TaskGroup
            from euroeval.data_models import GenerativeModelOutput

            _CLOZE_TASK_GROUPS = {
                TaskGroup.SEQUENCE_CLASSIFICATION,
                TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
            }
            _orig_litellm_generate = LiteLLMModel.generate
            _cloze_base = api_base.rstrip("/")
            _cloze_norm = args.cloze_norm
            _scores_dir = Path(args.output_dir) / "cloze_scores"
            _scores_dir.mkdir(parents=True, exist_ok=True)
            _sample_counters: dict = {}

            def _flatten_messages(msgs):
                # Match the server's chat-completions flattening: concatenate
                # `content` fields with newlines.
                return "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))

            def _cloze_generate(self, inputs):
                tg = getattr(self.dataset_config.task, "task_group", None)
                if tg not in _CLOZE_TASK_GROUPS:
                    return _orig_litellm_generate(self, inputs)

                if "text" in inputs:
                    prompts = list(inputs["text"])
                elif "messages" in inputs:
                    prompts = [_flatten_messages(m) for m in inputs["messages"]]
                else:
                    return _orig_litellm_generate(self, inputs)

                local_labels = [
                    self.dataset_config.prompt_label_mapping[label].strip()
                    for label in self.dataset_config.labels
                ]
                if not local_labels:
                    return _orig_litellm_generate(self, inputs)

                is_mc = tg == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
                ds_name = getattr(self.dataset_config, "name", "unknown")
                sidecar_path = _scores_dir / f"{args.model_name}-{ds_name}.jsonl"

                sequences: list = []
                failed: list = []
                with httpx.Client(timeout=120.0) as client, open(sidecar_path, "a") as sidecar:
                    for i, prompt in enumerate(prompts):
                        sample_idx = _sample_counters.get(ds_name, 0)
                        _sample_counters[ds_name] = sample_idx + 1

                        # Candidates: full choice texts for MC (fall back to the
                        # option letters if parsing fails), label words otherwise.
                        parse_fallback = False
                        if is_mc:
                            candidates = parse_mc_choices(prompt, local_labels)
                            if candidates is None:
                                candidates = local_labels
                                parse_fallback = True
                        else:
                            candidates = local_labels

                        try:
                            r = client.post(
                                f"{_cloze_base}/loglikelihood",
                                json={"prompt": prompt, "continuations": candidates},
                            )
                            r.raise_for_status()
                            data = r.json()["data"]
                        except Exception as e:
                            failed.append({
                                "sample_index": i,
                                "error": f"loglikelihood call failed: {e!r}",
                            })
                            sequences.append("")
                            continue

                        for d in data:
                            d["logprob_per_byte"] = d["logprob"] / max(1, d["n_bytes"])
                        key = "logprob_per_byte" if _cloze_norm == "byte" else "logprob"
                        best = max(range(len(data)), key=lambda j: data[j][key])
                        # Report the label EuroEval expects: the option letter
                        # for MC, the label word for classification.
                        sequences.append(local_labels[best])

                        sidecar.write(json.dumps({
                            "sample_index": sample_idx,
                            "norm": _cloze_norm,
                            "parse_fallback": parse_fallback,
                            "chosen_label": local_labels[best],
                            "candidates": [{
                                "label": local_labels[j],
                                "text": candidates[j],
                                "logprob": data[j]["logprob"],
                                "n_bytes": data[j]["n_bytes"],
                                "logprob_per_byte": data[j]["logprob_per_byte"],
                            } for j in range(len(data))],
                        }, ensure_ascii=False) + "\n")

                # scores=None forces EuroEval's label extractor to skip the
                # first-token logprob path and use prefix-match on `sequences`,
                # which will find our exact-label string unambiguously.
                return GenerativeModelOutput(
                    sequences=sequences, scores=None, failed_instances=failed,
                )

            LiteLLMModel.generate = _cloze_generate
            print(
                "[run_euroeval] Cloze scoring ON for "
                f"{{{', '.join(t.name for t in _CLOZE_TASK_GROUPS)}}} tasks "
                f"(full-choice-text candidates, norm={_cloze_norm}, "
                f"via {_cloze_base}/loglikelihood; scores -> {_scores_dir}).",
                flush=True,
            )

        # Redirect the per-batch model-outputs JSON that --debug writes:
        # default location is CWD with name "<model_id>-<dataset>-model-outputs.json"
        # (euroeval/generation.py:62-72). We rewrite the path to args.output_dir
        # and inject the prompt style (base vs instruct) into the filename so
        # base- and instruction-tuned runs of the same (model, dataset) don't
        # overwrite each other.
        if args.debug:
            from euroeval.model_cache import ModelCache as _ModelCache

            out_dir_abs = Path(args.output_dir).resolve()
            out_dir_abs.mkdir(parents=True, exist_ok=True)
            style_tag = "base" if gtype == GenerativeType.BASE else "instruct"

            if not getattr(_ModelCache, "_hnet_path_patch", False):
                _orig_mc_init = _ModelCache.__init__

                def _patched_mc_init(self, model_cache_dir, cache_name, *a, **kw):
                    if cache_name.endswith("-model-outputs.json"):
                        cache_name = cache_name.replace(
                            "-model-outputs.json",
                            f"-{style_tag}-model-outputs.json",
                        )
                        model_cache_dir = out_dir_abs
                    _orig_mc_init(self, model_cache_dir, cache_name, *a, **kw)

                _ModelCache.__init__ = _patched_mc_init
                _ModelCache._hnet_path_patch = True
                print(
                    f"[run_euroeval] Debug model-outputs will be written to "
                    f"{out_dir_abs} with -{style_tag}- in the filename.",
                    flush=True,
                )

        if args.limit is not None and args.limit > 0:
            import euroeval.benchmarker as _bench_mod
            _orig_load_data = _bench_mod.load_data

            def _limited_load_data(*a, **kw):
                datasets = _orig_load_data(*a, **kw)
                # `datasets` is list[DatasetDict] of length num_iterations.
                # Truncate every split in every iteration to args.limit rows.
                for d in datasets:
                    for split_name in list(d.keys()):
                        if len(d[split_name]) > args.limit:
                            d[split_name] = d[split_name].select(range(args.limit))
                return datasets

            _bench_mod.load_data = _limited_load_data
            print(f"[run_euroeval] --limit {args.limit}: truncating each split.",
                  flush=True)

        bench = Benchmarker(
            api_key=args.api_key,
            api_base=api_base,
            progress_bar=True,
            verbose=args.verbose or args.debug,
            debug=args.debug,
            force=args.force,
            generative_type=gtype,
            cache_dir=args.cache_dir,
        )

        run_kwargs: dict = {"model": args.model_name}
        if args.tasks:
            run_kwargs["task"] = [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.datasets:
            run_kwargs["dataset"] = [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.languages:
            run_kwargs["language"] = [l.strip() for l in args.languages.split(",") if l.strip()]
        # if args.cloze:
        #     run_kwargs["use_bits_per_character"] = True

        print(f"Running Benchmarker.benchmark({run_kwargs})\n", flush=True)
        results = bench.benchmark(**run_kwargs)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"euroeval_{args.model_name}_{run_id}.jsonl"
        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(_result_to_payload(r), default=str) + "\n")

        print(f"\nResults written to {out_path}")
        print("EuroEval also writes its canonical log to euroeval_benchmark_results.jsonl in the CWD.")

        print("\nSummary:")
        for r in results:
            ds = getattr(r, "dataset", None) or getattr(r, "dataset_config", None) or "<unknown>"
            scores = getattr(r, "results", None) or getattr(r, "scores", None) or "<no scores>"
            print(f"  {ds}: {scores}")

    finally:
        if server_proc is not None:
            print("Stopping H-Net server...", flush=True)
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
