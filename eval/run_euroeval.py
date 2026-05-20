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
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    parser.add_argument("--max-gen-tokens", type=int, default=1024)
    parser.add_argument("--max-context-length", type=int, default=8192)
    parser.add_argument("--languages", default="en",
                        help="Comma-separated language codes (e.g. en,bg).")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated EuroEval task names.")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated EuroEval dataset names.")
    parser.add_argument("--output-dir", default="eval/results/euroeval")
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
        )

        run_kwargs: dict = {"model": args.model_name}
        if args.tasks:
            run_kwargs["task"] = [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.datasets:
            run_kwargs["dataset"] = [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.languages:
            run_kwargs["language"] = [l.strip() for l in args.languages.split(",") if l.strip()]

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
