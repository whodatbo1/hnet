"""OpenAI-compatible HTTP server wrapping H-Net.

Lets EuroEval (via its `LiteLLMModel` path) treat a locally-loaded H-Net as a
hosted OpenAI endpoint. Exposes:

    GET  /v1/models                 -> static model list
    POST /v1/completions            -> text-completion endpoint (primary path for base LMs)
    POST /v1/chat/completions       -> chat endpoint (concatenates messages as a base-LM prompt)
    POST /v1/loglikelihood          -> non-standard: score candidate continuations of a prompt.
                                       Used by EuroEval cloze-scoring mode for classification.

Unsupported OpenAI features (response_format, tools, streaming) are rejected
with HTTP 422 and an `unsupported_param` error body; LiteLLM translates that
to `UnsupportedParamsError`, on which EuroEval automatically retries without
the offending field.

Usage:
    python eval/hnet_openai_server.py \\
        --model-path /path/to/latest.pt \\
        --config-path /path/to/hnet.json \\
        --port 8765
"""

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained
from hnet.utils.byte_tokenizer import ByteTokenizer


MODEL = None
TOKENIZER: Optional[ByteTokenizer] = None
DEVICE: Optional[torch.device] = None
MODEL_NAME = "hnet-base"
MAX_GEN_TOKENS = 1024
MAX_CONTEXT_LENGTH = 8192
GEN_LOCK = asyncio.Lock()


def _byte_to_token_str(b: int) -> str:
    if 0x20 <= b < 0x7F:
        return chr(b)
    return f"<0x{b:02X}>"


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumprob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumprob > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    drop_idx = sorted_indices[remove]
    logits[drop_idx] = -float("inf")
    return logits


@torch.inference_mode()
def _generate(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list,
    seed: Optional[int],
    want_logprobs: bool,
    top_logprobs_k: int,
) -> dict:
    enc = TOKENIZER.encode([prompt], add_bos=True)[0]["input_ids"].tolist()
    if len(enc) > MAX_CONTEXT_LENGTH:
        enc = [TOKENIZER.bos_idx] + enc[-(MAX_CONTEXT_LENGTH - 1):]

    max_tokens = max(1, min(int(max_tokens), MAX_GEN_TOKENS))
    if seed is not None:
        torch.manual_seed(int(seed))

    input_ids = torch.tensor([enc], dtype=torch.long, device=DEVICE)
    cache = MODEL.allocate_inference_cache(
        1, len(enc) + max_tokens, dtype=torch.bfloat16
    )

    autocast_device = "cuda" if DEVICE.type == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        out = MODEL.forward(input_ids, mask=mask, inference_params=cache)
    logits = out.logits[0, -1, :].float()

    generated_bytes: list = []
    logprob_records: list = []
    finish_reason = "length"
    text_so_far = ""

    for _ in range(max_tokens):
        if temperature <= 0:
            chosen = int(torch.argmax(logits).item())
        else:
            scaled = logits / max(temperature, 1e-6)
            scaled = _apply_top_p(scaled, top_p)
            probs = torch.softmax(scaled, dim=-1)
            chosen = int(torch.multinomial(probs, 1).item())

        if want_logprobs:
            base_lp = F.log_softmax(logits, dim=-1)
            chosen_lp = float(base_lp[chosen].item())
            k = max(1, min(int(top_logprobs_k) or 1, 20))
            topk_vals, topk_idx = torch.topk(base_lp, k=k)
            top_lp_records = [
                {
                    "token": _byte_to_token_str(int(i.item())),
                    "logprob": float(v.item()),
                    "bytes": [int(i.item())],
                }
                for v, i in zip(topk_vals, topk_idx)
            ]
            logprob_records.append({
                "token": _byte_to_token_str(chosen),
                "logprob": chosen_lp,
                "bytes": [chosen],
                "top_logprobs": top_lp_records,
            })

        if chosen == TOKENIZER.eos_idx:
            finish_reason = "stop"
            break

        generated_bytes.append(chosen)

        try:
            text_so_far = TOKENIZER.decode(generated_bytes, errors="replace")
        except Exception:
            text_so_far = ""
        hit = next((s for s in (stop or []) if s and s in text_so_far), None)
        if hit is not None:
            text_so_far = text_so_far.split(hit, 1)[0]
            finish_reason = "stop"
            break

        next_ids = torch.tensor([[chosen]], dtype=torch.long, device=DEVICE)
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            out = MODEL.step(next_ids, cache)
        logits = out.logits[0, -1, :].float()

    if finish_reason == "length":
        text_so_far = TOKENIZER.decode(generated_bytes, errors="replace")

    return {
        "text": text_so_far,
        "finish_reason": finish_reason,
        "logprobs": logprob_records if want_logprobs else None,
        "prompt_tokens": len(enc),
        "completion_tokens": len(generated_bytes),
    }


# Fields we silently accept but ignore. EuroEval/LiteLLM sets response_format
# for constrained classification; if we reject with 422 LiteLLM raises
# BadRequestError, which EuroEval's retry loop doesn't catch (it only handles
# its own client-side UnsupportedParamsError). So we accept and generate
# free-form text. Downstream parsing may still fail per-sample, but the
# benchmark completes instead of aborting.
SILENTLY_IGNORED_FIELDS = ("response_format", "tools", "tool_choice", "functions", "function_call")


def _reject_streaming(body: dict) -> Optional[JSONResponse]:
    if body.get("stream"):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Streaming is not supported by this server",
                    "type": "invalid_request_error",
                    "param": "stream",
                    "code": "stream_unsupported",
                }
            },
        )
    return None


def _common_params(body: dict) -> dict:
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]

    # Accept BOTH logprob conventions:
    #   - Chat-completions: logprobs=True (bool) + top_logprobs=N (int)
    #   - Legacy text-completion (OpenAI v1/completions): logprobs=N (int),
    #     no top_logprobs field. LiteLLM converts to this form when its
    #     text-completion adapter rejects top_logprobs.
    lp = body.get("logprobs", False)
    tlp = body.get("top_logprobs")
    if isinstance(lp, bool):
        want_logprobs = lp
        top_logprobs_k = int(tlp or 0)
    else:
        # lp is numeric (legacy form)
        want_logprobs = bool(lp)
        top_logprobs_k = max(int(lp or 0), int(tlp or 0))

    return {
        "max_tokens": int(body.get("max_completion_tokens") or body.get("max_tokens") or 256),
        "temperature": float(body.get("temperature", 0.0)),
        "top_p": float(body.get("top_p", 1.0)),
        "stop": stop,
        "seed": body.get("seed"),
        "want_logprobs": want_logprobs,
        "top_logprobs_k": top_logprobs_k,
    }


app = FastAPI()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "hnet",
        }],
    }


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    rej = _reject_streaming(body)
    if rej is not None:
        return rej

    prompt = body.get("prompt")
    if isinstance(prompt, list):
        if len(prompt) != 1 or not isinstance(prompt[0], str):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "prompt must be a string or a single-element list of strings", "type": "invalid_request_error"}},
            )
        prompt = prompt[0]
    if not isinstance(prompt, str):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "prompt is required and must be a string", "type": "invalid_request_error"}},
        )

    params = _common_params(body)

    async with GEN_LOCK:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            _generate,
            prompt,
            params["max_tokens"],
            params["temperature"],
            params["top_p"],
            params["stop"],
            params["seed"],
            params["want_logprobs"],
            params["top_logprobs_k"],
        )

    choice: dict[str, Any] = {
        "index": 0,
        "text": result["text"],
        "finish_reason": result["finish_reason"],
        "logprobs": None,
    }
    if params["want_logprobs"] and result["logprobs"] is not None:
        choice["logprobs"] = {
            "tokens": [r["token"] for r in result["logprobs"]],
            "token_logprobs": [r["logprob"] for r in result["logprobs"]],
            "top_logprobs": [
                {alt["token"]: alt["logprob"] for alt in r["top_logprobs"]}
                for r in result["logprobs"]
            ],
            "text_offset": [],
        }

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [choice],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    rej = _reject_streaming(body)
    if rej is not None:
        return rej

    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "messages must be a non-empty list", "type": "invalid_request_error"}},
        )

    parts = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if isinstance(content, str):
            parts.append(content)
    prompt = "\n".join(parts)

    params = _common_params(body)

    async with GEN_LOCK:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            _generate,
            prompt,
            params["max_tokens"],
            params["temperature"],
            params["top_p"],
            params["stop"],
            params["seed"],
            params["want_logprobs"],
            params["top_logprobs_k"],
        )

    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": result["text"]},
        "finish_reason": result["finish_reason"],
        "logprobs": None,
    }
    if params["want_logprobs"] and result["logprobs"] is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "token": r["token"],
                    "logprob": r["logprob"],
                    "bytes": r["bytes"],
                    "top_logprobs": r["top_logprobs"],
                }
                for r in result["logprobs"]
            ]
        }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [choice],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


@torch.inference_mode()
def _loglikelihood_impl(prompt: str, continuations: list) -> list:
    """For each (prompt, continuation), compute sum_i log P(byte_i | prompt + cont[:i]).

    Equivalent to lm-eval-harness' `loglikelihood` primitive but exposed as
    HTTP so EuroEval can call it. Used to do continuation-cloze scoring of
    classification/MCQ labels: score each candidate label as a continuation
    and pick the one with the highest log-probability.

    Returns a list of dicts {continuation, logprob, n_bytes, per_token_logprobs}
    in the same order as the input.
    """
    prompt_ids = TOKENIZER.encode([prompt], add_bos=True)[0]["input_ids"].tolist()
    cont_ids_list = [
        TOKENIZER.encode([c], add_bos=False, add_eos=False)[0]["input_ids"].tolist()
        for c in continuations
    ]
    if any(len(c) == 0 for c in cont_ids_list):
        raise ValueError("All continuations must encode to >= 1 byte.")

    # Left-truncate the prompt so prompt + longest-continuation fits in context.
    # Preserve BOS at index 0 because dynamic chunking needs a boundary token.
    max_cont = max(len(c) for c in cont_ids_list)
    if len(prompt_ids) + max_cont > MAX_CONTEXT_LENGTH:
        budget = MAX_CONTEXT_LENGTH - max_cont
        budget = max(2, budget)
        prompt_ids = [TOKENIZER.bos_idx] + prompt_ids[-(budget - 1):]

    sequences = [prompt_ids + c for c in cont_ids_list]
    max_len = max(len(s) for s in sequences)

    # Pad with null bytes (vocab idx 0). Matches the padding scheme used by
    # eval/lm_eval_wrapper.py:_loglikelihood_tokens — logits at padded positions
    # are garbage but never read; we slice only the continuation region.
    padded = torch.zeros(
        (len(sequences), max_len), dtype=torch.long, device=DEVICE
    )
    for i, s in enumerate(sequences):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=DEVICE)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = MODEL(padded, mask=None)
    logits = output.logits.float()  # (B, L, V)

    results = []
    prompt_len = len(prompt_ids)
    for i, cont_ids in enumerate(cont_ids_list):
        cont_len = len(cont_ids)
        # Token at position t is predicted by logits at position t-1.
        # Continuation tokens occupy positions [prompt_len .. prompt_len+cont_len-1],
        # so we slice logits at [prompt_len-1 .. prompt_len+cont_len-1).
        cont_logits = logits[i, prompt_len - 1 : prompt_len + cont_len - 1, :]
        cont_tensor = torch.tensor(cont_ids, dtype=torch.long, device=DEVICE)
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_lp = log_probs.gather(
            dim=-1, index=cont_tensor.unsqueeze(-1)
        ).squeeze(-1)
        total = float(token_lp.sum().item())
        results.append({
            "continuation": continuations[i],
            "logprob": total,
            "n_bytes": cont_len,
            "per_byte_logprobs": [float(x) for x in token_lp.tolist()],
        })
    return results


@app.post("/v1/loglikelihood")
async def loglikelihood(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    continuations = body.get("continuations") or []
    if not isinstance(prompt, str) or not isinstance(continuations, list) or not continuations:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": "Need string 'prompt' and non-empty list 'continuations'.",
                "type": "invalid_request_error",
            }},
        )
    if not all(isinstance(c, str) and c for c in continuations):
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": "All continuations must be non-empty strings.",
                "type": "invalid_request_error",
            }},
        )
    async with GEN_LOCK:
        data = await asyncio.get_running_loop().run_in_executor(
            None, _loglikelihood_impl, prompt, continuations,
        )
    return {"object": "loglikelihood", "model": MODEL_NAME, "data": data}


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible HTTP server for H-Net (for EuroEval / LiteLLM)."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-name", default="hnet-base",
                        help="Model id reported by /v1/models and accepted in requests.")
    parser.add_argument("--max-gen-tokens", type=int, default=1024)
    parser.add_argument("--max-context-length", type=int, default=8192)
    args = parser.parse_args()

    global MODEL, TOKENIZER, DEVICE, MODEL_NAME, MAX_GEN_TOKENS, MAX_CONTEXT_LENGTH
    print("Loading H-Net...")
    MODEL = load_from_pretrained(args.model_path, args.config_path)
    MODEL.eval()
    DEVICE = next(MODEL.parameters()).device
    TOKENIZER = ByteTokenizer()
    MODEL_NAME = args.model_name
    MAX_GEN_TOKENS = args.max_gen_tokens
    MAX_CONTEXT_LENGTH = args.max_context_length
    print(f"H-Net loaded on {DEVICE}. Serving as '{MODEL_NAME}' on http://{args.host}:{args.port}/v1")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
