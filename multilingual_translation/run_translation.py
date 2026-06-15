"""Evaluate H-Net machine-translation quality on a FLORES+ language pair.

This is an H-Net analogue of the multilingual-MT experiments in NJUNLP/MMT-LLM
("Multilingual Machine Translation with Large Language Models", Zhu et al.,
2023). Like that work we:

  * use FLORES as the parallel evaluation corpus (here FLORES+, the
    community-maintained successor, via the HF dataset `openlanguagedata/
    flores_plus` — the same source `scripts/compression_flores.py` already uses);
  * prompt the LM with a few-shot in-context template (MMT-LLM defaults to
    8 demonstrations) and let it complete the target sentence;
  * score with spBLEU (sacreBLEU's `flores200` SentencePiece tokenizer, the
    standard FLORES metric) plus chrF++. COMET is reported too when the
    `comet` package is installed.

Unlike the upstream OpenICL-based runner, we drive H-Net directly: load the
checkpoint with `generate.load_from_pretrained`, tokenize bytes with
`ByteTokenizer`, and greedy-decode via the prefill+step pattern used by
`eval/run_cute.py`. No HTTP server is needed.

FLORES protocol: demonstrations are drawn from the `dev` split, evaluation is
on `devtest`; the two splits are disjoint so there is no leakage. Source and
target sentences are aligned by their shared `id`.

Usage:

    # English -> German, 5-shot, full devtest:
    python multilingual_translation/run_translation.py \\
        --model-path checkpoints/latest.pt \\
        --config-path configs/.../hnet.json \\
        --model-name hnet-s \\
        --src-lang eng_Latn --tgt-lang deu_Latn \\
        --num-shots 5

    # Smoke test (first 20 sentences):
    python multilingual_translation/run_translation.py \\
        --model-path ... --config-path ... \\
        --src-lang eng_Latn --tgt-lang fra_Latn \\
        --limit 20
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained
from hnet.utils.byte_tokenizer import ByteTokenizer


# FLORES+ language code ("<iso639-3>_<script>") -> English display name used in
# the prompt. Covers the high/medium/low-resource languages most commonly used
# in MMT-LLM-style studies. Unknown codes fall back to the title-cased ISO part
# (with a warning), so the script still runs on any FLORES+ config.
LANG_NAMES = {
    "eng_Latn": "English",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "ita_Latn": "Italian",
    "nld_Latn": "Dutch",
    "ron_Latn": "Romanian",
    "swe_Latn": "Swedish",
    "dan_Latn": "Danish",
    "nob_Latn": "Norwegian",
    "fin_Latn": "Finnish",
    "pol_Latn": "Polish",
    "ces_Latn": "Czech",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian",
    "ell_Grek": "Greek",
    "bul_Cyrl": "Bulgarian",
    "rus_Cyrl": "Russian",
    "ukr_Cyrl": "Ukrainian",
    "srp_Cyrl": "Serbian",
    "tur_Latn": "Turkish",
    "arb_Arab": "Arabic",
    "heb_Hebr": "Hebrew",
    "pes_Arab": "Persian",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
    "zho_Hans": "Chinese",
    "zho_Hant": "Traditional Chinese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Malay",
    "swh_Latn": "Swahili",
    "isl_Latn": "Icelandic",
    "est_Latn": "Estonian",
    "lvs_Latn": "Latvian",
    "lit_Latn": "Lithuanian",
    "cat_Latn": "Catalan",
    "eus_Latn": "Basque",
    "glg_Latn": "Galician",
}


def lang_name(code: str) -> str:
    if code in LANG_NAMES:
        return LANG_NAMES[code]
    iso = code.split("_")[0]
    name = iso.capitalize()
    print(
        f"[warn] No display name for FLORES+ code '{code}'; "
        f"using '{name}'. Add it to LANG_NAMES for a nicer prompt.",
        flush=True,
    )
    return name


def load_parallel(src_lang: str, tgt_lang: str, split: str) -> list[tuple[int, str, str]]:
    """Return (id, src_text, tgt_text) triples aligned by FLORES+ sentence id."""
    src_ds = load_dataset("openlanguagedata/flores_plus", src_lang, split=split)
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_lang, split=split)
    src_by_id = {int(r["id"]): r["text"] for r in src_ds}
    tgt_by_id = {int(r["id"]): r["text"] for r in tgt_ds}
    common = sorted(set(src_by_id) & set(tgt_by_id))
    return [(i, src_by_id[i], tgt_by_id[i]) for i in common]


def build_prompt(
    shots: list[tuple[int, str, str]],
    src_text: str,
    src_name: str,
    tgt_name: str,
) -> str:
    """MMT-LLM-style few-shot prompt: parallel `Lang: sentence` blocks.

    Each demonstration is two lines (`{src}: ...` / `{tgt}: ...`); blocks are
    separated by a blank line; the final block leaves the target side open for
    the model to complete on one line.
    """
    blocks = [f"{src_name}: {s}\n{tgt_name}: {t}" for _, s, t in shots]
    blocks.append(f"{src_name}: {src_text}\n{tgt_name}:")
    return "\n\n".join(blocks)


def parse_translation(text: str) -> str:
    """First non-empty line of the completion, stripped of surrounding space."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


@torch.no_grad()
def translate(
    model,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    max_context: int,
    device: torch.device,
) -> str:
    """Greedy-decode (temperature 0) until newline or EOS.

    Mirrors `eval/run_cute.py:generate_answer` — the same prefill+step loop used
    elsewhere in this repo for H-Net generation.
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
            if "\n" in partial:
                break
        except (UnicodeDecodeError, ValueError):
            pass
        step_input = torch.tensor([[nt]], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.step(step_input, cache)
        logits = out.logits[0, -1, :]

    try:
        text = tokenizer.decode(gen, errors="replace")
    except (UnicodeDecodeError, ValueError):
        text = ""
    return text


def compute_metrics(hyps: list[str], refs: list[str]) -> dict:
    """Corpus spBLEU (FLORES tokenizer) + chrF++; signatures for reproducibility."""
    import sacrebleu

    metrics: dict = {}
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200")
    metrics["spbleu"] = round(bleu.score, 4)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    metrics["chrf2"] = round(chrf.score, 4)
    try:
        metrics["spbleu_signature"] = str(bleu.get_signature())
        metrics["chrf2_signature"] = str(chrf.get_signature())
    except AttributeError:
        pass
    return metrics


def compute_comet(srcs: list[str], hyps: list[str], refs: list[str],
                  model_name: str, batch_size: int) -> Optional[float]:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("[warn] --comet requested but the `comet` package is not installed; "
              "skipping COMET. Install with `pip install unbabel-comet`.", flush=True)
        return None
    ckpt = download_model(model_name)
    comet_model = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = comet_model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    return round(float(out["system_score"]), 4)


def main():
    parser = argparse.ArgumentParser(
        description="FLORES+ machine-translation evaluation for H-Net (MMT-LLM-style)."
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--config-path", required=True,
                        help="Path to the model configuration (.json file)")
    parser.add_argument("--model-name", default="hnet")
    parser.add_argument("--src-lang", required=True,
                        help="FLORES+ source code, e.g. eng_Latn")
    parser.add_argument("--tgt-lang", required=True,
                        help="FLORES+ target code, e.g. deu_Latn")
    parser.add_argument("--num-shots", type=int, default=5,
                        help="In-context demonstrations from the dev split "
                             "(MMT-LLM uses 8; default 5).")
    parser.add_argument("--test-split", default="devtest", choices=["dev", "devtest"],
                        help="Split to evaluate on (default: devtest).")
    parser.add_argument("--shot-split", default="dev", choices=["dev", "devtest"],
                        help="Split to draw demonstrations from (default: dev).")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for sampling the fixed demonstration set.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test sentences (smoke test).")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max generated bytes per translation.")
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--comet", action="store_true",
                        help="Also compute COMET (requires `unbabel-comet`).")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="multilingual_translation/results")
    args = parser.parse_args()

    if args.num_shots > 0 and args.shot_split == args.test_split:
        print(f"[warn] shot-split == test-split ({args.test_split}); demonstrations "
              f"will be excluded from the test set to avoid leakage.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path}", flush=True)
    model = load_from_pretrained(args.model_path, args.config_path)
    model.eval()
    tokenizer = ByteTokenizer()

    src_name, tgt_name = lang_name(args.src_lang), lang_name(args.tgt_lang)
    print(f"Translation direction: {src_name} ({args.src_lang}) -> "
          f"{tgt_name} ({args.tgt_lang})", flush=True)

    # Demonstration pool + test set.
    shots: list[tuple[int, str, str]] = []
    if args.num_shots > 0:
        pool = load_parallel(args.src_lang, args.tgt_lang, args.shot_split)
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(pool), generator=g).tolist()
        shots = [pool[i] for i in perm[:args.num_shots]]
        print(f"Using {len(shots)} fixed demonstrations from '{args.shot_split}' "
              f"(seed={args.seed}).", flush=True)

    test = load_parallel(args.src_lang, args.tgt_lang, args.test_split)
    if args.shot_split == args.test_split and shots:
        shot_ids = {i for i, _, _ in shots}
        test = [row for row in test if row[0] not in shot_ids]
    if args.limit is not None:
        test = test[:args.limit]
    print(f"Evaluating on {len(test)} sentences from '{args.test_split}'.\n", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    pair_tag = f"{args.src_lang}-{args.tgt_lang}"
    detail_path = out_dir / f"mt_{args.model_name}_{pair_tag}_{run_id}.jsonl"
    summary_path = out_dir / f"mt_{args.model_name}_{pair_tag}_{run_id}_summary.json"

    srcs: list[str] = []
    refs: list[str] = []
    hyps: list[str] = []
    with detail_path.open("w") as f:
        for sid, src_text, ref_text in tqdm(test, desc=pair_tag, unit="sent"):
            prompt = build_prompt(shots, src_text, src_name, tgt_name)
            raw = translate(
                model, tokenizer, prompt,
                args.max_new_tokens, args.max_context, device,
            )
            hyp = parse_translation(raw)
            srcs.append(src_text)
            refs.append(ref_text)
            hyps.append(hyp)
            f.write(json.dumps({
                "id": sid,
                "source": src_text,
                "reference": ref_text,
                "hypothesis": hyp,
                "raw_output": raw,
            }, ensure_ascii=False) + "\n")

    print("\nScoring...", flush=True)
    metrics = compute_metrics(hyps, refs)
    if args.comet:
        comet_score = compute_comet(
            srcs, hyps, refs, args.comet_model, args.comet_batch_size
        )
        if comet_score is not None:
            metrics["comet"] = comet_score
            metrics["comet_model"] = args.comet_model

    summary = {
        "model": args.model_name,
        "model_path": args.model_path,
        "config_path": args.config_path,
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "direction": f"{src_name}->{tgt_name}",
        "test_split": args.test_split,
        "shot_split": args.shot_split,
        "num_shots": len(shots),
        "shot_ids": [i for i, _, _ in shots],
        "seed": args.seed,
        "n": len(test),
        "max_new_tokens": args.max_new_tokens,
        "run_id": run_id,
        "metrics": metrics,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nPer-sentence outputs: {detail_path}")
    print(f"Summary: {summary_path}")
    print(f"\n{src_name} -> {tgt_name}  ({len(test)} sentences, {len(shots)}-shot)")
    print(f"  spBLEU : {metrics['spbleu']:.2f}")
    print(f"  chrF++ : {metrics['chrf2']:.2f}")
    if "comet" in metrics:
        print(f"  COMET  : {metrics['comet']:.4f}")


if __name__ == "__main__":
    main()
