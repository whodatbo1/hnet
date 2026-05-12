"""Measure per-language patch-boundary rate on FLORES+ sentences.

For each requested language, runs every FLORES+ sentence through the H-Net
model and accumulates total boundaries and total text bytes (excluding the
BOS-forced boundary at position 0). Reports total_boundaries / total_bytes
and its inverse (mean bytes per patch).

Usage:
    python scripts/compression_flores.py \\
        --model-path checkpoints/latest.pt \\
        --config-path checkpoints/config.json \\
        --langs eng_Latn,bul_Cyrl \\
        --split devtest
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained
from hnet.utils.byte_tokenizer import ByteTokenizer


@torch.no_grad()
def measure_sample(model, tokenizer, text, device, stage=0, max_length=None):
    """Run the model on a single sentence and count boundaries on its text bytes.

    Position 0 (BOS) is dropped from both the boundary count and the byte
    count, so the returned numbers reflect only the learned routing policy
    on actual text bytes.

    Returns (n_text_bytes, n_boundaries_on_text_bytes).
    """
    enc = tokenizer.encode([text], add_bos=True)[0]
    ids = enc["input_ids"]
    if max_length and len(ids) > max_length:
        ids = ids[:max_length]
    if len(ids) <= 1:
        return 0, 0
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    out = model(input_ids, mask=mask)
    bm = out.bpred_output[stage].boundary_mask.squeeze(0)
    bm = bm[1:]
    return int(bm.numel()), int(bm.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Per-language patch-boundary rate on FLORES+."
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--config-path", required=True,
                        help="Path to the model configuration (.json file)")
    parser.add_argument("--langs", default="eng_Latn,bul_Cyrl",
                        help="Comma-separated FLORES+ language codes")
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--stage", type=int, default=0,
                        help="Routing stage to measure (0 = byte-level outermost)")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Truncate any sentence longer than this many bytes")
    parser.add_argument("--output-json", default=None,
                        help="Optional: dump per-language and per-sample stats")
    args = parser.parse_args()

    print("Loading model...")
    model = load_from_pretrained(args.model_path, args.config_path)
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()
    print("Model loaded.\n")

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    results = {}
    for lang in langs:
        print(f"Loading FLORES+ {lang} / {args.split}...")
        ds = load_dataset("openlanguagedata/flores_plus", lang, split=args.split)
        per_sample = []
        total_bytes = total_bnd = 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for row in tqdm(ds, desc=f"  {lang}", unit="sent"):
                n_b, n_bnd = measure_sample(
                    model, tokenizer, row["text"], device,
                    stage=args.stage, max_length=args.max_length,
                )
                if n_b == 0:
                    continue
                total_bytes += n_b
                total_bnd += n_bnd
                per_sample.append({
                    "bytes": n_b,
                    "boundaries": n_bnd,
                    "rate": n_bnd / n_b,
                })
        rate = total_bnd / total_bytes if total_bytes else float("nan")
        results[lang] = {
            "split": args.split,
            "n_samples": len(per_sample),
            "total_bytes": total_bytes,
            "total_boundaries": total_bnd,
            "boundary_rate": rate,
            "bytes_per_patch": (1.0 / rate) if rate else float("inf"),
            "per_sample": per_sample,
        }

    print(f"\nFLORES+ {args.split} — boundary rate per language "
          f"(BOS-excluded, stage {args.stage})")
    print("-" * 88)
    print(f"{'Language':<14} {'#Sent':>6} {'#Bytes':>9} {'#Bnd':>8} "
          f"{'Bnd/Byte':>10} {'Bytes/Patch':>12}")
    print("-" * 88)
    for lang, r in results.items():
        print(f"{lang:<14} {r['n_samples']:>6} {r['total_bytes']:>9} "
              f"{r['total_boundaries']:>8} {r['boundary_rate']:>10.4f} "
              f"{r['bytes_per_patch']:>11.2f}x")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nSaved per-sample stats to {args.output_json}")


if __name__ == "__main__":
    main()
