#!/usr/bin/env python3
"""Estimate total bytes and characters in pre-tokenized binary training data.

The .bin files produced by scripts/prepare_data.py contain flat UTF-8 bytes
with BOS (254) prepended and EOS (255) appended to each document.

This script memory-maps a .bin file, draws a random contiguous sample,
splits it into documents at EOS boundaries, decodes UTF-8, then extrapolates
total-bytes and total-character counts to the full file.

Usage:
    # English
    python experiments/estimate_dataset_stats.py \
        --data-dir /scratch-shared/mivanov1/hnet/data \
        --subset sample-10BT \
        --split train \
        --sample-mb 512

    # Chinese
    python experiments/estimate_dataset_stats.py \
        --dataset fineweb-edu-chinese \
        --data-dir /scratch-shared/mivanov1/hnet/data \
        --subset 3_4 \
        --split both \
        --sample-mb 512
"""

import argparse
from pathlib import Path

import numpy as np

BOS = 254
EOS = 255


def estimate_stats(bin_path: Path, sample_bytes: int, seed: int = 42) -> dict:
    """Sample `sample_bytes` bytes from *bin_path*, decode documents, extrapolate.

    Returns a dict with both sample-level and extrapolated statistics.
    """
    file_size = bin_path.stat().st_size
    print(f"File      : {bin_path}")
    print(f"File size : {file_size / 1e9:.3f} GB  ({file_size:,} bytes)")

    sample_bytes = min(sample_bytes, file_size)
    rng = np.random.default_rng(seed)
    max_offset = max(0, file_size - sample_bytes)
    offset = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0

    print(f"\nSampling  : {sample_bytes / 1e6:.1f} MB  "
          f"at offset {offset / 1e9:.3f} GB  ({offset:,} bytes)")

    data = np.memmap(bin_path, dtype=np.uint8, mode="r")
    chunk = bytes(data[offset : offset + sample_bytes])

    # Split on EOS to recover individual documents.
    # Format in file: ... BOS <utf-8 text bytes> EOS BOS <utf-8 text bytes> EOS ...
    # The first and last parts may be truncated at the sample boundary — skip them.
    parts = chunk.split(bytes([EOS]))

    total_docs = 0
    total_raw_bytes = 0   # sum of len(utf-8 text bytes) for complete documents
    total_chars = 0       # sum of len(decoded text) for complete documents
    decode_errors = 0

    # Skip the very first part (potentially truncated) and the very last part
    # (potentially truncated). Only interior parts are guaranteed complete.
    interior = parts[1:-1] if len(parts) > 2 else []

    for part in interior:
        # Each complete document starts with BOS; strip it.
        if part and part[0] == BOS:
            part = part[1:]
        if not part:
            continue

        total_raw_bytes += len(part)
        try:
            text = part.decode("utf-8", errors="strict")
            total_chars += len(text)
            total_docs += 1
        except UnicodeDecodeError:
            # Shouldn't happen for interior documents, but handle gracefully.
            decode_errors += 1

    if total_docs == 0:
        print("\nNo complete documents found in sample. Try a larger --sample-mb.")
        return {}

    bytes_per_char = total_raw_bytes / total_chars

    # BOS/EOS add 2 bytes per document; estimate overhead fraction in the sample.
    bos_eos_sample = total_docs * 2
    overhead_fraction = bos_eos_sample / sample_bytes

    print(f"\n{'─'*55}")
    print(f"  Sample statistics  ({sample_bytes / 1e6:.1f} MB sampled)")
    print(f"{'─'*55}")
    print(f"  Complete documents : {total_docs:>15,}")
    if decode_errors:
        print(f"  Decode errors      : {decode_errors:>15,}  (skipped)")
    print(f"  Text bytes         : {total_raw_bytes:>15,}")
    print(f"  Characters         : {total_chars:>15,}")
    print(f"  Bytes / char       : {bytes_per_char:>18.4f}")
    print(f"  BOS/EOS overhead   : {overhead_fraction * 100:>17.4f}%")

    # Extrapolate: text_fraction of file_size is real UTF-8 bytes.
    text_fraction = 1.0 - overhead_fraction
    total_text_bytes_est = int(file_size * text_fraction)
    total_chars_est = int(total_text_bytes_est / bytes_per_char)

    print(f"\n{'─'*55}")
    print(f"  Extrapolated to full file  ({file_size / 1e9:.3f} GB)")
    print(f"{'─'*55}")
    print(f"  Total text bytes   : {total_text_bytes_est:>15,}  "
          f"({total_text_bytes_est / 1e9:.3f} GB)")
    print(f"  Total characters   : {total_chars_est:>15,}  "
          f"({total_chars_est / 1e9:.3f} B chars)")

    return {
        "file_size": file_size,
        "sample_bytes": sample_bytes,
        "total_docs_in_sample": total_docs,
        "bytes_per_char": bytes_per_char,
        "overhead_fraction": overhead_fraction,
        "total_text_bytes_est": total_text_bytes_est,
        "total_chars_est": total_chars_est,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate total bytes and characters in a pre-tokenised .bin file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb-edu",
        choices=["fineweb-edu", "fineweb-edu-chinese"],
        help="Which dataset to analyse (default: fineweb-edu)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/scratch-shared/mivanov1/hnet/data"),
        help="Root data directory (default: /scratch-shared/mivanov1/hnet/data)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=(
            "Subset/score-range to analyse. "
            "fineweb-edu: e.g. 'sample-10BT' (default). "
            "fineweb-edu-chinese: '2_3', '3_4', or '4_5' (default: '3_4')."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "both"],
        help="Which split(s) to analyse (default: train)",
    )
    parser.add_argument(
        "--sample-mb",
        type=float,
        default=512,
        help="Approximate size of the random sample to draw, in MB (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for choosing the sample offset (default: 42)",
    )
    args = parser.parse_args()

    if args.subset is None:
        args.subset = "3_4" if args.dataset == "fineweb-edu-chinese" else "sample-10BT"

    if args.dataset == "fineweb-edu-chinese":
        subset_dir = args.data_dir / f"fineweb-edu-chinese-{args.subset}"
    else:
        subset_dir = args.data_dir / f"fineweb-edu-{args.subset}"

    sample_bytes = int(args.sample_mb * 1024 * 1024)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    all_results = {}

    for split in splits:
        bin_path = subset_dir / f"{split}.bin"
        if not bin_path.exists():
            print(f"\n[SKIP] {bin_path} not found. "
                  f"Run scripts/prepare_data.py first.")
            continue

        print(f"\n{'═'*55}")
        print(f"  Split: {split}")
        print(f"{'═'*55}")
        result = estimate_stats(bin_path, sample_bytes=sample_bytes, seed=args.seed)
        all_results[split] = result

    if len(all_results) > 1:
        print(f"\n{'═'*55}")
        print("  Combined totals (train + val)")
        print(f"{'═'*55}")
        total_bytes_est = sum(r["total_text_bytes_est"] for r in all_results.values())
        total_chars_est = sum(r["total_chars_est"] for r in all_results.values())
        print(f"  Total text bytes   : {total_bytes_est:>15,}  "
              f"({total_bytes_est / 1e9:.3f} GB)")
        print(f"  Total characters   : {total_chars_est:>15,}  "
              f"({total_chars_est / 1e9:.3f} B chars)")


if __name__ == "__main__":
    main()
