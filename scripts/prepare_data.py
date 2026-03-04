"""Pre-tokenize FineWeb-Edu parquet files into flat uint8 binary files.

This only needs to be run once. Training then uses np.memmap on the resulting
.bin files, which are shared across all concurrent processes via the OS page cache.

Output layout:
    fineweb-edu (English):
        <data_dir>/fineweb-edu-<subset>/train.bin
        <data_dir>/fineweb-edu-<subset>/val.bin

    fineweb-edu-chinese:
        <data_dir>/fineweb-edu-chinese-<score_range>/train.bin
        <data_dir>/fineweb-edu-chinese-<score_range>/val.bin

Each file is a flat sequence of raw UTF-8 bytes with BOS (254) prepended and
EOS (255) appended to each document.

Usage:
    # English
    python -u scripts/prepare_data.py \\
        --data-dir /scratch-shared/mivanov1/hnet/data \\
        --subset sample-10BT \\
        --val-fraction 0.005 \\
        --seed 42 \\
        --num-workers 24

    # Chinese (score range 3_4)
    python -u scripts/prepare_data.py \\
        --dataset fineweb-edu-chinese \\
        --data-dir /scratch-shared/mivanov1/hnet/data \\
        --subset 3_4 \\
        --val-fraction 0.005 \\
        --seed 42 \\
        --num-workers 24
"""

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

BOS = 254
EOS = 255


def _encode_shard(shard_idx: int, num_shards: int, arrow_dir: str, tmp_path: str) -> int:
    """Encode one shard of a pre-saved Arrow dataset to a binary file.

    Each worker loads its own shard from the Arrow cache (memory-mapped, fast)
    and encodes documents to raw bytes. Returns total bytes written.
    """
    ds = load_from_disk(arrow_dir)
    shard = ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True)

    total_bytes = 0
    with open(tmp_path, "wb") as f, tqdm(
        total=len(shard),
        desc=f"shard-{shard_idx:02d}",
        position=shard_idx,
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=10,
    ) as pbar:
        for example in shard:
            text = example.get("text", "")
            if text:
                encoded = bytes([BOS]) + text.encode("utf-8") + bytes([EOS])
                f.write(encoded)
                total_bytes += len(encoded)
            pbar.update()
            pbar.set_postfix(gb=f"{total_bytes / 1e9:.2f}", refresh=False)

    return total_bytes


def prepare(data_dir: str, subset: str, val_fraction: float, seed: int, num_workers: int,
            dataset: str = "fineweb-edu"):
    if dataset == "fineweb-edu-chinese":
        # Score-range dirs are already the correct path component (e.g. "3_4")
        repo_path = subset
        dir_name = f"fineweb-edu-chinese-{subset}"
    else:
        # English: "sample-10BT" -> "sample/10BT"
        repo_path = subset.replace("-", "/")
        dir_name = f"fineweb-edu-{subset}"

    parquet_dir = str(Path(data_dir) / dir_name / repo_path)
    output_dir = Path(data_dir) / dir_name
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load parquet once, split, save as Arrow ----
    # Arrow files are memory-mapped, so workers can load their shard quickly
    # without re-decompressing the parquet.
    train_arrow = tmp_dir / "train_arrow"
    val_arrow = tmp_dir / "val_arrow"

    if train_arrow.exists() and val_arrow.exists():
        print("Arrow cache found, skipping parquet load.")
    else:
        print(f"Loading parquet from {parquet_dir} ...")
        ds = load_dataset("parquet", data_dir=parquet_dir, split="train")
        print(f"Loaded {len(ds):,} documents. Splitting and saving to Arrow ...")
        split = ds.train_test_split(test_size=val_fraction, seed=seed)
        split["train"].save_to_disk(str(train_arrow))
        split["test"].save_to_disk(str(val_arrow))
        print(f"Arrow saved: train={len(split['train']):,}  val={len(split['test']):,}\n")

    arrow_dirs = {"train": str(train_arrow), "val": str(val_arrow)}

    # ---- Step 2: encode each split in parallel ----
    for split_name, arrow_dir in arrow_dirs.items():
        n_docs = len(load_from_disk(arrow_dir))
        outfile = output_dir / f"{split_name}.bin"
        n_shards = min(num_workers, n_docs)

        print(f"=== {split_name} ({n_docs:,} docs, {n_shards} shards) -> {outfile} ===")

        shard_files = [tmp_dir / f"{split_name}_shard{i}.bin" for i in range(n_shards)]
        args = [
            (i, n_shards, arrow_dir, str(shard_files[i]))
            for i in range(n_shards)
        ]

        total_bytes = 0
        with ProcessPoolExecutor(max_workers=n_shards) as executor:
            futures = {executor.submit(_encode_shard, *a): a[0] for a in args}
            for fut in as_completed(futures):
                total_bytes += fut.result()

        print(f"\nConcatenating {n_shards} shards -> {outfile} ...")
        with open(outfile, "wb") as out:
            for sf in shard_files:
                with open(sf, "rb") as f:
                    shutil.copyfileobj(f, out)
                sf.unlink()

        print(f"Done: {outfile.stat().st_size / 1e9:.2f} GB\n")

    # Clean up Arrow cache and tmp dir
    shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize FineWeb-Edu to binary")
    parser.add_argument("--dataset", default="fineweb-edu",
                        choices=["fineweb-edu", "fineweb-edu-chinese"],
                        help="Dataset to prepare (default: fineweb-edu)")
    parser.add_argument("--data-dir", default="/scratch-shared/mivanov1/hnet/data")
    parser.add_argument("--subset", default=None,
                        help=(
                            "Subset/score-range to prepare. "
                            "fineweb-edu: e.g. 'sample-10BT' (default). "
                            "fineweb-edu-chinese: '2_3', '3_4', or '4_5' (default: '3_4')."
                        ))
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel encoding processes")
    args = parser.parse_args()

    if args.subset is None:
        args.subset = "3_4" if args.dataset == "fineweb-edu-chinese" else "sample-10BT"

    prepare(args.data_dir, args.subset, args.val_fraction, args.seed, args.num_workers,
            dataset=args.dataset)


if __name__ == "__main__":
    main()
