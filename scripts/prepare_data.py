"""Pre-tokenize parquet files into flat uint8 binary files.

This only needs to be run once. Training then uses np.memmap on the resulting
.bin files, which are shared across all concurrent processes via the OS page cache.

Output layout:
    fineweb-edu (English):
        <data_dir>/fineweb-edu-<subset>/train.bin
        <data_dir>/fineweb-edu-<subset>/val.bin

    fineweb-edu-chinese:
        <data_dir>/fineweb-edu-chinese-<score_range>/train.bin
        <data_dir>/fineweb-edu-chinese-<score_range>/val.bin

    the-stack-v2-smol:
        <data_dir>/the-stack-v2-smol-<language>/train.bin
        <data_dir>/the-stack-v2-smol-<language>/val.bin

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

    # Code (The Stack V2 Smol — run download_data.py first)
    python -u scripts/prepare_data.py \\
        --dataset the-stack-v2-smol \\
        --data-dir /scratch-shared/mivanov1/hnet/data \\
        --val-fraction 0.005 \\
        --seed 42 \\
        --num-workers 24

    # Code (StarCoderData — run download_data.py first)
    python -u scripts/prepare_data.py \\
        --dataset starcoderdata \\
        --subset python \\
        --data-dir /scratch-shared/mivanov1/hnet/data \\
        --val-fraction 0.005 \\
        --seed 42 \\
        --num-workers 24

    # FineWeb-2 (any language — run download_data.py first)
    python -u scripts/prepare_data.py \\
        --dataset fineweb-2 \\
        --subset kor_Hang \\
        --data-dir /scratch-shared/mivanov1/hnet/data \\
        --val-fraction 0.005 \\
        --seed 42 \\
        --num-workers 24
"""

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

BOS = 254
EOS = 255


def _encode_shard(shard_idx: int, num_shards: int, arrow_dir: str, tmp_path: str,
                  text_column: str = "text") -> int:
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
            text = example.get(text_column, "")
            if text:
                encoded = bytes([BOS]) + text.encode("utf-8") + bytes([EOS])
                f.write(encoded)
                total_bytes += len(encoded)
            pbar.update()
            pbar.set_postfix(gb=f"{total_bytes / 1e9:.2f}", refresh=False)

    return total_bytes


def _encode_indices(indices_path: str, parquet_dir: str, tmp_path: str,
                    text_column: str, shard_idx: int) -> int:
    """Encode a subset of a parquet dataset (by index) to a binary file.

    Avoids the expensive Arrow save_to_disk step by reading directly from
    the parquet-backed HF dataset using pre-computed indices.
    """
    indices = np.load(indices_path)
    ds = load_dataset("parquet", data_dir=parquet_dir, split="train")
    subset = ds.select(indices)

    total_bytes = 0
    with open(tmp_path, "wb") as f, tqdm(
        total=len(subset),
        desc=f"shard-{shard_idx:02d}",
        position=shard_idx,
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=10,
    ) as pbar:
        for example in subset:
            text = example.get(text_column, "")
            if text:
                encoded = bytes([BOS]) + text.encode("utf-8") + bytes([EOS])
                f.write(encoded)
                total_bytes += len(encoded)
            pbar.update()
            pbar.set_postfix(gb=f"{total_bytes / 1e9:.2f}", refresh=False)

    return total_bytes


def prepare(data_dir: str, subset: str, val_fraction: float, seed: int, num_workers: int,
            dataset: str = "fineweb-edu"):
    if dataset == "starcoderdata":
        # snapshot_download puts parquets under <language>/ subdirs
        if subset:
            dir_name = f"starcoderdata-{subset.lower()}"
            repo_path = subset.lower()
        else:
            dir_name = "starcoderdata"
            repo_path = ""
        text_column = "content"
    elif dataset == "the-stack-v2-smol":
        # Full dataset, parquet shards already in data/ from download_data.py
        dir_name = "the-stack-v2-smol"
        repo_path = "data"
        text_column = "content"
    elif dataset == "fineweb-edu-chinese":
        # Score-range dirs are already the correct path component (e.g. "3_4")
        repo_path = subset
        dir_name = f"fineweb-edu-chinese-{subset}"
        text_column = "text"
    elif dataset == "fineweb-2":
        # Language subsets downloaded under data/<lang>/train/ and data/<lang>/test/
        repo_path = f"data/{subset}/train"
        dir_name = f"fineweb-2-{subset}"
        text_column = "text"
    else:
        # English: "sample-10BT" -> "sample/10BT"
        repo_path = subset.replace("-", "/")
        dir_name = f"fineweb-edu-{subset}"
        text_column = "text"

    parquet_dir = str(Path(data_dir) / dir_name / repo_path)
    output_dir = Path(data_dir) / dir_name
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # For large code datasets (starcoderdata, the-stack-v2-smol), skip the
    # expensive Arrow save_to_disk step.  Instead, compute shuffled train/val
    # index arrays (cheap — just ints) and have each worker read directly from
    # the parquet-backed HF dataset.
    use_fast_path = dataset in ("starcoderdata", "the-stack-v2-smol", "fineweb-edu-chinese", "fineweb-2")

    if use_fast_path:
        _prepare_fast(parquet_dir, output_dir, tmp_dir, text_column,
                      val_fraction, seed, num_workers)
    else:
        _prepare_arrow(parquet_dir, output_dir, tmp_dir, text_column,
                       val_fraction, seed, num_workers)

    # Clean up tmp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _prepare_arrow(parquet_dir, output_dir, tmp_dir, text_column,
                   val_fraction, seed, num_workers):
    """Original path: load parquet → save as Arrow → encode from Arrow shards."""
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

    for split_name, arrow_dir in arrow_dirs.items():
        n_docs = len(load_from_disk(arrow_dir))
        outfile = output_dir / f"{split_name}.bin"
        n_shards = min(num_workers, n_docs)

        print(f"=== {split_name} ({n_docs:,} docs, {n_shards} shards) -> {outfile} ===")

        shard_files = [tmp_dir / f"{split_name}_shard{i}.bin" for i in range(n_shards)]
        args = [
            (i, n_shards, arrow_dir, str(shard_files[i]), text_column)
            for i in range(n_shards)
        ]

        total_bytes = 0
        with ProcessPoolExecutor(max_workers=n_shards) as executor:
            futures = {executor.submit(_encode_shard, *a): a[0] for a in args}
            for fut in as_completed(futures):
                total_bytes += fut.result()

        _concat_shards(shard_files, outfile, n_shards)


def _prepare_fast(parquet_dir, output_dir, tmp_dir, text_column,
                  val_fraction, seed, num_workers):
    """Fast path for large datasets: split by index, encode directly from parquet."""
    print(f"Loading parquet from {parquet_dir} ...")
    ds = load_dataset("parquet", data_dir=parquet_dir, split="train")
    n = len(ds)
    print(f"Loaded {n:,} documents.")

    # Compute shuffled train/val indices (cheap — just int arrays)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_indices = np.sort(perm[:n_val])
    train_indices = np.sort(perm[n_val:])
    print(f"Split: train={len(train_indices):,}  val={len(val_indices):,}")

    # Free the dataset handle from the main process
    del ds

    splits = {"train": train_indices, "val": val_indices}
    for split_name, indices in splits.items():
        outfile = output_dir / f"{split_name}.bin"
        n_shards = min(num_workers, len(indices))

        print(f"\n=== {split_name} ({len(indices):,} docs, {n_shards} shards) -> {outfile} ===")

        # Save per-shard index arrays so workers can load them
        shard_files = []
        args = []
        index_chunks = np.array_split(indices, n_shards)
        for i, chunk in enumerate(index_chunks):
            idx_path = tmp_dir / f"{split_name}_indices_{i}.npy"
            np.save(str(idx_path), chunk)
            shard_bin = tmp_dir / f"{split_name}_shard{i}.bin"
            shard_files.append(shard_bin)
            args.append((str(idx_path), parquet_dir, str(shard_bin), text_column, i))

        total_bytes = 0
        with ProcessPoolExecutor(max_workers=n_shards) as executor:
            futures = {executor.submit(_encode_indices, *a): a[0] for a in args}
            for fut in as_completed(futures):
                total_bytes += fut.result()

        # Clean up index files
        for i in range(n_shards):
            idx_path = tmp_dir / f"{split_name}_indices_{i}.npy"
            idx_path.unlink(missing_ok=True)

        _concat_shards(shard_files, outfile, n_shards)


def _concat_shards(shard_files, outfile, n_shards):
    print(f"\nConcatenating {n_shards} shards -> {outfile} ...")
    with open(outfile, "wb") as out:
        for sf in shard_files:
            with open(sf, "rb") as f:
                shutil.copyfileobj(f, out)
            sf.unlink()
    print(f"Done: {outfile.stat().st_size / 1e9:.2f} GB\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize FineWeb-Edu to binary")
    parser.add_argument("--dataset", default="fineweb-edu",
                        choices=["fineweb-edu", "fineweb-edu-chinese", "fineweb-2", "the-stack-v2-smol", "starcoderdata"],
                        help="Dataset to prepare (default: fineweb-edu)")
    parser.add_argument("--data-dir", default="/scratch-shared/mivanov1/hnet/data")
    parser.add_argument("--subset", default=None,
                        help=(
                            "Subset/score-range to prepare. "
                            "fineweb-edu: e.g. 'sample-10BT' (default). "
                            "fineweb-edu-chinese: '2_3', '3_4', or '4_5' (default: '3_4'). "
                            "For starcoderdata: language, e.g. 'python' (default: all). "
                            "Not used for the-stack-v2-smol."
                        ))
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel encoding processes")
    args = parser.parse_args()

    if args.subset is None:
        if args.dataset == "fineweb-edu-chinese":
            args.subset = "3_4"
        elif args.dataset == "the-stack-v2-smol":
            args.subset = ""  # not used, single directory
        else:
            args.subset = "sample-10BT"

    prepare(args.data_dir, args.subset, args.val_fraction, args.seed, args.num_workers,
            dataset=args.dataset)


if __name__ == "__main__":
    main()
