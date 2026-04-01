#!/usr/bin/env python3
"""
Download FineWeb-Edu (English), FineWeb-Edu-Chinese, FineWeb-2, or The Stack V2 Smol from HuggingFace.

Usage:
    # English (default)
    python download_data.py
    python download_data.py --output-dir /scratch-shared/mivanov1/hnet/data --subset sample-10BT

    # Chinese
    python download_data.py --dataset fineweb-edu-chinese --subset 3_4
    python download_data.py --dataset fineweb-edu-chinese --subset 4_5

    # FineWeb-2 (any language — see https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
    python download_data.py --dataset fineweb-2 --subset kor_Hang     # Korean (~98 GB)
    python download_data.py --dataset fineweb-2 --subset arb_Arab     # Arabic

    # Code (The Stack V2 Smol) — requires AWS credentials for SWH S3 access
    python download_data.py --dataset the-stack-v2-smol
    python download_data.py --dataset the-stack-v2-smol --max-files 100000

    # Code (StarCoderData) — gated, accept terms at HF first
    python download_data.py --dataset starcoderdata                    # all languages (~783 GB)
    python download_data.py --dataset starcoderdata --subset python    # single language
"""

import argparse
import gzip
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download


def download_fineweb_edu(output_dir: str, subset: str = "sample-10BT"):
    """Download a FineWeb-Edu (English) subset from HuggingFace.

    Args:
        output_dir: Root directory to save the dataset.
        subset: Dataset config/subset name (e.g. "sample-10BT", "sample-100BT").
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "HuggingFaceFW/fineweb-edu"
    local_dir = output_path / f"fineweb-edu-{subset}"

    print(f"Downloading {repo_id} ({subset}) to {local_dir.absolute()}...")
    print("This may take a while (~28GB for sample-10BT).")

    # The HF repo stores subsets under sample/<size>/ (e.g. sample/10BT/)
    # The subset CLI arg uses the HF config name "sample-10BT" -> repo path "sample/10BT"
    repo_path = subset.replace("-", "/")
    model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[f"{repo_path}/*"],
    )

    print(f"Successfully downloaded to: {model_path}")
    return model_path


def download_fineweb_edu_chinese(output_dir: str, score_range: str = "3_4"):
    """Download a FineWeb-Edu-Chinese-V2.1 score-range subset from HuggingFace.

    The repo is organised into three score-range directories:
        2_3  (~1.4 TB) — potentially useful content with limitations
        3_4  (~800 GB) — suitable educational content
        4_5  (~70 GB)  — high-quality educational content

    Downloaded data lands at:
        <output_dir>/fineweb-edu-chinese-<score_range>/<score_range>/

    This matches the path expected by prepare_data.py --dataset fineweb-edu-chinese.

    Args:
        output_dir: Root directory to save the dataset.
        score_range: Score-range subdirectory to download ("2_3", "3_4", or "4_5").
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "opencsg/Fineweb-Edu-Chinese-V2.1"
    local_dir = output_path / f"fineweb-edu-chinese-{score_range}"

    print(f"Downloading {repo_id} ({score_range}) to {local_dir.absolute()}...")

    model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[f"{score_range}/*"],
    )

    print(f"Successfully downloaded to: {model_path}")
    return model_path


def download_fineweb_2(output_dir: str, subset: str = "kor_Hang"):
    """Download a FineWeb-2 language subset from HuggingFace.

    FineWeb-2 covers 1000+ languages with subsets named by BCP-47 style codes
    (e.g. kor_Hang, arb_Arab, fra_Latn). See the full list at:
    https://huggingface.co/datasets/HuggingFaceFW/fineweb-2#languages-and-available-subsets

    Downloaded data lands at:
        <output_dir>/fineweb-2-<subset>/data/<subset>/train/
        <output_dir>/fineweb-2-<subset>/data/<subset>/test/

    Args:
        output_dir: Root directory to save the dataset.
        subset: Language config name (e.g. "kor_Hang").
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "HuggingFaceFW/fineweb-2"
    local_dir = output_path / f"fineweb-2-{subset}"

    print(f"Downloading {repo_id} ({subset}) to {local_dir.absolute()}...")

    model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[f"data/{subset}/*"],
    )

    print(f"Successfully downloaded to: {model_path}")
    return model_path


def download_starcoderdata(output_dir: str, subset: str = None):
    """Download StarCoderData from HuggingFace (bigcode/starcoderdata).

    Contains source code with a ``content`` column across 86 programming
    languages.  Gated dataset — accept terms at
    https://huggingface.co/datasets/bigcode/starcoderdata first, then
    ``huggingface-cli login``.

    Args:
        output_dir: Root directory to save the dataset.
        subset: Programming language to download (e.g. "python").
                If None, downloads all languages (~783 GB).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "bigcode/starcoderdata"

    if subset:
        dir_name = f"starcoderdata-{subset.lower()}"
        allow_patterns = [f"{subset.lower()}/*"]
    else:
        dir_name = "starcoderdata"
        allow_patterns = None

    local_dir = output_path / dir_name
    size_hint = f" ({subset})" if subset else " (all languages, ~783 GB)"
    print(f"Downloading {repo_id}{size_hint} to {local_dir.absolute()} ...")

    model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )

    print(f"Successfully downloaded to: {model_path}")
    return model_path


def download_the_stack_v2_smol(output_dir: str, num_workers: int = 32,
                               max_files: int = None, shard_size: int = 50_000):
    """Download the full The Stack V2 Train Smol dataset (~70 GB).

    The HF dataset is repo-level: each row has a ``files`` list of file dicts
    (with ``blob_id``, ``src_encoding``, ``language``, ``path``, etc.).
    This function flattens repos into individual files, downloads their content
    from the Software Heritage S3 bucket, and saves parquet shards with a
    ``content`` column ready for prepare_data.py.

    Requirements:
        - HuggingFace account with accepted dataset terms (huggingface-cli login)
        - AWS credentials: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars
        - Bulk access agreement with Software Heritage (datasets@softwareheritage.org)
        - pip install boto3 datasets

    Args:
        output_dir: Root directory to save the dataset.
        num_workers: Number of parallel S3 download threads.
        max_files: If set, cap the total number of individual files to download.
        shard_size: Number of files per parquet shard.
    """
    import boto3
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from tqdm import tqdm

    output_path = Path(output_dir)
    local_dir = output_path / "the-stack-v2-smol"
    parquet_dir = local_dir / "data"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # --- Check AWS credentials ---
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not aws_key or not aws_secret:
        print("Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set.")
        print("Bulk download from the Software Heritage S3 bucket requires")
        print("an access agreement. Contact: datasets@softwareheritage.org")
        sys.exit(1)

    session = boto3.Session(aws_access_key_id=aws_key,
                            aws_secret_access_key=aws_secret)
    s3 = session.client("s3")

    # --- Step 1: Load repo-level dataset and flatten to individual files ---
    print("Loading bigcode/the-stack-v2-train-smol-ids ...")
    ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train")
    print(f"Loaded {len(ds):,} repos")

    print("Flattening repos into individual files ...")
    flat_files = []  # list of (blob_id, src_encoding, language, path)
    for row in tqdm(ds, desc="Flattening", dynamic_ncols=True):
        for f in row["files"]:
            flat_files.append((
                f["blob_id"],
                f.get("src_encoding") or "utf-8",
                f.get("language", ""),
                f.get("path", ""),
            ))
            if max_files is not None and len(flat_files) >= max_files:
                break
        if max_files is not None and len(flat_files) >= max_files:
            break

    n_total = len(flat_files)
    print(f"Total files: {n_total:,}")

    # Free the repo-level dataset
    del ds

    # --- Step 2: Download content from S3, saving in parquet shards ---
    def _fetch_one(blob_id: str, src_encoding: str):
        try:
            resp = s3.get_object(Bucket="softwareheritage",
                                 Key=f"content/{blob_id}")
            raw = gzip.decompress(resp["Body"].read())
            return raw.decode(src_encoding, errors="replace")
        except Exception as e:
            return f"__ERROR__:{type(e).__name__}:{e}"

    shard_idx = 0
    failed = 0
    downloaded = 0

    for chunk_start in range(0, n_total, shard_size):
        chunk_end = min(chunk_start + shard_size, n_total)
        chunk_files = flat_files[chunk_start:chunk_end]

        contents = [""] * len(chunk_files)
        desc = f"Shard {shard_idx} [{chunk_start}:{chunk_end}]"
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_fetch_one, blob_id, enc): i
                for i, (blob_id, enc, _, _) in enumerate(chunk_files)
            }
            for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx),
                            desc=desc, dynamic_ncols=True):
                idx = future_to_idx[fut]
                contents[idx] = fut.result()
                if contents[idx].startswith("__ERROR__:"):
                    failed += 1
                    if failed <= 5:
                        print(f"\n  [error] blob_id={chunk_files[idx][0]}: {contents[idx]}")
                    contents[idx] = ""
                else:
                    downloaded += 1

        # Build parquet shard with content + metadata, dropping empty rows
        rows = []
        for i, (blob_id, src_encoding, language, path) in enumerate(chunk_files):
            if contents[i]:
                rows.append({
                    "blob_id": blob_id,
                    "language": language,
                    "path": path,
                    "content": contents[i],
                })

        if rows:
            table = pa.Table.from_pylist(rows)
            shard_path = parquet_dir / f"shard-{shard_idx:05d}.parquet"
            pq.write_table(table, str(shard_path))
            print(f"  Saved {shard_path.name} ({len(rows):,} files)")

        shard_idx += 1

    print(f"\nDone: {downloaded:,} files downloaded, {failed:,} failed")
    print(f"Output: {parquet_dir}")
    return str(local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu (English/Chinese), FineWeb-2, or The Stack V2 Smol from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb-edu",
        choices=["fineweb-edu", "fineweb-edu-chinese", "fineweb-2", "the-stack-v2-smol", "starcoderdata"],
        help="Which dataset to download (default: fineweb-edu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/projects/0/hpmlprjs/interns/marko/hnet/data/",
        help="Directory to save dataset (default: /scratch-shared/mivanov1/hnet/data)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=(
            "Subset to download. "
            "For fineweb-edu: e.g. 'sample-10BT' (default). "
            "For fineweb-edu-chinese: score range '2_3', '3_4', or '4_5' (default: '3_4'). "
            "For fineweb-2: language code, e.g. 'kor_Hang' (default). "
            "For starcoderdata: language, e.g. 'python' (default: all). "
            "Not used for the-stack-v2-smol (downloads all languages)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of parallel S3 download threads (the-stack-v2-smol only, default: 32)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap the number of files to download (the-stack-v2-smol only, useful for testing)",
    )

    args = parser.parse_args()

    if args.dataset == "fineweb-edu-chinese":
        score_range = args.subset if args.subset is not None else "3_4"
        download_fineweb_edu_chinese(args.output_dir, score_range)
    elif args.dataset == "fineweb-2":
        subset = args.subset if args.subset is not None else "kor_Hang"
        download_fineweb_2(args.output_dir, subset)
    elif args.dataset == "starcoderdata":
        download_starcoderdata(args.output_dir, subset=args.subset)
    elif args.dataset == "the-stack-v2-smol":
        download_the_stack_v2_smol(args.output_dir,
                                   num_workers=args.num_workers,
                                   max_files=args.max_files)
    else:
        subset = args.subset if args.subset is not None else "sample-10BT"
        download_fineweb_edu(args.output_dir, subset)
