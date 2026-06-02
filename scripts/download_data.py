#!/usr/bin/env python3
"""
Download FineWeb-Edu (English), FineWeb-Edu-Chinese, FineWeb-2, or HPLT v3.0.

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

    # HPLT v3.0 (https://hplt-project.org/datasets/v3.0) — fixed-byte slice per language
    python download_data.py --dataset hplt --subset nld_Latn          # Dutch (10 GB text default)
    python download_data.py --dataset hplt --subset eng_Latn --target-bytes 10000000000
    python download_data.py --dataset hplt --subset cmn_Hans --target-bytes 5000000000

    # Code (StarCoderData) — gated, accept terms at HF first
    python download_data.py --dataset starcoderdata                    # all languages (~783 GB)
    python download_data.py --dataset starcoderdata --subset python    # single language
"""

import argparse
import sys
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


def download_hplt(output_dir: str, subset: str = "nld_Latn",
                  target_bytes: int = 10_000_000_000,
                  shard_doc_count: int = 100_000):
    """Download a fixed-byte slice of HPLT v3.0 for one language.

    HPLT v3.0 (https://hplt-project.org/datasets/v3.0) ships per-language
    `.jsonl.zst` shards on data.hplt-project.org, named `<bin>_<idx>.jsonl.zst`
    where `bin` is a quality score (10 = best, 5 = lowest). The `.map` index
    lists the shards in lexicographic order (`10_*`, `5_*`, `6_*`, ... `9_*`),
    which is NOT quality order, so we re-sort by quality bin below.

    This function streams shards best-quality-first, decompresses on the fly,
    extracts the `text` field from each JSON document, and writes parquet
    shards (`text` column only) until ``target_bytes`` of UTF-8 text has
    been written. Critical for large languages (e.g. eng_Latn, cmn_Hans)
    where a single source shard already exceeds the target slice size.

    Output layout (matches what prepare_data.py expects for `--dataset hplt`):
        <output_dir>/hplt-<subset>/<subset>/shard-NNNNN.parquet

    Args:
        output_dir: Root directory to save the dataset.
        subset: HPLT language code, e.g. "nld_Latn", "fin_Latn", "eng_Latn",
                "bul_Cyrl", "cmn_Hans". See manifest at
                https://data.hplt-project.org/three/sorted/manifest.json
        target_bytes: UTF-8 text bytes to extract before stopping.
        shard_doc_count: Documents per output parquet shard.
    """
    import io
    import json
    import urllib.request

    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard as zstd

    base = "https://data.hplt-project.org/three/sorted"

    output_path = Path(output_dir)
    local_dir = output_path / f"hplt-{subset}"
    parquet_dir = local_dir / subset
    parquet_dir.mkdir(parents=True, exist_ok=True)

    map_url = f"{base}/{subset}.map"
    print(f"Fetching shard list from {map_url}")
    with urllib.request.urlopen(map_url, timeout=60) as r:
        shard_urls = [line.decode().strip() for line in r if line.strip()]

    # The .map lists shards lexicographically (10_*, 5_*, 6_*, ... 9_*), which
    # is NOT quality order: after the top 10_* bin it drops to the lowest 5_*
    # bin. Re-sort by quality bin descending (then shard index ascending) so a
    # fixed-byte slice keeps the highest-quality documents.
    def _quality_key(url: str):
        name = url.rsplit("/", 1)[-1].split(".", 1)[0]  # ".../10_1.jsonl.zst" -> "10_1"
        try:
            quality, idx = (int(p) for p in name.split("_"))
        except ValueError:
            return (1, 0)  # unrecognised name -> sort last
        return (-quality, idx)

    shard_urls.sort(key=_quality_key)

    print(f"Found {len(shard_urls)} shards. "
          f"Targeting {target_bytes / 1e9:.2f} GB of text into {parquet_dir}")

    written_bytes = 0
    written_docs = 0
    shard_idx = 0
    buffer: list[dict] = []

    def _flush():
        nonlocal shard_idx, buffer
        if not buffer:
            return
        path = parquet_dir / f"shard-{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buffer), str(path))
        print(f"  Wrote {path.name} ({len(buffer):,} docs)")
        shard_idx += 1
        buffer = []

    for url in shard_urls:
        if written_bytes >= target_bytes:
            break
        print(f"\nShard {url.rsplit('/', 1)[-1]}  "
              f"(progress {written_bytes / 1e9:.2f}/{target_bytes / 1e9:.2f} GB)")
        req = urllib.request.Request(url, headers={"User-Agent": "hnet-hplt/1.0"})
        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except Exception as e:
            print(f"  [error] failed to open {url}: {e}", file=sys.stderr)
            continue

        with resp:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(resp) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8",
                                               errors="replace")
                for line in text_stream:
                    if written_bytes >= target_bytes:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = doc.get("text") or ""
                    if not text:
                        continue
                    buffer.append({"text": text})
                    written_bytes += len(text.encode("utf-8"))
                    written_docs += 1
                    if len(buffer) >= shard_doc_count:
                        _flush()

    _flush()

    if written_bytes < target_bytes:
        print(f"\nWarning: exhausted shards with {written_bytes / 1e9:.2f} GB "
              f"(< target {target_bytes / 1e9:.2f} GB).", file=sys.stderr)
    print(f"\nDone: {written_bytes / 1e9:.2f} GB across "
          f"{shard_idx} parquet shards ({written_docs:,} docs).")
    print(f"Output: {parquet_dir}")
    return str(local_dir)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu (English/Chinese), FineWeb-2, or HPLT v3.0"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb-edu",
        choices=["fineweb-edu", "fineweb-edu-chinese", "fineweb-2", "hplt",
                 "starcoderdata"],
        help="Which dataset to download (default: fineweb-edu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/projects/0/hpmlprjs/interns/marko/hnet/data/",
        help="Directory to save dataset (default: /projects/0/hpmlprjs/interns/marko/hnet/data/)",
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
            "For hplt: language code, e.g. 'nld_Latn' (default). "
            "For starcoderdata: language, e.g. 'python' (default: all)."
        ),
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=10_000_000_000,
        help="UTF-8 text byte budget per language (hplt only, default 10 GB)",
    )

    args = parser.parse_args()

    if args.dataset == "fineweb-edu-chinese":
        score_range = args.subset if args.subset is not None else "3_4"
        download_fineweb_edu_chinese(args.output_dir, score_range)
    elif args.dataset == "fineweb-2":
        subset = args.subset if args.subset is not None else "kor_Hang"
        download_fineweb_2(args.output_dir, subset)
    elif args.dataset == "hplt":
        subset = args.subset if args.subset is not None else "nld_Latn"
        download_hplt(args.output_dir, subset, target_bytes=args.target_bytes)
    elif args.dataset == "starcoderdata":
        download_starcoderdata(args.output_dir, subset=args.subset)
    else:
        subset = args.subset if args.subset is not None else "sample-10BT"
        download_fineweb_edu(args.output_dir, subset)
