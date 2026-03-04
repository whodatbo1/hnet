#!/usr/bin/env python3
"""
Download FineWeb-Edu (English) or FineWeb-Edu-Chinese from HuggingFace.

Usage:
    # English (default)
    python download_data.py
    python download_data.py --output-dir /scratch-shared/mivanov1/hnet/data --subset sample-10BT

    # Chinese
    python download_data.py --dataset fineweb-edu-chinese --subset 3_4
    python download_data.py --dataset fineweb-edu-chinese --subset 4_5
"""

import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu (English or Chinese) from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb-edu",
        choices=["fineweb-edu", "fineweb-edu-chinese"],
        help="Which dataset to download (default: fineweb-edu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch-shared/mivanov1/hnet/data",
        help="Directory to save dataset (default: /scratch-shared/mivanov1/hnet/data)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=(
            "Subset to download. "
            "For fineweb-edu: e.g. 'sample-10BT' (default). "
            "For fineweb-edu-chinese: score range '2_3', '3_4', or '4_5' (default: '3_4')."
        ),
    )

    args = parser.parse_args()

    if args.dataset == "fineweb-edu-chinese":
        score_range = args.subset if args.subset is not None else "3_4"
        download_fineweb_edu_chinese(args.output_dir, score_range)
    else:
        subset = args.subset if args.subset is not None else "sample-10BT"
        download_fineweb_edu(args.output_dir, subset)
