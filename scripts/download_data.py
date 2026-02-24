#!/usr/bin/env python3
"""
Download the FineWeb-Edu dataset (sample-10BT) from HuggingFace to local disk.

Usage:
    python download_data.py
    python download_data.py --output-dir /scratch-shared/mivanov1/hnet/data --subset sample-10BT
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_fineweb_edu(output_dir: str, subset: str = "sample-10BT"):
    """Download a FineWeb-Edu subset from HuggingFace.

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu dataset from HuggingFace"
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
        default="sample-10BT",
        help="Dataset subset to download (default: sample-10BT)",
    )

    args = parser.parse_args()
    download_fineweb_edu(args.output_dir, args.subset)
