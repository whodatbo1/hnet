#!/usr/bin/env python3
"""
Script to download Hnet model weights from Hugging Face.
Downloads the model from cartesia-ai/hnet_2stage_XL and saves to ./checkpoints
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_NAMES = [
    "hnet_1stage_L",
    "hnet_1stage_XL",
    "hnet_2stage_L",
    "hnet_2stage_XL",
    "hnet_2stage_XL_chinese",
    "hnet_2stage_XL_code",
]

REPOS = ["cartesia-ai/" + model_name for model_name in MODEL_NAMES]

def download_hnet_weights(output_dir: str = "checkpoints", model: str = "hnet_1stage_L"):
    """
    Download Hnet model weights from Hugging Face.

    Args:
        output_dir: Directory to save the model weights (default: "checkpoints")
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Hnet model weights to {output_path.absolute()}...")
    print(f"Model: cartesia-ai/{model}")

    try:
        # Download the model
        model_path = snapshot_download(
            repo_id=f"cartesia-ai/{model}",
            local_dir=output_path / model,
            local_dir_use_symlinks=False,
        )

        print(f"✓ Successfully downloaded model weights to: {model_path}")
        return model_path

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Hnet model weights from Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model weights (default: checkpoints)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="hnet_1stage_L",
        choices=MODEL_NAMES,
        help="Model name",
    )

    parser.add_argument(
        "--download-all",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    if args.download_all:
        print("download-all flag set to true. Downloading all model checkpoints...")
        for model in MODEL_NAMES:
            download_hnet_weights(args.output_dir, model)
    else:        
        download_hnet_weights(args.output_dir, args.model)
