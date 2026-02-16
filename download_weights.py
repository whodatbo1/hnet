#!/usr/bin/env python3
"""
Script to download Hnet model weights from Hugging Face.
Downloads the model from cartesia-ai/hnet_2stage_XL and saves to ./checkpoints
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_hnet_weights(output_dir: str = "checkpoints"):
    """
    Download Hnet model weights from Hugging Face.

    Args:
        output_dir: Directory to save the model weights (default: "checkpoints")
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Hnet model weights to {output_path.absolute()}...")
    print("Model: cartesia-ai/hnet_2stage_XL")

    try:
        # Download the model
        model_path = snapshot_download(
            repo_id="cartesia-ai/hnet_2stage_XL",
            local_dir=output_path / "hnet_2stage_XL",
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

    args = parser.parse_args()
    download_hnet_weights(args.output_dir)
