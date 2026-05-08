# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates a compressor."""

import argparse
import functools
import json
import os
import sys
import time
from collections.abc import Generator
from typing import Callable

import tqdm

import constants
import data_loaders
from compressors import compressor
from compressors import language_model

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Make the repo root importable so `from generate import load_from_pretrained`
# resolves regardless of where this script is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def evaluate_compressor_chunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
    count_header_only_once: bool = True,
    mask_fn: Callable[[bytes], tuple[bytes, int]] | None = None,
    use_tqdm: bool = True,
) -> tuple[float, float]:
    """Evaluates the compressor on the chunked dataset.

    Args:
        compress_fn: The function that compresses data.
        get_data_generator_fn: The function that creates a data generator.
        num_chunks: The number of chunks to consider.
        count_header_only_once: Whether to count the header as part of the
            compressed output only once for the whole dataset or for every chunk
            individually.
        mask_fn: The function that masks the data in case the compressor cannot
            handle all possible byte values (e.g., language models can only
            process ASCII-decodable data).
        use_tqdm: Whether to use a progress bar or not.

    Returns:
        The compression rate and the total running time.
    """
    num_missed_bits = running_time = raw_length = compressed_length = 0

    data_generator = get_data_generator_fn()
    if use_tqdm:
        data_generator = tqdm.tqdm(data_generator, total=num_chunks)

    for data in data_generator:
        if mask_fn is not None:
            data, missed_bits = mask_fn(data)
            num_missed_bits += missed_bits

        t0 = time.perf_counter()
        compressed_data = compress_fn(data)
        t1 = time.perf_counter()

        running_time += t1 - t0
        raw_length += len(data)
        compressed_length += len(compressed_data)

    # Since language models are trained on ASCII strings, they cannot handle all
    # byte values. Thus, we mask the data to be ASCII-decodable by zeroing
    # `num_missed_bits` of the most significant bits. However, this means that
    # we are effectively only compressing `num_bits - num_missed_bits` bits, so
    # we rescale the `compressed_length` to account for this.
    if mask_fn is not None:
        num_bits = 8 * num_chunks * constants.CHUNK_SIZE_BYTES
        compressed_length *= num_bits / (num_bits - num_missed_bits)

    # We only count the header once for classical compressors.
    if count_header_only_once:
        header_length = len(compress_fn((0).to_bytes(1, 'little')))
        compressed_length -= header_length * (num_chunks - 1)

    return compressed_length / raw_length, running_time


def evaluate_compressor_unchunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
) -> tuple[float, float]:
    """Evaluates the compressor on the unchunked dataset.

    Args:
        compress_fn: The function that compresses data.
        get_data_generator_fn: The function that creates a data generator.
        num_chunks: The number of chunks to consider.

    Returns:
        The compression rate and the total running time.
    """
    all_data = bytearray()
    for data in tqdm.tqdm(get_data_generator_fn(), total=num_chunks):
        all_data += data
    all_data = bytes(all_data)

    t0 = time.perf_counter()
    compressed_data = compress_fn(all_data)
    t1 = time.perf_counter()

    return len(compressed_data) / len(all_data), t1 - t0


def _build_language_model_compress_fn(model_path: str, config_path: str):
    """Loads the H-Net once and returns a `compress_fn` bound to it."""
    # Imported lazily so the classical compressors don't pay the torch / hnet
    # import cost when they don't need it.
    from generate import load_from_pretrained

    model = load_from_pretrained(model_path, config_path)
    model.eval()
    predict_fn = language_model.make_hnet_predict_fn(model)
    return functools.partial(language_model.compress, predict_fn=predict_fn)


def _save_results(args, results: dict) -> str:
    """Writes a JSON record of the run to RESULTS_DIR and returns the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "compressor": args.compressor,
        "dataset": args.dataset,
        "num_chunks": args.num_chunks,
        "chunk_size_bytes": args.chunk_size,
        "model_path": args.model_path,
        "config_path": args.config_path,
        "model_name": args.result_model_name,
        "timestamp": timestamp,
        **results,
    }
    if "chunked_rate" in payload:
        payload["chunked_bits_per_byte"] = 8 * payload["chunked_rate"]
    if "unchunked_rate" in payload:
        payload["unchunked_bits_per_byte"] = 8 * payload["unchunked_rate"]

    name_suffix = f"_{args.result_model_name}" if args.result_model_name else ""
    filename = (
        f"{args.compressor}_{args.dataset}"
        f"_n{args.num_chunks}{name_suffix}_{timestamp}.json"
    )
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a lossless compressor on a dataset."
    )
    parser.add_argument(
        "--compressor",
        type=str,
        default="gzip",
        choices=list(compressor.COMPRESS_FN_DICT.keys()),
        help="Compressor to use (default: gzip)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="enwik9",
        choices=list(data_loaders.GET_DATA_GENERATOR_FN_DICT.keys()),
        help="Dataset to use (default: enwik9)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=constants.NUM_CHUNKS,
        help=f"Number of chunks (default: {constants.NUM_CHUNKS})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=constants.CHUNK_SIZE_BYTES,
        help=f"Bytes per chunk; only honored by enwik9 and random datasets "
             f"(default: {constants.CHUNK_SIZE_BYTES})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model checkpoint (.pt file). "
             "Required for the language_model compressor.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to the model configuration (.json file). "
             "Required for the language_model compressor.",
    )
    parser.add_argument(
        "--result-model-name",
        type=str,
        default=None,
        help="Short tag appended to the results filename to disambiguate "
             "multiple H-Net variants (e.g., 'XXS_10B').",
    )
    args = parser.parse_args()

    print(f"Compressor: {args.compressor}")
    print(f"Dataset: {args.dataset}")

    if args.compressor == "language_model":
        if args.model_path is None or args.config_path is None:
            parser.error(
                "--model-path and --config-path are required for the "
                "language_model compressor"
            )
        compress_fn = _build_language_model_compress_fn(
            args.model_path, args.config_path
        )
    else:
        compress_fn = compressor.COMPRESS_FN_DICT[args.compressor]

    data_loader_fn = data_loaders.GET_DATA_GENERATOR_FN_DICT[args.dataset]
    if args.dataset in ("enwik9", "random"):
        get_data_generator_fn = functools.partial(
            data_loader_fn,
            num_chunks=args.num_chunks,
            sequence_length=args.chunk_size,
        )
    else:
        if args.chunk_size != constants.CHUNK_SIZE_BYTES:
            parser.error(
                f"--chunk-size is only supported for enwik9 and random datasets "
                f"(imagenet/librispeech use fixed-size patches)."
            )
        get_data_generator_fn = functools.partial(
            data_loader_fn, num_chunks=args.num_chunks
        )

    if args.compressor in compressor.COMPRESSOR_TYPES["classical"]:
        unchunked_rate, unchunked_time = evaluate_compressor_unchunked(
            compress_fn=compress_fn,
            get_data_generator_fn=get_data_generator_fn,
            num_chunks=args.num_chunks,
        )
        chunked_rate, chunked_time = evaluate_compressor_chunked(
            compress_fn=compress_fn,
            get_data_generator_fn=get_data_generator_fn,
            num_chunks=args.num_chunks,
            count_header_only_once=True,
            mask_fn=None,
        )
        print(f"Unchunked: {100 * unchunked_rate:.1f} [{unchunked_time:.1f}s]")
        print(f"Chunked:   {100 * chunked_rate:.1f} [{chunked_time:.1f}s]")

        results = {
            "unchunked_rate": unchunked_rate,
            "unchunked_time_s": unchunked_time,
            "chunked_rate": chunked_rate,
            "chunked_time_s": chunked_time,
        }

    elif args.compressor in compressor.COMPRESSOR_TYPES["arithmetic_coding"]:
        # The byte-level H-Net was trained on raw bytes, so feed chunks
        # directly to the model without ASCII / right-shift masking.
        chunked_rate, chunked_time = evaluate_compressor_chunked(
            compress_fn=compress_fn,
            get_data_generator_fn=get_data_generator_fn,
            num_chunks=args.num_chunks,
            count_header_only_once=False,
            mask_fn=None,
        )
        print(f"Chunked: {100 * chunked_rate:.1f} [{chunked_time:.1f}s]")

        results = {
            "chunked_rate": chunked_rate,
            "chunked_time_s": chunked_time,
        }

    saved_path = _save_results(args, results)
    print(f"Saved results to {saved_path}")


if __name__ == "__main__":
    main()
