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

"""Implements a lossless compressor with language models (arithmetic coding)."""

from collections.abc import Iterator
from typing import Callable

import numpy as np
import torch

import arithmetic_coder
import constants
import utils

# A predict_fn takes a numpy array of shape [1, N] (uint8) and returns
# log-probabilities of shape [1, N, vocab_size] (float). Position i of the
# returned array must contain log P(seq[i] | seq[:i]) — i.e., the distribution
# over the i-th token given everything before it.
PredictFn = Callable[[np.ndarray], np.ndarray]


def make_hnet_predict_fn(model, bos_idx: int = 254) -> PredictFn:
    """Builds a `predict_fn` for an HNetForCausalLM byte-level model.

    The HNet returns logits whose position i predicts the token at input
    position i+1. We prepend a BOS token so that the prediction at output
    position i becomes the distribution over the i-th data byte conditioned on
    all preceding bytes (and BOS).

    Args:
        model: A loaded HNetForCausalLM in eval mode.
        bos_idx: BOS token id used during training (254 for ByteTokenizer).
    """
    device = next(model.parameters()).device

    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        seq = sequence_array[0]
        n = len(seq)

        input_ids_np = np.empty(n + 1, dtype=np.int64)
        input_ids_np[0] = bos_idx
        input_ids_np[1:] = seq
        input_ids = torch.from_numpy(input_ids_np).unsqueeze(0).to(device)
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        with torch.inference_mode():
            output = model.forward(input_ids, mask=mask)

        # logits: [1, n+1, V]; positions 0..n-1 predict seq[0..n-1].
        logits = output.logits[:, :n, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.cpu().numpy()

    return predict_fn


def compress(
    data: bytes,
    predict_fn: PredictFn,
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = False,
) -> bytes | tuple[bytes, int]:
    """Compresses `data` with arithmetic coding driven by `predict_fn`.

    Args:
        data: The data to be compressed.
        predict_fn: Callable returning log P(seq[i] | seq[:i]) for each i;
            see `PredictFn`.
        return_num_padded_bits: Whether to also return the number of zero bits
            appended to make the bitstream byte-aligned.
        use_slow_lossless_compression: If True, recompute the predictive
            distribution one position at a time (O(n^2)). This matches what
            decompress does, so use it when you intend to round-trip. The
            default O(n) path is correct for evaluation.
    """
    sequence_array = np.frombuffer(data, dtype=np.uint8)

    if use_slow_lossless_compression:
        log_probs = []
        for subsequence_length in range(len(sequence_array)):
            subsequence_log_probs = predict_fn(
                sequence_array[None, : subsequence_length + 1]
            )
            log_probs.append(subsequence_log_probs[0, -1])
        log_probs = np.vstack(log_probs)
    else:
        log_probs = predict_fn(sequence_array[None])[0, ...]
    probs = np.exp(log_probs)

    output = []
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,
    )
    for pdf, symbol in zip(probs, sequence_array):
        encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
    encoder.terminate()

    compressed_bits = ''.join(map(str, output))
    compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

    if return_num_padded_bits:
        return compressed_bytes, num_padded_bits

    return compressed_bytes


def decompress(
    data: bytes,
    predict_fn: PredictFn,
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
) -> bytes:
    """Decompresses `data` with arithmetic coding and `predict_fn`.

    See https://en.wikipedia.org/wiki/Arithmetic_coding for details.
    """
    data_iter = iter(utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))

    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )
    # The dummy trailing token preserves the array layout; only the predictive
    # distribution at positions 0..len(sequence_array)-1 is consumed, and our
    # predict_fn never reads the dummy because of the BOS shift.
    sequence_array = np.empty((1,), dtype=np.uint8)
    probs = np.exp(predict_fn(sequence_array[None])[0, ...])

    for idx in range(uncompressed_length):
        token = decoder.decode(
            utils.normalize_pdf_for_arithmetic_coding(probs[idx])
        )
        sequence_array = np.insert(sequence_array, -1, token)
        probs = np.exp(predict_fn(sequence_array[None])[0, ...])

    return sequence_array[:-1].tobytes()
