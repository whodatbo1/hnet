"""
Data loading utilities for HNet training.

Expects data to be pre-tokenized into flat uint8 binary files via
scripts/prepare_data.py. Training processes open the same .bin file via
np.memmap so the OS page cache keeps the data in RAM once, shared across
all concurrent runs.
"""

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset


# Subsets whose dir name is used verbatim under <data_dir>/. Anything else
# resolves to <data_dir>/fineweb-edu-<subset>/ for backwards compat with the
# original FineWeb-Edu setup.
_RAW_SUBSET_PREFIXES = ("the-stack-v2-smol", "starcoderdata", "fineweb-2", "hplt")


def _resolve_subset_dir(data_dir: Path, subset: str) -> Path:
    if subset.startswith(_RAW_SUBSET_PREFIXES):
        return data_dir / subset
    return data_dir / f"fineweb-edu-{subset}"


class MemmapByteDataset(IterableDataset):
    """Reads fixed-length chunks from a pre-tokenized uint8 binary file.

    The file is opened as a read-only np.memmap, so all processes on the same
    machine share a single copy in the OS page cache at no extra memory cost.

    Each yielded item is a LongTensor of length seq_len + 1. The training loop
    splits it into input_ids = item[:-1] and targets = item[1:].

    Args:
        bin_path: Path to the flat uint8 binary file (train.bin or val.bin).
        seq_len: Number of input tokens per sample (output length = seq_len + 1).
        seed: Base random seed. Each worker gets seed + global_worker_id.
        shuffle: Shuffle chunk order each epoch (True for train, False for val).
        max_samples: Stop after this many chunks (used to cap validation).
    """

    def __init__(self, bin_path, seq_len, seed=42, shuffle=True, max_samples=None):
        self.bin_path = str(bin_path)
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle = shuffle
        self.max_samples = max_samples

        num_bytes = Path(bin_path).stat().st_size
        # Each chunk is seq_len + 1 bytes (input + 1 target)
        self.num_chunks = num_bytes // (seq_len + 1)
        assert self.num_chunks > 0, f"Binary file too small for seq_len={seq_len}: {bin_path}"

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        # Each worker owns a strided slice of chunks
        worker_chunks = np.arange(global_worker_id, self.num_chunks, total_workers)

        rng = np.random.default_rng(self.seed + global_worker_id)
        if self.shuffle:
            rng.shuffle(worker_chunks)

        # Open memmap once per worker iteration; the OS keeps it in page cache
        data = np.memmap(self.bin_path, dtype=np.uint8, mode="r")

        samples_yielded = 0
        chunk_len = self.seq_len + 1
        for chunk_id in worker_chunks:
            start = int(chunk_id) * chunk_len
            chunk = torch.from_numpy(data[start : start + chunk_len].astype(np.int64))
            yield chunk
            samples_yielded += 1
            if self.max_samples is not None and samples_yielded >= self.max_samples:
                return


class MultilingualByteDataset(IterableDataset):
    """Multiplex several MemmapByteDatasets at configured weights.

    On every yield, picks one source by weighted-random choice and pulls its
    next chunk. When a source's per-worker iterator is exhausted, it is
    re-iterated (so smaller languages cycle more often — that is the whole
    point of the weight knob). DDP/worker sharding is delegated to each
    sub-dataset, which already shards by global_worker_id.

    Args:
        sub_datasets: List of MemmapByteDataset instances, one per source.
        weights: Sampling weights (will be normalized to sum to 1).
        seed: Base seed for the language-choice RNG. Per-worker seeds are
              derived as seed + 9973 * global_worker_id, distinct from the
              chunk-shuffle seed used inside each sub-dataset.
        max_samples: If set, cap total per-worker yields (used to cap val).
    """

    def __init__(self, sub_datasets, weights, seed=42, max_samples=None):
        assert len(sub_datasets) == len(weights) and len(sub_datasets) > 0
        self.sub_datasets = list(sub_datasets)
        weights = np.asarray(weights, dtype=np.float64)
        assert (weights >= 0).all() and weights.sum() > 0, "weights must be non-negative and sum > 0"
        self.weights = weights / weights.sum()
        self.seed = seed
        self.max_samples = max_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        global_worker_id = rank * num_workers + worker_id
        rng = np.random.default_rng(self.seed + 9973 * global_worker_id)

        # Local mutable copy so we can zero out exhausted (or empty-on-this-worker) sources
        weights = self.weights.copy()
        iters = [iter(d) for d in self.sub_datasets]
        n = len(iters)

        samples_yielded = 0
        while True:
            if self.max_samples is not None and samples_yielded >= self.max_samples:
                return
            if weights.sum() == 0:
                return
            i = int(rng.choice(n, p=weights / weights.sum()))
            try:
                chunk = next(iters[i])
            except StopIteration:
                # Restart this source for the next epoch on this worker
                iters[i] = iter(self.sub_datasets[i])
                try:
                    chunk = next(iters[i])
                except StopIteration:
                    # Source has zero chunks for this worker — drop it from the mixture
                    weights[i] = 0
                    continue
            yield chunk
            samples_yielded += 1


class SFTByteDataset(IterableDataset):
    """Reads aligned (token, label) chunks for supervised fine-tuning.

    Like MemmapByteDataset but opens TWO flat files written by
    scripts/prepare_sft_data.py:
        - token file (uint8): BOS + prompt + target + EOS, packed across examples
        - label file (int16): -100 on BOS+prompt bytes, real byte id on target+EOS

    Both streams are byte-for-byte aligned, so the same chunk indexing/shuffle is
    applied to each. Each yielded item is a LongTensor of shape (2, seq_len + 1):
    row 0 = token ids, row 1 = labels. The SFT branch in train.py splits it into
    input_ids = tokens[:, :-1] and the cross-entropy targets = labels[:, 1:]
    (with -100 outside the target region), while passing the dense token row to
    the model's `targets=` arg (used by the router's boundary-matching loss).

    Args mirror MemmapByteDataset; `label_path` points at the int16 label file.
    """

    def __init__(self, bin_path, label_path, seq_len, seed=42, shuffle=True, max_samples=None):
        self.bin_path = str(bin_path)
        self.label_path = str(label_path)
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle = shuffle
        self.max_samples = max_samples

        num_tokens = Path(bin_path).stat().st_size            # uint8 -> 1 byte/token
        num_label_bytes = Path(label_path).stat().st_size     # int16 -> 2 bytes/label
        assert num_label_bytes == 2 * num_tokens, (
            f"Label file {label_path} ({num_label_bytes} B) is not int16-aligned with "
            f"token file {bin_path} ({num_tokens} tokens). Re-run prepare_sft_data.py."
        )
        self.num_chunks = num_tokens // (seq_len + 1)
        assert self.num_chunks > 0, f"SFT data too small for seq_len={seq_len}: {bin_path}"

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        worker_chunks = np.arange(global_worker_id, self.num_chunks, total_workers)
        rng = np.random.default_rng(self.seed + global_worker_id)
        if self.shuffle:
            rng.shuffle(worker_chunks)

        tokens = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
        labels = np.memmap(self.label_path, dtype=np.int16, mode="r")

        samples_yielded = 0
        chunk_len = self.seq_len + 1
        for chunk_id in worker_chunks:
            start = int(chunk_id) * chunk_len
            tok = torch.from_numpy(tokens[start : start + chunk_len].astype(np.int64))
            lab = torch.from_numpy(labels[start : start + chunk_len].astype(np.int64))
            yield torch.stack([tok, lab])  # (2, seq_len + 1)
            samples_yielded += 1
            if self.max_samples is not None and samples_yielded >= self.max_samples:
                return


def _build_dataset(subset_dirs, weights, file_name, seq_len, seed, shuffle, max_samples):
    """Return one MemmapByteDataset for a single source, else a MultilingualByteDataset.

    `max_samples` (per-worker cap) is applied at the mixture level when
    multiple sources are present, otherwise at the sub-dataset.
    """
    if len(subset_dirs) == 1:
        return MemmapByteDataset(
            bin_path=subset_dirs[0] / file_name,
            seq_len=seq_len,
            seed=seed,
            shuffle=shuffle,
            max_samples=max_samples,
        )
    sub_datasets = [
        MemmapByteDataset(
            bin_path=d / file_name,
            seq_len=seq_len,
            seed=seed,
            shuffle=shuffle,
            max_samples=None,  # cap at the mixture level instead
        )
        for d in subset_dirs
    ]
    return MultilingualByteDataset(
        sub_datasets=sub_datasets,
        weights=weights,
        seed=seed,
        max_samples=max_samples,
    )


def create_dataloaders(data_dir, dataset_config, dataset_mixture,
                       seq_len, seed, val_batches, batch_size, num_workers):
    """Create train and validation DataLoaders from config.

    Single-source mode (dataset_mixture is None / empty): looks for pre-tokenized
    binary files at <data_dir>/<resolved>/{train,val}.bin, where <resolved> is:
        - <data_subset>                   for prefixes the-stack-v2-smol,
                                          starcoderdata, fineweb-2, hplt
        - fineweb-edu-<data_subset>       otherwise (e.g. "sample-10BT")

    Mixture mode: dataset_mixture is a list of dicts, e.g.
        [{name: hplt-eng_Latn, weight: 0.75}, {name: hplt-nld_Latn, weight: 0.25}]
    Each `name` is resolved with the same rules above. Weights are normalized
    to 1. Train and val both use the same mixture so the reported val_loss is
    directly comparable to train loss. Per-yield weighted sampling means batch
    composition matches the weights in expectation.

    Run scripts/prepare_data.py once per source to generate the .bin files.
    """
    if dataset_mixture:
        sources = [(s["name"], float(s["weight"])) for s in dataset_mixture]
    else:
        sources = [(dataset_config, 1.0)]

    subset_names, weights = zip(*sources)
    subset_dirs = [_resolve_subset_dir(data_dir, name) for name in subset_names]

    for subset, d in zip(subset_names, subset_dirs):
        for fname in ("train.bin", "val.bin"):
            p = d / fname
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run scripts/prepare_data.py first:\n"
                    f"  python scripts/prepare_data.py --data-dir {data_dir} --subset {subset}"
                )

    train_dataset = _build_dataset(
        subset_dirs, weights, "train.bin",
        seq_len=seq_len, seed=seed, shuffle=True, max_samples=None,
    )
    val_dataset = _build_dataset(
        subset_dirs, weights, "val.bin",
        seq_len=seq_len, seed=seed, shuffle=False,
        max_samples=val_batches * batch_size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def create_sft_dataloaders(data_dir, sft_name, seq_len, seed,
                           val_batches, batch_size, num_workers):
    """Create train/val DataLoaders for ALMA-style SFT (paired token+label streams).

    Reads the four files written by scripts/prepare_sft_data.py under
    <data_dir>/<sft_name>/:
        train.bin / val.bin               uint8 packed token stream
        labels_train.bin / labels_val.bin int16 byte-aligned label stream

    Single source: both translation directions are already interleaved in the
    packed stream, so no mixture is needed. Each batch is shaped
    (batch_size, 2, seq_len + 1) — row 0 tokens, row 1 labels.
    """
    sft_dir = Path(data_dir) / sft_name
    files = {
        "train_tok": sft_dir / "train.bin",
        "train_lab": sft_dir / "labels_train.bin",
        "val_tok": sft_dir / "val.bin",
        "val_lab": sft_dir / "labels_val.bin",
    }
    for key, p in files.items():
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/prepare_sft_data.py first:\n"
                f"  python scripts/prepare_sft_data.py --data-dir {data_dir} "
                f"--out-name {sft_name} --pairs eng_Latn:bul_Cyrl bul_Cyrl:eng_Latn"
            )

    train_dataset = SFTByteDataset(
        bin_path=files["train_tok"], label_path=files["train_lab"],
        seq_len=seq_len, seed=seed, shuffle=True, max_samples=None,
    )
    val_dataset = SFTByteDataset(
        bin_path=files["val_tok"], label_path=files["val_lab"],
        seq_len=seq_len, seed=seed, shuffle=False,
        max_samples=val_batches * batch_size,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    return train_dataloader, val_dataloader
