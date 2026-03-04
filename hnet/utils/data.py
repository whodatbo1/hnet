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


def create_dataloaders(data_dir, data_subset, seq_len, seed, val_batches, batch_size, num_workers):
    """Create train and validation DataLoaders from config.

    Looks for pre-tokenized binary files at:
        <data_dir>/fineweb-edu-<data_subset>/train.bin   (e.g. data_subset="sample-10BT")
        <data_dir>/fineweb-edu-<data_subset>/val.bin

    For the Chinese dataset use data_subset="chinese-<score_range>", e.g. "chinese-3_4",
    which resolves to <data_dir>/fineweb-edu-chinese-3_4/{train,val}.bin.

    Run scripts/prepare_data.py once to generate them.
    """
    subset_dir = data_dir / f"fineweb-edu-{data_subset}"

    train_bin = subset_dir / "train.bin"
    val_bin = subset_dir / "val.bin"

    for p in (train_bin, val_bin):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/prepare_data.py first:\n"
                f"  python scripts/prepare_data.py --data-dir {data_dir} --subset {data_subset}"
            )

    train_dataset = MemmapByteDataset(
        bin_path=train_bin,
        seq_len=seq_len,
        seed=seed,
        shuffle=True,
    )
    val_dataset = MemmapByteDataset(
        bin_path=val_bin,
        seq_len=seq_len,
        seed=seed,
        shuffle=False,
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
