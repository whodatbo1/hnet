"""
Data loading utilities for HNet training.

Loads FineWeb-Edu (or similar) datasets from local parquet files,
performs train/val splitting, byte-tokenizes on-the-fly, and packs
into fixed-length sequences.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_dataset

from hnet.utils.tokenizers import ByteTokenizer


class PackedByteDataset(IterableDataset):
    """Loads text from local parquet files, encodes to bytes, packs into fixed-length chunks.

    The dataset is loaded from disk (not streaming), split into train/val using HF's
    train_test_split, and then iterated as an IterableDataset for memory efficiency
    (documents are tokenized and packed on-the-fly).

    In distributed training, each rank processes a disjoint shard of documents.

    Args:
        hf_dataset: A HuggingFace Dataset object (already split into train or val).
        seq_len: Fixed sequence length for packed chunks.
        tokenizer: ByteTokenizer instance.
        seed: Random seed for shuffling.
        shuffle: Whether to shuffle documents each epoch.
        max_samples: If set, stop after yielding this many chunks.
    """

    def __init__(self, hf_dataset, seq_len, tokenizer, seed=42, shuffle=True,
                 max_samples=None):
        self.hf_dataset = hf_dataset
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.seed = seed
        self.shuffle = shuffle
        self.max_samples = max_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Total parallelism = world_size * num_workers
        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        # Shard the dataset across all workers
        ds = self.hf_dataset.shard(num_shards=total_workers, index=global_worker_id)

        if self.shuffle:
            ds = ds.shuffle(seed=self.seed + global_worker_id)

        buffer = []
        samples_yielded = 0

        for example in ds:
            text = example.get("text", "")
            if not text:
                continue

            encoded = self.tokenizer.encode([text], add_bos=True, add_eos=True)[0]
            token_ids = encoded["input_ids"].tolist()
            buffer.extend(token_ids)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                yield torch.tensor(chunk, dtype=torch.long)
                samples_yielded += 1
                if self.max_samples is not None and samples_yielded >= self.max_samples:
                    return


def load_local_dataset(data_dir, subset="sample-10BT"):
    """Load a FineWeb-Edu subset from local parquet files.

    Args:
        data_dir: Root data directory (e.g. /scratch-shared/mivanov1/hnet/data).
        subset: Subset name (e.g. "sample-10BT").

    Returns:
        A HuggingFace Dataset object.
    """
    # After download, parquets live at <data_dir>/fineweb-edu-<subset>/sample/<size>/
    # e.g. subset="sample-10BT" -> repo_path="sample/10BT"
    repo_path = subset.replace("-", "/")
    parquet_dir = f"{data_dir}/fineweb-edu-{subset}/{repo_path}"
    ds = load_dataset("parquet", data_dir=parquet_dir, split="train")
    return ds


def create_dataloaders(cfg):
    """Create train and validation DataLoaders from config.

    Expects cfg to have:
        - data_dir: path to local data root
        - dataset_config: subset name (e.g. "sample-10BT")
        - seq_len: sequence length
        - seed: random seed
        - val_fraction: fraction of data for validation
        - val_batches: number of val micro-batches
        - batch_size: micro batch size per GPU
        - num_workers: dataloader workers

    Returns:
        (train_dataloader, val_dataloader) tuple.
    """
    tokenizer = ByteTokenizer()

    data_dir = cfg.data_dir
    subset = cfg.get("dataset_config", "sample-10BT")
    val_fraction = cfg.get("val_fraction", 0.005)

    print(f"Loading dataset from {data_dir} (subset={subset})...")
    full_dataset = load_local_dataset(data_dir, subset)
    print(f"Loaded {len(full_dataset):,} documents.")

    # Split into train/val
    split = full_dataset.train_test_split(test_size=val_fraction, seed=cfg.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Train: {len(train_ds):,} docs, Val: {len(val_ds):,} docs")

    # Train dataset
    train_dataset = PackedByteDataset(
        hf_dataset=train_ds,
        seq_len=cfg.seq_len,
        tokenizer=tokenizer,
        seed=cfg.seed,
        shuffle=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val dataset
    val_batches = cfg.get("val_batches", 50)
    val_dataset = PackedByteDataset(
        hf_dataset=val_ds,
        seq_len=cfg.seq_len,
        tokenizer=tokenizer,
        seed=cfg.seed,
        shuffle=False,
        max_samples=val_batches * cfg.batch_size,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader, val_dataloader
