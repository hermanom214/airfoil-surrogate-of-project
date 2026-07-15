from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class DatasetSplit:
    train_set: Dataset
    val_set: Dataset
    train_indices: list[int]
    val_indices: list[int]
    train_size: int
    val_size: int


def compute_train_val_indices(
    dataset_len: int,
    validation_split: float,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0")

    train_size = max(1, int((1.0 - validation_split) * dataset_len))
    val_size = dataset_len - train_size

    if val_size == 0:
        all_indices = list(range(dataset_len))
        return all_indices, all_indices

    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    return train_indices, val_indices


def build_train_val_split(
    dataset: Dataset,
    validation_split: float,
    split_seed: int,
) -> DatasetSplit:
    dataset_len = len(dataset)
    logical_train_size = max(1, int((1.0 - validation_split) * dataset_len))
    logical_val_size = dataset_len - logical_train_size

    train_indices, val_indices = compute_train_val_indices(
        dataset_len=dataset_len,
        validation_split=validation_split,
        split_seed=split_seed,
    )

    if logical_val_size == 0 or train_indices == val_indices:
        return DatasetSplit(
            train_set=dataset,
            val_set=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            train_size=logical_train_size,
            val_size=logical_val_size,
        )

    return DatasetSplit(
        train_set=Subset(dataset, train_indices),
        val_set=Subset(dataset, val_indices),
        train_indices=train_indices,
        val_indices=val_indices,
        train_size=logical_train_size,
        val_size=logical_val_size,
    )