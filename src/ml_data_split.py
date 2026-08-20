from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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


@dataclass(frozen=True)
class DevelopmentTestSplit:
    development_indices: list[int]
    test_indices: list[int]
    development_case_ids: list[str]
    test_case_ids: list[str]
    dataset_case_ids: list[str]
    dataset_fingerprint: str
    test_fraction: float
    seed: int


@dataclass(frozen=True)
class CVFold:
    fold_id: int
    train_indices: list[int]
    val_indices: list[int]
    train_case_ids: list[str]
    val_case_ids: list[str]
    validation_group_ids: list[str]


def compute_train_val_indices(dataset_len: int, validation_split: float,
                              split_seed: int) -> tuple[list[int], list[int]]:
    """Legacy deterministic holdout kept for backward compatibility."""
    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0")
    train_size = max(1, int((1.0 - validation_split) * dataset_len))
    val_size = dataset_len - train_size
    if val_size == 0:
        all_indices = list(range(dataset_len))
        return all_indices, all_indices
    permutation = torch.randperm(
        dataset_len, generator=torch.Generator().manual_seed(split_seed)
    ).tolist()
    return permutation[:train_size], permutation[train_size:]


def build_train_val_split(dataset: Dataset, validation_split: float,
                          split_seed: int) -> DatasetSplit:
    dataset_len = len(dataset)
    logical_train_size = max(1, int((1.0 - validation_split) * dataset_len))
    logical_val_size = dataset_len - logical_train_size
    train_indices, val_indices = compute_train_val_indices(
        dataset_len, validation_split, split_seed
    )
    if logical_val_size == 0 or train_indices == val_indices:
        return DatasetSplit(dataset, dataset, train_indices, val_indices,
                            logical_train_size, logical_val_size)
    return DatasetSplit(Subset(dataset, train_indices), Subset(dataset, val_indices),
                        train_indices, val_indices, logical_train_size, logical_val_size)


def build_holdout_fold(development_indices: Sequence[int], case_ids: Sequence[str],
                       validation_split: float, seed: int) -> CVFold:
    """Build one deterministic train/validation fold inside the development set."""
    development = [int(i) for i in development_indices]
    local_train, local_val = compute_train_val_indices(
        len(development), validation_split, seed
    )
    train = [development[i] for i in local_train]
    val = [development[i] for i in local_val]
    return CVFold(
        fold_id=1,
        train_indices=train,
        val_indices=val,
        train_case_ids=[str(case_ids[i]) for i in train],
        val_case_ids=[str(case_ids[i]) for i in val],
        validation_group_ids=[],
    )


def dataset_fingerprint(case_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(v) for v in case_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _new_development_test_split(case_ids: Sequence[str], test_fraction: float,
                                seed: int) -> DevelopmentTestSplit:
    if len(case_ids) < 2:
        raise ValueError("At least two valid cases are required for a fixed test split")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_ids must be unique")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    n_test = min(len(case_ids) - 1, max(1, round(len(case_ids) * test_fraction)))
    perm = torch.randperm(len(case_ids), generator=torch.Generator().manual_seed(seed)).tolist()
    test_set = set(perm[:n_test])
    development = [i for i in range(len(case_ids)) if i not in test_set]
    test = [i for i in range(len(case_ids)) if i in test_set]
    ids = [str(v) for v in case_ids]
    return DevelopmentTestSplit(
        development, test, [ids[i] for i in development], [ids[i] for i in test],
        ids, dataset_fingerprint(ids), float(test_fraction), int(seed)
    )


def load_or_create_fixed_test_split(
    case_ids: Sequence[str], metadata_path: Path, test_fraction: float = 0.2,
    seed: int = 42, *, regenerate_on_dataset_change: bool = False,
) -> DevelopmentTestSplit:
    """Persist a test split by case ID; fold/search settings never affect it."""
    current = [str(v) for v in case_ids]
    if metadata_path.exists() and regenerate_on_dataset_change:
        warnings.warn(
            f"Explicit regeneration requested; replacing fixed test split at {metadata_path}.",
            RuntimeWarning,
        )
    elif metadata_path.exists():
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_ids = [str(v) for v in raw["dataset_case_ids"]]
        saved_fingerprint = str(raw.get("dataset_fingerprint", ""))
        current_fingerprint = dataset_fingerprint(current)
        if (len(saved_ids) == len(current) and set(saved_ids) == set(current)
                and saved_fingerprint == current_fingerprint):
            id_to_idx = {case_id: i for i, case_id in enumerate(current)}
            test_ids = [str(v) for v in raw["test_case_ids"]]
            dev_ids = [str(v) for v in raw["development_case_ids"]]
            return DevelopmentTestSplit(
                [id_to_idx[v] for v in dev_ids], [id_to_idx[v] for v in test_ids],
                dev_ids, test_ids, current, dataset_fingerprint(current),
                float(raw["test_fraction"]), int(raw["seed"])
            )
        message = (
            f"Dataset fingerprint changed for {metadata_path}; existing fixed test split "
            "is incompatible and explicit regeneration is required."
        )
        raise RuntimeError(message)
    split = _new_development_test_split(current, test_fraction, seed)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    return split


def naca_group_from_case_id(case_id: str) -> str:
    match = re.search(r"naca[_-]?(\d{4,5})", case_id, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract NACA code from case ID: {case_id}")
    return f"NACA{match.group(1)}"


def build_cv_folds(development_indices: Sequence[int], case_ids: Sequence[str],
                   n_folds: int, seed: int, strategy: str = "kfold") -> list[CVFold]:
    dev = [int(i) for i in development_indices]
    if n_folds < 2 or n_folds > len(dev):
        raise ValueError("n_folds must be between 2 and development dataset size")
    strategy = strategy.lower().strip()
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    if strategy == "kfold":
        order = torch.randperm(len(dev), generator=torch.Generator().manual_seed(seed)).tolist()
        for position, local_idx in enumerate(order):
            buckets[position % n_folds].append(dev[local_idx])
    elif strategy == "group_kfold_naca":
        groups: dict[str, list[int]] = {}
        for idx in dev:
            groups.setdefault(naca_group_from_case_id(str(case_ids[idx])), []).append(idx)
        if n_folds > len(groups):
            raise ValueError("n_folds cannot exceed the number of NACA groups")
        group_names = sorted(groups)
        permutation = torch.randperm(
            len(group_names), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        for pos in permutation:
            target = min(range(n_folds), key=lambda i: (len(buckets[i]), i))
            buckets[target].extend(groups[group_names[pos]])
    else:
        raise ValueError("cv_strategy must be 'kfold' or 'group_kfold_naca'")
    dev_set = set(dev)
    folds: list[CVFold] = []
    for fold_id, val in enumerate(buckets, start=1):
        val = sorted(val)
        train = sorted(dev_set.difference(val))
        validation_groups = (
            sorted({naca_group_from_case_id(str(case_ids[i])) for i in val})
            if strategy == "group_kfold_naca" else []
        )
        folds.append(CVFold(fold_id, train, val,
                            [str(case_ids[i]) for i in train],
                            [str(case_ids[i]) for i in val], validation_groups))
    return folds


def save_cv_folds(path: Path, folds: Sequence[CVFold], *, strategy: str,
                  n_folds: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cv_strategy": strategy, "n_folds": n_folds, "seed": seed,
               "folds": [asdict(fold) for fold in folds]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
