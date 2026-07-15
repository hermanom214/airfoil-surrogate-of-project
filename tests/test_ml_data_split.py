from __future__ import annotations

import inspect

import torch
from torch.utils.data import TensorDataset

import scripts.evaluate_ml_model as evaluate_script
import scripts.train_ml_model as train_script
from src.ml_data_split import build_train_val_split, compute_train_val_indices


def test_compute_train_val_indices_is_deterministic_for_same_seed() -> None:
    train_a, val_a = compute_train_val_indices(50, 0.2, 42)
    train_b, val_b = compute_train_val_indices(50, 0.2, 42)
    assert train_a == train_b
    assert val_a == val_b


def test_compute_train_val_indices_changes_with_different_seed() -> None:
    train_a, val_a = compute_train_val_indices(50, 0.2, 42)
    train_b, val_b = compute_train_val_indices(50, 0.2, 43)
    assert train_a != train_b or val_a != val_b


def test_split_indices_are_disjoint_and_cover_all_indices() -> None:
    train_idx, val_idx = compute_train_val_indices(73, 0.2, 7)
    train_set = set(train_idx)
    val_set = set(val_idx)

    assert train_set.isdisjoint(val_set)
    assert train_set.union(val_set) == set(range(73))


def test_build_train_val_split_returns_explicit_indices_and_sizes() -> None:
    dataset = TensorDataset(torch.arange(20, dtype=torch.float32).unsqueeze(1))
    split = build_train_val_split(dataset=dataset, validation_split=0.25, split_seed=11)

    assert split.train_size == len(split.train_indices)
    assert split.val_size == len(split.val_indices)
    assert split.train_size + split.val_size == len(dataset)


def test_build_train_val_split_val_zero_reuses_full_dataset() -> None:
    dataset = TensorDataset(torch.arange(1, dtype=torch.float32).unsqueeze(1))
    split = build_train_val_split(dataset=dataset, validation_split=0.2, split_seed=11)

    assert split.train_set is dataset
    assert split.val_set is dataset
    assert split.train_indices == [0]
    assert split.val_indices == [0]
    assert split.val_size == 0


def test_training_and_evaluation_scripts_use_shared_split_function() -> None:
    train_source = inspect.getsource(train_script)
    eval_source = inspect.getsource(evaluate_script)

    assert "build_train_val_split(" in train_source
    assert "build_train_val_split(" in eval_source
