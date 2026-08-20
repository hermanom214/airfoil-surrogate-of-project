from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from src.ml_data_split import (
    build_cv_folds,
    build_holdout_fold,
    load_or_create_fixed_test_split,
    naca_group_from_case_id,
)
from src.ml_hyperparameter_search import generate_search_configurations
from src.ml_clcd_dataset import AirfoilClCdDataset
from src.ml_dataset import AirfoilFlowDataset
from src.ml_experiment import experiment_mode, run_cross_validation, write_experiment_results
from scripts.train_ml_model import parse_args


def _ids() -> list[str]:
    return [f"case_{i:03d}_naca{code}_aoa0p0_u10p0" for i, code in enumerate(
        ["0012", "0012", "2412", "2412", "4412", "4412", "0015", "0015", "2415", "2415"]
    )]


def test_fixed_test_split_is_persisted_and_independent_of_cv(tmp_path) -> None:
    path = tmp_path / "test_cases.json"
    first = load_or_create_fixed_test_split(_ids(), path, 0.2, 42)
    second = load_or_create_fixed_test_split(_ids(), path, 0.4, 999)
    assert first.test_case_ids == second.test_case_ids
    assert json.loads(path.read_text())["test_case_ids"] == first.test_case_ids


def test_dataset_change_requires_explicit_regeneration(tmp_path) -> None:
    path = tmp_path / "test_cases.json"
    load_or_create_fixed_test_split(_ids(), path, 0.2, 42)
    changed = _ids()[:-1]
    original = path.read_text()
    with pytest.raises(RuntimeError, match="explicit regeneration is required"):
        load_or_create_fixed_test_split(changed, path, 0.2, 42)
    assert path.read_text() == original


def test_explicit_regeneration_is_deterministic(tmp_path) -> None:
    path = tmp_path / "test_cases.json"
    load_or_create_fixed_test_split(_ids(), path, 0.2, 42)
    changed = _ids()[:-1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load_or_create_fixed_test_split(
            changed, path, 0.2, 42, regenerate_on_dataset_change=True
        )
    assert caught
    assert result.dataset_case_ids == changed
    path.unlink()
    repeated = load_or_create_fixed_test_split(changed, path, 0.2, 42)
    assert result.test_case_ids == repeated.test_case_ids


def test_regenerate_split_cli_flag_is_explicit() -> None:
    args = parse_args(["--model", "simple_unet", "--regenerate-split"])
    assert args.regenerate_split is True


def test_training_mode_cli_flags_are_mutually_exclusive() -> None:
    assert parse_args(["--single-run"]).single_run is True
    assert parse_args(["--cross-validation"]).cross_validation is True
    with pytest.raises(SystemExit):
        parse_args(["--single-run", "--cross-validation"])


def test_test_ids_never_enter_cv_and_each_dev_is_validation_once(tmp_path) -> None:
    split = load_or_create_fixed_test_split(_ids(), tmp_path / "split.json", 0.2, 42)
    folds = build_cv_folds(split.development_indices, _ids(), 4, 42, "kfold")
    test = set(split.test_indices)
    validation = [idx for fold in folds for idx in fold.val_indices]
    assert test.isdisjoint(validation)
    assert sorted(validation) == sorted(split.development_indices)
    assert len(folds) == 4


def test_single_run_holdout_stays_inside_development_set(tmp_path) -> None:
    split = load_or_create_fixed_test_split(_ids(), tmp_path / "split.json", 0.2, 42)
    fold = build_holdout_fold(split.development_indices, _ids(), 0.25, 42)
    assert set(fold.train_indices).isdisjoint(fold.val_indices)
    assert set(fold.train_indices) | set(fold.val_indices) == set(split.development_indices)
    assert set(split.test_indices).isdisjoint(fold.train_indices)
    assert set(split.test_indices).isdisjoint(fold.val_indices)
    assert len(fold.val_indices) == 2


def test_cv_and_search_configuration_do_not_change_fixed_test(tmp_path) -> None:
    path = tmp_path / "fixed.json"
    split = load_or_create_fixed_test_split(_ids(), path, 0.2, 42)
    build_cv_folds(split.development_indices, _ids(), 2, 1, "kfold")
    generate_search_configurations("none", {"x": 1})
    build_cv_folds(split.development_indices, _ids(), 4, 999, "kfold")
    generate_search_configurations(
        "randomized", {}, search_space={"x": [1, 2]}, n_iter=2, seed=999
    )
    reused = load_or_create_fixed_test_split(_ids(), path, 0.2, 42)
    assert reused.test_case_ids == split.test_case_ids


def test_group_kfold_is_only_within_development_dataset(tmp_path) -> None:
    split = load_or_create_fixed_test_split(_ids(), tmp_path / "fixed.json", 0.2, 42)
    folds = build_cv_folds(split.development_indices, _ids(), 3, 42, "group_kfold_naca")
    test_indices = set(split.test_indices)
    assert all(test_indices.isdisjoint(fold.train_indices) for fold in folds)
    assert all(test_indices.isdisjoint(fold.val_indices) for fold in folds)


def test_group_kfold_keeps_naca_geometry_together() -> None:
    ids = _ids()
    folds = build_cv_folds(range(len(ids)), ids, 5, 42, "group_kfold_naca")
    for fold in folds:
        train_groups = {naca_group_from_case_id(ids[i]) for i in fold.train_indices}
        val_groups = {naca_group_from_case_id(ids[i]) for i in fold.val_indices}
        assert train_groups.isdisjoint(val_groups)
        assert fold.validation_group_ids == sorted(val_groups)


def test_randomized_search_n_iter_and_seed() -> None:
    kwargs = dict(strategy="randomized", base_config={}, search_space={"a": [1, 2, 3], "b": [4, 5]}, n_iter=4, seed=7)
    first = generate_search_configurations(**kwargs)
    second = generate_search_configurations(**kwargs)
    assert len(first) == 4
    assert first == second


def test_none_search_is_single_base_configuration() -> None:
    result = generate_search_configurations("none", {"learning_rate": 0.001}, n_iter=99)
    assert len(result) == 1
    assert result[0].hyperparameters == {"learning_rate": 0.001}


def test_normalization_uses_only_training_fold() -> None:
    dataset = AirfoilClCdDataset.__new__(AirfoilClCdDataset)
    dataset.samples = [
        (np.full(5, value, dtype=np.float32), np.full(2, value, dtype=np.float32), f"case_{value}")
        for value in (1.0, 3.0, 100.0)
    ]
    dataset.fit_normalization([0, 1])
    assert np.allclose(dataset.feature_mean, 2.0)
    assert np.allclose(dataset.target_mean, 2.0)
    assert not np.allclose(dataset.feature_mean, np.mean([1.0, 3.0, 100.0]))


def test_flow_normalization_excludes_validation_extreme(tmp_path) -> None:
    dataset = AirfoilFlowDataset.__new__(AirfoilFlowDataset)
    dataset.sample_params = [(0.0, 10.0), (2.0, 20.0), (4.0, 1000.0)]
    dataset.files = []
    for index, value in enumerate((1.0, 3.0, 1000.0)):
        path = tmp_path / f"sample_{index}.npz"
        np.savez(path, p=np.full((2, 2), value),
                 U=np.full((2, 2, 2), value), fluid_mask=np.ones((2, 2)))
        dataset.files.append(path)
    dataset.fit_condition_normalization([0, 1])
    dataset.fit_target_normalization([0, 1])
    assert dataset.inlet_u_mean == pytest.approx(15.0)
    assert dataset.p_mean == pytest.approx(2.0)


def test_final_normalization_uses_development_not_test() -> None:
    dataset = AirfoilClCdDataset.__new__(AirfoilClCdDataset)
    dataset.samples = [
        (np.full(5, value, dtype=np.float32), np.full(2, value, dtype=np.float32), str(i))
        for i, value in enumerate((1.0, 3.0, 1000.0))
    ]
    development_indices, test_indices = [0, 1], [2]
    dataset.fit_normalization(development_indices)
    assert dataset.feature_mean[0] == pytest.approx(2.0)
    assert test_indices[0] not in development_indices


def test_cv_and_search_modes_are_independent() -> None:
    assert experiment_mode(True, "randomized") == "search_with_cv"
    assert experiment_mode(True, "none") == "cv_without_search"
    assert experiment_mode(False, "none") == "single_run"
    with pytest.raises(ValueError, match="requires cv_enabled=true"):
        experiment_mode(False, "randomized")


def test_none_strategy_with_cv_runs_every_fold() -> None:
    folds = build_cv_folds(range(6), _ids()[:6], 3, 42)
    configs = generate_search_configurations("none", {"x": 1})
    calls: list[int] = []
    results, _ = run_cross_validation(
        configs, folds,
        lambda hp, fold: (calls.append(fold.fold_id) or {"objective": float(fold.fold_id)}),
    )
    assert calls == [1, 2, 3]
    assert len(results[0]["fold_metrics"]) == 3


def test_smoke_sized_workflow_writes_all_experiment_artifacts(tmp_path) -> None:
    split = load_or_create_fixed_test_split(_ids(), tmp_path / "fixed.json", 0.2, 42)
    folds = build_cv_folds(split.development_indices, _ids(), 2, 42)
    configs = generate_search_configurations(
        "randomized", {}, search_space={"learning_rate": [0.001, 0.0003]},
        n_iter=2, seed=42,
    )
    results, best = run_cross_validation(
        configs, folds,
        lambda hp, fold: {"objective": float(hp["learning_rate"] + fold.fold_id)},
    )
    output = tmp_path / "smoke"
    write_experiment_results(
        output, model_name="synthetic", split=split, folds=folds,
        cv_strategy="kfold", search_strategy="randomized",
        sampled_configurations=configs, cv_results=results, best=best,
        final_test_metrics={"objective": 1.0},
    )
    assert {path.name for path in output.iterdir()} == {
        "split_metadata.json", "cv_results.json",
        "hyperparameter_search_results.csv", "final_test_metrics.json",
    }


def test_excluded_case_cannot_enter_splits_when_dataset_supplies_only_valid_ids(tmp_path) -> None:
    valid_ids = _ids()
    excluded_id = "case_invalid_naca9999_aoa0p0_u10p0"
    split = load_or_create_fixed_test_split(valid_ids, tmp_path / "split.json", 0.2, 42)
    folds = build_cv_folds(split.development_indices, valid_ids, 4, 42)
    all_saved_ids = split.test_case_ids + [case for fold in folds for case in fold.val_case_ids]
    assert excluded_id not in all_saved_ids
