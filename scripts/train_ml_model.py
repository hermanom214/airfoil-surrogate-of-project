"""Reproducible fixed-test, cross-validation and tuning entry point."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_ml_models_config, load_paths
from src.ml_clcd_dataset import AirfoilClCdDataset
from src.ml_data_split import (
    build_holdout_fold,
    build_cv_folds,
    build_train_val_split,  # legacy API remains importable for older callers
    load_or_create_fixed_test_split,
)
from src.ml_dataset import AirfoilFlowDataset
from src.ml_experiment import experiment_mode, run_cross_validation, write_experiment_results
from src.ml_hyperparameter_search import generate_search_configurations
from src.ml_models import AVAILABLE_MODELS, build_model
from src.ml_device import loader_device_kwargs, log_device, resolve_device
from src.ml_training import evaluate, train_one_epoch, train_one_epoch_physics


ROOT = Path(__file__).resolve().parents[1]
PATHS = load_paths(ROOT / "configs" / "paths.yaml")
ML_CFG = load_ml_models_config(ROOT / "configs" / "ml_models_config.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=AVAILABLE_MODELS, default=ML_CFG.model.name)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=ML_CFG.device)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training.epochs (useful for matched CPU/CUDA benchmarks).",
    )
    parser.add_argument("--cv-strategy", choices=("kfold", "group_kfold_naca"))
    parser.add_argument("--search-strategy", choices=("none", "randomized", "manual_coarse"))
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--single-run", action="store_true",
        help="Disable CV explicitly; requires search strategy 'none'.",
    )
    mode_group.add_argument(
        "--cross-validation", action="store_true",
        help="Enable cross-validation explicitly, overriding cv_enabled from config.",
    )
    parser.add_argument(
        "--regenerate-split", action="store_true",
        help="Explicitly replace an incompatible persisted fixed test split.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run the same workflow with 2 folds, at most 2 candidates and 2 epochs.",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def case_ids_for(dataset: Any, model_name: str) -> list[str]:
    if model_name == "clcd_mlp":
        return [str(sample[2]) for sample in dataset.samples]
    return [path.parent.name for path in dataset.files]


def fit_normalization(dataset: Any, model_name: str, train_indices: Sequence[int]) -> None:
    """The only normalization gateway: callers must pass fold-train or full-development IDs."""
    if model_name == "clcd_mlp":
        dataset.fit_normalization(train_indices)
    else:
        dataset.fit_condition_normalization(train_indices)
        dataset.fit_target_normalization(train_indices)


def normalization_metadata(dataset: Any, model_name: str) -> dict[str, Any]:
    if model_name == "clcd_mlp":
        return {
            "feature_mean": dataset.feature_mean.tolist(),
            "feature_std": dataset.feature_std.tolist(),
            "target_mean": dataset.target_mean.tolist(),
            "target_std": dataset.target_std.tolist(),
            "feature_order": list(dataset.FEATURE_ORDER),
            "target_order": list(dataset.TARGET_ORDER),
        }
    return {key: float(getattr(dataset, key)) for key in (
        "aoa_mean", "aoa_std", "inlet_u_mean", "inlet_u_std", "p_mean", "p_std",
        "ux_mean", "ux_std", "uy_mean", "uy_std",
    )}


def model_kwargs(model_name: str, hp: Mapping[str, Any]) -> dict[str, Any]:
    base = asdict(getattr(ML_CFG.model.params, model_name))
    allowed = {
        "clcd_mlp": {"input_dim", "hidden_dims", "output_dim", "dropout"},
        "simple_unet": {"in_channels", "out_channels", "encoder_channels", "bottleneck_channels"},
        "rans_pinn": {"in_channels", "out_channels", "hidden_channels", "depth"},
    }[model_name]
    return {key: hp.get(key, value) for key, value in base.items() if key in allowed}


def base_hyperparameters(model_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "learning_rate": ML_CFG.training.learning_rate,
        "batch_size": ML_CFG.training.batch_size,
        "weight_decay": 0.0,
    }
    values.update(asdict(getattr(ML_CFG.model.params, model_name)))
    if model_name == "clcd_mlp":
        for key in ("cases_subdir", "force_coeffs_relpath", "tail_window"):
            values.pop(key, None)
    if model_name == "rans_pinn":
        values["physics_loss.weight"] = ML_CFG.physics_loss.weight
        values["physics_loss.warmup_epochs"] = ML_CFG.physics_loss.warmup_epochs
    return values


def train_clcd(dataset: AirfoilClCdDataset, train_indices: Sequence[int],
               eval_indices: Sequence[int], hp: Mapping[str, Any], device: torch.device,
               seed: int, epochs: int) -> tuple[torch.nn.Module, dict[str, float]]:
    seed_everything(seed)
    fit_normalization(dataset, "clcd_mlp", train_indices)
    batch = int(hp["batch_size"])
    loader_kwargs = loader_device_kwargs(device)
    train_loader = DataLoader(Subset(dataset, list(train_indices)), batch_size=batch,
                              shuffle=True, generator=torch.Generator().manual_seed(seed),
                              **loader_kwargs)
    eval_loader = DataLoader(Subset(dataset, list(eval_indices)), batch_size=batch, shuffle=False,
                             **loader_kwargs)
    model = build_model("clcd_mlp", **model_kwargs("clcd_mlp", hp)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp["learning_rate"]),
                                 weight_decay=float(hp.get("weight_decay", 0.0)))
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for features, targets in train_loader:
            non_blocking = device.type == "cuda"
            features = features.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Cl/Cd training loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    losses: list[float] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in eval_loader:
            pred = model(features.to(device, non_blocking=device.type == "cuda")).detach().cpu()
            losses.append(float(criterion(pred, targets).item()))
            truths.append(targets.numpy() * dataset.target_std + dataset.target_mean)
            predictions.append(pred.numpy() * dataset.target_std + dataset.target_mean)
    true = np.concatenate(truths); pred = np.concatenate(predictions)
    error = pred - true
    return model, {
        "objective": float(np.mean(losses)),
        "cl_mae": float(np.mean(np.abs(error[:, 0]))),
        "cd_mae": float(np.mean(np.abs(error[:, 1]))),
        "cl_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "cd_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
    }


def train_field(dataset: AirfoilFlowDataset, model_name: str,
                train_indices: Sequence[int], eval_indices: Sequence[int],
                hp: Mapping[str, Any], device: torch.device, seed: int,
                dx: float, dy: float, epochs: int) -> tuple[torch.nn.Module, dict[str, float]]:
    seed_everything(seed)
    fit_normalization(dataset, model_name, train_indices)
    batch = int(hp["batch_size"])
    loader_kwargs = loader_device_kwargs(device)
    train_loader = DataLoader(Subset(dataset, list(train_indices)), batch_size=batch,
                              shuffle=True, generator=torch.Generator().manual_seed(seed),
                              **loader_kwargs)
    eval_loader = DataLoader(Subset(dataset, list(eval_indices)), batch_size=batch, shuffle=False,
                             **loader_kwargs)
    model = build_model(model_name, **model_kwargs(model_name, hp)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp["learning_rate"]),
                                 weight_decay=float(hp.get("weight_decay", 0.0)))
    for epoch in range(epochs):
        if model_name == "rans_pinn":
            warmup = int(hp.get("physics_loss.warmup_epochs", ML_CFG.physics_loss.warmup_epochs))
            physics_weight = 0.0 if epoch < warmup else float(
                hp.get("physics_loss.weight", ML_CFG.physics_loss.weight)
            )
            train_one_epoch_physics(
                model, train_loader, optimizer, device, dx, dy, ML_CFG.physics_loss.nu,
                dataset.p_mean, dataset.p_std, dataset.ux_mean, dataset.ux_std,
                dataset.uy_mean, dataset.uy_std, physics_weight,
                ML_CFG.physics_loss.pressure_is_kinematic, ML_CFG.physics_loss.mask_erode_pixels,
            )
        else:
            train_one_epoch(model, train_loader, optimizer, device)
    objective = evaluate(model, eval_loader, device)
    return model, {"objective": float(objective), "normalized_masked_mse": float(objective)}


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")
    model_name = args.model
    experiment = ML_CFG.experiment_for(model_name)
    split_cfg = experiment.data_split
    search_cfg = experiment.hyperparameter_search
    cv_strategy = args.cv_strategy or split_cfg.cv_strategy
    search_strategy = args.search_strategy or search_cfg.strategy
    cv_enabled = (
        True if args.cross_validation else False if args.single_run else split_cfg.cv_enabled
    )
    mode = experiment_mode(cv_enabled, search_strategy)
    configured_epochs = args.epochs if args.epochs is not None else ML_CFG.training.epochs
    epochs = min(2, configured_epochs) if args.smoke_test else configured_epochs
    n_folds = min(2, split_cfg.n_folds) if args.smoke_test else split_cfg.n_folds
    n_iter = min(2, search_cfg.n_iter) if args.smoke_test else search_cfg.n_iter
    device = resolve_device(args.device)
    log_device(args.device, device)
    seed_everything(split_cfg.seed)

    if model_name == "clcd_mlp":
        cfg = ML_CFG.model.params.clcd_mlp
        dataset: Any = AirfoilClCdDataset(
            PATHS.flow_fields_output / cfg.cases_subdir, cfg.tail_window,
            force_coeffs_relpath=cfg.force_coeffs_relpath,
        )
        dx = dy = 0.0
    else:
        dataset = AirfoilFlowDataset(PATHS.flow_fields_output)
        with np.load(dataset.files[0], allow_pickle=False) as raw:
            xy = raw["xy"]
        dx = float(np.median(np.abs(np.diff(xy[:, :, 0], axis=1))[np.abs(np.diff(xy[:, :, 0], axis=1)) > 0]))
        dy = float(np.median(np.abs(np.diff(xy[:, :, 1], axis=0))[np.abs(np.diff(xy[:, :, 1], axis=0)) > 0]))

    case_ids = case_ids_for(dataset, model_name)
    model_root = PATHS.project_root / ML_CFG.output.models_subdir / model_name
    output_dir = model_root / "smoke_test" if args.smoke_test else model_root
    fixed_split_path = model_root / "fixed_test_split.json"
    split_existed = fixed_split_path.exists()
    split = load_or_create_fixed_test_split(
        case_ids, fixed_split_path, split_cfg.test_fraction,
        split_cfg.seed,
        regenerate_on_dataset_change=(
            args.regenerate_split or split_cfg.regenerate_on_dataset_change
        ),
    )
    if args.regenerate_split and split_existed:
        print(f"[WARN] Fixed test split was explicitly regenerated: {fixed_split_path}")
    folds = (
        build_cv_folds(split.development_indices, case_ids, n_folds, split_cfg.seed, cv_strategy)
        if cv_enabled else [build_holdout_fold(
            split.development_indices, case_ids, ML_CFG.training.validation_split,
            split_cfg.seed,
        )]
    )
    configurations = generate_search_configurations(
        search_strategy, base_hyperparameters(model_name),
        search_space=search_cfg.search_space,
        manual_coarse_configs=search_cfg.manual_coarse_configs,
        n_iter=n_iter, seed=search_cfg.seed,
    )
    if args.smoke_test:
        configurations = configurations[:2]

    def run_fold(hp: Mapping[str, Any], fold: Any) -> Mapping[str, float]:
        fold_seed = split_cfg.seed + fold.fold_id
        print(
            f"[INFO] Fold {fold.fold_id}: normalization fitted from "
            f"{len(fold.train_indices)} train cases; validation cases={len(fold.val_indices)}"
        )
        print(
            f"[INFO] Fold {fold.fold_id} train ID preview: {fold.train_case_ids[:5]}"
        )
        print(
            f"[INFO] Fold {fold.fold_id} validation ID preview: {fold.val_case_ids[:5]}"
        )
        if fold.validation_group_ids:
            print(f"[INFO] Fold {fold.fold_id} validation NACA groups: {fold.validation_group_ids}")
        if model_name == "clcd_mlp":
            return train_clcd(dataset, fold.train_indices, fold.val_indices, hp, device,
                              fold_seed, epochs)[1]
        return train_field(dataset, model_name, fold.train_indices, fold.val_indices,
                           hp, device, fold_seed, dx, dy, epochs)[1]

    if mode == "single_run":
        validation_metrics = dict(run_fold(configurations[0].hyperparameters, folds[0]))
        cv_results = [{
            "config_id": configurations[0].config_id,
            "hyperparameters": configurations[0].hyperparameters,
            "fold_metrics": [{"fold_id": 1, **validation_metrics}],
            "mean_cv_metric": validation_metrics["objective"],
            "std_cv_metric": 0.0,
        }]
        best = {
            "config_id": configurations[0].config_id,
            "hyperparameters": configurations[0].hyperparameters,
            "fold_metrics": cv_results[0]["fold_metrics"],
            "mean_cv_metric": validation_metrics["objective"],
            "std_cv_metric": 0.0,
        }
    else:
        cv_results, best = run_cross_validation(configurations, folds, run_fold)
    best_hp = best["hyperparameters"]
    final_seed = split_cfg.seed + 10000
    print(
        f"[INFO] Final normalization fitted from all {len(split.development_indices)} "
        f"development cases; fixed test cases excluded={len(split.test_indices)}"
    )
    if model_name == "clcd_mlp":
        final_model, test_metrics = train_clcd(
            dataset, split.development_indices, split.test_indices, best_hp, device,
            final_seed, epochs,
        )
    else:
        final_model, test_metrics = train_field(
            dataset, model_name, split.development_indices, split.test_indices,
            best_hp, device, final_seed, dx, dy, epochs,
        )
    checkpoint = {
        "model_name": model_name,
        "model_state_dict": final_model.state_dict(),
        "model_config": model_kwargs(model_name, best_hp),
        **normalization_metadata(dataset, model_name),
        "dataset_size": len(case_ids),
        "dataset_fingerprint": split.dataset_fingerprint,
        "development_case_ids": split.development_case_ids,
        "test_case_ids": split.test_case_ids,
        "cv_strategy": cv_strategy,
        "n_folds": len(folds),
        "best_hyperparameters": best_hp,
        "split_seed": split.seed,
        "hyperparameter_search_strategy": search_strategy,
        "cv_enabled": cv_enabled,
        "experiment_mode": mode,
        "smoke_test": bool(args.smoke_test),
        "training_epochs": epochs,
        "runtime_device_requested": args.device,
        "runtime_device_selected": device.type,
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / ML_CFG.output.filename_template.format(model_name=model_name)
    torch.save(checkpoint, checkpoint_path)
    test_payload = {"split": "test", "used_for_model_selection": False, **test_metrics}
    write_experiment_results(
        output_dir, model_name=model_name, split=split, folds=folds,
        cv_strategy=cv_strategy, search_strategy=search_strategy,
        sampled_configurations=configurations, cv_results=cv_results, best=best,
        final_test_metrics=test_payload,
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "best": best,
                      "final_test_metrics": test_payload}, indent=2))


if __name__ == "__main__":
    main()
