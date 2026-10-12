"""Train the isolated high-resolution local wake SimpleUNet experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_ml_model import normalization_metadata, seed_everything, train_field
from src.ml_data_split import build_cv_folds, load_or_create_fixed_test_split
from src.ml_dataset import AirfoilFlowDataset
from src.ml_models import build_model
from src.ml_supervised import SupervisedLossConfig, near_wall_region_mask, wake_region_mask


DATA_ROOT = ROOT / "data" / "flow_fields_highres_wake"
MODEL_ROOT = ROOT / "data" / "models" / "simple_unet_local_wake_v1"
SPLIT_PATH = ROOT / "data" / "models" / "simple_unet" / "fixed_test_split.json"
CHECKPOINT_PATH = MODEL_ROOT / "simple_unet_airfoil.pt"
EXPECTED_SPLIT_SHA256 = "a4b523c2c8effc02536bf958f507efa6e83faecd87aab659b492c3abaa92b12a"
EXPECTED_BOUNDS = np.array([0.70, 1.75, -0.15, 0.15])
EXPECTED_SHAPE = (480, 272)
HP = {
    "learning_rate": 0.001, "batch_size": 2, "weight_decay": 0.0,
    "in_channels": 5, "out_channels": 3,
    "encoder_channels": [32, 64, 128], "bottleneck_channels": 256,
}
LOSS_CONFIG = SupervisedLossConfig(
    target_representation="nondimensional_perturbation",
    wall_weight=0.0, wake_weight=0.0,
    wall_distance_fraction=0.05, wake_x_start=0.9, wake_x_end=1.75,
    wake_half_height=0.20, gradient_loss_weight=0.1,
    gradient_channel_weights=(1.0, 1.0, 1.0), p_inf=0.0,
)


def validate_dataset(dataset: AirfoilFlowDataset) -> tuple[object, float, float]:
    split_sha = hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Fixed split changed: {split_sha}")
    case_ids = [path.parent.name for path in dataset.files]
    split = load_or_create_fixed_test_split(case_ids, SPLIT_PATH, 0.2, 42)
    if len(split.development_indices) != 465 or len(split.test_indices) != 116:
        raise RuntimeError("Expected exact development/test counts 465/116")
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != set(split.development_case_ids + split.test_case_ids):
        raise RuntimeError("Local dataset membership/uniqueness differs from fixed split")
    expected_dx = (EXPECTED_BOUNDS[1]-EXPECTED_BOUNDS[0])/(EXPECTED_SHAPE[1]-1)
    expected_dy = (EXPECTED_BOUNDS[3]-EXPECTED_BOUNDS[2])/(EXPECTED_SHAPE[0]-1)
    for path in dataset.files:
        with np.load(path, allow_pickle=False) as z:
            if z["p"].shape != EXPECTED_SHAPE or not np.array_equal(z["patch_bounds_c"], EXPECTED_BOUNDS):
                raise RuntimeError(f"Invalid patch geometry: {path}")
            if not np.isclose(float(z["dx_c"]), expected_dx, rtol=0, atol=1e-12) or not np.isclose(float(z["dy_c"]), expected_dy, rtol=0, atol=1e-12):
                raise RuntimeError(f"Invalid patch spacing: {path}")
    return split, expected_dx, expected_dy


def region_statistics(dataset: AirfoilFlowDataset, indices: list[int]) -> tuple[list[dict[str, float | str]], dict[str, object]]:
    rows: list[dict[str, float | str]] = []
    for index in indices:
        path = dataset.files[index]
        with np.load(path, allow_pickle=False) as z:
            xy, fluid = z["xy"], z["fluid_mask"] > 0.5
            chord = float(z["chord"])
        wall = near_wall_region_mask(xy, fluid, chord, LOSS_CONFIG.wall_distance_fraction)
        wake = wake_region_mask(xy, fluid, chord, 0.0, LOSS_CONFIG.wake_x_start, LOSS_CONFIG.wake_x_end, LOSS_CONFIG.wake_half_height)
        remaining = fluid & ~wall & ~wake
        total = fluid.size
        rows.append({
            "case_id": path.parent.name, "fluid": np.count_nonzero(fluid)/total,
            "solid": np.count_nonzero(~fluid)/total, "near_wall": np.count_nonzero(wall)/total,
            "wake": np.count_nonzero(wake)/total, "remaining_local_fluid": np.count_nonzero(remaining)/total,
            "wall_wake_overlap": np.count_nonzero(wall & wake)/total,
        })
    summary: dict[str, object] = {"population": "465 development cases only", "case_count": len(rows), "fractions_of_all_patch_pixels": {}}
    for key in ("fluid", "solid", "near_wall", "wake", "remaining_local_fluid", "wall_wake_overlap"):
        values = np.array([float(row[key]) for row in rows])
        summary["fractions_of_all_patch_pixels"][key] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()),
        }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if MODEL_ROOT.exists():
        raise FileExistsError(f"Refusing experiment collision: {MODEL_ROOT}")
    if args.epochs != 30 and not args.preflight_only:
        raise ValueError("Final v1 protocol requires exactly 30 epochs")
    dataset = AirfoilFlowDataset(DATA_ROOT, supervised_config=LOSS_CONFIG, return_regions=True)
    split, dx, dy = validate_dataset(dataset)
    case_ids = [path.parent.name for path in dataset.files]
    folds = build_cv_folds(split.development_indices, case_ids, 5, 42, "kfold")
    if any((len(f.train_indices), len(f.val_indices)) != (372, 93) for f in folds):
        raise RuntimeError("Expected every CV fold to contain 372/93 cases")
    model_config = {key: HP[key] for key in ("in_channels", "out_channels", "encoder_channels", "bottleneck_channels")}
    probe_model = build_model("simple_unet", **model_config)
    parameter_count = sum(p.numel() for p in probe_model.parameters())
    print("[PREFLIGHT]")
    for key, value in {
        "experiment_name": MODEL_ROOT.name, "patch_shape": EXPECTED_SHAPE,
        "patch_bounds_c": EXPECTED_BOUNDS.tolist(), "dx_c": dx, "dy_c": dy,
        "target_representation": LOSS_CONFIG.target_representation,
        "coordinate_normalization": "per-grid mean/std (identical fixed local grid for every case)",
        "condition_normalization": "fold-train only in CV; all 465 development cases for final fit",
        "model_parameters": parameter_count, "development_count": 465, "test_count": 116,
        "cv_fold_sizes": "372 train / 93 validation", "batch_size": 2,
        "learning_rate": 0.001, "weight_decay": 0.0, "epochs": args.epochs,
        "gradient_loss_weight": 0.1, "wall_weight": 0.0, "wake_weight": 0.0,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "fixed_split_sha256": hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
    }.items():
        print(f"{key}={value}")
    if args.preflight_only:
        return

    MODEL_ROOT.mkdir(parents=True)
    started = time.perf_counter()
    region_rows, region_summary = region_statistics(dataset, split.development_indices)
    with (MODEL_ROOT / "local_region_fractions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(region_rows[0])); writer.writeheader(); writer.writerows(region_rows)
    (MODEL_ROOT / "local_region_fraction_summary.json").write_text(json.dumps(region_summary, indent=2), encoding="utf-8")
    print("[REGION FRACTIONS]")
    print(json.dumps(region_summary, indent=2))

    # One real forward/backward batch catches CUDA-memory incompatibility before CV.
    seed_everything(42)
    dataset.fit_condition_normalization(folds[0].train_indices)
    inputs, targets, mask, regions = dataset[folds[0].train_indices[0]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_model = probe_model.to(device)
    from src.ml_training import supervised_loss_components
    losses = supervised_loss_components(probe_model(inputs[None].to(device)), targets[None].to(device),
                                        mask[None].to(device), regions[None].to(device), LOSS_CONFIG, dx, dy)
    losses["total_loss"].backward()
    print(f"[MEMORY SMOKE] device={device.type} total_loss={losses['total_loss'].item():.6g}")
    del probe_model, inputs, targets, mask, regions, losses
    if device.type == "cuda": torch.cuda.empty_cache()

    cv_results = []
    for fold in folds:
        print(f"[CV] fold={fold.fold_id} train=372 validation=93")
        fold_started = time.perf_counter()
        model, metrics = train_field(dataset, "simple_unet", fold.train_indices, fold.val_indices,
                                     HP, device, 42 + fold.fold_id, dx, dy, args.epochs, LOSS_CONFIG)
        record = {"fold_id": fold.fold_id, **metrics, "runtime_seconds": time.perf_counter()-fold_started}
        cv_results.append(record)
        (MODEL_ROOT / "cv_progress.json").write_text(json.dumps(cv_results, indent=2), encoding="utf-8")
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    print("[FINAL] train=465 test=116")
    final_started = time.perf_counter()
    final_model, test_metrics = train_field(dataset, "simple_unet", split.development_indices,
                                            split.test_indices, HP, device, 10042, dx, dy,
                                            args.epochs, LOSS_CONFIG)
    final_runtime = time.perf_counter()-final_started
    checkpoint = {
        "model_name": "simple_unet", "experiment_name": MODEL_ROOT.name,
        "model_state_dict": final_model.state_dict(), "model_config": model_config,
        **normalization_metadata(dataset, "simple_unet"),
        "dataset_size": len(dataset), "dataset_root": str(DATA_ROOT),
        "development_case_ids": split.development_case_ids, "test_case_ids": split.test_case_ids,
        "dataset_fingerprint": split.dataset_fingerprint, "fixed_split_sha256": EXPECTED_SPLIT_SHA256,
        "split_seed": 42, "n_folds": 5, "cv_strategy": "kfold", "training_epochs": 30,
        "best_hyperparameters": HP, "target_representation": LOSS_CONFIG.target_representation,
        "supervised_loss_config": LOSS_CONFIG.metadata(), "coordinate_frame": "body",
        "coordinate_normalization": "per-grid mean/std; fixed local geometry",
        "patch_shape_ny_nx": list(EXPECTED_SHAPE), "patch_bounds_c": EXPECTED_BOUNDS.tolist(),
        "dx_c": dx, "dy_c": dy, "parameter_count": parameter_count,
        "runtime_device_selected": device.type, "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    cv_payload = {
        "model_name": "simple_unet", "experiment_name": MODEL_ROOT.name,
        "dataset_size": len(dataset), "seed": 42, "cv_strategy": "kfold", "n_folds": 5,
        "hyperparameter_search_strategy": "none", "hyperparameters": HP,
        "fold_metrics": cv_results,
        "mean_cv_objective": float(np.mean([r["objective"] for r in cv_results])),
        "std_cv_objective": float(np.std([r["objective"] for r in cv_results])),
    }
    (MODEL_ROOT / "cv_results.json").write_text(json.dumps(cv_payload, indent=2), encoding="utf-8")
    (MODEL_ROOT / "final_test_metrics.json").write_text(json.dumps({"split": "test", "used_for_model_selection": False,
        **test_metrics, "runtime_seconds": final_runtime}, indent=2), encoding="utf-8")
    (MODEL_ROOT / "split_metadata.json").write_text(json.dumps({
        "development_case_ids": split.development_case_ids, "test_case_ids": split.test_case_ids,
        "dataset_fingerprint": split.dataset_fingerprint, "fixed_split_sha256": EXPECTED_SPLIT_SHA256,
        "folds": [asdict(fold) for fold in folds],
    }, indent=2), encoding="utf-8")
    runtime = {"total_seconds": time.perf_counter()-started, "final_training_seconds": final_runtime,
               "cv_fold_seconds": [r["runtime_seconds"] for r in cv_results]}
    (MODEL_ROOT / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(CHECKPOINT_PATH), "cv": cv_payload, "test": test_metrics,
                      "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
