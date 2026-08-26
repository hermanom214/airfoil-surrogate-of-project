from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_ml_models_config, load_paths
from src.evaluate_io import (
    evaluate_extract_case_meta,
    evaluate_get_model_kwargs,
    sanitize_for_json,
)
from src.evaluate_metrics import (
    evaluate_build_case_metrics,
    evaluate_case_metrics_to_row,
    evaluate_finalize_global_aggregator,
    evaluate_init_global_aggregator,
    evaluate_summarize_numeric_columns,
    evaluate_update_global_aggregator,
)
from src.evaluate_plotting import (
    evaluate_build_velocity_annotations,
    evaluate_generate_dataset_level_plot,
    evaluate_plot_fields_comparison,
    evaluate_plot_velocity_comparison,
)
from src.ml_data_split import build_train_val_split
from src.ml_dataset import AirfoilFlowDataset, parse_case_params
from src.ml_models import build_model
from src.ml_device import log_device, resolve_device
from src.ml_protocols import PhysicsLossModel
from src.ml_training import compute_grid_spacing_from_xy, masked_mse


@torch.no_grad()
def run_validation(device_requested: str | None = None) -> None:
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    ml_config_path = PROJECT_ROOT / "configs" / "ml_models_config.yaml"

    paths = load_paths(config_path)
    ml_cfg = load_ml_models_config(ml_config_path)

    data_dir = paths.flow_fields_output
    model_name = ml_cfg.model.name
    model_filename = ml_cfg.output.filename_template.format(model_name=model_name)
    model_path = paths.project_root / ml_cfg.output.models_subdir / model_name / model_filename
    legacy_model_path = paths.project_root / ml_cfg.output.models_subdir / model_filename
    if not model_path.exists() and legacy_model_path.exists():
        model_path = legacy_model_path

    validation_root = paths.project_root / ml_cfg.output.models_subdir / f"{model_name}_validation"
    metrics_dir = validation_root / "metrics"
    velocity_fig_dir = validation_root / "velocity_figures"
    field_fig_dir = validation_root / "field_figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    velocity_fig_dir.mkdir(parents=True, exist_ok=True)
    field_fig_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    dataset = AirfoilFlowDataset(data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty in: {data_dir}")

    try:
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(model_path, map_location="cpu")
    checkpoint = loaded if isinstance(loaded, dict) and "model_state_dict" in loaded else None
    if checkpoint and checkpoint.get("test_case_ids"):
        case_ids = [path.parent.name for path in dataset.files]
        id_to_idx = {case_id: idx for idx, case_id in enumerate(case_ids)}
        missing = [case_id for case_id in checkpoint["test_case_ids"] if case_id not in id_to_idx]
        if missing:
            raise RuntimeError(f"Saved test cases are missing from dataset: {missing}")
        val_indices = [id_to_idx[case_id] for case_id in checkpoint["test_case_ids"]]
        val_set = Subset(dataset, val_indices)
        train_size = len(checkpoint.get("development_case_ids", []))
        val_size = len(val_indices)
        for key in ("aoa_mean", "aoa_std", "inlet_u_mean", "inlet_u_std", "p_mean",
                    "p_std", "ux_mean", "ux_std", "uy_mean", "uy_std"):
            setattr(dataset, key, float(checkpoint[key]))
    else:
        split = build_train_val_split(dataset, ml_cfg.training.validation_split,
                                      ml_cfg.training.split_seed)
        val_set = split.val_set
        val_indices = split.val_indices
        train_size, val_size = split.train_size, split.val_size
        dataset.fit_condition_normalization(split.train_indices)
        dataset.fit_target_normalization(split.train_indices)

    if len(val_set) == 0:
        raise RuntimeError("Validation set is empty after split.")

    requested = device_requested or ml_cfg.device
    device = resolve_device(requested)
    log_device(requested, device)
    model_kwargs = (checkpoint or {}).get("model_config") or evaluate_get_model_kwargs(model_name, ml_cfg)
    model = build_model(model_name, **model_kwargs).to(device)

    try:
        state_dict = (checkpoint or torch.load(model_path, map_location=device, weights_only=True))
    except TypeError:
        state_dict = torch.load(
            model_path,
            map_location=device,
        )
    model.load_state_dict(state_dict["model_state_dict"] if checkpoint else state_dict)

    model.eval()
    physics_available = model_name == "rans_pinn" and hasattr(model, "rans_residual_loss")

    if model_name == "rans_pinn":
        print(
            "[WARN] Existing rans_pinn weights trained before the output-channel-order fix "
            "are not compatible semantically and should be retrained."
        )

    csv_path = metrics_dir / "validation_metrics_per_case.csv"
    summary_path = metrics_dir / "validation_summary.json"
    dataset_plot_path = metrics_dir / "validation_case_errors.png"

    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []
    warned_mask_metadata_missing = False

    global_agg = evaluate_init_global_aggregator()

    for local_idx in range(len(val_set)):
        try:
            original_idx = val_indices[local_idx]
            npz_path = dataset.files[original_idx]

            inp, target_norm, _ = dataset[original_idx]
            inp_batch = inp.unsqueeze(0).to(device)
            pred_norm = model(inp_batch)[0].cpu()

            pred_batch = pred_norm.unsqueeze(0)
            target_batch = target_norm.unsqueeze(0)

            pred_np = pred_norm.numpy()
            target_np = target_norm.numpy()

            with np.load(npz_path, allow_pickle=False) as npz:
                xy = npz["xy"].astype(np.float32)
                p_true = npz["p"].astype(np.float32)
                u_true = npz["U"].astype(np.float32)
                fluid_mask_npz = npz["fluid_mask"].astype(np.float32) > 0.5
                fluid_mask = fluid_mask_npz
                ux_true = u_true[:, :, 0]
                uy_true = u_true[:, :, 1]

                if "mask_rotation_applied" in npz.files:
                    if not bool(np.array(npz["mask_rotation_applied"]).item()):
                        print(f"[WARN] mask_rotation_applied is False in {npz_path.name}.")
                elif not warned_mask_metadata_missing:
                    print("[WARN] Mask rotation metadata not found; mask is used as stored in NPZ.")
                    warned_mask_metadata_missing = True

                parsed = parse_case_params(npz_path.name)
                meta = evaluate_extract_case_meta(npz, npz_path, parsed)

            mask_batch = torch.from_numpy(fluid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            normalized_loss = float(masked_mse(pred_batch, target_batch, mask_batch).item())

            p_pred = pred_np[0] * dataset.p_std + dataset.p_mean
            ux_pred = pred_np[1] * dataset.ux_std + dataset.ux_mean
            uy_pred = pred_np[2] * dataset.uy_std + dataset.uy_mean

            p_target = target_np[0] * dataset.p_std + dataset.p_mean
            ux_target = target_np[1] * dataset.ux_std + dataset.ux_mean
            uy_target = target_np[2] * dataset.uy_std + dataset.uy_mean

            dx, dy = compute_grid_spacing_from_xy(torch.from_numpy(xy.astype(np.float32)))

            if not (
                np.allclose(p_target[fluid_mask], p_true[fluid_mask], rtol=1e-4, atol=1e-4)
                and np.allclose(ux_target[fluid_mask], ux_true[fluid_mask], rtol=1e-4, atol=1e-4)
                and np.allclose(uy_target[fluid_mask], uy_true[fluid_mask], rtol=1e-4, atol=1e-4)
            ):
                print(
                    "[WARN] Denormalized target does not fully match NPZ values in fluid cells "
                    f"for {npz_path.name}."
                )

            speed_true = np.sqrt(ux_true ** 2 + uy_true ** 2)
            speed_pred = np.sqrt(ux_pred ** 2 + uy_pred ** 2)
            velocity_vector_error = np.sqrt((ux_pred - ux_true) ** 2 + (uy_pred - uy_true) ** 2)

            if physics_available:
                physics_model = cast(PhysicsLossModel, model)
                phys = physics_model.rans_residual_loss(
                    pred=pred_batch.to(device),
                    fluid_mask=mask_batch.to(device),
                    dx=dx,
                    dy=dy,
                    nu=ml_cfg.physics_loss.nu,
                    p_mean=dataset.p_mean,
                    p_std=dataset.p_std,
                    ux_mean=dataset.ux_mean,
                    ux_std=dataset.ux_std,
                    uy_mean=dataset.uy_mean,
                    uy_std=dataset.uy_std,
                    pressure_is_kinematic=ml_cfg.physics_loss.pressure_is_kinematic,
                    mask_erode_pixels=ml_cfg.physics_loss.mask_erode_pixels,
                )
                physics_loss = float(phys["physics_loss"].item())
                continuity_loss = float(phys["continuity_loss"].item())
                momentum_x_loss = float(phys["momentum_x_loss"].item())
                momentum_y_loss = float(phys["momentum_y_loss"].item())
            else:
                physics_loss = float("nan")
                continuity_loss = float("nan")
                momentum_x_loss = float("nan")
                momentum_y_loss = float("nan")

            case_metrics = evaluate_build_case_metrics(
                case_id=str(meta["case_id"]),
                filename=npz_path.name,
                naca_code=str(meta["naca_code"]),
                chord=float(meta["chord"]),
                aoa_deg=float(meta["aoa_deg"]),
                inlet_velocity_mps=float(meta["inlet_velocity"]),
                latest_time=str(meta["latest_time"]),
                normalized_masked_mse=normalized_loss,
                p_pred=p_pred,
                p_true=p_true,
                ux_pred=ux_pred,
                ux_true=ux_true,
                uy_pred=uy_pred,
                uy_true=uy_true,
                speed_pred=speed_pred,
                speed_true=speed_true,
                velocity_vector_error=velocity_vector_error,
                fluid_mask=fluid_mask,
                physics_loss=physics_loss,
                continuity_loss=continuity_loss,
                momentum_x_loss=momentum_x_loss,
                momentum_y_loss=momentum_y_loss,
            )
            rows.append(evaluate_case_metrics_to_row(case_metrics))

            evaluate_update_global_aggregator(
                agg=global_agg,
                p_pred=p_pred,
                p_true=p_true,
                ux_pred=ux_pred,
                ux_true=ux_true,
                uy_pred=uy_pred,
                uy_true=uy_true,
                speed_pred=speed_pred,
                speed_true=speed_true,
                velocity_vector_error=velocity_vector_error,
                fluid_mask=fluid_mask,
            )

            annotations = evaluate_build_velocity_annotations(
                xy=xy,
                fluid_mask=fluid_mask,
                velocity_vector_error=velocity_vector_error,
                chord=float(meta["chord"]),
                aoa_deg=float(meta["aoa_deg"]),
            )

            case_label = (
                f"{meta['case_id']} | naca={meta['naca_code']} | chord={float(meta['chord']):.3f} m | "
                f"AoA={float(meta['aoa_deg']):.3f} deg | U_in={float(meta['inlet_velocity']):.3f} m/s"
            )
            subtitle = (
                f"normalized masked MSE={normalized_loss:.6e} | "
                f"velocity vector RMSE={case_metrics.velocity_vector_rmse_mps:.6e} m/s | "
                "sampling locations are orientational"
            )

            evaluate_plot_velocity_comparison(
                output_path=velocity_fig_dir / f"{meta['case_id']}_velocity_comparison.png",
                xy=xy,
                fluid_mask=fluid_mask,
                speed_true=speed_true,
                speed_pred=speed_pred,
                velocity_vector_error=velocity_vector_error,
                case_label=case_label,
                subtitle=subtitle,
                annotations=annotations,
            )

            evaluate_plot_fields_comparison(
                output_path=field_fig_dir / f"{meta['case_id']}_fields_comparison.png",
                xy=xy,
                fluid_mask=fluid_mask,
                p_true=p_true,
                p_pred=p_pred,
                ux_true=ux_true,
                ux_pred=ux_pred,
                uy_true=uy_true,
                uy_pred=uy_pred,
                case_label=case_label,
            )

        except Exception as exc:
            failed_cases.append({"local_validation_index": str(local_idx), "error": str(exc)})
            print(f"[ERROR] Validation failed for local index {local_idx}: {exc}")

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by="case_id").reset_index(drop=True)
    metrics_df.to_csv(csv_path, index=False)

    if not metrics_df.empty:
        numeric_cols = [
            c for c in metrics_df.columns if c not in ["case_id", "filename", "naca_code", "latest_time"]
        ]
        per_metric_stats = evaluate_summarize_numeric_columns(metrics_df, numeric_cols)

        best_row = metrics_df.loc[metrics_df["velocity_vector_rmse_mps"].idxmin()].to_dict()
        worst_row = metrics_df.loc[metrics_df["velocity_vector_rmse_mps"].idxmax()].to_dict()

        summary = {
            "model_name": model_name,
            "evaluation_split": "test" if checkpoint else "legacy_validation",
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "validation_case_count": int(len(metrics_df)),
            "validation_split": float(ml_cfg.training.validation_split),
            "split_seed": int(ml_cfg.training.split_seed),
            "device": device,
            "normalization_statistics": {
                "aoa_mean": float(dataset.aoa_mean),
                "aoa_std": float(dataset.aoa_std),
                "inlet_u_mean": float(dataset.inlet_u_mean),
                "inlet_u_std": float(dataset.inlet_u_std),
                "p_mean": float(dataset.p_mean),
                "p_std": float(dataset.p_std),
                "ux_mean": float(dataset.ux_mean),
                "ux_std": float(dataset.ux_std),
                "uy_mean": float(dataset.uy_mean),
                "uy_std": float(dataset.uy_std),
            },
            "best_case_by_velocity_vector_rmse": best_row,
            "worst_case_by_velocity_vector_rmse": worst_row,
            "per_case_metric_distribution": per_metric_stats,
            "global_pixel_weighted_metrics": evaluate_finalize_global_aggregator(global_agg),
            "failed_cases": failed_cases,
            "train_size": int(train_size),
            "val_size": int(val_size),
        }

        if physics_available:
            physics_cols = [
                "physics_loss",
                "continuity_loss",
                "momentum_x_loss",
                "momentum_y_loss",
            ]
            physics_stats = evaluate_summarize_numeric_columns(metrics_df, physics_cols)
            summary["physics_metrics"] = physics_stats
    else:
        summary = {
            "model_name": model_name,
            "evaluation_split": "test" if checkpoint else "legacy_validation",
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "validation_case_count": 0,
            "validation_split": float(ml_cfg.training.validation_split),
            "split_seed": int(ml_cfg.training.split_seed),
            "device": device,
            "normalization_statistics": {
                "aoa_mean": float(dataset.aoa_mean),
                "aoa_std": float(dataset.aoa_std),
                "inlet_u_mean": float(dataset.inlet_u_mean),
                "inlet_u_std": float(dataset.inlet_u_std),
                "p_mean": float(dataset.p_mean),
                "p_std": float(dataset.p_std),
                "ux_mean": float(dataset.ux_mean),
                "ux_std": float(dataset.ux_std),
                "uy_mean": float(dataset.uy_mean),
                "uy_std": float(dataset.uy_std),
            },
            "best_case_by_velocity_vector_rmse": None,
            "worst_case_by_velocity_vector_rmse": None,
            "per_case_metric_distribution": {},
            "global_pixel_weighted_metrics": {},
            "failed_cases": failed_cases,
            "train_size": int(train_size),
            "val_size": int(val_size),
        }

    sanitized_summary = sanitize_for_json(summary)
    summary_path.write_text(
        json.dumps(sanitized_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if not metrics_df.empty:
        evaluate_generate_dataset_level_plot(metrics_df, dataset_plot_path)

        print("[VALIDATION SUMMARY]")
        print(f"Model: {model_name}")
        print(f"Cases: {len(metrics_df)}")
        print(f"Normalized masked MSE: mean={metrics_df['normalized_masked_mse'].mean():.6e}")
        print(f"Pressure RMSE: mean={metrics_df['p_rmse'].mean():.6e}")
        print(f"Ux RMSE: mean={metrics_df['ux_rmse_mps'].mean():.6e}")
        print(f"Uy RMSE: mean={metrics_df['uy_rmse_mps'].mean():.6e}")
        print(f"Speed RMSE: mean={metrics_df['speed_rmse_mps'].mean():.6e}")
        print(f"Velocity vector RMSE: mean={metrics_df['velocity_vector_rmse_mps'].mean():.6e}")
        best_case = metrics_df.loc[metrics_df['velocity_vector_rmse_mps'].idxmin()]
        worst_case = metrics_df.loc[metrics_df['velocity_vector_rmse_mps'].idxmax()]
        print(
            "Best case: "
            f"{best_case['case_id']} (velocity_vector_rmse_mps={best_case['velocity_vector_rmse_mps']:.6e})"
        )
        print(
            "Worst case: "
            f"{worst_case['case_id']} (velocity_vector_rmse_mps={worst_case['velocity_vector_rmse_mps']:.6e})"
        )

        if physics_available:
            print("[PHYSICS VALIDATION]")
            print(f"Physics residual: mean={metrics_df['physics_loss'].dropna().mean():.6e}")
            print(f"Continuity residual: mean={metrics_df['continuity_loss'].dropna().mean():.6e}")
            print(f"Momentum X residual: mean={metrics_df['momentum_x_loss'].dropna().mean():.6e}")
            print(f"Momentum Y residual: mean={metrics_df['momentum_y_loss'].dropna().mean():.6e}")
    else:
        print("[VALIDATION SUMMARY]")
        print(f"Model: {model_name}")
        print("Cases: 0")
        print("Normalized masked MSE: n/a")
        print("Pressure RMSE: n/a")
        print("Ux RMSE: n/a")
        print("Uy RMSE: n/a")
        print("Speed RMSE: n/a")
        print("Velocity vector RMSE: n/a")
        print("Best case: n/a")
        print("Worst case: n/a")

    print(f"Model path: {model_path}")
    print(f"CSV path: {csv_path}")
    print(f"Summary JSON path: {summary_path}")
    print(f"Velocity figures directory: {velocity_fig_dir}")
    print(f"Field figures directory: {field_fig_dir}")
    print(f"Successfully processed validation cases: {len(rows)}")
    print(f"Failed cases: {len(failed_cases)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained flow-field model.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = parser.parse_args()
    run_validation(args.device)


if __name__ == "__main__":
    main()
