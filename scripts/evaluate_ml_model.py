from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import matplotlib
from torch.utils.data import Subset

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_ml_models_config, load_paths
from src.aero_coefficients_evaluation import build_aero_summary, generate_aero_plots
from src.aero_coefficients_from_fields import (
    FieldCoefficientSettings,
    ForceCoefficientConfig,
    derive_aero_coefficients_from_fields,
)
from src.ml_clcd_dataset import read_cl_cd_tail_average
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
    evaluate_symmetric_error_limit,
)
from src.ml_data_split import build_train_val_split
from src.ml_dataset import AirfoilFlowDataset, parse_case_params
from src.ml_supervised import (
    RAW_TARGETS,
    normalize_physical_targets,
    supervised_config_from_checkpoint,
    targets_to_physical,
)
from src.ml_supervised import near_wall_region_mask, wake_region_mask
from src.ml_models import build_model
from src.ml_device import log_device, resolve_device
from src.ml_protocols import PhysicsLossModel
from src.ml_training import compute_grid_spacing_from_xy, masked_mse


@torch.no_grad()
def run_validation(
    device_requested: str | None = None,
    experiment_name: str | None = None,
) -> None:
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    ml_config_path = PROJECT_ROOT / "configs" / "ml_models_config.yaml"

    paths = load_paths(config_path)
    ml_cfg = load_ml_models_config(ml_config_path)

    data_dir = paths.flow_fields_output
    model_name = ml_cfg.model.name
    model_filename = ml_cfg.output.filename_template.format(model_name=model_name)
    checkpoint_subdir = experiment_name or model_name
    model_path = paths.project_root / ml_cfg.output.models_subdir / checkpoint_subdir / model_filename
    legacy_model_path = paths.project_root / ml_cfg.output.models_subdir / model_filename
    if not model_path.exists() and legacy_model_path.exists():
        model_path = legacy_model_path

    validation_root = paths.project_root / ml_cfg.output.models_subdir / f"{checkpoint_subdir}_validation"
    metrics_dir = validation_root / "metrics"
    velocity_fig_dir = validation_root / "velocity_figures"
    field_fig_dir = validation_root / "field_figures"
    aero_fig_dir = validation_root / "aero_coefficient_figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    velocity_fig_dir.mkdir(parents=True, exist_ok=True)
    field_fig_dir.mkdir(parents=True, exist_ok=True)
    aero_fig_dir.mkdir(parents=True, exist_ok=True)

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
    dataset.supervised_config = supervised_config_from_checkpoint(checkpoint)
    target_representation = dataset.supervised_config.target_representation
    raw_metric_stats: dict[str, float]
    if checkpoint and checkpoint.get("test_case_ids"):
        case_ids = [path.parent.name for path in dataset.files]
        id_to_idx = {case_id: idx for idx, case_id in enumerate(case_ids)}
        missing = [case_id for case_id in checkpoint["test_case_ids"] if case_id not in id_to_idx]
        if missing:
            raise RuntimeError(f"Saved test cases are missing from dataset: {missing}")
        val_indices = [id_to_idx[case_id] for case_id in checkpoint["test_case_ids"]]
        development_indices = [
            id_to_idx[case_id] for case_id in checkpoint.get("development_case_ids", [])
        ]
        val_set = Subset(dataset, val_indices)
        train_size = len(checkpoint.get("development_case_ids", []))
        val_size = len(val_indices)
        for key in ("aoa_mean", "aoa_std", "inlet_u_mean", "inlet_u_std", "p_mean",
                    "p_std", "ux_mean", "ux_std", "uy_mean", "uy_std"):
            setattr(dataset, key, float(checkpoint[key]))
        if target_representation == RAW_TARGETS:
            raw_metric_stats = {
                key: float(checkpoint[key])
                for key in ("p_mean", "p_std", "ux_mean", "ux_std", "uy_mean", "uy_std")
            }
        else:
            raw_metric_stats = dataset.compute_raw_target_normalization(development_indices)
    else:
        split = build_train_val_split(dataset, ml_cfg.training.validation_split,
                                      ml_cfg.training.split_seed)
        val_set = split.val_set
        val_indices = split.val_indices
        train_size, val_size = split.train_size, split.val_size
        dataset.fit_condition_normalization(split.train_indices)
        dataset.fit_target_normalization(split.train_indices)
        raw_metric_stats = {
            key: float(getattr(dataset, key))
            for key in ("p_mean", "p_std", "ux_mean", "ux_std", "uy_mean", "uy_std")
        }

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
    plot_cases: list[dict[str, Any]] = []
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
            native_objective = float(masked_mse(pred_batch, target_batch, mask_batch).item())

            if target_representation == RAW_TARGETS:
                p_pred = pred_np[0] * dataset.p_std + dataset.p_mean
                ux_pred = pred_np[1] * dataset.ux_std + dataset.ux_mean
                uy_pred = pred_np[2] * dataset.uy_std + dataset.uy_mean
                p_target = target_np[0] * dataset.p_std + dataset.p_mean
                ux_target = target_np[1] * dataset.ux_std + dataset.ux_mean
                uy_target = target_np[2] * dataset.uy_std + dataset.uy_mean
            else:
                inlet_velocity = float(meta["inlet_velocity"])
                p_pred, ux_pred, uy_pred = targets_to_physical(
                    pred_np, inlet_velocity, target_representation,
                    dataset.supervised_config.p_inf,
                )
                p_target, ux_target, uy_target = targets_to_physical(
                    target_np, inlet_velocity, target_representation,
                    dataset.supervised_config.p_inf,
                )

            physical_pred_norm = normalize_physical_targets(
                p_pred, ux_pred, uy_pred, raw_metric_stats,
            )
            physical_true_norm = normalize_physical_targets(
                p_true, ux_true, uy_true, raw_metric_stats,
            )
            common_normalized_masked_mse = float(masked_mse(
                torch.from_numpy(physical_pred_norm).unsqueeze(0),
                torch.from_numpy(physical_true_norm).unsqueeze(0),
                mask_batch,
            ).item())

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
                normalized_masked_mse=common_normalized_masked_mse,
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
            case_row = evaluate_case_metrics_to_row(case_metrics)
            # Keep the legacy column as a backwards-compatible alias, but expose
            # unambiguous names for cross-representation comparisons.
            case_row["native_objective"] = native_objective
            case_row["common_normalized_masked_mse"] = common_normalized_masked_mse
            wall_region = near_wall_region_mask(
                xy, fluid_mask, float(meta["chord"]),
                dataset.supervised_config.wall_distance_fraction,
            )
            wake_region = wake_region_mask(
                xy, fluid_mask, float(meta["chord"]), float(meta["aoa_deg"]),
                dataset.supervised_config.wake_x_start, dataset.supervised_config.wake_x_end,
                dataset.supervised_config.wake_half_height,
            )
            outer_region = fluid_mask & ~wall_region & ~wake_region
            for region_name, region in (
                ("global", fluid_mask), ("near_wall", wall_region),
                ("wake", wake_region), ("outer", outer_region),
            ):
                if not np.any(region):
                    for quantity in ("p", "ux", "uy", "velocity_vector"):
                        case_row[f"{region_name}_{quantity}_rmse"] = float("nan")
                    continue
                case_row[f"{region_name}_p_rmse"] = float(np.sqrt(np.mean((p_pred[region] - p_true[region]) ** 2)))
                case_row[f"{region_name}_ux_rmse"] = float(np.sqrt(np.mean((ux_pred[region] - ux_true[region]) ** 2)))
                case_row[f"{region_name}_uy_rmse"] = float(np.sqrt(np.mean((uy_pred[region] - uy_true[region]) ** 2)))
                case_row[f"{region_name}_velocity_vector_rmse"] = float(
                    np.sqrt(np.mean(velocity_vector_error[region] ** 2))
                )
            rows.append(case_row)

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
                f"common normalized masked MSE={common_normalized_masked_mse:.6e} | "
                f"velocity vector RMSE={case_metrics.velocity_vector_rmse_mps:.6e} m/s | "
                "sampling locations are orientational"
            )

            plot_cases.append(
                {
                    "case_id": str(meta["case_id"]),
                    "xy": xy,
                    "fluid_mask": fluid_mask,
                    "speed_true": speed_true,
                    "speed_pred": speed_pred,
                    "velocity_vector_error": velocity_vector_error,
                    "p_true": p_true,
                    "p_pred": p_pred,
                    "ux_true": ux_true,
                    "ux_pred": ux_pred,
                    "uy_true": uy_true,
                    "uy_pred": uy_pred,
                    "case_label": case_label,
                    "subtitle": subtitle,
                    "annotations": annotations,
                    "meta": meta,
                }
            )

        except Exception as exc:
            failed_cases.append({"local_validation_index": str(local_idx), "error": str(exc)})
            print(f"[ERROR] Validation failed for local index {local_idx}: {exc}")

    if plot_cases:
        error_limits = {
            name: evaluate_symmetric_error_limit(
                np.concatenate(
                    [(case[f"{name}_pred"] - case[f"{name}_true"])[case["fluid_mask"]] for case in plot_cases]
                )
            )
            for name in ("p", "ux", "uy")
        }
        vector_error_limit = evaluate_symmetric_error_limit(
            np.concatenate([case["velocity_vector_error"][case["fluid_mask"]] for case in plot_cases])
        )

        for case in plot_cases:
            case_id = case["case_id"]
            evaluate_plot_velocity_comparison(
                output_path=velocity_fig_dir / f"{case_id}_velocity_comparison.png",
                xy=case["xy"],
                fluid_mask=case["fluid_mask"],
                speed_true=case["speed_true"],
                speed_pred=case["speed_pred"],
                velocity_vector_error=case["velocity_vector_error"],
                case_label=case["case_label"],
                subtitle=case["subtitle"],
                annotations=case["annotations"],
                vector_error_limit=vector_error_limit,
            )
            evaluate_plot_fields_comparison(
                output_path=field_fig_dir / f"{case_id}_fields_comparison.png",
                xy=case["xy"],
                fluid_mask=case["fluid_mask"],
                p_true=case["p_true"],
                p_pred=case["p_pred"],
                ux_true=case["ux_true"],
                ux_pred=case["ux_pred"],
                uy_true=case["uy_true"],
                uy_pred=case["uy_pred"],
                case_label=case["case_label"],
                error_limits=error_limits,
            )

    aero_rows: list[dict[str, Any]] = []
    is_fixed_test_evaluation = bool(checkpoint and checkpoint.get("test_case_ids"))
    aero_settings = FieldCoefficientSettings(nu=float(ml_cfg.physics_loss.nu))
    if is_fixed_test_evaluation:
        force_relpath = Path(ml_cfg.model.params.clcd_mlp.force_coeffs_relpath)
        local_case_root = paths.project_root / "run" / "airfoil_surrogate_cases" / "blockmesh_cases"
        solver_case_root = paths.openfoam_case_sim / "blockmesh_cases"
        for case in plot_cases:
            meta = case["meta"]
            params = meta.get("params", {})
            span = float(params.get("span", 0.1))
            chord = float(meta["chord"])
            force_config = ForceCoefficientConfig(
                reference_area=float(params.get("Aref", chord * span)),
                reference_length=float(params.get("lRef", chord)),
                span=span,
            )
            common = dict(
                xy=case["xy"], fluid_mask=case["fluid_mask"],
                naca_code=str(meta["naca_code"]), chord=chord,
                aoa_deg=float(meta["aoa_deg"]), inlet_velocity=float(meta["inlet_velocity"]),
                force_config=force_config, settings=aero_settings,
            )
            true_result = derive_aero_coefficients_from_fields(
                p=case["p_true"], U=np.stack((case["ux_true"], case["uy_true"]), axis=-1), **common
            )
            pred_result = derive_aero_coefficients_from_fields(
                p=case["p_pred"], U=np.stack((case["ux_pred"], case["uy_pred"]), axis=-1), **common
            )
            cl_openfoam = cd_openfoam = float("nan")
            reference_error = ""
            candidates = [
                paths.flow_fields_output / case["case_id"] / force_relpath,
                local_case_root / case["case_id"] / force_relpath,
                solver_case_root / case["case_id"] / force_relpath,
            ]
            reference_path = next((path for path in candidates if path.is_file()), None)
            if reference_path is not None:
                try:
                    cl_openfoam, cd_openfoam = read_cl_cd_tail_average(reference_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    reference_error = str(exc)
            else:
                reference_error = "forceCoeffs.dat not found"

            row = {
                "case_id": case["case_id"], "naca": str(meta["naca_code"]),
                "aoa_deg": float(meta["aoa_deg"]), "inlet_velocity": float(meta["inlet_velocity"]),
                "cl_openfoam": cl_openfoam, "cd_openfoam": cd_openfoam,
                "reference_path": str(reference_path) if reference_path else "", "reference_error": reference_error,
                "pressure_valid": bool(true_result.pressure_valid and pred_result.pressure_valid),
                "viscous_valid": bool(true_result.viscous_valid and pred_result.viscous_valid),
                "true_pressure_valid": true_result.pressure_valid, "pred_pressure_valid": pred_result.pressure_valid,
                "true_viscous_valid": true_result.viscous_valid, "pred_viscous_valid": pred_result.viscous_valid,
                "true_diagnostics": json.dumps(sanitize_for_json(true_result.diagnostics), sort_keys=True),
                "pred_diagnostics": json.dumps(sanitize_for_json(pred_result.diagnostics), sort_keys=True),
            }
            for prefix, result in (("true_grid", true_result), ("pred", pred_result)):
                for coefficient in ("cl", "cd"):
                    for contribution in ("pressure", "viscous", "total"):
                        row[f"{coefficient}_{prefix}_{contribution}"] = getattr(result, f"{coefficient}_{contribution}")
            for coefficient in ("cl", "cd"):
                of_value = row[f"{coefficient}_openfoam"]
                true_value = row[f"{coefficient}_true_grid_pressure"]
                pred_value = row[f"{coefficient}_pred_pressure"]
                row[f"{coefficient}_true_grid_vs_openfoam_abs_error"] = abs(true_value - of_value)
                row[f"{coefficient}_pred_vs_openfoam_abs_error"] = abs(pred_value - of_value)
                row[f"{coefficient}_pred_vs_true_grid_abs_error"] = abs(pred_value - true_value)
            aero_rows.append(row)

        aero_df = pd.DataFrame(aero_rows)
        aero_df.to_csv(metrics_dir / "aero_coefficients_test.csv", index=False)
        settings_payload = {
            **sanitize_for_json(aero_settings.__dict__),
            "force_coefficients": {
                "lift_dir": [0.0, 1.0], "drag_dir": [1.0, 0.0],
                "lRef": "case chord", "Aref": "case chord * span", "span": "case metadata",
                "pressure": "kinematic", "rho_cancels": True,
            },
            "regression_method": "least-squares linear fit through no-slip U_t(0)=0",
        }
        aero_summary = build_aero_summary(aero_df, settings_payload)
        (metrics_dir / "aero_coefficients_summary.json").write_text(
            json.dumps(sanitize_for_json(aero_summary), indent=2, allow_nan=False), encoding="utf-8"
        )
        generate_aero_plots(aero_df, aero_fig_dir)

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
            "experiment_name": checkpoint_subdir,
            "target_representation": target_representation,
            "metric_semantics": {
                "native_objective": "masked MSE in the checkpoint's training-target space",
                "common_normalized_masked_mse": (
                    "masked MSE after physical decoding and train-only raw-field z-scoring"
                ),
                "normalized_masked_mse": "backwards-compatible alias of common_normalized_masked_mse",
            },
            "common_raw_normalization": raw_metric_stats,
            "common_raw_normalization_population": "checkpoint development_case_ids only",
            "evaluation_split": "test" if checkpoint else "legacy_validation",
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "validation_case_count": int(len(metrics_df)),
            "validation_split": float(ml_cfg.training.validation_split),
            "split_seed": int(ml_cfg.training.split_seed),
            "device": str(device),
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
            "experiment_name": checkpoint_subdir,
            "target_representation": target_representation,
            "metric_semantics": {
                "native_objective": "masked MSE in the checkpoint's training-target space",
                "common_normalized_masked_mse": (
                    "masked MSE after physical decoding and train-only raw-field z-scoring"
                ),
                "normalized_masked_mse": "backwards-compatible alias of common_normalized_masked_mse",
            },
            "common_raw_normalization": raw_metric_stats,
            "common_raw_normalization_population": "training split only",
            "evaluation_split": "test" if checkpoint else "legacy_validation",
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "validation_case_count": 0,
            "validation_split": float(ml_cfg.training.validation_split),
            "split_seed": int(ml_cfg.training.split_seed),
            "device": str(device),
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
        print(
            "Common normalized masked MSE: "
            f"mean={metrics_df['common_normalized_masked_mse'].mean():.6e}"
        )
        print(f"Native objective: mean={metrics_df['native_objective'].mean():.6e}")
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
    parser.add_argument(
        "--experiment-name", default=None,
        help="Checkpoint/output subdirectory; defaults to the model name for legacy baselines.",
    )
    args = parser.parse_args()
    run_validation(args.device, args.experiment_name)


if __name__ == "__main__":
    main()
