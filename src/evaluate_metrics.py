from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CaseMetrics:
    case_id: str
    filename: str
    naca_code: str
    chord: float
    aoa_deg: float
    inlet_velocity_mps: float
    latest_time: str
    normalized_masked_mse: float
    p_mae: float
    p_rmse: float
    p_max_abs_error: float
    p_bias: float
    ux_mae_mps: float
    ux_rmse_mps: float
    ux_max_abs_error_mps: float
    ux_bias_mps: float
    uy_mae_mps: float
    uy_rmse_mps: float
    uy_max_abs_error_mps: float
    uy_bias_mps: float
    speed_mae_mps: float
    speed_rmse_mps: float
    speed_max_abs_error_mps: float
    speed_bias_mps: float
    velocity_vector_mean_error_mps: float
    velocity_vector_rmse_mps: float
    velocity_vector_max_error_mps: float
    velocity_vector_p95_error_mps: float
    velocity_vector_p99_error_mps: float
    physics_loss: float
    continuity_loss: float
    momentum_x_loss: float
    momentum_y_loss: float
    fluid_cell_count: int
    total_cell_count: int


def _evaluate_select_values(values: np.ndarray, fluid_mask: np.ndarray) -> np.ndarray:
    vals = values[fluid_mask]
    vals = vals[np.isfinite(vals)]
    return vals


def evaluate_masked_mae(pred: np.ndarray, true: np.ndarray, fluid_mask: np.ndarray) -> float:
    vals = _evaluate_select_values(np.abs(pred - true), fluid_mask)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def evaluate_masked_rmse(pred: np.ndarray, true: np.ndarray, fluid_mask: np.ndarray) -> float:
    vals = _evaluate_select_values((pred - true) ** 2, fluid_mask)
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals)))


def evaluate_masked_max_abs_error(pred: np.ndarray, true: np.ndarray, fluid_mask: np.ndarray) -> float:
    vals = _evaluate_select_values(np.abs(pred - true), fluid_mask)
    if vals.size == 0:
        return float("nan")
    return float(np.max(vals))


def evaluate_masked_bias(pred: np.ndarray, true: np.ndarray, fluid_mask: np.ndarray) -> float:
    vals = _evaluate_select_values(pred - true, fluid_mask)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def evaluate_masked_percentile_abs_error(
    pred: np.ndarray,
    true: np.ndarray,
    fluid_mask: np.ndarray,
    percentile: float,
) -> float:
    vals = _evaluate_select_values(np.abs(pred - true), fluid_mask)
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, percentile))


def evaluate_build_case_metrics(
    case_id: str,
    filename: str,
    naca_code: str,
    chord: float,
    aoa_deg: float,
    inlet_velocity_mps: float,
    latest_time: str,
    normalized_masked_mse: float,
    p_pred: np.ndarray,
    p_true: np.ndarray,
    ux_pred: np.ndarray,
    ux_true: np.ndarray,
    uy_pred: np.ndarray,
    uy_true: np.ndarray,
    speed_pred: np.ndarray,
    speed_true: np.ndarray,
    velocity_vector_error: np.ndarray,
    fluid_mask: np.ndarray,
    physics_loss: float,
    continuity_loss: float,
    momentum_x_loss: float,
    momentum_y_loss: float,
) -> CaseMetrics:
    zeros = np.zeros_like(velocity_vector_error)

    return CaseMetrics(
        case_id=case_id,
        filename=filename,
        naca_code=naca_code,
        chord=chord,
        aoa_deg=aoa_deg,
        inlet_velocity_mps=inlet_velocity_mps,
        latest_time=latest_time,
        normalized_masked_mse=normalized_masked_mse,
        p_mae=evaluate_masked_mae(p_pred, p_true, fluid_mask),
        p_rmse=evaluate_masked_rmse(p_pred, p_true, fluid_mask),
        p_max_abs_error=evaluate_masked_max_abs_error(p_pred, p_true, fluid_mask),
        p_bias=evaluate_masked_bias(p_pred, p_true, fluid_mask),
        ux_mae_mps=evaluate_masked_mae(ux_pred, ux_true, fluid_mask),
        ux_rmse_mps=evaluate_masked_rmse(ux_pred, ux_true, fluid_mask),
        ux_max_abs_error_mps=evaluate_masked_max_abs_error(ux_pred, ux_true, fluid_mask),
        ux_bias_mps=evaluate_masked_bias(ux_pred, ux_true, fluid_mask),
        uy_mae_mps=evaluate_masked_mae(uy_pred, uy_true, fluid_mask),
        uy_rmse_mps=evaluate_masked_rmse(uy_pred, uy_true, fluid_mask),
        uy_max_abs_error_mps=evaluate_masked_max_abs_error(uy_pred, uy_true, fluid_mask),
        uy_bias_mps=evaluate_masked_bias(uy_pred, uy_true, fluid_mask),
        speed_mae_mps=evaluate_masked_mae(speed_pred, speed_true, fluid_mask),
        speed_rmse_mps=evaluate_masked_rmse(speed_pred, speed_true, fluid_mask),
        speed_max_abs_error_mps=evaluate_masked_max_abs_error(speed_pred, speed_true, fluid_mask),
        speed_bias_mps=evaluate_masked_bias(speed_pred, speed_true, fluid_mask),
        velocity_vector_mean_error_mps=evaluate_masked_mae(velocity_vector_error, zeros, fluid_mask),
        velocity_vector_rmse_mps=evaluate_masked_rmse(velocity_vector_error, zeros, fluid_mask),
        velocity_vector_max_error_mps=evaluate_masked_max_abs_error(velocity_vector_error, zeros, fluid_mask),
        velocity_vector_p95_error_mps=evaluate_masked_percentile_abs_error(
            velocity_vector_error,
            zeros,
            fluid_mask,
            percentile=95.0,
        ),
        velocity_vector_p99_error_mps=evaluate_masked_percentile_abs_error(
            velocity_vector_error,
            zeros,
            fluid_mask,
            percentile=99.0,
        ),
        physics_loss=physics_loss,
        continuity_loss=continuity_loss,
        momentum_x_loss=momentum_x_loss,
        momentum_y_loss=momentum_y_loss,
        fluid_cell_count=int(np.count_nonzero(fluid_mask)),
        total_cell_count=int(fluid_mask.size),
    )


def evaluate_case_metrics_to_row(metrics: CaseMetrics) -> dict[str, Any]:
    return asdict(metrics)


def evaluate_summarize_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return out


def evaluate_init_global_aggregator() -> dict[str, float | int]:
    return {
        "fluid_cells": 0,
        "p_abs_sum": 0.0,
        "p_sq_sum": 0.0,
        "p_sum": 0.0,
        "ux_abs_sum": 0.0,
        "ux_sq_sum": 0.0,
        "ux_sum": 0.0,
        "uy_abs_sum": 0.0,
        "uy_sq_sum": 0.0,
        "uy_sum": 0.0,
        "speed_abs_sum": 0.0,
        "speed_sq_sum": 0.0,
        "speed_sum": 0.0,
        "vec_abs_sum": 0.0,
        "vec_sq_sum": 0.0,
    }


def evaluate_update_global_aggregator(
    agg: dict[str, float | int],
    p_pred: np.ndarray,
    p_true: np.ndarray,
    ux_pred: np.ndarray,
    ux_true: np.ndarray,
    uy_pred: np.ndarray,
    uy_true: np.ndarray,
    speed_pred: np.ndarray,
    speed_true: np.ndarray,
    velocity_vector_error: np.ndarray,
    fluid_mask: np.ndarray,
) -> None:
    p_diff = (p_pred - p_true)[fluid_mask]
    ux_diff = (ux_pred - ux_true)[fluid_mask]
    uy_diff = (uy_pred - uy_true)[fluid_mask]
    speed_diff = (speed_pred - speed_true)[fluid_mask]
    vec_diff = velocity_vector_error[fluid_mask]

    p_diff = p_diff[np.isfinite(p_diff)]
    ux_diff = ux_diff[np.isfinite(ux_diff)]
    uy_diff = uy_diff[np.isfinite(uy_diff)]
    speed_diff = speed_diff[np.isfinite(speed_diff)]
    vec_diff = vec_diff[np.isfinite(vec_diff)]

    n_cells = int(min(p_diff.size, ux_diff.size, uy_diff.size, speed_diff.size, vec_diff.size))
    if n_cells <= 0:
        return

    p_diff = p_diff[:n_cells]
    ux_diff = ux_diff[:n_cells]
    uy_diff = uy_diff[:n_cells]
    speed_diff = speed_diff[:n_cells]
    vec_diff = vec_diff[:n_cells]

    agg["fluid_cells"] = int(agg["fluid_cells"]) + n_cells
    agg["p_abs_sum"] = float(agg["p_abs_sum"]) + float(np.abs(p_diff).sum())
    agg["p_sq_sum"] = float(agg["p_sq_sum"]) + float((p_diff ** 2).sum())
    agg["p_sum"] = float(agg["p_sum"]) + float(p_diff.sum())
    agg["ux_abs_sum"] = float(agg["ux_abs_sum"]) + float(np.abs(ux_diff).sum())
    agg["ux_sq_sum"] = float(agg["ux_sq_sum"]) + float((ux_diff ** 2).sum())
    agg["ux_sum"] = float(agg["ux_sum"]) + float(ux_diff.sum())
    agg["uy_abs_sum"] = float(agg["uy_abs_sum"]) + float(np.abs(uy_diff).sum())
    agg["uy_sq_sum"] = float(agg["uy_sq_sum"]) + float((uy_diff ** 2).sum())
    agg["uy_sum"] = float(agg["uy_sum"]) + float(uy_diff.sum())
    agg["speed_abs_sum"] = float(agg["speed_abs_sum"]) + float(np.abs(speed_diff).sum())
    agg["speed_sq_sum"] = float(agg["speed_sq_sum"]) + float((speed_diff ** 2).sum())
    agg["speed_sum"] = float(agg["speed_sum"]) + float(speed_diff.sum())
    agg["vec_abs_sum"] = float(agg["vec_abs_sum"]) + float(np.abs(vec_diff).sum())
    agg["vec_sq_sum"] = float(agg["vec_sq_sum"]) + float((vec_diff ** 2).sum())


def evaluate_finalize_global_aggregator(agg: dict[str, float | int]) -> dict[str, float | int]:
    n = max(int(agg["fluid_cells"]), 1)
    return {
        "global_pixel_weighted_p_mae": float(agg["p_abs_sum"]) / n,
        "global_pixel_weighted_p_rmse": float(np.sqrt(float(agg["p_sq_sum"]) / n)),
        "global_pixel_weighted_p_bias": float(agg["p_sum"]) / n,
        "global_pixel_weighted_ux_mae_mps": float(agg["ux_abs_sum"]) / n,
        "global_pixel_weighted_ux_rmse_mps": float(np.sqrt(float(agg["ux_sq_sum"]) / n)),
        "global_pixel_weighted_ux_bias_mps": float(agg["ux_sum"]) / n,
        "global_pixel_weighted_uy_mae_mps": float(agg["uy_abs_sum"]) / n,
        "global_pixel_weighted_uy_rmse_mps": float(np.sqrt(float(agg["uy_sq_sum"]) / n)),
        "global_pixel_weighted_uy_bias_mps": float(agg["uy_sum"]) / n,
        "global_pixel_weighted_speed_mae_mps": float(agg["speed_abs_sum"]) / n,
        "global_pixel_weighted_speed_rmse_mps": float(np.sqrt(float(agg["speed_sq_sum"]) / n)),
        "global_pixel_weighted_speed_bias_mps": float(agg["speed_sum"]) / n,
        "global_pixel_weighted_velocity_vector_mean_error_mps": float(agg["vec_abs_sum"]) / n,
        "global_pixel_weighted_velocity_vector_rmse_mps": float(np.sqrt(float(agg["vec_sq_sum"]) / n)),
        "global_fluid_cell_count": int(agg["fluid_cells"]),
    }
