from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def evaluate_parse_case_metadata(case_id: str) -> dict[str, Any]:
    naca_match = re.search(r"naca(\d{4})", case_id)
    naca = naca_match.group(1) if naca_match else "unknown"

    if naca_match:
        camber = int(naca[0])
        camber_position = int(naca[1])
        thickness = int(naca[2:])
    else:
        camber = -1
        camber_position = -1
        thickness = -1

    return {
        "NACA": naca,
        "camber": camber,
        "camber_position": camber_position,
        "thickness": thickness,
    }


def evaluate_build_clcd_case_row(
    case_id: str,
    features: np.ndarray,
    true_targets: np.ndarray,
    pred_targets: np.ndarray,
    normalized_mse: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32).reshape(-1)
    y_true = np.asarray(true_targets, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(pred_targets, dtype=np.float32).reshape(-1)

    if x.size != 5:
        raise RuntimeError(f"Feature vector for case '{case_id}' must have size 5")
    if y_true.size != 2 or y_pred.size != 2:
        raise RuntimeError(f"Target vectors for case '{case_id}' must have size 2")

    if not (np.isfinite(x).all() and np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise RuntimeError(f"Non-finite values in case row data for case '{case_id}'")

    meta = evaluate_parse_case_metadata(case_id)

    cl_true = float(y_true[0])
    cd_true = float(y_true[1])
    cl_pred = float(y_pred[0])
    cd_pred = float(y_pred[1])

    cl_error = cl_pred - cl_true
    cd_error = cd_pred - cd_true

    return {
        "case_id": case_id,
        "NACA": meta["NACA"],
        "camber": int(meta["camber"]),
        "camber_position": int(meta["camber_position"]),
        "thickness": int(meta["thickness"]),
        "AoA": float(x[3]),
        "inlet_velocity": float(x[4]),
        "Cl_true": cl_true,
        "Cl_pred": cl_pred,
        "Cl_error": float(cl_error),
        "Cl_abs_error": float(abs(cl_error)),
        "Cd_true": cd_true,
        "Cd_pred": cd_pred,
        "Cd_error": float(cd_error),
        "Cd_abs_error": float(abs(cd_error)),
        "normalized_MSE": float(normalized_mse),
    }


def _scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]

    if yt.size == 0:
        return {
            "MAE": None,
            "RMSE": None,
            "Bias": None,
            "Median_absolute_error": None,
            "Maximum_absolute_error": None,
            "R2": None,
        }

    diff = yp - yt
    abs_diff = np.abs(diff)

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12:
        r2 = None
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    return {
        "MAE": float(np.mean(abs_diff)),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "Bias": float(np.mean(diff)),
        "Median_absolute_error": float(np.median(abs_diff)),
        "Maximum_absolute_error": float(np.max(abs_diff)),
        "R2": r2,
    }


def evaluate_compute_clcd_global_metrics(metrics_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if metrics_df.empty:
        return {"Cl": _scalar_metrics(np.array([]), np.array([])), "Cd": _scalar_metrics(np.array([]), np.array([]))}

    cl_metrics = _scalar_metrics(metrics_df["Cl_true"].to_numpy(), metrics_df["Cl_pred"].to_numpy())
    cd_metrics = _scalar_metrics(metrics_df["Cd_true"].to_numpy(), metrics_df["Cd_pred"].to_numpy())
    return {
        "Cl": cl_metrics,
        "Cd": cd_metrics,
    }
