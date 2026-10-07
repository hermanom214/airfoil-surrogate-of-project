from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scalar_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int | None]:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[valid], est[valid]
    if ref.size == 0:
        return {"count": 0, "MAE": None, "RMSE": None, "bias": None,
                "median_absolute_error": None, "max_absolute_error": None, "R2": None}
    error = est - ref
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    return {
        "count": int(ref.size), "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))), "bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
        "R2": None if ss_tot <= 1e-12 else float(1.0 - np.sum(error**2) / ss_tot),
    }


def build_aero_summary(df: pd.DataFrame, numerical_settings: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "true_grid_vs_openfoam": ("openfoam", "true_grid"),
        "prediction_vs_openfoam": ("openfoam", "pred"),
        "prediction_vs_true_grid": ("true_grid", "pred"),
    }
    metric_blocks: dict[str, Any] = {}
    for label, (ref, est) in comparisons.items():
        metric_blocks[label] = {}
        for coefficient in ("cl", "cd"):
            # Pressure is universally available; total remains experimental and is reported separately.
            metric_blocks[label][coefficient] = scalar_metrics(
                df[f"{coefficient}_{ref}" if ref == "openfoam" else f"{coefficient}_{ref}_pressure"].to_numpy(),
                df[f"{coefficient}_{est}_pressure"].to_numpy(),
            )
            total_ref_col = (
                f"{coefficient}_openfoam" if ref == "openfoam" else f"{coefficient}_{ref}_total"
            )
            metric_blocks[label][f"{coefficient}_total_experimental"] = scalar_metrics(
                df[total_ref_col].to_numpy(), df[f"{coefficient}_{est}_total"].to_numpy()
            )
    cd_ref = df["cd_openfoam"].to_numpy(dtype=np.float64)
    cd_est = df["cd_pred_pressure"].to_numpy(dtype=np.float64)
    safe = np.isfinite(cd_ref) & np.isfinite(cd_est) & (np.abs(cd_ref) >= 1e-3)
    cd_relative_mae = None if not np.any(safe) else float(np.mean(np.abs((cd_est[safe] - cd_ref[safe]) / cd_ref[safe])))
    return {
        "evaluated_case_count": int(len(df)),
        "openfoam_reference_count": int(np.isfinite(df["cl_openfoam"]).sum()),
        "valid_pressure_integrations": int(df["pressure_valid"].sum()),
        "valid_viscous_integrations": int(df["viscous_valid"].sum()),
        "metrics": metric_blocks,
        "cd_prediction_vs_openfoam_relative_mae_for_abs_reference_ge_1e-3": cd_relative_mae,
        "numerical_settings": numerical_settings,
        "interpretation": {
            "pressure": "field-derived pressure contribution",
            "viscous": "experimental near-wall Cartesian-grid estimate",
            "total": "experimental pressure plus viscous estimate",
        },
    }


def _scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str, output: Path) -> None:
    x, y = df[xcol].to_numpy(float), df[ycol].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    x, y = x[valid], y[valid]
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    margin = max((hi - lo) * 0.05, 1e-8)
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    ax.scatter(x, y, alpha=.75); ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], "k--")
    ax.set(xlabel=xcol, ylabel=ycol, title=title, xlim=(lo-margin, hi+margin), ylim=(lo-margin, hi+margin))
    ax.grid(alpha=.25); fig.savefig(output, dpi=160); plt.close(fig)


def _error_plot(df: pd.DataFrame, xcol: str, coefficient: str, output: Path) -> None:
    x = df[xcol].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    made = False
    for label, estimate in (("true grid - OpenFOAM", "true_grid"), ("prediction - OpenFOAM", "pred")):
        error = df[f"{coefficient}_{estimate}_pressure"].to_numpy(float) - df[f"{coefficient}_openfoam"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(error)
        if np.any(valid):
            ax.scatter(x[valid], error[valid], alpha=.7, label=label); made = True
    if not made:
        plt.close(fig); return
    ax.axhline(0, color="k", ls="--"); ax.set(xlabel=xcol, ylabel=f"{coefficient} error", title=f"{coefficient} error versus {xcol}")
    ax.grid(alpha=.25); ax.legend(); fig.savefig(output, dpi=160); plt.close(fig)


def generate_aero_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for coefficient in ("cl", "cd"):
        _scatter(df, f"{coefficient}_openfoam", f"{coefficient}_pred_pressure",
                 f"OpenFOAM vs predicted-field pressure {coefficient}", output_dir / f"{coefficient}_openfoam_vs_prediction.png")
        _scatter(df, f"{coefficient}_openfoam", f"{coefficient}_true_grid_pressure",
                 f"OpenFOAM vs true-grid pressure {coefficient}", output_dir / f"{coefficient}_openfoam_vs_true_grid.png")
        _scatter(df, f"{coefficient}_true_grid_pressure", f"{coefficient}_pred_pressure",
                 f"True-grid vs predicted-field pressure {coefficient}", output_dir / f"{coefficient}_true_grid_vs_prediction.png")
        _error_plot(df, "aoa_deg", coefficient, output_dir / f"{coefficient}_error_vs_aoa.png")
        _error_plot(df, "inlet_velocity", coefficient, output_dir / f"{coefficient}_error_vs_inlet_velocity.png")
