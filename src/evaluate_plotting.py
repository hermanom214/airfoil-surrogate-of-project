from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def evaluate_robust_limits(arr: np.ndarray, low: float, high: float) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, low))
    hi = float(np.percentile(vals, high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def evaluate_mask_to_nan(arr: np.ndarray, fluid_mask: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float64, copy=True)
    out[~fluid_mask] = np.nan
    return out


def evaluate_find_nearest_fluid_cell(
    xy: np.ndarray,
    fluid_mask: np.ndarray,
    requested_x: float,
    requested_y: float,
) -> tuple[int, int]:
    candidates = np.argwhere(fluid_mask > 0.5)
    if candidates.size == 0:
        raise RuntimeError("No fluid cells available for nearest-cell lookup")

    points = xy[candidates[:, 0], candidates[:, 1], :]
    dx = points[:, 0] - requested_x
    dy = points[:, 1] - requested_y
    dist2 = dx * dx + dy * dy
    best = int(np.argmin(dist2))
    return int(candidates[best, 0]), int(candidates[best, 1])


def evaluate_rotate_point_about_quarter_chord(x: float, y: float, aoa_deg: float, chord: float) -> tuple[float, float]:
    cx = 0.25 * chord
    cy = 0.0
    angle = np.deg2rad(-aoa_deg)

    dx = x - cx
    dy = y - cy

    xr = dx * np.cos(angle) - dy * np.sin(angle)
    yr = dx * np.sin(angle) + dy * np.cos(angle)

    return float(xr + cx), float(yr + cy)


def evaluate_build_velocity_annotations(
    xy: np.ndarray,
    fluid_mask: np.ndarray,
    velocity_vector_error: np.ndarray,
    chord: float,
    aoa_deg: float,
) -> list[dict[str, float | str]]:
    # Exactly four orientation points requested by user.
    requested_body = [
        ("upstream", -0.10 * chord, 0.0),
        ("upper_front", 0.33 * chord, 0.10 * chord),
        ("lower_front", 0.33 * chord, -0.10 * chord),
        ("wake", 1.20 * chord, 0.0),
    ]

    span_x = float(xy[:, :, 0].max() - xy[:, :, 0].min())
    span_y = float(xy[:, :, 1].max() - xy[:, :, 1].min())
    offsets = [
        (0.025 * span_x, 0.11 * span_y),
        (0.025 * span_x, 0.13 * span_y),
        (0.025 * span_x, -0.13 * span_y),
        (0.025 * span_x, 0.11 * span_y),
    ]

    out: list[dict[str, float | str]] = []
    for idx, (label, x_body, y_body) in enumerate(requested_body):
        x_req, y_req = evaluate_rotate_point_about_quarter_chord(x_body, y_body, aoa_deg, chord)
        jj, ii = evaluate_find_nearest_fluid_cell(xy, fluid_mask, x_req, y_req)
        out.append(
            {
                "label": label,
                "x": float(xy[jj, ii, 0]),
                "y": float(xy[jj, ii, 1]),
                "value": float(velocity_vector_error[jj, ii]),
                "dx": float(offsets[idx][0]),
                "dy": float(offsets[idx][1]),
            }
        )

    return out


def _evaluate_blue_to_red_colormap() -> Any:
    # Blue for low values, red for high values.
    cmap = cm.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="#efefef")
    return cmap


def evaluate_plot_velocity_comparison(
    output_path: Any,
    xy: np.ndarray,
    fluid_mask: np.ndarray,
    speed_true: np.ndarray,
    speed_pred: np.ndarray,
    velocity_vector_error: np.ndarray,
    case_label: str,
    subtitle: str,
    annotations: list[dict[str, float | str]],
) -> None:
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    speed_true_m = evaluate_mask_to_nan(speed_true, fluid_mask)
    speed_pred_m = evaluate_mask_to_nan(speed_pred, fluid_mask)
    err_m = evaluate_mask_to_nan(velocity_vector_error, fluid_mask)

    combined = np.concatenate([speed_true[fluid_mask], speed_pred[fluid_mask]])
    vmin, vmax = evaluate_robust_limits(combined, 1.0, 99.5)
    emax = float(np.percentile(velocity_vector_error[fluid_mask], 99.5))
    if emax <= 0:
        emax = float(np.max(velocity_vector_error[fluid_mask])) + 1e-6

    cmap = _evaluate_blue_to_red_colormap()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    panels = [
        (speed_true_m, "OpenFOAM |U| [m/s]", vmin, vmax),
        (speed_pred_m, "Predicted |U| [m/s]", vmin, vmax),
        (err_m, "Vector error |U_pred - U_OF| [m/s]", 0.0, emax),
    ]

    for ax, (field, title, lo, hi) in zip(axes, panels):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            aspect="equal",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Value")

    err_ax = axes[2]
    for ann in annotations:
        x0 = float(ann["x"])
        y0 = float(ann["y"])
        dx = float(ann["dx"])
        dy = float(ann["dy"])
        du = float(ann["value"])
        label = str(ann["label"])

        err_ax.plot(x0, y0, marker="o", markersize=4, color="white", markeredgecolor="black")
        err_ax.annotate(
            f"{label}: ΔU={du:.2f} m/s",
            xy=(x0, y0),
            xytext=(x0 + dx, y0 + dy),
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.2},
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7},
        )

    fig.suptitle(f"{case_label}\n{subtitle}", fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_plot_fields_comparison(
    output_path: Any,
    xy: np.ndarray,
    fluid_mask: np.ndarray,
    p_true: np.ndarray,
    p_pred: np.ndarray,
    ux_true: np.ndarray,
    ux_pred: np.ndarray,
    uy_true: np.ndarray,
    uy_pred: np.ndarray,
    case_label: str,
) -> None:
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    def prep(true_arr: np.ndarray, pred_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        true_m = evaluate_mask_to_nan(true_arr, fluid_mask)
        pred_m = evaluate_mask_to_nan(pred_arr, fluid_mask)
        err = np.abs(pred_arr - true_arr)
        err_m = evaluate_mask_to_nan(err, fluid_mask)

        both = np.concatenate([true_arr[fluid_mask], pred_arr[fluid_mask]])
        vmin, vmax = evaluate_robust_limits(both, 1.0, 99.0)
        emax = float(np.percentile(err[fluid_mask], 99.5))
        if emax <= 0:
            emax = float(np.max(err[fluid_mask])) + 1e-6
        return true_m, pred_m, err_m, vmin, vmax, emax

    p_t, p_p, p_e, p_lo, p_hi, p_ehi = prep(p_true, p_pred)
    ux_t, ux_p, ux_e, ux_lo, ux_hi, ux_ehi = prep(ux_true, ux_pred)
    uy_t, uy_p, uy_e, uy_lo, uy_hi, uy_ehi = prep(uy_true, uy_pred)

    cmap = _evaluate_blue_to_red_colormap()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    fields = [
        (p_t, "OpenFOAM p [OpenFOAM kinematic-pressure units]", p_lo, p_hi),
        (p_p, "Predicted p [OpenFOAM kinematic-pressure units]", p_lo, p_hi),
        (p_e, "|p_pred - p_true|", 0.0, p_ehi),
        (ux_t, "OpenFOAM Ux [m/s]", ux_lo, ux_hi),
        (ux_p, "Predicted Ux [m/s]", ux_lo, ux_hi),
        (ux_e, "|Ux_pred - Ux_true| [m/s]", 0.0, ux_ehi),
        (uy_t, "OpenFOAM Uy [m/s]", uy_lo, uy_hi),
        (uy_p, "Predicted Uy [m/s]", uy_lo, uy_hi),
        (uy_e, "|Uy_pred - Uy_true| [m/s]", 0.0, uy_ehi),
    ]

    for ax, (field, title, lo, hi) in zip(axes.flat, fields):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            aspect="equal",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Value")

    fig.suptitle(case_label, fontsize=12)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_generate_dataset_level_plot(metrics_df: pd.DataFrame, output_path: Any) -> None:
    if metrics_df.empty:
        return

    sorted_df = metrics_df.sort_values(by="velocity_vector_rmse_mps", ascending=True).reset_index(drop=True)
    n = len(sorted_df)

    if n > 25:
        fig, ax = plt.subplots(figsize=(14, max(5, min(0.35 * n + 2, 14))), constrained_layout=True)
        y = np.arange(n)
        ax.barh(y - 0.25, sorted_df["velocity_vector_rmse_mps"], height=0.25, label="Velocity vector RMSE")
        ax.barh(y, sorted_df["ux_rmse_mps"], height=0.25, label="Ux RMSE")
        ax.barh(y + 0.25, sorted_df["uy_rmse_mps"], height=0.25, label="Uy RMSE")
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_df["case_id"])
        ax.set_xlabel("RMSE [m/s]")
    else:
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        x = np.arange(n)
        w = 0.28
        ax.bar(x - w, sorted_df["velocity_vector_rmse_mps"], width=w, label="Velocity vector RMSE")
        ax.bar(x, sorted_df["ux_rmse_mps"], width=w, label="Ux RMSE")
        ax.bar(x + w, sorted_df["uy_rmse_mps"], width=w, label="Uy RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_df["case_id"], rotation=70, ha="right")
        ax.set_ylabel("RMSE [m/s]")

    ax.set_title("Validation case errors (sorted by velocity vector RMSE)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
