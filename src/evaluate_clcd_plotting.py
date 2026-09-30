from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_FIGURE_SIZE = (13.0, 6.0)
SUMMARY_FIGURE_DPI = 250


def _scatter_true_vs_pred(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    t = np.asarray(true_values, dtype=np.float64)
    p = np.asarray(pred_values, dtype=np.float64)

    valid = np.isfinite(t) & np.isfinite(p)
    t = t[valid]
    p = p[valid]
    if t.size == 0:
        return

    lo = float(min(np.min(t), np.min(p)))
    hi = float(max(np.max(t), np.max(p)))
    if hi <= lo:
        hi = lo + 1e-8

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.scatter(t, p, alpha=0.8)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _residual_plot(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    t = np.asarray(true_values, dtype=np.float64)
    p = np.asarray(pred_values, dtype=np.float64)

    valid = np.isfinite(t) & np.isfinite(p)
    t = t[valid]
    residuals = (p - t)[valid]
    if t.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.scatter(t, residuals, alpha=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _bar_abs_errors(
    case_ids: list[str],
    abs_errors: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    errs = np.asarray(abs_errors, dtype=np.float64)
    if errs.size == 0:
        return

    order = np.argsort(errs)[::-1]
    sorted_ids = [case_ids[int(i)] for i in order]
    sorted_errs = errs[order]

    fig, ax = plt.subplots(
        figsize=(max(9.0, 0.26 * len(sorted_ids) + 2.5), 5.5),
        constrained_layout=True,
    )
    x = np.arange(len(sorted_ids))
    ax.bar(x, sorted_errs)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_ids, rotation=70, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _summary_axis_limits(
    true_values: np.ndarray,
    pred_values: np.ndarray,
) -> tuple[float, float]:
    """Return honest, equal x/y limits with a small margin for one coefficient."""
    lo = float(min(np.min(true_values), np.min(pred_values)))
    hi = float(max(np.max(true_values), np.max(pred_values)))
    span = hi - lo
    margin = 0.05 * span if span > 0.0 else max(abs(lo) * 0.05, 1e-8)
    return lo - margin, hi + margin


def _format_summary_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _clcd_summary_scatter(
    metrics_df: pd.DataFrame,
    global_metrics: Mapping[str, Mapping[str, float | None]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=SUMMARY_FIGURE_SIZE,
        constrained_layout=True,
    )

    for ax, coefficient, panel_title in zip(
        axes,
        ("Cl", "Cd"),
        ("Lift coefficient, Cl", "Drag coefficient, Cd"),
    ):
        true_values = metrics_df[f"{coefficient}_true"].to_numpy(dtype=np.float64)
        pred_values = metrics_df[f"{coefficient}_pred"].to_numpy(dtype=np.float64)
        valid = np.isfinite(true_values) & np.isfinite(pred_values)
        true_values = true_values[valid]
        pred_values = pred_values[valid]

        if true_values.size == 0:
            continue

        lo, hi = _summary_axis_limits(true_values, pred_values)
        ax.scatter(true_values, pred_values, s=34, alpha=0.78, edgecolors="none")
        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            linewidth=1.4,
            color="0.25",
            label="Perfect prediction",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
        ax.set_xlabel(f"CFD {coefficient}", fontsize=12)
        ax.set_ylabel(f"ML-predicted {coefficient}", fontsize=12)
        ax.set_title(panel_title, fontsize=14)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right", frameon=False, fontsize=9)

        coefficient_metrics = global_metrics[coefficient]
        metrics_text = (
            f"R² = {_format_summary_metric(coefficient_metrics['R2'])}\n"
            f"MAE = {_format_summary_metric(coefficient_metrics['MAE'])}"
        )
        ax.text(
            0.04,
            0.96,
            metrics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": "0.8",
            },
        )

    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI)
    plt.close(fig)


def evaluate_generate_clcd_plots(
    metrics_df: pd.DataFrame,
    figures_dir: Path,
    global_metrics: Mapping[str, Mapping[str, float | None]],
) -> None:
    if metrics_df.empty:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    _scatter_true_vs_pred(
        true_values=metrics_df["Cl_true"].to_numpy(),
        pred_values=metrics_df["Cl_pred"].to_numpy(),
        xlabel="Cl true",
        ylabel="Cl predicted",
        title="Cl: True vs Predicted",
        output_path=figures_dir / "cl_scatter.png",
    )

    _scatter_true_vs_pred(
        true_values=metrics_df["Cd_true"].to_numpy(),
        pred_values=metrics_df["Cd_pred"].to_numpy(),
        xlabel="Cd true",
        ylabel="Cd predicted",
        title="Cd: True vs Predicted",
        output_path=figures_dir / "cd_scatter.png",
    )

    _residual_plot(
        true_values=metrics_df["Cl_true"].to_numpy(),
        pred_values=metrics_df["Cl_pred"].to_numpy(),
        xlabel="Cl true",
        ylabel="Cl residual (pred - true)",
        title="Cl residuals",
        output_path=figures_dir / "cl_residual.png",
    )

    _residual_plot(
        true_values=metrics_df["Cd_true"].to_numpy(),
        pred_values=metrics_df["Cd_pred"].to_numpy(),
        xlabel="Cd true",
        ylabel="Cd residual (pred - true)",
        title="Cd residuals",
        output_path=figures_dir / "cd_residual.png",
    )

    _bar_abs_errors(
        case_ids=[str(v) for v in metrics_df["case_id"].tolist()],
        abs_errors=metrics_df["Cl_abs_error"].to_numpy(),
        ylabel="Absolute Cl error",
        title="Absolute Cl error by case (sorted)",
        output_path=figures_dir / "cl_errors.png",
    )

    _bar_abs_errors(
        case_ids=[str(v) for v in metrics_df["case_id"].tolist()],
        abs_errors=metrics_df["Cd_abs_error"].to_numpy(),
        ylabel="Absolute Cd error",
        title="Absolute Cd error by case (sorted)",
        output_path=figures_dir / "cd_errors.png",
    )

    _clcd_summary_scatter(
        metrics_df=metrics_df,
        global_metrics=global_metrics,
        output_path=figures_dir / "clcd_summary_scatter.png",
    )
