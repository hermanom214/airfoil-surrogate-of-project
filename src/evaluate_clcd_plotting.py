from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def evaluate_generate_clcd_plots(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
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
