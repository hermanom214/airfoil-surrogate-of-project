"""Generate final persisted-artifact reports for the improved Phase-A A/B run."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.unet_pinn_reporting import plot_histories, validate_pair, validate_run


ROOTS = {
    "supervised": Path("data/models/unet_improved_supervised"),
    "physics_informed": Path("data/models/unet_improved_pinn"),
}
PROFILES = {
    "supervised": "unet_improved_supervised",
    "physics_informed": "unet_improved_pinn",
}
OUTPUT = Path("outputs/experiments/unet_improved_pinn_ab")


def pct(a, b):
    return 100.0 * (b - a) / a if a else None


def main():
    runs = {label: validate_run(ROOTS[label], PROFILES[label]) for label in ROOTS}
    validate_pair(runs["supervised"][0], runs["physics_informed"][0])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = ["epoch_history.json", "epoch_history.csv", "selected_checkpoint_report.json", "report.md"]
    targets += ["supervised_loss.png", "supervised_weighted_data_loss.png",
                "supervised_gradient_loss.png", "raw_continuity_loss.png",
                "raw_momentum_x_loss.png", "raw_momentum_y_loss.png"]
    collisions = [name for name in targets if (OUTPUT / name).exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite report artifacts: {collisions}")
    histories = {label: run[1] for label, run in runs.items()}
    best = {label: run[0]["best_validation_epoch"] for label, run in runs.items()}
    figures = plot_histories(histories, best, OUTPUT, improved=True)
    (OUTPUT / "epoch_history.json").write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUTPUT / "epoch_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "epoch", "improved_supervised_validation_loss",
            "region_weighted_data_loss", "gradient_matching_loss",
            "continuity_sqrt_mean_batch_mse", "momentum_x_sqrt_mean_batch_mse",
            "momentum_y_sqrt_mean_batch_mse", "elapsed_seconds"])
        for label, history in histories.items():
            for epoch in history:
                val = epoch["validation"]
                writer.writerow([label, epoch["epoch"], val["supervised_loss"],
                    val["supervised_weighted_data_loss"], val["supervised_gradient_loss"],
                    math.sqrt(val["raw_continuity_loss"]),
                    math.sqrt(val["raw_momentum_x_loss"]),
                    math.sqrt(val["raw_momentum_y_loss"]), epoch["elapsed_training_seconds"]])
    metrics = {label: run[2] for label, run in runs.items()}
    selected = {}
    keys = ["supervised_loss", "supervised_weighted_data_loss", "supervised_gradient_loss",
            "pooled_full_p_mae", "pooled_full_p_rmse", "pooled_full_ux_mae",
            "pooled_full_ux_rmse", "pooled_full_uy_mae", "pooled_full_uy_rmse",
            "case_mean_full_p_mae", "case_mean_full_ux_mae", "case_mean_full_uy_mae",
            "pooled_near_airfoil_p_mae", "pooled_near_airfoil_ux_mae",
            "pooled_near_airfoil_uy_mae", "pooled_wake_p_mae", "pooled_wake_ux_mae",
            "pooled_wake_uy_mae"]
    for key in keys:
        a, b = metrics["supervised"][key], metrics["physics_informed"][key]
        selected[key] = {"supervised": a, "physics_informed": b,
                         "difference": b - a, "percentage_difference": pct(a, b)}
    for output_key, source_key in (("continuity_rms", "raw_continuity_loss"),
            ("momentum_x_rms", "raw_momentum_x_loss"),
            ("momentum_y_rms", "raw_momentum_y_loss")):
        a, b = (math.sqrt(metrics[label][source_key]) for label in ROOTS)
        selected[output_key] = {"supervised": a, "physics_informed": b,
                                "difference": b - a, "percentage_difference": pct(a, b),
                                "aggregation": "sqrt(mean batch masked MSE); nondimensional"}
    report = {
        "controlled_comparison": True,
        "selected_epochs": best,
        "metrics": selected,
        "training": {label: {key: runs[label][0][key] for key in (
            "training_duration_seconds", "peak_cuda_memory_allocated_mib",
            "peak_cuda_memory_reserved_mib", "gradient_batches_checked")}
            for label in ROOTS},
        "metric_definitions": {
            "supervised_loss": "batch-averaged region-weighted data loss + 0.1 * gradient-matching loss",
            "physical_field_errors": "pooled_* uses all valid validation cells; case_mean_* averages per-case MAE",
            "physics_history": "sqrt(mean batch masked MSE), nondimensional; not globally pooled-cell RMS",
        },
        "figures": figures,
    }
    (OUTPUT / "selected_checkpoint_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Phase A v1 Improved U-Net + PINN A/B report", "",
        "Both controlled 30-epoch CUDA runs completed and passed strict compatibility validation.", "",
        f"Best validation epochs: A={best['supervised']}, B={best['physics_informed']}.", "",
        "Historical physics curves are nondimensional sqrt(mean batch masked MSE), not globally pooled-cell RMS.",
        "Physical field `pooled_*` errors pool all validation cells; `case_mean_*` values average per-case MAE.", "",
        "## Figures", ""] + [f"- [{name}]({name})" for name in figures]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
