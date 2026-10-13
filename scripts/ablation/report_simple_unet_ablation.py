"""Build the controlled five-model simple-U-Net ablation report from saved evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
OUT = ROOT / "data" / "ablation_logs"
EXPERIMENTS = [
    "simple_unet_V0",
    "simple_unet_ablation_targets",
    "simple_unet_ablation_regions",
    "simple_unet_ablation_gradient",
    "simple_unet_improved",
]
FIELD_COLUMNS = {
    "common_normalized_mse": "common_normalized_masked_mse",
    "velocity_rmse": "velocity_vector_rmse_mps",
    "p_rmse": "p_rmse",
    "ux_rmse": "ux_rmse_mps",
    "uy_rmse": "uy_rmse_mps",
    "wake_rmse": "wake_velocity_vector_rmse",
    "near_wall_rmse": "near_wall_velocity_vector_rmse",
    "outer_rmse": "outer_velocity_vector_rmse",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields: dict[str, pd.DataFrame] = {}
    aero: dict[str, pd.DataFrame] = {}
    for name in EXPERIMENTS:
        metrics = MODELS / f"{name}_validation" / "metrics"
        fields[name] = pd.read_csv(metrics / "validation_metrics_per_case.csv").set_index("case_id")
        aero[name] = pd.read_csv(metrics / "aero_coefficients_test.csv").set_index("case_id")
    expected = set(fields[EXPERIMENTS[0]].index)
    for name in EXPERIMENTS:
        if set(fields[name].index) != expected or set(aero[name].index) != expected:
            raise RuntimeError(f"Evaluation case mismatch for {name}")
        if len(fields[name]) != 116:
            raise RuntimeError(f"Expected 116 evaluated cases for {name}, got {len(fields[name])}")

    rows = []
    for name in EXPERIMENTS:
        row = {"experiment": name}
        row.update({label: float(fields[name][column].mean()) for label, column in FIELD_COLUMNS.items()})
        valid = aero[name][["cl_openfoam", "cl_pred_pressure"]].dropna()
        error = valid["cl_pred_pressure"].to_numpy() - valid["cl_openfoam"].to_numpy()
        row.update({
            "cl_mae": float(np.mean(np.abs(error))),
            "cl_rmse": float(np.sqrt(np.mean(error**2))),
            "cl_r2": r2(valid["cl_openfoam"].to_numpy(), valid["cl_pred_pressure"].to_numpy()),
        })
        rows.append(row)
    table = pd.DataFrame(rows).set_index("experiment")
    table.to_csv(OUT / "ablation_table.csv")

    errors = [*FIELD_COLUMNS, "cl_mae", "cl_rmse"]
    baseline = table.loc[EXPERIMENTS[0]]
    improvement = pd.DataFrame(index=table.index)
    for column in errors:
        improvement[f"{column}_pct"] = 100.0 * (baseline[column] - table[column]) / baseline[column]
    improvement["cl_r2_delta"] = table["cl_r2"] - baseline["cl_r2"]
    improvement.to_csv(OUT / "improvement_vs_baseline.csv")

    robustness: dict[str, object] = {}
    base_fields = fields[EXPERIMENTS[0]]
    base_aero = aero[EXPERIMENTS[0]]
    for name in EXPERIMENTS:
        item: dict[str, object] = {}
        for region, column in (("wake", FIELD_COLUMNS["wake_rmse"]),
                               ("near_wall", FIELD_COLUMNS["near_wall_rmse"])):
            relative = (base_fields[column] - fields[name][column]) / base_fields[column]
            item[region] = {
                "better": int((relative > 0).sum()), "worse": int((relative < 0).sum()),
                "equal": int((relative == 0).sum()),
                "median_relative_improvement_pct": float(100.0 * relative.median()),
                "worst_regression_pct": float(100.0 * relative.min()),
                "worst_case": str(relative.idxmin()),
            }
        base_abs = (base_aero["cl_pred_pressure"] - base_aero["cl_openfoam"]).abs()
        current_abs = (aero[name]["cl_pred_pressure"] - aero[name]["cl_openfoam"]).abs()
        change = base_abs - current_abs
        item["cl"] = {
            "better": int((change > 0).sum()), "worse": int((change < 0).sum()),
            "equal": int((change == 0).sum()),
            "median_absolute_error_improvement": float(change.median()),
            "worst_regression": float(change.min()), "worst_case": str(change.idxmin()),
        }
        robustness[name] = item

    individual = EXPERIMENTS[1:4]
    synergy = {}
    for column in errors:
        reductions = (baseline[column] - table.loc[individual, column]).sum()
        combined = baseline[column] - table.loc["simple_unet_improved", column]
        synergy[column] = {
            "combined_reduction": float(combined),
            "sum_individual_reductions": float(reductions),
            "interaction_excess_reduction": float(combined - reductions),
        }

    hashes = {
        "baseline_checkpoint": sha256(MODELS / "simple_unet_V0" / "simple_unet_airfoil.pt"),
        "improved_checkpoint": sha256(MODELS / "simple_unet_improved" / "simple_unet_airfoil.pt"),
        "fixed_test_split": sha256(MODELS / "simple_unet" / "fixed_test_split.json"),
    }
    payload = {"robustness": robustness, "synergy": synergy, "sha256": hashes}
    (OUT / "ablation_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(table.to_string())
    print("\nImprovement vs baseline (%; R2 is absolute delta):")
    print(improvement.to_string())
    print("\nArtifacts written to", OUT)


if __name__ == "__main__":
    main()
