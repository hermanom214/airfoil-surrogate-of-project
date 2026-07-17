from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.clcd_inference import (
    build_clcd_model_from_checkpoint,
    load_clcd_checkpoint,
    normalize_features,
    predict_clcd,
)
from src.config import load_ml_models_config, load_paths
from src.evaluate_clcd_metrics import (
    evaluate_build_clcd_case_row,
    evaluate_compute_clcd_global_metrics,
)
from src.evaluate_clcd_plotting import evaluate_generate_clcd_plots
from src.evaluate_io import sanitize_for_json
from src.evaluate_metrics import evaluate_summarize_numeric_columns
from src.ml_clcd_dataset import AirfoilClCdDataset
from src.ml_data_split import build_train_val_split


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    out: list[int] = []
    for value in values:
        out.append(int(value))
    return out


def _to_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    return [str(v) for v in values]


def _pick_validation_indices(
    dataset: AirfoilClCdDataset,
    checkpoint: dict[str, Any],
    validation_split: float,
    split_seed: int,
) -> tuple[list[int], list[str], int]:
    case_names = [sample[2] for sample in dataset.samples]
    name_to_index = {name: idx for idx, name in enumerate(case_names)}

    val_case_ids = _to_str_list(checkpoint.get("val_case_ids"))
    train_case_ids = _to_str_list(checkpoint.get("train_case_ids"))
    val_indices_ckpt = _to_int_list(checkpoint.get("val_indices"))
    train_indices_ckpt = _to_int_list(checkpoint.get("train_indices"))

    if val_case_ids:
        resolved = [name_to_index[c] for c in val_case_ids if c in name_to_index]
        missing = [c for c in val_case_ids if c not in name_to_index]
        if missing:
            print(
                "[WARN] Some checkpoint val_case_ids are missing in dataset and will be ignored: "
                f"{missing}"
            )
        if resolved:
            train_count = len(train_case_ids) if train_case_ids else len(train_indices_ckpt)
            return resolved, [case_names[i] for i in resolved], int(train_count)

    if val_indices_ckpt:
        resolved = [i for i in val_indices_ckpt if 0 <= i < len(dataset.samples)]
        if resolved:
            if len(resolved) != len(val_indices_ckpt):
                print("[WARN] Some checkpoint val_indices were out of range and were ignored.")
            train_count = len(train_case_ids) if train_case_ids else len(train_indices_ckpt)
            return resolved, [case_names[i] for i in resolved], int(train_count)

    print("[WARN] No validation cases in checkpoint; rebuilding split from config seed.")
    split = build_train_val_split(
        dataset=dataset,
        validation_split=validation_split,
        split_seed=split_seed,
    )
    val_indices = list(split.val_indices)
    return val_indices, [case_names[i] for i in val_indices], int(len(split.train_indices))


@torch.no_grad()
def run_clcd_validation() -> None:
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    ml_config_path = PROJECT_ROOT / "configs" / "ml_models_config.yaml"

    paths = load_paths(config_path)
    ml_cfg = load_ml_models_config(ml_config_path)

    model_name = "clcd_mlp"
    model_filename = ml_cfg.output.filename_template.format(model_name=model_name)
    checkpoint_path = paths.project_root / ml_cfg.output.models_subdir / model_filename

    validation_root = paths.project_root / ml_cfg.output.models_subdir / f"{model_name}_validation"
    metrics_dir = validation_root / "metrics"
    figures_dir = validation_root / "figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "validation_metrics_per_case.csv"
    summary_path = metrics_dir / "validation_summary.json"

    checkpoint = load_clcd_checkpoint(checkpoint_path=checkpoint_path)

    clcd_cfg = ml_cfg.model.params.clcd_mlp
    cases_root = paths.openfoam_case_sim / clcd_cfg.cases_subdir

    dataset = AirfoilClCdDataset(
        cases_root=cases_root,
        tail_window=clcd_cfg.tail_window,
        force_coeffs_relpath=clcd_cfg.force_coeffs_relpath,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Cl/Cd dataset is empty in: {cases_root}")

    model = build_clcd_model_from_checkpoint(
        checkpoint=checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fallback_model_config=asdict(ml_cfg.model.params.clcd_mlp),
    )

    val_indices, val_case_ids, train_case_count = _pick_validation_indices(
        dataset=dataset,
        checkpoint=checkpoint,
        validation_split=float(ml_cfg.training.validation_split),
        split_seed=int(ml_cfg.training.split_seed),
    )

    if not val_indices:
        raise RuntimeError("Validation split produced zero validation cases")

    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []

    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]

    for idx in val_indices:
        try:
            features, targets_true, case_id = dataset.samples[idx]
            pred_denorm, pred_norm = predict_clcd(
                model=model,
                features=features,
                feature_mean=checkpoint["feature_mean"],
                feature_std=checkpoint["feature_std"],
                target_mean=target_mean,
                target_std=target_std,
                device=next(model.parameters()).device,
            )

            y_true = np.asarray(targets_true, dtype=np.float32).reshape(1, -1)
            y_true_norm = (y_true - target_mean.reshape(1, -1)) / target_std.reshape(1, -1)
            normalized_mse = float(np.mean((pred_norm - y_true_norm) ** 2, dtype=np.float64))

            row = evaluate_build_clcd_case_row(
                case_id=case_id,
                features=features,
                true_targets=y_true.reshape(-1),
                pred_targets=pred_denorm.reshape(-1),
                normalized_mse=normalized_mse,
            )
            rows.append(row)
        except Exception as exc:
            failed_cases.append({"case_id": dataset.samples[idx][2], "error": str(exc)})
            print(f"[ERROR] Validation failed for case {dataset.samples[idx][2]}: {exc}")

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by="case_id").reset_index(drop=True)
    metrics_df.to_csv(csv_path, index=False)

    global_metrics = evaluate_compute_clcd_global_metrics(metrics_df)

    if not metrics_df.empty:
        numeric_cols = [
            "Cl_true",
            "Cl_pred",
            "Cl_error",
            "Cl_abs_error",
            "Cd_true",
            "Cd_pred",
            "Cd_error",
            "Cd_abs_error",
            "normalized_MSE",
        ]
        per_metric_stats = evaluate_summarize_numeric_columns(metrics_df, numeric_cols)

        best_row = metrics_df.loc[metrics_df["normalized_MSE"].idxmin()].to_dict()
        worst_row = metrics_df.loc[metrics_df["normalized_MSE"].idxmax()].to_dict()
    else:
        per_metric_stats = {}
        best_row = None
        worst_row = None

    summary = {
        "model_name": str(checkpoint.get("model_name", model_name)),
        "checkpoint": str(checkpoint_path),
        "validation_case_count": int(len(metrics_df)),
        "train_case_count": int(train_case_count),
        "validation_case_ids": val_case_ids,
        "feature_normalization": {
            "mean": checkpoint["feature_mean"],
            "std": checkpoint["feature_std"],
            "order": checkpoint.get("feature_order", []),
        },
        "target_normalization": {
            "mean": checkpoint["target_mean"],
            "std": checkpoint["target_std"],
            "order": checkpoint.get("target_order", []),
        },
        "global_cl_metrics": global_metrics["Cl"],
        "global_cd_metrics": global_metrics["Cd"],
        "best_case": best_row,
        "worst_case": worst_row,
        "per_case_metric_distribution": per_metric_stats,
        "failed_cases": failed_cases,
    }

    summary_path.write_text(
        json.dumps(sanitize_for_json(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    evaluate_generate_clcd_plots(metrics_df, figures_dir)

    print("[CLCD VALIDATION SUMMARY]")
    print(f"Model: {summary['model_name']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation cases: {summary['validation_case_count']}")
    print(f"Train cases: {summary['train_case_count']}")
    if summary["validation_case_count"] > 0:
        cl = summary["global_cl_metrics"]
        cd = summary["global_cd_metrics"]
        print(
            "Cl metrics | "
            f"MAE={cl['MAE']:.6e} | RMSE={cl['RMSE']:.6e} | Bias={cl['Bias']:.6e} | "
            f"MedAE={cl['Median_absolute_error']:.6e} | MaxAE={cl['Maximum_absolute_error']:.6e} | "
            f"R2={cl['R2'] if cl['R2'] is not None else 'n/a'}"
        )
        print(
            "Cd metrics | "
            f"MAE={cd['MAE']:.6e} | RMSE={cd['RMSE']:.6e} | Bias={cd['Bias']:.6e} | "
            f"MedAE={cd['Median_absolute_error']:.6e} | MaxAE={cd['Maximum_absolute_error']:.6e} | "
            f"R2={cd['R2'] if cd['R2'] is not None else 'n/a'}"
        )
        print(f"Best case (normalized_MSE): {summary['best_case']['case_id']}")
        print(f"Worst case (normalized_MSE): {summary['worst_case']['case_id']}")
    else:
        print("No successful validation cases.")

    print(f"Metrics CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Figures directory: {figures_dir}")
    print(f"Failed cases: {len(failed_cases)}")


def main() -> None:
    run_clcd_validation()


if __name__ == "__main__":
    main()
