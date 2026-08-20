from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.ml_data_split import CVFold, DevelopmentTestSplit
from src.ml_hyperparameter_search import (
    SearchConfiguration,
    serialize_hyperparameters,
    summarize_cv_results,
)


FoldRunner = Callable[[Mapping[str, Any], CVFold], Mapping[str, float]]


def experiment_mode(cv_enabled: bool, search_strategy: str) -> str:
    """Keep CV and search semantics independent and reject ambiguous combinations."""
    strategy = search_strategy.lower().strip()
    if not cv_enabled:
        if strategy != "none":
            raise ValueError("Hyperparameter search requires cv_enabled=true")
        return "single_run"
    return "cv_without_search" if strategy == "none" else "search_with_cv"


def run_cross_validation(
    configurations: Sequence[SearchConfiguration], folds: Sequence[CVFold],
    run_fold: FoldRunner,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run candidates/folds and rank by normalized validation objective (lower is better)."""
    results: list[dict[str, Any]] = []
    for candidate in configurations:
        fold_results: list[dict[str, Any]] = []
        objectives: list[float] = []
        for fold in folds:
            metrics = {str(k): float(v) for k, v in run_fold(candidate.hyperparameters, fold).items()}
            if "objective" not in metrics:
                raise ValueError("Fold runner must return an 'objective' metric")
            objectives.append(metrics["objective"])
            fold_results.append({"fold_id": fold.fold_id, **metrics})
        mean, std = summarize_cv_results(objectives)
        results.append({
            "config_id": candidate.config_id,
            "hyperparameters": dict(candidate.hyperparameters),
            "fold_metrics": fold_results,
            "mean_cv_metric": mean,
            "std_cv_metric": std,
        })
    if not results:
        raise RuntimeError("Hyperparameter search generated no configurations")
    best = min(results, key=lambda row: (row["mean_cv_metric"], row["config_id"]))
    return results, best


def write_experiment_results(
    output_dir: Path, *, model_name: str, split: DevelopmentTestSplit,
    folds: Sequence[CVFold], cv_strategy: str, search_strategy: str,
    sampled_configurations: Sequence[SearchConfiguration],
    cv_results: Sequence[Mapping[str, Any]], best: Mapping[str, Any],
    final_test_metrics: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_payload = asdict(split) | {
        "model_name": model_name,
        "cv_strategy": cv_strategy,
        "n_folds": len(folds),
        "folds": [asdict(fold) for fold in folds],
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    cv_payload = {
        "model_name": model_name,
        "dataset_size": len(split.dataset_case_ids),
        "seed": split.seed,
        "cv_strategy": cv_strategy,
        "cv_label": "geometry-held-out CV" if cv_strategy == "group_kfold_naca" else "random/interpolation CV",
        "n_folds": len(folds),
        "hyperparameter_search_strategy": search_strategy,
        "sampled_hyperparameter_configurations": [asdict(v) for v in sampled_configurations],
        "results": list(cv_results),
        "best_configuration": dict(best),
    }
    (output_dir / "cv_results.json").write_text(
        json.dumps(cv_payload, indent=2), encoding="utf-8"
    )
    max_folds = len(folds)
    metric_names = sorted({
        key
        for result in cv_results
        for fold_metric in result["fold_metrics"]
        for key in fold_metric
        if key not in {"fold_id", "objective"}
    })
    fieldnames = ["config_id", "hyperparameters"] + [
        f"fold_{i}_metric" for i in range(1, max_folds + 1)
    ] + ["mean_cv_metric", "std_cv_metric"] + [f"mean_{name}" for name in metric_names]
    with (output_dir / "hyperparameter_search_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in cv_results:
            row: dict[str, Any] = {
                "config_id": result["config_id"],
                "hyperparameters": serialize_hyperparameters(result["hyperparameters"]),
                "mean_cv_metric": result["mean_cv_metric"],
                "std_cv_metric": result["std_cv_metric"],
            }
            for metric in result["fold_metrics"]:
                row[f"fold_{metric['fold_id']}_metric"] = metric["objective"]
            for name in metric_names:
                values = [float(metric[name]) for metric in result["fold_metrics"] if name in metric]
                row[f"mean_{name}"] = sum(values) / len(values) if values else ""
            writer.writerow(row)
    (output_dir / "final_test_metrics.json").write_text(
        json.dumps(dict(final_test_metrics), indent=2), encoding="utf-8"
    )
