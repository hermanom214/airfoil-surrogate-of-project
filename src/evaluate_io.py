from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import random_split


def evaluate_parse_params_json(params_raw: Any) -> dict[str, Any]:
    if params_raw is None:
        return {}

    if isinstance(params_raw, np.ndarray):
        if params_raw.shape == ():
            params_raw = params_raw.item()
        else:
            try:
                params_raw = params_raw.tolist()
            except Exception:
                return {}

    if isinstance(params_raw, bytes):
        text = params_raw.decode("utf-8", errors="replace")
    elif isinstance(params_raw, str):
        text = params_raw
    else:
        text = str(params_raw)

    text = text.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return parsed
    return {}


def evaluate_fallback_naca_from_filename(filename: str) -> str:
    match = re.search(r"naca(\d{4})", filename)
    if match:
        return match.group(1)
    return "unknown"


def evaluate_npz_scalar_to_str(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            value = value.tolist()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def evaluate_extract_case_meta(npz_data: Any, npz_path: Any, parsed_params: tuple[float, float] | None) -> dict[str, Any]:
    npz_keys = set(npz_data.files) if hasattr(npz_data, "files") else set()
    params_raw = npz_data["params"] if "params" in npz_keys else None
    params = evaluate_parse_params_json(params_raw)

    case_id = str(params.get("case_id", "")).strip()
    if not case_id:
        if "case_id" in npz_keys:
            case_id = evaluate_npz_scalar_to_str(npz_data["case_id"])
        else:
            case_id = npz_path.stem.replace("_flow", "")

    naca_code = str(params.get("naca_code", "")).strip() or evaluate_fallback_naca_from_filename(npz_path.name)
    chord = float(params.get("chord", 1.0))

    if "aoa_deg" in params:
        aoa_deg = float(params["aoa_deg"])
    elif parsed_params is not None:
        aoa_deg = float(parsed_params[0])
    else:
        aoa_deg = 0.0

    if "inlet_velocity" in params:
        inlet_velocity = float(params["inlet_velocity"])
    elif parsed_params is not None:
        inlet_velocity = float(parsed_params[1])
    else:
        inlet_velocity = 0.0

    latest_time = evaluate_npz_scalar_to_str(npz_data["latest_time"]) if "latest_time" in npz_keys else ""

    return {
        "params": params,
        "case_id": case_id,
        "naca_code": naca_code,
        "chord": chord,
        "aoa_deg": aoa_deg,
        "inlet_velocity": inlet_velocity,
        "latest_time": latest_time,
    }


def evaluate_get_model_kwargs(model_name: str, ml_cfg: Any) -> dict[str, Any]:
    if model_name == "simple_unet":
        return asdict(ml_cfg.model.params.simple_unet)
    if model_name == "rans_pinn":
        return asdict(ml_cfg.model.params.rans_pinn)
    raise ValueError(f"Unsupported model in config: {model_name}")


def evaluate_build_eval_split(dataset: Any, ml_cfg: Any) -> tuple[Any, Any, list[int], int, int]:
    train_size = max(1, int((1.0 - ml_cfg.training.validation_split) * len(dataset)))
    val_size = len(dataset) - train_size

    if val_size == 0:
        print("[WARN] Validation split produced 0 samples. Reusing full dataset for validation.")
        train_set = dataset
        val_set = dataset
        train_indices = list(range(len(dataset)))
        return train_set, val_set, train_indices, train_size, val_size

    generator = torch.Generator().manual_seed(ml_cfg.training.split_seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    if hasattr(train_set, "indices"):
        train_indices = list(train_set.indices)
    else:
        train_indices = list(range(len(dataset)))

    return train_set, val_set, train_indices, train_size, val_size


def evaluate_compute_split_indices(
    dataset_len: int,
    validation_split: float,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic train/val indices with the same logic as training script."""
    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0")

    train_size = max(1, int((1.0 - validation_split) * dataset_len))
    val_size = dataset_len - train_size

    if val_size == 0:
        all_idx = list(range(dataset_len))
        return all_idx, all_idx

    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    train_idx = permutation[:train_size]
    val_idx = permutation[train_size:]
    return train_idx, val_idx


def evaluate_get_original_dataset_index(val_set: Any, local_idx: int) -> int:
    if hasattr(val_set, "indices"):
        return int(val_set.indices[local_idx])
    return int(local_idx)
