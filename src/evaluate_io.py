from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


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


def sanitize_for_json(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}

    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]

    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return sanitize_for_json(value.item())
        return [sanitize_for_json(v) for v in value.tolist()]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        scalar = float(value)
        if not np.isfinite(scalar):
            return None
        return scalar

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    if isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.str_):
        return str(value)

    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            pass

    return value
