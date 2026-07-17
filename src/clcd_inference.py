from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect

import numpy as np
import torch

from src.ml_models import build_model, ClCdMLP, PhysicsInformedCNN, SimpleUNet


def _as_float_vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError(f"Checkpoint field '{name}' is empty")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"Checkpoint field '{name}' contains non-finite values")
    return arr


def load_clcd_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location=device)

    if not isinstance(raw, dict):
        raise RuntimeError(
            "Checkpoint must be a dictionary containing model weights and normalization stats"
        )

    if "model_state_dict" not in raw:
        raise RuntimeError("Checkpoint missing required key: model_state_dict")

    required_norm_keys = ["feature_mean", "feature_std", "target_mean", "target_std"]
    for key in required_norm_keys:
        if key not in raw:
            raise RuntimeError(f"Checkpoint missing required key: {key}")

    feature_mean = _as_float_vector(raw["feature_mean"], "feature_mean")
    feature_std = _as_float_vector(raw["feature_std"], "feature_std")
    target_mean = _as_float_vector(raw["target_mean"], "target_mean")
    target_std = _as_float_vector(raw["target_std"], "target_std")

    if np.any(feature_std <= 1e-8):
        raise RuntimeError("Invalid checkpoint feature_std (<= 1e-8)")
    if np.any(target_std <= 1e-8):
        raise RuntimeError("Invalid checkpoint target_std (<= 1e-8)")

    checkpoint: dict[str, Any] = dict(raw)
    checkpoint["model_name"] = str(raw.get("model_name", "clcd_mlp"))
    checkpoint["feature_mean"] = feature_mean
    checkpoint["feature_std"] = feature_std
    checkpoint["target_mean"] = target_mean
    checkpoint["target_std"] = target_std
    checkpoint["feature_order"] = list(raw.get("feature_order", []))
    checkpoint["target_order"] = list(raw.get("target_order", []))

    return checkpoint


def build_clcd_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: str | torch.device = "cpu",
    fallback_model_config: dict[str, Any] | None = None,
) -> torch.nn.Module:
    model_name = str(checkpoint.get("model_name", "clcd_mlp"))
    model_config = checkpoint.get("model_config", None)
    if model_config is None:
        model_config = fallback_model_config

    if model_config is None:
        raise RuntimeError(
            "Checkpoint does not contain model_config and no fallback_model_config was provided"
        )

    cfg = dict(model_config)
    model_ctor_by_name = {
        "clcd_mlp": ClCdMLP,
        "simple_unet": SimpleUNet,
        "rans_pinn": PhysicsInformedCNN,
    }
    if model_name not in model_ctor_by_name:
        raise RuntimeError(f"Unsupported model_name in checkpoint: {model_name}")

    signature = inspect.signature(model_ctor_by_name[model_name].__init__)
    valid_params = {
        name
        for name in signature.parameters
        if name not in {"self", "args", "kwargs"}
    }
    model_kwargs = {k: v for k, v in cfg.items() if k in valid_params}

    model = build_model(model_name, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def normalize_features(
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape[1] != int(feature_mean.shape[0]):
        raise RuntimeError(
            f"Feature dimension mismatch: got {x.shape[1]}, expected {int(feature_mean.shape[0])}"
        )
    if not np.isfinite(x).all():
        raise RuntimeError("Features contain non-finite values")

    x_norm = (x - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)
    if not np.isfinite(x_norm).all():
        raise RuntimeError("Normalized features contain non-finite values")
    return x_norm.astype(np.float32)


def denormalize_outputs(
    outputs_norm: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    y_norm = np.asarray(outputs_norm, dtype=np.float32)
    if y_norm.ndim == 1:
        y_norm = y_norm.reshape(1, -1)

    if y_norm.shape[1] != int(target_mean.shape[0]):
        raise RuntimeError(
            f"Output dimension mismatch: got {y_norm.shape[1]}, expected {int(target_mean.shape[0])}"
        )

    y = y_norm * target_std.reshape(1, -1) + target_mean.reshape(1, -1)
    if not np.isfinite(y).all():
        raise RuntimeError("Denormalized outputs contain non-finite values")
    return y.astype(np.float32)


@torch.no_grad()
def predict_clcd(
    model: torch.nn.Module,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    x_norm = normalize_features(features, feature_mean, feature_std)
    x_tensor = torch.from_numpy(x_norm).to(device)

    pred_norm_tensor = model(x_tensor)
    if not torch.isfinite(pred_norm_tensor).all():
        raise RuntimeError("Model produced non-finite normalized predictions")

    pred_norm = pred_norm_tensor.detach().cpu().numpy().astype(np.float32)
    pred = denormalize_outputs(pred_norm, target_mean, target_std)
    return pred, pred_norm
