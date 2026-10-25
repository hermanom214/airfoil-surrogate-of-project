"""Reproducibility primitives for the controlled Phase-A U-Net A/B experiment."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


AB_EXPERIMENTS = ("unet_pinn_supervised", "unet_pinn_physics")
ALLOWED_DIFFERENCES = {"experiment_name", "output_name", "loss_weights", "pilot"}


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def membership_sha256(case_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, case_ids))).encode()).hexdigest()


def dataset_artifact_sha256(files: Sequence[Path], root: Path) -> str:
    """Hash paths and bytes of every immutable case artifact."""
    digest = hashlib.sha256()
    for path in sorted(map(Path, files)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def environment_metadata(device: torch.device) -> dict[str, Any]:
    total_vram = None
    if device.type == "cuda":
        total_vram = int(torch.cuda.get_device_properties(device).total_memory)
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": total_vram,
        "gpu_total_memory_mib": total_vram / (1024**2) if total_vram is not None else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def validate_finite_gradients(model: torch.nn.Module) -> int:
    """Fail before optimizer.step if any existing parameter gradient is NaN or Inf."""
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        checked += 1
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient detected for parameter: {name}")
    return checked


def cuda_peak_memory(device: torch.device) -> dict[str, int | float | None]:
    """Return synchronized CUDA peaks, or explicit unavailable values on CPU."""
    if device.type != "cuda":
        return {
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_allocated_mib": None,
            "peak_cuda_memory_reserved_bytes": None,
            "peak_cuda_memory_reserved_mib": None,
        }
    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "peak_cuda_memory_allocated_bytes": allocated,
        "peak_cuda_memory_allocated_mib": allocated / (1024**2),
        "peak_cuda_memory_reserved_bytes": reserved,
        "peak_cuda_memory_reserved_mib": reserved / (1024**2),
    }


def compare_metadata(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    required = ("dataset_fingerprint", "fixed_test_fingerprint", "train_fingerprint",
                "validation_fingerprint", "initial_state_hash", "model_config",
                "training_config", "selection_metric")
    problems = [f"missing {key}" for key in required if key not in left or key not in right]
    for key in required:
        if key in left and key in right and left[key] != right[key]:
            problems.append(f"{key} differs")
    lw, rw = left.get("loss_weights", {}), right.get("loss_weights", {})
    if lw.get("supervised_weight") != rw.get("supervised_weight"):
        problems.append("supervised_weight differs")
    return problems


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
