from __future__ import annotations

import torch


SUPPORTED_DEVICES = ("auto", "cpu", "cuda")


def resolve_device(requested: str) -> torch.device:
    """Resolve a configured device without silently weakening explicit CUDA requests."""
    value = str(requested).strip().lower()
    if value not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device {requested!r}; expected one of: {', '.join(SUPPORTED_DEVICES)}"
        )
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Device 'cuda' was requested, but torch.cuda.is_available() is False. "
            "Install/use a CUDA-enabled PyTorch build and verify the NVIDIA driver."
        )
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def log_device(requested: str, device: torch.device) -> None:
    print(f"Device requested: {requested}")
    print(f"Device selected: {device.type}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'not available'}")


def loader_device_kwargs(device: torch.device) -> dict[str, bool]:
    """Pinned host memory benefits host-to-GPU batch copies, but not CPU runs."""
    return {"pin_memory": device.type == "cuda"}
