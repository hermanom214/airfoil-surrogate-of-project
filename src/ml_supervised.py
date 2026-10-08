from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


RAW_TARGETS = "raw"
NONDIMENSIONAL_TARGETS = "nondimensional_perturbation"
TARGET_REPRESENTATIONS = {RAW_TARGETS, NONDIMENSIONAL_TARGETS}


@dataclass(frozen=True)
class SupervisedLossConfig:
    target_representation: str = RAW_TARGETS
    wall_weight: float = 0.0
    wake_weight: float = 0.0
    wall_distance_fraction: float = 0.05
    wake_x_start: float = 0.9
    wake_x_end: float = 1.75
    wake_half_height: float = 0.20
    gradient_loss_weight: float = 0.0
    gradient_channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    p_inf: float = 0.0

    def __post_init__(self) -> None:
        if self.target_representation not in TARGET_REPRESENTATIONS:
            raise ValueError(f"Unsupported target representation: {self.target_representation}")
        if min(self.wall_weight, self.wake_weight, self.wall_distance_fraction,
               self.wake_half_height, self.gradient_loss_weight) < 0.0:
            raise ValueError("Supervised loss weights and region sizes must be non-negative")
        if self.wake_x_end <= self.wake_x_start:
            raise ValueError("wake_x_end must be greater than wake_x_start")
        if len(self.gradient_channel_weights) != 3:
            raise ValueError("gradient_channel_weights must contain three values")

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def supervised_config_from_checkpoint(
    checkpoint: Mapping[str, object] | None,
) -> SupervisedLossConfig:
    """Decode loss metadata, treating absent metadata as the legacy raw-target mode."""
    if not checkpoint:
        return SupervisedLossConfig()
    representation = str(checkpoint.get("target_representation", RAW_TARGETS))
    metadata = checkpoint.get("supervised_loss_config")
    if not isinstance(metadata, Mapping):
        return SupervisedLossConfig(target_representation=representation)
    return SupervisedLossConfig(**dict(metadata))


def normalize_physical_targets(
    p: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    stats: Mapping[str, float],
) -> np.ndarray:
    """Normalize decoded physical fields with one common raw-field convention."""
    return np.stack((
        (p - float(stats["p_mean"])) / float(stats["p_std"]),
        (ux - float(stats["ux_mean"])) / float(stats["ux_std"]),
        (uy - float(stats["uy_mean"])) / float(stats["uy_std"]),
    )).astype(np.float32)


def physical_to_targets(
    p: np.ndarray, ux: np.ndarray, uy: np.ndarray, inlet_velocity: float,
    representation: str, p_inf: float = 0.0,
) -> np.ndarray:
    if representation == RAW_TARGETS:
        return np.stack((p, ux, uy), axis=0).astype(np.float32)
    if representation != NONDIMENSIONAL_TARGETS or inlet_velocity <= 0.0:
        raise ValueError("Invalid target representation or inlet velocity")
    return np.stack((
        (p - p_inf) / (0.5 * inlet_velocity**2),
        (ux - inlet_velocity) / inlet_velocity,
        uy / inlet_velocity,
    ), axis=0).astype(np.float32)


def targets_to_physical(
    targets: np.ndarray, inlet_velocity: float, representation: str, p_inf: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(targets)
    if values.shape[0] != 3:
        raise ValueError("targets must have three channels")
    if representation == RAW_TARGETS:
        return values[0], values[1], values[2]
    if representation != NONDIMENSIONAL_TARGETS or inlet_velocity <= 0.0:
        raise ValueError("Invalid target representation or inlet velocity")
    return (
        values[0] * (0.5 * inlet_velocity**2) + p_inf,
        inlet_velocity * (1.0 + values[1]),
        inlet_velocity * values[2],
    )


def body_coordinates(xy: np.ndarray, chord: float, aoa_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Undo the project's -AoA geometry rotation about quarter chord."""
    x, y = np.asarray(xy)[..., 0], np.asarray(xy)[..., 1]
    cx = 0.25 * chord
    angle = np.deg2rad(aoa_deg)
    dx = x - cx
    return cx + dx * np.cos(angle) - y * np.sin(angle), dx * np.sin(angle) + y * np.cos(angle)


def wake_region_mask(
    xy: np.ndarray, fluid_mask: np.ndarray, chord: float, aoa_deg: float,
    x_start: float, x_end: float, half_height: float,
) -> np.ndarray:
    xb, yb = body_coordinates(xy, chord, aoa_deg)
    return ((fluid_mask > 0.5) & (xb / chord >= x_start) & (xb / chord <= x_end)
            & (np.abs(yb) / chord <= half_height))


def near_wall_region_mask(
    xy: np.ndarray, fluid_mask: np.ndarray, chord: float, distance_fraction: float,
) -> np.ndarray:
    """Deterministic physical wall distance from the stored geometry-derived raster mask."""
    fluid = np.asarray(fluid_mask) > 0.5
    dx = float(np.median(np.abs(np.diff(np.asarray(xy)[0, :, 0]))))
    dy = float(np.median(np.abs(np.diff(np.asarray(xy)[:, 0, 1]))))
    distance = distance_transform_edt(fluid, sampling=(dy, dx))
    return fluid & (distance > 0.0) & (distance / chord <= distance_fraction)


def build_region_weights(
    fluid_mask: torch.Tensor, wall_mask: torch.Tensor, wake_mask: torch.Tensor,
    wall_weight: float, wake_weight: float,
) -> torch.Tensor:
    return fluid_mask * (1.0 + wall_weight * wall_mask + wake_weight * wake_mask)


def gradient_matching_loss(
    pred: torch.Tensor, target: torch.Tensor, fluid_mask: torch.Tensor,
    dx: float, dy: float, channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    base = fluid_mask[:, 0] > 0.5
    valid = (base[:, 1:-1, 1:-1] & base[:, 1:-1, :-2] & base[:, 1:-1, 2:]
             & base[:, :-2, 1:-1] & base[:, 2:, 1:-1])
    if not torch.any(valid):
        return pred.sum() * 0.0
    dpdx = (pred[:, :, 1:-1, 2:] - pred[:, :, 1:-1, :-2]) / (2.0 * dx)
    dtdx = (target[:, :, 1:-1, 2:] - target[:, :, 1:-1, :-2]) / (2.0 * dx)
    dpdy = (pred[:, :, 2:, 1:-1] - pred[:, :, :-2, 1:-1]) / (2.0 * dy)
    dtdy = (target[:, :, 2:, 1:-1] - target[:, :, :-2, 1:-1]) / (2.0 * dy)
    weights = pred.new_tensor(channel_weights).view(1, 3, 1, 1)
    squared = weights * ((dpdx - dtdx).square() + (dpdy - dtdy).square())
    valid4 = valid[:, None].expand_as(squared)
    return squared[valid4].mean()
