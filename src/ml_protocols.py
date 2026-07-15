from __future__ import annotations

from typing import Protocol

import torch


class PhysicsLossModel(Protocol):
    def rans_residual_loss(
        self,
        pred: torch.Tensor,
        fluid_mask: torch.Tensor,
        dx: float,
        dy: float,
        nu: float,
        p_mean: float,
        p_std: float,
        ux_mean: float,
        ux_std: float,
        uy_mean: float,
        uy_std: float,
        nu_t: torch.Tensor | None = None,
        pressure_is_kinematic: bool = True,
        mask_erode_pixels: int = 1,
    ) -> dict[str, torch.Tensor]:
        ...