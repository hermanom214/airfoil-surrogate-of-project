from __future__ import annotations

import torch
from typing import cast

from src.ml_protocols import PhysicsLossModel


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / (
        mask.sum() * pred.shape[1] + 1e-8
    )


def compute_grid_spacing_from_xy(xy: torch.Tensor) -> tuple[float, float]:
    """Infer representative dx/dy from a structured xy grid tensor [H, W, 2]."""
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("xy grid tensor must have shape [H, W, 2]")

    x = xy[:, :, 0]
    y = xy[:, :, 1]

    dx_candidates = torch.abs(x[:, 1:] - x[:, :-1]).reshape(-1)
    dy_candidates = torch.abs(y[1:, :] - y[:-1, :]).reshape(-1)

    dx_valid = dx_candidates[dx_candidates > 0]
    dy_valid = dy_candidates[dy_candidates > 0]

    if dx_valid.numel() == 0 or dy_valid.numel() == 0:
        raise RuntimeError("Unable to infer positive dx/dy from xy grid")

    dx = float(dx_valid.median().item())
    dy = float(dy_valid.median().item())
    return dx, dy


def train_one_epoch(model, loader, optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0

    for inp, target, mask in loader:
        inp = inp.to(device)
        target = target.to(device)
        mask = mask.to(device)

        pred = model(inp)
        loss = masked_mse(pred, target, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device: str) -> float:
    model.eval()
    total_loss = 0.0

    for inp, target, mask in loader:
        inp = inp.to(device)
        target = target.to(device)
        mask = mask.to(device)

        pred = model(inp)
        loss = masked_mse(pred, target, mask)

        total_loss += loss.item()

    return total_loss / len(loader)


def train_one_epoch_physics(
    model,
    loader,
    optimizer,
    device: str,
    dx: float,
    dy: float,
    nu: float,
    p_mean: float,
    p_std: float,
    ux_mean: float,
    ux_std: float,
    uy_mean: float,
    uy_std: float,
    physics_weight: float,
    pressure_is_kinematic: bool,
    mask_erode_pixels: int,
) -> dict[str, float]:
    if not hasattr(model, "rans_residual_loss"):
        raise TypeError("Selected model does not implement rans_residual_loss")

    physics_model = cast(PhysicsLossModel, model)
    model.train()

    totals = {
        "loss_total": 0.0,
        "loss_data": 0.0,
        "loss_physics": 0.0,
        "loss_continuity": 0.0,
        "loss_momentum_x": 0.0,
        "loss_momentum_y": 0.0,
    }

    for inp, target, mask in loader:
        inp = inp.to(device)
        target = target.to(device)
        mask = mask.to(device)

        pred = model(inp)

        loss_data = masked_mse(pred, target, mask)
        phys = physics_model.rans_residual_loss(
            pred=pred,
            fluid_mask=mask,
            dx=dx,
            dy=dy,
            nu=nu,
            p_mean=p_mean,
            p_std=p_std,
            ux_mean=ux_mean,
            ux_std=ux_std,
            uy_mean=uy_mean,
            uy_std=uy_std,
            pressure_is_kinematic=pressure_is_kinematic,
            mask_erode_pixels=mask_erode_pixels,
        )
        loss_total = loss_data + physics_weight * phys["physics_loss"]

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        totals["loss_total"] += float(loss_total.item())
        totals["loss_data"] += float(loss_data.item())
        totals["loss_physics"] += float(phys["physics_loss"].item())
        totals["loss_continuity"] += float(phys["continuity_loss"].item())
        totals["loss_momentum_x"] += float(phys["momentum_x_loss"].item())
        totals["loss_momentum_y"] += float(phys["momentum_y_loss"].item())

    n_batches = len(loader)
    if n_batches == 0:
        raise RuntimeError("Training loader has zero batches")

    return {k: v / n_batches for k, v in totals.items()}