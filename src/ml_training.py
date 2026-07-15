from __future__ import annotations

import torch


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