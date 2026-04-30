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