from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_case_params(filename: str) -> tuple[float, float]:
    aoa_match = re.search(r"aoa(-?\d+)p(\d+)", filename)
    u_match = re.search(r"u(\d+)p(\d+)", filename)

    aoa = 0.0
    inlet_u = 20.0

    if aoa_match:
        aoa = float(f"{aoa_match.group(1)}.{aoa_match.group(2)}")

    if u_match:
        inlet_u = float(f"{u_match.group(1)}.{u_match.group(2)}")

    return aoa, inlet_u


class AirfoilFlowDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.files = sorted(data_dir.glob("*_flow.npz"))

        if not self.files:
            raise RuntimeError(f"No *_flow.npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path)

        xy = data["xy"].astype(np.float32)
        p = data["p"].astype(np.float32)
        U = data["U"].astype(np.float32)
        mask = data["fluid_mask"].astype(np.float32)

        aoa, inlet_u = parse_case_params(path.name)

        x = xy[:, :, 0]
        y = xy[:, :, 1]

        x_norm = (x - x.mean()) / (x.std() + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)

        aoa_channel = np.full_like(mask, aoa / 20.0)
        inlet_channel = np.full_like(mask, inlet_u / 50.0)

        inp = np.stack(
            [
                mask,
                x_norm,
                y_norm,
                aoa_channel,
                inlet_channel,
            ],
            axis=0,
        )

        target = np.stack(
            [
                p / 1000.0,
                U[:, :, 0] / 50.0,
                U[:, :, 1] / 50.0,
            ],
            axis=0,
        )

        return (
            torch.tensor(inp),
            torch.tensor(target),
            torch.tensor(mask[None, :, :]),
        )