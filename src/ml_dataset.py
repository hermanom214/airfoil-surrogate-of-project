# PyTorch Dataset for airfoil CFD flow-field data.
# Reads pre-processed .npz files (grid coordinates, pressure, velocity, fluid mask),
# parses flow conditions (angle of attack, inlet velocity) from the filename,
# normalises all fields, and returns (input tensor, target tensor, mask tensor)
# ready for training the surrogate model.

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ml_quality_filter import load_excluded_cases
from src.evaluate_io import evaluate_parse_params_json
from src.ml_supervised import (
    NONDIMENSIONAL_TARGETS,
    RAW_TARGETS,
    SupervisedLossConfig,
    physical_to_targets,
    near_wall_region_mask,
    wake_region_mask,
)


def parse_case_params(filename: str) -> tuple[float, float] | None:
    """Extract angle of attack [deg] and inlet velocity [m/s] from a case filename.

    Filename convention: ..._aoaXpY_uApB_... where XpY encodes X.Y (e.g. aoa2p5 -> 2.5 deg).
    Returns None if either parameter is missing.
    """
    aoa_match = re.search(r"aoa([m-]?\d+)p(\d+)", filename)
    u_match = re.search(r"u(\d+)p(\d+)", filename)

    if not aoa_match or not u_match:
        return None

    aoa_int = aoa_match.group(1)
    # Support both aoa-4p0 and aoam4p0 notations for negative AoA.
    if aoa_int.startswith("m"):
        aoa_int = f"-{aoa_int[1:]}"
    aoa = float(f"{aoa_int}.{aoa_match.group(2)}")
    inlet_u = float(f"{u_match.group(1)}.{u_match.group(2)}")

    return aoa, inlet_u


class AirfoilFlowDataset(Dataset):
    """Dataset that loads inspected flow-field NPZ files from case directories."""

    def __init__(
        self,
        data_dir: Path,
        supervised_config: SupervisedLossConfig | None = None,
        return_regions: bool = False,
    ):
        # Collect and sort all flow-field files in the given directory
        all_files = sorted(data_dir.glob("*/*.npz"))
        excluded_cases = load_excluded_cases(data_dir)
        self.files: list[Path] = []
        self.sample_params: list[tuple[float, float]] = []
        self.aoa_mean = 0.0
        self.aoa_std = 1.0
        self.inlet_u_mean = 0.0
        self.inlet_u_std = 1.0
        self.p_mean = 0.0
        self.p_std = 1.0
        self.ux_mean = 0.0
        self.ux_std = 1.0
        self.uy_mean = 0.0
        self.uy_std = 1.0
        self.supervised_config = supervised_config or SupervisedLossConfig()
        self.return_regions = return_regions
        self._packed_wall_masks: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}

        for path in all_files:
            if path.parent.name in excluded_cases:
                print(
                    f"[INFO] Excluding nOK case from dataset: {path.parent.name} | "
                    f"{excluded_cases[path.parent.name]}"
                )
                continue
            params = parse_case_params(path.name)
            if params is None:
                print(
                    "[WARN] Excluding file without required aoa/u in filename: "
                    f"{path.name}"
                )
                continue
            self.files.append(path)
            self.sample_params.append(params)

        if not all_files:
            raise RuntimeError(f"No flow-field NPZ files found in case directories under {data_dir}")

        if not self.files:
            raise RuntimeError(
                "No valid flow-field NPZ files left after quality and filename filtering "
                f"in {data_dir}"
            )

        # Default to stats computed on all available samples; caller can override
        # this with training-only indices after train/val split.
        self.fit_condition_normalization()
        self.fit_target_normalization()

    def fit_condition_normalization(self, indices: Sequence[int] | None = None) -> None:
        """Fit AoA and inlet-U normalisation from selected sample indices.

        Args:
            indices: Dataset indices used to estimate mean/std. If None, all
                     currently valid dataset samples are used.
        """
        if indices is None:
            selected = self.sample_params
        else:
            selected = [self.sample_params[i] for i in indices]

        if not selected:
            raise RuntimeError("Cannot fit condition normalization on an empty index set")

        aoa_values = np.array([p[0] for p in selected], dtype=np.float32)
        inlet_values = np.array([p[1] for p in selected], dtype=np.float32)

        self.aoa_mean = float(aoa_values.mean())
        self.aoa_std = float(aoa_values.std() + 1e-8)
        self.inlet_u_mean = float(inlet_values.mean())
        self.inlet_u_std = float(inlet_values.std() + 1e-8)

    def fit_target_normalization(self, indices: Sequence[int] | None = None) -> None:
        """Fit p/Ux/Uy normalisation from selected sample indices.

        Statistics are estimated on fluid cells only (where fluid_mask > 0.5).

        Args:
            indices: Dataset indices used to estimate mean/std. If None, all
                     currently valid dataset samples are used.
        """
        supervised_config = getattr(self, "supervised_config", SupervisedLossConfig())
        if supervised_config.target_representation == NONDIMENSIONAL_TARGETS:
            self.p_mean = self.ux_mean = self.uy_mean = 0.0
            self.p_std = self.ux_std = self.uy_std = 1.0
            return

        stats = self.compute_raw_target_normalization(indices)
        for name, value in stats.items():
            setattr(self, name, value)

    def compute_raw_target_normalization(
        self, indices: Sequence[int] | None = None,
    ) -> dict[str, float]:
        """Return train-subset raw p/U statistics, independent of target mode."""

        if indices is None:
            selected_indices = list(range(len(self.files)))
        else:
            selected_indices = list(indices)

        if not selected_indices:
            raise RuntimeError("Cannot fit target normalization on an empty index set")

        p_sum = p_sumsq = 0.0
        ux_sum = ux_sumsq = 0.0
        uy_sum = uy_sumsq = 0.0
        count = 0

        for i in selected_indices:
            with np.load(self.files[i], allow_pickle=False) as data:
                p = data["p"].astype(np.float32)
                U = data["U"].astype(np.float32)
                mask = data["fluid_mask"].astype(np.float32) > 0.5

            p_vals = p[mask]
            ux_vals = U[:, :, 0][mask]
            uy_vals = U[:, :, 1][mask]

            p_sum += float(p_vals.sum())
            p_sumsq += float((p_vals ** 2).sum())
            ux_sum += float(ux_vals.sum())
            ux_sumsq += float((ux_vals ** 2).sum())
            uy_sum += float(uy_vals.sum())
            uy_sumsq += float((uy_vals ** 2).sum())
            count += int(mask.sum())

        if count == 0:
            raise RuntimeError("Cannot fit target normalization: no fluid cells found")

        p_mean = p_sum / count
        ux_mean = ux_sum / count
        uy_mean = uy_sum / count

        p_var = max(p_sumsq / count - p_mean ** 2, 0.0)
        ux_var = max(ux_sumsq / count - ux_mean ** 2, 0.0)
        uy_var = max(uy_sumsq / count - uy_mean ** 2, 0.0)

        return {
            "p_mean": float(p_mean),
            "p_std": float(np.sqrt(p_var) + 1e-8),
            "ux_mean": float(ux_mean),
            "ux_std": float(np.sqrt(ux_var) + 1e-8),
            "uy_mean": float(uy_mean),
            "uy_std": float(np.sqrt(uy_var) + 1e-8),
        }

    def __len__(self) -> int:
        # Return total number of flow-field samples
        return len(self.files)

    def __getitem__(self, idx: int):
        """Load one sample and return (input, target, mask) tensors.

        Input channels  : fluid mask, normalised x, normalised y, AoA channel, inlet-U channel.
        Target channels : normalised pressure p, normalised Ux, normalised Uy.
        Mask            : fluid_mask with an added channel dimension (1, H, W).
        """
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as data:
            # Load raw arrays from the .npz file and copy into memory before closing NPZ.
            xy = data["xy"].astype(np.float32)
            p = data["p"].astype(np.float32)
            U = data["U"].astype(np.float32)
            mask = data["fluid_mask"].astype(np.float32)
            params_meta = evaluate_parse_params_json(data["params"] if "params" in data.files else None)

        # Parse flow conditions from the filename
        params = parse_case_params(path.name)
        if params is None:
            raise RuntimeError(
                f"Missing aoa/u in filename for sample unexpectedly present in dataset: {path.name}"
            )
        aoa, inlet_u = params

        # Separate and normalise grid coordinates (zero-mean, unit-std)
        x = xy[:, :, 0]
        y = xy[:, :, 1]
        x_norm = (x - x.mean()) / (x.std() + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)

        # Create constant channels encoding flow conditions (normalised using
        # statistics fitted on the training split).
        aoa_channel = np.full_like(mask, (aoa - self.aoa_mean) / self.aoa_std)
        inlet_channel = np.full_like(
            mask, (inlet_u - self.inlet_u_mean) / self.inlet_u_std
        )

        # Stack all input channels into shape (5, H, W)
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

        target = physical_to_targets(
            p, U[:, :, 0], U[:, :, 1], inlet_u,
            self.supervised_config.target_representation,
            self.supervised_config.p_inf,
        )
        if self.supervised_config.target_representation == RAW_TARGETS:
            target = (target - np.asarray(
                [self.p_mean, self.ux_mean, self.uy_mean], dtype=np.float32
            )[:, None, None]) / np.asarray(
                [self.p_std, self.ux_std, self.uy_std], dtype=np.float32
            )[:, None, None]

        base = (
            torch.from_numpy(inp),
            torch.from_numpy(target),
            torch.from_numpy(mask[None, :, :]),  # mask: (1, H, W)
        )
        if not self.return_regions:
            return base

        chord = float(params_meta.get("chord", 1.0))
        naca_match = re.search(r"naca(\d{4})", path.parent.name)
        naca_code = str(params_meta.get("naca_code") or (naca_match.group(1) if naca_match else ""))
        if not naca_code:
            raise RuntimeError(f"Cannot determine NACA code for {path}")
        if idx not in self._packed_wall_masks:
            wall = near_wall_region_mask(
                xy, mask, chord, self.supervised_config.wall_distance_fraction
            )
            self._packed_wall_masks[idx] = (np.packbits(wall.reshape(-1)), wall.shape)
        packed, shape = self._packed_wall_masks[idx]
        wall = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
        wake = wake_region_mask(
            xy, mask, chord, aoa, self.supervised_config.wake_x_start,
            self.supervised_config.wake_x_end, self.supervised_config.wake_half_height,
        )
        outer = (mask > 0.5) & ~wall & ~wake
        regions = np.stack((wall, wake, outer), axis=0).astype(np.float32)
        return (*base, torch.from_numpy(regions))
