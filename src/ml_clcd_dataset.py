from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_FORCECOEFFS_COL_COUNT = 6


def parse_case_features_from_name(case_name: str) -> tuple[float, float, float, float, float] | None:
    """Parse case features from case directory name.

    Returns tuple in order:
    (camber_percent, camber_position_tenths, thickness_percent, aoa_deg, inlet_velocity)
    """
    naca_match = re.search(r"naca(\d{4})", case_name)
    aoa_match = re.search(r"aoa([m-]?\d+)p(\d+)", case_name)
    u_match = re.search(r"u(\d+)p(\d+)", case_name)

    if not naca_match or not aoa_match or not u_match:
        return None

    naca = naca_match.group(1)
    camber_percent = float(int(naca[0]))
    camber_position_tenths = float(int(naca[1]))
    thickness_percent = float(int(naca[2:]))

    aoa_int = aoa_match.group(1)
    if aoa_int.startswith("m"):
        aoa_int = f"-{aoa_int[1:]}"
    aoa_deg = float(f"{aoa_int}.{aoa_match.group(2)}")

    inlet_velocity = float(f"{u_match.group(1)}.{u_match.group(2)}")

    return (
        camber_percent,
        camber_position_tenths,
        thickness_percent,
        aoa_deg,
        inlet_velocity,
    )


def read_cl_cd_tail_average(
    force_coeffs_path: Path,
    tail_window: int = 200,
    min_valid_rows: int = 100,
) -> tuple[float, float]:
    """Read forceCoeffs.dat and return (Cl, Cd) averages over last tail_window rows."""
    if tail_window <= 0:
        raise ValueError("tail_window must be > 0")
    if min_valid_rows <= 0:
        raise ValueError("min_valid_rows must be > 0")

    rows: list[tuple[float, float, float]] = []
    with force_coeffs_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < _FORCECOEFFS_COL_COUNT:
                continue

            try:
                time = float(parts[0])
                cd = float(parts[2])
                cl = float(parts[3])
            except ValueError:
                continue

            if not (np.isfinite(time) and np.isfinite(cl) and np.isfinite(cd)):
                continue

            rows.append((time, cl, cd))

    if not rows:
        raise RuntimeError(
            f"forcecoeffs_invalid: no finite numeric rows parsed from {force_coeffs_path}"
        )

    if len(rows) < min_valid_rows:
        raise RuntimeError(
            "too_few_valid_rows: "
            f"parsed {len(rows)} finite rows from {force_coeffs_path}, "
            f"required at least {min_valid_rows}"
        )

    tail = rows[-tail_window:] if len(rows) >= tail_window else rows
    cl_avg = float(np.mean([v[1] for v in tail], dtype=np.float64))
    cd_avg = float(np.mean([v[2] for v in tail], dtype=np.float64))
    if not (np.isfinite(cl_avg) and np.isfinite(cd_avg)):
        raise RuntimeError(
            f"forcecoeffs_invalid: non-finite tail averages in {force_coeffs_path}"
        )

    return cl_avg, cd_avg


class AirfoilClCdDataset(Dataset):
    """Dataset for scalar regression of aerodynamic coefficients [Cl, Cd]."""

    FEATURE_ORDER: tuple[str, ...] = (
        "camber_percent",
        "camber_position_tenths",
        "thickness_percent",
        "aoa_deg",
        "inlet_velocity",
    )
    TARGET_ORDER: tuple[str, ...] = ("Cl", "Cd")

    def __init__(
        self,
        cases_root: Path,
        tail_window: int = 200,
        min_valid_rows: int = 100,
        force_coeffs_relpath: str = "postProcessing/forceCoeffs1/0/forceCoeffs.dat",
    ):
        self.samples: list[tuple[np.ndarray, np.ndarray, str]] = []
        self.skipped_samples: list[dict[str, str]] = []
        self.feature_mean = np.zeros(5, dtype=np.float32)
        self.feature_std = np.ones(5, dtype=np.float32)
        self.target_mean = np.zeros(2, dtype=np.float32)
        self.target_std = np.ones(2, dtype=np.float32)

        for case_dir in sorted(cases_root.glob("case_*")):
            if not case_dir.is_dir():
                continue

            features = parse_case_features_from_name(case_dir.name)
            if features is None:
                self.skipped_samples.append(
                    {"case": case_dir.name, "reason": "cannot_parse_case_name"}
                )
                continue

            coeff_path = case_dir / force_coeffs_relpath
            if not coeff_path.exists():
                self.skipped_samples.append(
                    {"case": case_dir.name, "reason": "forcecoeffs_missing"}
                )
                continue

            try:
                cl_avg, cd_avg = read_cl_cd_tail_average(
                    coeff_path,
                    tail_window=tail_window,
                    min_valid_rows=min_valid_rows,
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "too_few_valid_rows" in msg:
                    reason = "too_few_valid_rows"
                else:
                    reason = "forcecoeffs_invalid"
                self.skipped_samples.append({"case": case_dir.name, "reason": reason})
                continue

            x = np.asarray(features, dtype=np.float32)
            y = np.asarray([cl_avg, cd_avg], dtype=np.float32)

            if not np.isfinite(y).all():
                self.skipped_samples.append({"case": case_dir.name, "reason": "nonfinite_target"})
                continue

            if not np.isfinite(x).all() or not np.isfinite(y).all():
                self.skipped_samples.append({"case": case_dir.name, "reason": "nonfinite_target"})
                continue

            self.samples.append((x, y, case_dir.name))

        print(f"[INFO] Number of valid samples: {len(self.samples)}")
        print(f"[INFO] Number of skipped samples: {len(self.skipped_samples)}")
        for skipped in self.skipped_samples:
            print(f"[SKIP] {skipped['case']} | reason: {skipped['reason']}")

        if not self.samples:
            raise RuntimeError(
                "No valid Cl/Cd samples found. Check case folder names and forceCoeffs paths."
            )

        self.fit_normalization()

    def fit_normalization(self, indices: Sequence[int] | None = None) -> None:
        if indices is None:
            selected = self.samples
        else:
            selected = [self.samples[i] for i in indices]

        if not selected:
            raise RuntimeError("Cannot fit normalization on empty index set")

        nonfinite_cases: list[str] = []
        for feat, targ, case_name in selected:
            if not np.isfinite(feat).all() or not np.isfinite(targ).all():
                nonfinite_cases.append(case_name)

        if nonfinite_cases:
            case_list = ", ".join(sorted(set(nonfinite_cases)))
            raise RuntimeError(
                "Non-finite features/targets found before normalization for cases: "
                f"{case_list}"
            )

        x = np.stack([s[0] for s in selected], axis=0).astype(np.float32)
        y = np.stack([s[1] for s in selected], axis=0).astype(np.float32)

        self.feature_mean = x.mean(axis=0)
        self.feature_std = x.std(axis=0)
        self.target_mean = y.mean(axis=0)
        self.target_std = y.std(axis=0)

        if not (
            np.isfinite(self.feature_mean).all()
            and np.isfinite(self.feature_std).all()
            and np.isfinite(self.target_mean).all()
            and np.isfinite(self.target_std).all()
        ):
            raise RuntimeError("Non-finite normalization statistics detected")

        if np.any(self.feature_std <= 1e-8):
            raise RuntimeError(
                f"Invalid feature_std detected (<= 1e-8): {self.feature_std.tolist()}"
            )
        if np.any(self.target_std <= 1e-8):
            raise RuntimeError(
                f"Invalid target_std detected (<= 1e-8): {self.target_std.tolist()}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_raw, y_raw, _ = self.samples[idx]
        x = (x_raw - self.feature_mean) / self.feature_std
        y = (y_raw - self.target_mean) / self.target_std
        return torch.from_numpy(x), torch.from_numpy(y)
