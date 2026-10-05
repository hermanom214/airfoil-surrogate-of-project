from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STAT_NAMES = ("mean_abs", "rms", "median_abs", "p95_abs", "p99_abs", "max_abs")


@dataclass(frozen=True)
class GridInfo:
    ny: int
    nx: int
    dx: float
    dy: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def inspect_uniform_xy(xy: np.ndarray, rtol: float = 1e-4) -> GridInfo:
    """Validate meshgrid-style xy[physical_y, physical_x, coordinate]."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[-1] != 2 or min(xy.shape[:2]) < 5:
        raise ValueError(f"xy must have shape (ny>=5, nx>=5, 2), got {xy.shape}")
    x, y = xy[..., 0], xy[..., 1]
    if not np.allclose(x, x[0:1, :], rtol=rtol, atol=1e-10):
        raise ValueError("xy[...,0] varies along array axis 0; expected physical x on axis 1")
    if not np.allclose(y, y[:, 0:1], rtol=rtol, atol=1e-10):
        raise ValueError("xy[...,1] varies along array axis 1; expected physical y on axis 0")
    dxs, dys = np.diff(x[0]), np.diff(y[:, 0])
    dx, dy = float(np.mean(dxs)), float(np.mean(dys))
    if dx <= 0 or dy <= 0 or not np.allclose(dxs, dx, rtol=rtol, atol=1e-10) or not np.allclose(dys, dy, rtol=rtol, atol=1e-10):
        raise ValueError("xy grid must be uniformly increasing in physical x and y")
    return GridInfo(x.shape[0], x.shape[1], dx, dy, float(x.min()), float(x.max()), float(y.min()), float(y.max()))


def stencil_valid_mask(base_valid: np.ndarray, order: int = 2) -> np.ndarray:
    """Exact cross-shaped validity requirement for centered first derivatives."""
    base = np.asarray(base_valid, dtype=bool)
    if base.ndim != 2 or order not in (2, 4):
        raise ValueError("base_valid must be 2-D and order must be 2 or 4")
    radius = order // 2
    valid = base.copy()
    for offset in range(1, radius + 1):
        shifted = np.zeros_like(base)
        shifted[offset:, :] = base[:-offset, :]
        valid &= shifted
        shifted[:] = False
        shifted[:-offset, :] = base[offset:, :]
        valid &= shifted
        shifted[:] = False
        shifted[:, offset:] = base[:, :-offset]
        valid &= shifted
        shifted[:] = False
        shifted[:, :-offset] = base[:, offset:]
        valid &= shifted
    return valid


def continuity_residual(U: np.ndarray, xy: np.ndarray, valid_mask: np.ndarray, order: int = 2) -> np.ndarray:
    """Return dimensional dUx/dx+dUy/dy [1/s], NaN outside valid_mask."""
    U = np.asarray(U, dtype=np.float64)
    info = inspect_uniform_xy(xy)
    mask = np.asarray(valid_mask, dtype=bool)
    if U.shape != (info.ny, info.nx, 2) or mask.shape != U.shape[:2]:
        raise ValueError("U, xy, and valid_mask shapes are inconsistent")
    result = np.full(mask.shape, np.nan, dtype=np.float64)
    if order == 2:
        values = (U[1:-1, 2:, 0] - U[1:-1, :-2, 0]) / (2 * info.dx)
        values += (U[2:, 1:-1, 1] - U[:-2, 1:-1, 1]) / (2 * info.dy)
        result[1:-1, 1:-1] = values
    elif order == 4:
        values = (-U[2:-2, 4:, 0] + 8 * U[2:-2, 3:-1, 0] - 8 * U[2:-2, 1:-3, 0] + U[2:-2, :-4, 0]) / (12 * info.dx)
        values += (-U[4:, 2:-2, 1] + 8 * U[3:-1, 2:-2, 1] - 8 * U[1:-3, 2:-2, 1] + U[:-4, 2:-2, 1]) / (12 * info.dy)
        result[2:-2, 2:-2] = values
    else:
        raise ValueError("order must be 2 or 4")
    result[~mask] = np.nan
    return result


def nondimensionalize_residual(residual_dim: np.ndarray, chord: float, u_inf: float) -> np.ndarray:
    if not np.isfinite(chord) or not np.isfinite(u_inf) or chord <= 0 or u_inf <= 0:
        raise ValueError("chord and U_inf must be finite and positive")
    return np.asarray(residual_dim, dtype=np.float64) * chord / u_inf


def residual_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, **{name: float("nan") for name in STAT_NAMES}}
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mean_abs": float(np.mean(absolute)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
    }


def region_masks(xy: np.ndarray, sdf: np.ndarray, valid: np.ndarray, chord: float, aoa_deg: float) -> dict[str, np.ndarray]:
    """Wall-distance bands plus an overlapping, trailing-edge-centred wake diagnostic."""
    if chord <= 0:
        raise ValueError("chord must be positive")
    x, y = np.asarray(xy)[..., 0] / chord, np.asarray(xy)[..., 1] / chord
    d = np.asarray(sdf) / chord
    valid = np.asarray(valid, dtype=bool)
    angle = np.deg2rad(aoa_deg)
    x_te = 0.25 + 0.75 * np.cos(angle)
    y_te = -0.75 * np.sin(angle)
    return {
        "near_wall": valid & (d > 0) & (d <= 0.02),
        "intermediate": valid & (d > 0.02) & (d <= 0.10),
        "outer": valid & (d > 0.10),
        "wake": valid & (x > x_te) & (x <= 1.75) & (np.abs(y - y_te) <= 0.25),
    }
