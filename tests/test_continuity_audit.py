from __future__ import annotations

import numpy as np

from src.continuity_audit import (
    continuity_residual,
    inspect_uniform_xy,
    nondimensionalize_residual,
    region_masks,
    stencil_valid_mask,
)


def grid(nx=15, ny=11):
    x, y = np.meshgrid(np.linspace(-1, 2, nx), np.linspace(-0.5, 0.5, ny))
    return np.stack((x, y), axis=-1)


def test_axes_and_constant_velocity() -> None:
    xy = grid()
    info = inspect_uniform_xy(xy)
    assert info.nx == 15 and info.ny == 11
    U = np.zeros((*xy.shape[:2], 2)); U[..., 0] = 3; U[..., 1] = -2
    r = continuity_residual(U, xy, np.ones(xy.shape[:2], bool))
    assert np.nanmax(np.abs(r)) == 0


def test_linear_divergence_free_and_nonzero() -> None:
    xy = grid(); valid = np.ones(xy.shape[:2], bool)
    U = np.stack((xy[..., 0], -xy[..., 1]), axis=-1)
    assert np.nanmax(np.abs(continuity_residual(U, xy, valid))) < 1e-12
    U[..., 1] = xy[..., 1]
    assert np.allclose(continuity_residual(U, xy, valid)[1:-1, 1:-1], 2)


def test_nondimensional_scaling() -> None:
    assert np.allclose(nondimensionalize_residual(np.array([2.0, -4.0]), 0.5, 10), [0.1, -0.2])


def test_mask_exclusion_and_stencil_requirement() -> None:
    xy = grid(); base = np.ones(xy.shape[:2], bool); base[5, 7] = False
    required = stencil_valid_mask(base, order=2)
    assert not required[5, 7] and not required[5, 6] and not required[5, 8]
    assert not required[4, 7] and not required[6, 7]
    U = np.zeros((*base.shape, 2)); residual = continuity_residual(U, xy, required)
    assert np.isnan(residual[5, 7]) and np.isnan(residual[5, 6])


def test_regions_are_deterministic_and_wall_bands_partition() -> None:
    xy = grid(); sdf = np.full(xy.shape[:2], 0.05); sdf[3, 3] = 0.01; sdf[4, 4] = 0.2
    masks1 = region_masks(xy, sdf, np.ones(sdf.shape, bool), 1.0, 0.0)
    masks2 = region_masks(xy, sdf, np.ones(sdf.shape, bool), 1.0, 0.0)
    assert all(np.array_equal(masks1[k], masks2[k]) for k in masks1)
    total = masks1["near_wall"].astype(int) + masks1["intermediate"] + masks1["outer"]
    assert np.all(total == 1)
