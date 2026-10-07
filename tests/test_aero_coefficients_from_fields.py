from __future__ import annotations

import numpy as np
import json
from pathlib import Path
import pytest

from src.aero_coefficients_from_fields import (
    FieldCoefficientSettings,
    ForceCoefficientConfig,
    derive_aero_coefficients_from_fields,
    reconstruct_surface,
)
from src.ml_clcd_dataset import read_cl_cd_tail_average


def _grid(n: int = 241):
    x = np.linspace(-0.4, 1.4, n)
    y = np.linspace(-0.45, 0.45, n // 2 + 1)
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx, yy), axis=-1), np.ones(xx.shape, dtype=bool)


def _derive(p, xy, mask, *, samples=300, U=None, config=None, include_viscous=False):
    if U is None:
        U = np.zeros(p.shape + (2,))
    return derive_aero_coefficients_from_fields(
        p, U, xy, mask, naca_code="0012", chord=1.0, aoa_deg=0.0,
        inlet_velocity=10.0, force_config=config or ForceCoefficientConfig(),
        settings=FieldCoefficientSettings(surface_samples_per_side=samples),
        include_viscous=include_viscous,
    )


def test_constant_pressure_and_uniform_offset_give_zero_force() -> None:
    xy, mask = _grid()
    first = _derive(np.full(mask.shape, 7.0), xy, mask)
    second = _derive(np.full(mask.shape, 107.0), xy, mask)
    assert first.pressure_valid and second.pressure_valid
    assert abs(first.cl_pressure) < 1e-10 and abs(first.cd_pressure) < 1e-10
    assert abs(second.cl_pressure) < 1e-10 and abs(second.cd_pressure) < 1e-10


def test_symmetric_pressure_has_zero_lift_at_zero_aoa() -> None:
    xy, mask = _grid()
    result = _derive(xy[..., 0] ** 2, xy, mask)
    assert result.pressure_valid
    assert abs(result.cl_pressure) < 1e-10


def test_surface_is_ccw_with_outward_normals_and_closes() -> None:
    points, tangents, normals, ds = reconstruct_surface("2412", 1.0, 4.0, 400)
    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.max(np.abs(np.sum(normals * ds[:, None], axis=0))) < 1e-10
    centroid = np.average(points, axis=0, weights=ds)
    assert np.mean(np.sum((points - centroid) * normals, axis=1)) > 0.0


def test_pressure_integration_converges_with_surface_refinement() -> None:
    xy, mask = _grid(321)
    p = xy[..., 1]
    coarse = _derive(p, xy, mask, samples=100)
    medium = _derive(p, xy, mask, samples=250)
    fine = _derive(p, xy, mask, samples=600)
    assert coarse.pressure_valid and medium.pressure_valid and fine.pressure_valid
    assert abs(fine.cl_pressure - medium.cl_pressure) < abs(medium.cl_pressure - coarse.cl_pressure)


def test_invalid_surface_interpolation_fails_closed() -> None:
    xy, mask = _grid()
    mask[:] = False
    result = _derive(np.zeros(mask.shape), xy, mask)
    assert not result.pressure_valid
    assert np.isnan(result.cl_pressure)


def test_viscous_estimator_fails_safely_without_finite_samples() -> None:
    xy, mask = _grid()
    U = np.full(mask.shape + (2,), np.nan)
    result = _derive(np.zeros(mask.shape), xy, mask, U=U, include_viscous=True)
    assert result.pressure_valid
    assert not result.viscous_valid
    assert np.isnan(result.cl_total)


def test_force_projection_uses_configured_directions() -> None:
    xy, mask = _grid()
    config = ForceCoefficientConfig(lift_dir=(1.0, 0.0), drag_dir=(0.0, 1.0))
    result = _derive(xy[..., 0], xy, mask, config=config)
    assert result.pressure_valid
    assert result.cl_pressure < 0.0
    assert abs(result.cd_pressure) < 1e-10


def test_real_cfd_case_pressure_regression() -> None:
    case = Path(__file__).resolve().parents[1] / "data/flow_fields/case_0001_naca0008_aoam4p0_u15p0"
    npz_files = list(case.glob("*.npz"))
    force_path = case / "postProcessing/forceCoeffs1/0/forceCoeffs.dat"
    if not npz_files or not force_path.is_file():
        pytest.skip("real CFD regression artifacts are not available")
    meta = json.loads((case / "params.json").read_text(encoding="utf-8"))
    with np.load(npz_files[0], allow_pickle=False) as data:
        result = derive_aero_coefficients_from_fields(
            data["p"], data["U"], data["xy"], data["fluid_mask"] > 0.5,
            naca_code=meta["naca_code"], chord=meta["chord"], aoa_deg=meta["aoa_deg"],
            inlet_velocity=meta["inlet_velocity"],
            force_config=ForceCoefficientConfig(
                reference_area=meta["Aref"], reference_length=meta["lRef"], span=meta["span"]
            ), include_viscous=False,
        )
    cl_openfoam, cd_openfoam = read_cl_cd_tail_average(force_path)
    assert result.pressure_valid
    assert abs(result.cl_pressure - cl_openfoam) < 0.03
    assert abs(result.cd_pressure - cd_openfoam) < 0.01
