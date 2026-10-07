from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.blockmesh_generator import NACA4Params, generate_naca4_upper_lower, rotate_points


@dataclass(frozen=True)
class ForceCoefficientConfig:
    lift_dir: tuple[float, float] = (0.0, 1.0)
    drag_dir: tuple[float, float] = (1.0, 0.0)
    reference_area: float = 0.1
    reference_length: float = 1.0
    span: float = 0.1
    rho_inf: float = 1.225
    pressure_is_kinematic: bool = True


@dataclass(frozen=True)
class FieldCoefficientSettings:
    surface_samples_per_side: int = 520
    le_cluster_exp: float = 2.8
    pressure_offset_grid_cells: float = 1.5
    viscous_distance_grid_cells: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)
    nu: float = 1.5e-5
    p_inf: float = 0.0


@dataclass
class AeroCoefficientResult:
    cl_pressure: float = np.nan
    cd_pressure: float = np.nan
    cl_viscous: float = np.nan
    cd_viscous: float = np.nan
    cl_total: float = np.nan
    cd_total: float = np.nan
    pressure_valid: bool = False
    viscous_valid: bool = False
    diagnostics: dict[str, Any] | None = None


def reconstruct_surface(
    naca_code: str,
    chord: float,
    aoa_deg: float,
    n_points: int,
    le_cluster_exp: float = 2.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CCW segment midpoints, tangents, outward normals and ds."""
    if len(naca_code) != 4 or not naca_code.isdigit():
        raise ValueError(f"Expected a four-digit NACA code, got {naca_code!r}")
    params = NACA4Params(int(naca_code[0]), int(naca_code[1]), int(naca_code[2:]), chord)
    upper, lower = generate_naca4_upper_lower(params, n_points, le_cluster_exp)
    upper = rotate_points(upper, -aoa_deg)
    lower = rotate_points(lower, -aoa_deg)

    # lower LE->TE, then upper TE->LE is counter-clockwise.
    vertices = np.asarray(lower + list(reversed(upper[1:-1])) + [lower[0]], dtype=np.float64)
    delta = np.diff(vertices, axis=0)
    ds = np.linalg.norm(delta, axis=1)
    keep = ds > np.finfo(np.float64).eps
    delta, ds = delta[keep], ds[keep]
    midpoints = (vertices[:-1][keep] + vertices[1:][keep]) * 0.5
    tangents = delta / ds[:, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    signed_area = 0.5 * np.sum(
        vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1]
    )
    if signed_area <= 0.0 or np.max(np.abs(np.sum(normals * ds[:, None], axis=0))) > 1e-8 * chord:
        raise RuntimeError("Surface orientation/closure consistency check failed")
    return midpoints, tangents, normals, ds


def _grid_interpolator(values: np.ndarray, xy: np.ndarray) -> RegularGridInterpolator:
    x = np.asarray(xy[0, :, 0], dtype=np.float64)
    y = np.asarray(xy[:, 0, 1], dtype=np.float64)
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("xy must describe an increasing regular Cartesian grid")
    return RegularGridInterpolator((y, x), np.asarray(values, dtype=np.float64), bounds_error=False, fill_value=np.nan)


def _sample(interpolator: RegularGridInterpolator, points: np.ndarray) -> np.ndarray:
    return np.asarray(interpolator(points[:, [1, 0]]), dtype=np.float64)


def _project(force: np.ndarray, config: ForceCoefficientConfig) -> tuple[float, float]:
    lift = np.asarray(config.lift_dir, dtype=np.float64)
    drag = np.asarray(config.drag_dir, dtype=np.float64)
    lift /= np.linalg.norm(lift)
    drag /= np.linalg.norm(drag)
    return float(force @ lift), float(force @ drag)


def derive_aero_coefficients_from_fields(
    p: np.ndarray,
    U: np.ndarray,
    xy: np.ndarray,
    fluid_mask: np.ndarray,
    *,
    naca_code: str,
    chord: float,
    aoa_deg: float,
    inlet_velocity: float,
    force_config: ForceCoefficientConfig,
    settings: FieldCoefficientSettings = FieldCoefficientSettings(),
    include_viscous: bool = True,
) -> AeroCoefficientResult:
    result = AeroCoefficientResult(diagnostics={"settings": asdict(settings), "experimental_viscous": True})
    if inlet_velocity <= 0.0 or chord <= 0.0:
        result.diagnostics["error"] = "non-positive inlet velocity or chord"
        return result
    try:
        points, tangents, normals, ds = reconstruct_surface(
            naca_code, chord, aoa_deg, settings.surface_samples_per_side, settings.le_cluster_exp
        )
        dx = float(np.median(np.diff(xy[0, :, 0])))
        dy = float(np.median(np.diff(xy[:, 0, 1])))
        grid_step = max(abs(dx), abs(dy))
        mask_interp = _grid_interpolator(np.asarray(fluid_mask, dtype=np.float64), xy)
        p_interp = _grid_interpolator(p, xy)
        pressure_points = points + settings.pressure_offset_grid_cells * grid_step * normals
        pressure_values = _sample(p_interp, pressure_points)
        pressure_fluid = _sample(mask_interp, pressure_points) >= 0.5
        valid_p = pressure_fluid & np.isfinite(pressure_values)
        result.diagnostics.update({
            "grid_spacing": [dx, dy], "surface_segment_count": int(ds.size),
            "pressure_valid_fraction": float(np.mean(valid_p)),
        })
        if not np.all(valid_p):
            result.diagnostics["pressure_error"] = "surface pressure interpolation left the fluid/grid"
            return result
        cp = (pressure_values - settings.p_inf) / (0.5 * inlet_velocity**2)
        # span/Aref is retained explicitly to match OpenFOAM's 3-D reference-area normalization.
        pressure_force = -np.sum(cp[:, None] * normals * ds[:, None], axis=0)
        pressure_force *= chord * force_config.span / force_config.reference_area / chord
        result.cl_pressure, result.cd_pressure = _project(pressure_force, force_config)
        result.pressure_valid = True

        if not include_viscous:
            return result
        u = np.asarray(U, dtype=np.float64)
        if u.shape != p.shape + (2,):
            result.diagnostics["viscous_error"] = "U must have shape p.shape + (2,)"
            return result
        ux_interp, uy_interp = _grid_interpolator(u[..., 0], xy), _grid_interpolator(u[..., 1], xy)
        gradients = np.full(ds.shape, np.nan, dtype=np.float64)
        distances = grid_step * np.asarray(settings.viscous_distance_grid_cells, dtype=np.float64)
        for i, (point, tangent, normal) in enumerate(zip(points, tangents, normals)):
            sample_points = point[None, :] + distances[:, None] * normal[None, :]
            if not np.all(_sample(mask_interp, sample_points) >= 0.5):
                continue
            velocities = np.column_stack((_sample(ux_interp, sample_points), _sample(uy_interp, sample_points)))
            ut = velocities @ tangent
            if not np.all(np.isfinite(ut)):
                continue
            # Least-squares wall slope constrained by no-slip U_t(0)=0.
            gradients[i] = float(distances @ ut / (distances @ distances))
        valid_v = np.isfinite(gradients)
        result.diagnostics["viscous_valid_fraction"] = float(np.mean(valid_v))
        if not np.all(valid_v):
            result.diagnostics["viscous_error"] = "insufficient fluid samples for wall-gradient fit"
            return result
        cf_local = 2.0 * settings.nu * gradients / inlet_velocity**2
        viscous_force = np.sum(cf_local[:, None] * tangents * ds[:, None], axis=0)
        viscous_force *= chord * force_config.span / force_config.reference_area / chord
        result.cl_viscous, result.cd_viscous = _project(viscous_force, force_config)
        result.cl_total = result.cl_pressure + result.cl_viscous
        result.cd_total = result.cd_pressure + result.cd_viscous
        result.viscous_valid = True
        return result
    except (ValueError, RuntimeError) as exc:
        result.diagnostics["error"] = str(exc)
        return result
