"""
Flow-field extraction utilities for the blockMesh OpenFOAM workflow.

Module responsibilities:
- run `foamToVTK` for a solved case,
- load pressure/velocity fields from VTK,
- interpolate fields to a regular 2D grid,
- build a fluid mask from NACA geometry,
- export compressed `.npz` samples and dataset index CSV.
"""

from __future__ import annotations

import csv
import importlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon

from src.case_runner import run_command, discover_case_dirs
from src.config import load_solver_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLVER_CONFIG_PATH = PROJECT_ROOT / "configs" / "solver_config.yaml"
SOLVER_CFG = load_solver_config(SOLVER_CONFIG_PATH)


@dataclass
class FlowExtractionResult:
    case_id: str
    case_path: str
    status: str
    output_file: str
    nx: int
    ny: int
    latest_time: str


def read_vtk_with_pyvista(vtk_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read points, pressure and velocity from a VTK file using PyVista."""
    pv = importlib.import_module("pyvista")
    mesh = pv.read(str(vtk_file))

    points = mesh.points  # (N, 3)

    # tlak
    if "p" in mesh.point_data:
        p = mesh.point_data["p"]
    elif "p" in mesh.cell_data:
        p = mesh.cell_data["p"]
        points = mesh.cell_centers().points
    else:
        raise RuntimeError(f"'p' not found in {vtk_file}")

    # rychlost
    if "U" in mesh.point_data:
        U = mesh.point_data["U"]
    elif "U" in mesh.cell_data:
        U = mesh.cell_data["U"]
        points = mesh.cell_centers().points
    else:
        raise RuntimeError(f"'U' not found in {vtk_file}")

    return np.asarray(points), np.asarray(p), np.asarray(U)


def find_latest_time_dir(case_dir: Path) -> str:
    """Return the latest numeric OpenFOAM time directory name in a case."""
    time_dirs = []

    for path in case_dir.iterdir():
        if not path.is_dir():
            continue

        try:
            value = float(path.name)
        except ValueError:
            continue

        time_dirs.append((value, path.name))

    if not time_dirs:
        raise RuntimeError(f"No OpenFOAM time directories found in {case_dir}")

    return sorted(time_dirs, key=lambda x: x[0])[-1][1]


def run_foam_to_vtk(case_dir: Path) -> int:
    """Run foamToVTK for latest time and return process return code."""
    result = run_command(
        command=SOLVER_CFG.foam_to_vtk.command,
        case_dir=case_dir,
        log_dir=case_dir / "logs",
        log_name=SOLVER_CFG.foam_to_vtk.log_name,
    )

    return result.returncode


def find_vtk_file(case_dir: Path) -> Path:
    """Find the newest top-level VTK file in case_dir/VTK."""
    vtk_root = case_dir / "VTK"

    if not vtk_root.exists():
        raise RuntimeError(f"VTK folder not found: {vtk_root}")

    vtk_files = sorted(vtk_root.glob("*.vtk"))

    if not vtk_files:
        raise RuntimeError(f"No main VTK files found in {vtk_root}")

    return vtk_files[-1]


def interpolate_to_grid(
    points: np.ndarray,
    p: np.ndarray,
    U: np.ndarray,
    nx: int,
    ny: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)

    grid_x, grid_y = np.meshgrid(xs, ys)
    xy_points = points[:, :2]

    p_grid = griddata(xy_points, p, (grid_x, grid_y), method="linear")
    ux_grid = griddata(xy_points, U[:, 0], (grid_x, grid_y), method="linear")
    uy_grid = griddata(xy_points, U[:, 1], (grid_x, grid_y), method="linear")

    # Fill NaNs outside convex hull with nearest interpolation.
    p_nearest = griddata(xy_points, p, (grid_x, grid_y), method="nearest")
    ux_nearest = griddata(xy_points, U[:, 0], (grid_x, grid_y), method="nearest")
    uy_nearest = griddata(xy_points, U[:, 1], (grid_x, grid_y), method="nearest")

    p_grid = np.where(np.isnan(p_grid), p_nearest, p_grid)
    ux_grid = np.where(np.isnan(ux_grid), ux_nearest, ux_grid)
    uy_grid = np.where(np.isnan(uy_grid), uy_nearest, uy_grid)

    xy_grid = np.stack([grid_x, grid_y], axis=-1)
    U_grid = np.stack([ux_grid, uy_grid], axis=-1)

    return xy_grid, p_grid, U_grid


def extract_single_case(
    case_dir: Path,
    output_root: Path,
    nx: int = 128,
    ny: int = 64,
    x_min: float = -0.5,
    x_max: float = 1.5,
    y_min: float = -0.5,
    y_max: float = 0.5,
) -> FlowExtractionResult:
    case_id = case_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    latest_time = find_latest_time_dir(case_dir)

    returncode = run_foam_to_vtk(case_dir)
    if returncode != 0:
        return FlowExtractionResult(
            case_id=case_id,
            case_path=str(case_dir),
            status=f"foamToVTK_failed({returncode})",
            output_file="",
            nx=nx,
            ny=ny,
            latest_time=latest_time,
        )

    vtk_file = find_vtk_file(case_dir)

    points, p, U = read_vtk_with_pyvista(vtk_file)

    xy_grid, p_grid, U_grid = interpolate_to_grid(
        points=points,
        p=p,
        U=U,
        nx=nx,
        ny=ny,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    
    params_path = case_dir / "params.json"
    params = {}

    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))

    naca_code = str(params["naca_code"])
    chord = float(params.get("chord", 1.0))

    fluid_mask = create_fluid_mask(
        xy_grid=xy_grid,
        naca_code=naca_code,
        chord=chord,
    )

    output_file = output_root / f"{case_id}_flow.npz"

    np.savez_compressed(
        output_file,
        xy=xy_grid,
        p=p_grid,
        U=U_grid,
        fluid_mask=fluid_mask,
        case_id=case_id,
        latest_time=latest_time,
        vtk_file=str(vtk_file),
        params=json.dumps(params),
    )

    return FlowExtractionResult(
        case_id=case_id,
        case_path=str(case_dir),
        status="ok",
        output_file=str(output_file),
        nx=nx,
        ny=ny,
        latest_time=latest_time,
    )


def write_flow_index(results: list[FlowExtractionResult], output_csv: Path) -> None:
    """Write extraction summary CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "case_path",
        "status",
        "output_file",
        "nx",
        "ny",
        "latest_time",
    ]

    rows = [asdict(r) for r in results]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if rows:
            writer.writerows(rows)


def generate_naca4_polygon(
    naca_code: str,
    chord: float = 1.0,
    n_points: int = 1200,
) -> Polygon:
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0

    x = np.linspace(0.0, chord, n_points)
    xc = x / chord

    yt = 5 * t * chord * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0 and p > 0:
        left = xc < p
        right = ~left

        yc[left] = m / p**2 * (2 * p * xc[left] - xc[left] ** 2) * chord
        yc[right] = m / (1 - p) ** 2 * (
            (1 - 2 * p) + 2 * p * xc[right] - xc[right] ** 2
        ) * chord

        dyc_dx[left] = 2 * m / p**2 * (p - xc[left])
        dyc_dx[right] = 2 * m / (1 - p) ** 2 * (p - xc[right])

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = list(zip(xu, yu))
    lower = list(zip(xl[::-1], yl[::-1]))

    coords = upper + lower

    return Polygon(coords)

def create_fluid_mask(
    xy_grid: np.ndarray,
    naca_code: str,
    chord: float = 1.0,
) -> np.ndarray:
    """Return 1.0 in fluid cells and 0.0 inside the airfoil polygon."""
    polygon = generate_naca4_polygon(
        naca_code=naca_code,
        chord=chord,
    )

    ny, nx, _ = xy_grid.shape
    solid_mask = np.zeros((ny, nx), dtype=bool)

    for j in range(ny):
        for i in range(nx):
            x, y = xy_grid[j, i]
            solid_mask[j, i] = polygon.contains(Point(float(x), float(y)))

    fluid_mask = ~solid_mask

    return fluid_mask.astype(np.float32)