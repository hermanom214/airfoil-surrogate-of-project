from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

from src.case_runner import run_command, discover_case_dirs
import pyvista as pv

from shapely.geometry import Point, Polygon

@dataclass
class FlowExtractionResult:
    case_id: str
    case_path: str
    status: str
    output_file: str
    nx: int
    ny: int
    latest_time: str





def read_vtk_with_pyvista(vtk_file: Path):
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

    return points, p, U

def find_latest_time_dir(case_dir: Path) -> str:
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
    result = run_command(
        command=[
            "foamToVTK",
            "-latestTime",
            "-ascii",
            "-fields",
            "'(p U)'",
        ],
        case_dir=case_dir,
        log_dir=case_dir / "logs",
        log_name="06_foamToVTK.log",
    )

    return result.returncode


def find_vtk_file(case_dir: Path) -> Path:
    vtk_root = case_dir / "VTK"

    if not vtk_root.exists():
        raise RuntimeError(f"VTK folder not found: {vtk_root}")

    vtk_files = sorted(vtk_root.glob("*.vtk"))

    if not vtk_files:
        raise RuntimeError(f"No main VTK files found in {vtk_root}")

    return vtk_files[-1]


def _read_numbers(lines: list[str], start_idx: int, count: int, dtype=float):
    values = []
    i = start_idx

    while len(values) < count and i < len(lines):
        parts = lines[i].strip().split()
        for part in parts:
            values.append(dtype(part))
            if len(values) == count:
                break
        i += 1

    return values, i


def read_openfoam_vtk(vtk_file: Path):
    lines = vtk_file.read_text(encoding="utf-8", errors="ignore").splitlines()

    points = []
    p = []
    U = []

    mode = None
    reading_points = False
    reading_p = False
    reading_U = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("POINTS"):
            reading_points = True
            mode = None
            continue

        if stripped.startswith("CELLS") or stripped.startswith("CELL_TYPES"):
            reading_points = False
            continue

        if stripped.startswith("POINT_DATA"):
            mode = "POINT_DATA"
            reading_points = False
            reading_p = False
            reading_U = False
            continue

        if stripped.startswith("CELL_DATA"):
            mode = "CELL_DATA"
            reading_p = False
            reading_U = False
            continue

        if mode != "POINT_DATA":
            if reading_points:
                vals = stripped.split()
                if len(vals) == 3:
                    points.append([float(v) for v in vals])
            continue

        if stripped.startswith("SCALARS p"):
            reading_p = True
            reading_U = False
            continue

        if stripped.startswith("LOOKUP_TABLE"):
            continue

        if stripped.startswith("VECTORS U"):
            reading_p = False
            reading_U = True
            continue

        if reading_p:
            vals = stripped.split()
            for v in vals:
                p.append(float(v))

        elif reading_U:
            vals = stripped.split()
            if len(vals) == 3:
                U.append([float(v) for v in vals])

    points = np.array(points, dtype=float)
    p = np.array(p, dtype=float)
    U = np.array(U, dtype=float)

    if len(points) == 0:
        raise RuntimeError(f"No POINTS found in {vtk_file}")

    if len(p) == 0:
        raise RuntimeError(f"No POINT_DATA pressure p found in {vtk_file}")

    if len(U) == 0:
        raise RuntimeError(f"No POINT_DATA velocity U found in {vtk_file}")

    if len(points) != len(p) or len(points) != len(U):
        raise RuntimeError(
            f"POINT_DATA size mismatch in {vtk_file}: "
            f"points={len(points)}, p={len(p)}, U={len(U)}"
        )

    return points, p, U


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
    n_points: int = 300,
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