from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple
import sys
from unittest import case


# ============================================================
# KONFIGURACE
# ============================================================

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import load_paths

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
PATHS = load_paths(CONFIG_PATH)

PROJECT_ROOT = PATHS.project_root
OUTPUT_DIR = PATHS.generated_profiles
CSV_PATH = PATHS.sampling_table

TEST_MODE = False

N_POINTS = 241

# Sweep pro CFD podminky (3 rychlosti, 3 AoA).
AOA_VALUES = [-4.0, 0.0, 4.0]
INLET_VELOCITY_VALUES = [15.0, 20.0, 25.0]

Z_MIN = -0.05
Z_MAX = 0.05

CHORD = 1.0

# Poslední část profilu pro samostatný STL region bez layers.
# 0.97 = poslední 3 % chordu.
TE_SPLIT_X = 0.95


# ============================================================
# DATOVÝ MODEL
# ============================================================

@dataclass
class AirfoilCase:
    camber_percent: int
    camber_position_tenths: int
    thickness_percent: int
    chord: float = 1.0
    aoa_deg: float = 0.0
    inlet_velocity: float = 20.0

    def naca_code(self) -> str:
        return f"{self.camber_percent}{self.camber_position_tenths}{self.thickness_percent:02d}"


# ============================================================
# GEOMETRIE
# ============================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cosine_spacing(n_points: int) -> List[float]:
    return [
        0.5 * (1.0 - math.cos(math.pi * i / (n_points - 1)))
        for i in range(n_points)
    ]


def thickness_distribution(x: float, thickness_fraction: float) -> float:
    yt = 5.0 * thickness_fraction * (
        0.2969 * math.sqrt(max(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    return yt


def camber_line_and_slope(
    x: float,
    camber_fraction: float,
    camber_pos_fraction: float,
) -> Tuple[float, float]:
    if camber_fraction == 0.0 or camber_pos_fraction == 0.0:
        return 0.0, 0.0

    m = camber_fraction
    p = camber_pos_fraction

    if x < p:
        yc = m / (p**2) * (2 * p * x - x**2)
        dyc_dx = 2 * m / (p**2) * (p - x)
    else:
        yc = m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x**2)
        dyc_dx = 2 * m / ((1 - p) ** 2) * (p - x)

    return yc, dyc_dx


def generate_naca4_coordinates(
    case: AirfoilCase,
    n_points: int,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    xs = cosine_spacing(n_points)

    m = case.camber_percent / 100.0
    p = case.camber_position_tenths / 10.0
    t = case.thickness_percent / 100.0
    c = case.chord

    upper_points: List[Tuple[float, float]] = []
    lower_points: List[Tuple[float, float]] = []

    for x_norm in xs:
        yt = thickness_distribution(x_norm, t)
        yc, dyc_dx = camber_line_and_slope(x_norm, m, p)
        theta = math.atan(dyc_dx)

        xu = x_norm - yt * math.sin(theta)
        yu = yc + yt * math.cos(theta)

        xl = x_norm + yt * math.sin(theta)
        yl = yc - yt * math.cos(theta)

        upper_points.append((xu * c, yu * c))
        lower_points.append((xl * c, yl * c))

    return upper_points, lower_points


def force_sharp_trailing_edge(
    upper_points: List[Tuple[float, float]],
    lower_points: List[Tuple[float, float]],
    chord: float = 1.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    upper = upper_points.copy()
    lower = lower_points.copy()

    y_te = 0.5 * (upper[-1][1] + lower[-1][1])
    x_te = chord

    upper[-1] = (x_te, y_te)
    lower[-1] = (x_te, y_te)

    return upper, lower

def rotate_point(
    point: Tuple[float, float],
    angle_deg: float,
    center: Tuple[float, float] = (0.25, 0.0),
) -> Tuple[float, float]:
    """
    Otočí bod kolem zadaného středu.

    center=(0.25, 0.0) odpovídá cca quarter-chord,
    což je běžný referenční bod pro airfoil AoA.
    """
    x, y = point
    cx, cy = center

    angle_rad = math.radians(angle_deg)

    x0 = x - cx
    y0 = y - cy

    xr = x0 * math.cos(angle_rad) - y0 * math.sin(angle_rad)
    yr = x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)

    return xr + cx, yr + cy


def rotate_profile_loop(
    loop_points: List[Tuple[float, float]],
    angle_deg: float,
    center: Tuple[float, float] = (0.25, 0.0),
) -> List[Tuple[float, float]]:
    """
    Otočí celý uzavřený profil o AoA.
    """
    return [rotate_point(p, angle_deg, center=center) for p in loop_points]

def build_closed_profile_loop(
    upper_points: List[Tuple[float, float]],
    lower_points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    upper_te_to_le = list(reversed(upper_points))
    lower_le_to_near_te = lower_points[1:-1]

    return upper_te_to_le + lower_le_to_near_te


# ============================================================
# EXPORT
# ============================================================

def write_dat_file(
    path: Path,
    case: AirfoilCase,
    loop_points: List[Tuple[float, float]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NACA {case.naca_code()}\n")
        for x, y in loop_points:
            f.write(f"{x:.8f} {y:.8f}\n")


def triangle_normal(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]

    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx

    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0:
        return 0.0, 0.0, 0.0

    return nx / norm, ny / norm, nz / norm

def format_float_for_name(value: float) -> str:
    """
    Převod čísla do názvu souboru:
    -4.0 -> m4p0
     0.0 -> 0p0
     4.0 -> 4p0
    """
    return str(value).replace("-", "m").replace(".", "p")

def write_facet(
    f,
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
) -> None:
    nx, ny, nz = triangle_normal(a, b, c)

    f.write(f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}\n")
    f.write("    outer loop\n")
    f.write(f"      vertex {a[0]:.8e} {a[1]:.8e} {a[2]:.8e}\n")
    f.write(f"      vertex {b[0]:.8e} {b[1]:.8e} {b[2]:.8e}\n")
    f.write(f"      vertex {c[0]:.8e} {c[1]:.8e} {c[2]:.8e}\n")
    f.write("    endloop\n")
    f.write("  endfacet\n")


def classify_stl_region(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    chord: float,
    te_split_x: float,
) -> str:
    """
    Segmenty profilu blízko trailing edge pošle do airfoil_TE.
    Zbytek do airfoil_main.

    Používáme střed segmentu, aby se regiony nerozbíjely bodově.
    """
    x_mid = 0.5 * (p1[0] + p2[0])

    if x_mid >= te_split_x * chord:
        return "airfoil_TE"

    return "airfoil_main"


def write_ascii_stl(
    path: Path,
    case: AirfoilCase,
    loop_points: List[Tuple[float, float]],
    z_min: float,
    z_max: float,
) -> None:
    """
    Extrudovaný STL pouze z bočního pláště.

    STL je rozdělené na dva solid/region bloky:
    - airfoil_main -> bude mít boundary layers
    - airfoil_TE   -> bez boundary layers

    Bez caps, protože cap triangulace dělala špatné fasety pro snappy.
    """
    front = [(x, y, z_min) for x, y in loop_points]
    back = [(x, y, z_max) for x, y in loop_points]
    n = len(loop_points)

    facets_by_region = {
        "airfoil_main": [],
        "airfoil_TE": [],
    }

    for i in range(n):
        j = (i + 1) % n

        region = classify_stl_region(
            loop_points[i],
            loop_points[j],
            chord=case.chord,
            te_split_x=TE_SPLIT_X,
        )

        a = front[i]
        b = front[j]
        c = back[j]
        d = back[i]

        facets_by_region[region].append((a, b, c))
        facets_by_region[region].append((a, c, d))

    with path.open("w", encoding="utf-8") as f:
        for region_name in ["airfoil_main", "airfoil_TE"]:
            facets = facets_by_region[region_name]

            f.write(f"solid {region_name}\n")

            for a, b, c in facets:
                write_facet(f, a, b, c)

            f.write(f"endsolid {region_name}\n")


def write_sampling_table(csv_path: Path, cases: List[AirfoilCase]) -> None:
    fieldnames = [
        "naca_code",
        "camber_percent",
        "camber_position_tenths",
        "thickness_percent",
        "chord",
        "aoa_deg",
        "inlet_velocity",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            row = asdict(case)
            row["naca_code"] = case.naca_code()
            writer.writerow(row)


# ============================================================
# CASE LISTY
# ============================================================

def get_test_cases() -> List[AirfoilCase]:
    base_cases = [
        AirfoilCase(camber_percent=0, camber_position_tenths=0, thickness_percent=12),
        AirfoilCase(camber_percent=2, camber_position_tenths=4, thickness_percent=12),
        AirfoilCase(camber_percent=4, camber_position_tenths=4, thickness_percent=15),
        AirfoilCase(camber_percent=2, camber_position_tenths=2, thickness_percent=8),
    ]

    return expand_cases_with_conditions(base_cases)


def get_real_run_cases() -> List[AirfoilCase]:
    base_cases: List[AirfoilCase] = []

    camber_values = [0, 2, 4]
    camber_pos_values = [2, 4]
    thickness_values = [8, 12, 16]

    for camber in camber_values:
        for camber_pos in camber_pos_values:
            for thickness in thickness_values:
                if camber == 0:
                    case = AirfoilCase(
                        camber_percent=0,
                        camber_position_tenths=0,
                        thickness_percent=thickness,
                    )
                else:
                    case = AirfoilCase(
                        camber_percent=camber,
                        camber_position_tenths=camber_pos,
                        thickness_percent=thickness,
                    )

                if not any(existing.naca_code() == case.naca_code() for existing in base_cases):
                    base_cases.append(case)

    return expand_cases_with_conditions(base_cases)


def expand_cases_with_conditions(base_cases: List[AirfoilCase]) -> List[AirfoilCase]:
    expanded: List[AirfoilCase] = []

    for case in base_cases:
        for aoa in AOA_VALUES:
            for inlet_velocity in INLET_VELOCITY_VALUES:
                expanded.append(
                    AirfoilCase(
                        camber_percent=case.camber_percent,
                        camber_position_tenths=case.camber_position_tenths,
                        thickness_percent=case.thickness_percent,
                        chord=case.chord,
                        aoa_deg=aoa,
                        inlet_velocity=inlet_velocity,
                    )
                )

    return expanded


def unique_geometry_cases(cases: List[AirfoilCase]) -> List[AirfoilCase]:
    unique_cases: List[AirfoilCase] = []
    seen: set[tuple[str, float]] = set()

    for case in cases:
        key = (case.naca_code(), case.aoa_deg)

        if key in seen:
            continue

        seen.add(key)
        unique_cases.append(case)

    return unique_cases


# ============================================================
# HLAVNÍ LOGIKA
# ============================================================

def generate_airfoil_files(cases: List[AirfoilCase], output_dir: Path) -> None:
    ensure_output_dir(output_dir)

    for case in unique_geometry_cases(cases):
        upper, lower = generate_naca4_coordinates(case, n_points=N_POINTS)
        upper, lower = force_sharp_trailing_edge(upper, lower, chord=case.chord)

        loop = build_closed_profile_loop(upper, lower)
        loop = rotate_profile_loop(loop, -case.aoa_deg)

        aoa_name = format_float_for_name(case.aoa_deg)
        dat_path = output_dir / f"naca{case.naca_code()}_aoa{aoa_name}.dat"
        stl_path = output_dir / f"naca{case.naca_code()}_aoa{aoa_name}.stl"

        write_dat_file(dat_path, case, loop)
        write_ascii_stl(stl_path, case, loop, z_min=Z_MIN, z_max=Z_MAX)

        print(f"Generated: {dat_path.name}, {stl_path.name}")


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    if TEST_MODE:
        cases = get_test_cases()
        print("Running in TEST_MODE")
    else:
        cases = get_real_run_cases()
        print("Running in REAL_RUN mode")

    generate_airfoil_files(cases, OUTPUT_DIR)
    write_sampling_table(CSV_PATH, cases)

    print(f"\nSaved sampling table to: {CSV_PATH}")
    print(f"Total CFD combinations (geometry x AoA x velocity): {len(cases)}")
    print(f"Unique generated geometries (DAT/STL): {len(unique_geometry_cases(cases))}")
    print(f"AoA sweep: {AOA_VALUES}")
    print(f"Velocity sweep: {INLET_VELOCITY_VALUES}")
    print(f"TE split starts at x/c = {TE_SPLIT_X}")


if __name__ == "__main__":
    main()