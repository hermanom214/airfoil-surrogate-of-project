from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple
import sys


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

TEST_MODE = True

# Body pro původní NACA výpočet.
N_POINTS = 161

# Body finální STL smyčky po arclength resamplingu.
# Cíl: žádné extrémně úzké segmenty u LE/TE.
STL_LOOP_POINTS = 600

# Tenká extruze pro STL v ose z.
Z_MIN = -0.05
Z_MAX = 0.05

CHORD = 1.0


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
# POMOCNÉ FUNKCE
# ============================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cosine_spacing(n_points: int) -> List[float]:
    xs = []
    for i in range(n_points):
        beta = math.pi * i / (n_points - 1)
        x = 0.5 * (1.0 - math.cos(beta))
        xs.append(x)
    return xs


def thickness_distribution(x: float, thickness_fraction: float) -> float:
    """
    NACA 4-digit half-thickness.

    Koeficient -0.1015 dává malou, ale nenulovou trailing-edge tloušťku.
    To je pro snappyHexMesh lepší než matematicky ostrý TE.
    """
    yt = 5.0 * thickness_fraction * (
        0.2969 * math.sqrt(max(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
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


def build_closed_profile_loop(
    upper_points: List[Tuple[float, float]],
    lower_points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Uzavřená smyčka:
    - horní strana TE -> LE
    - dolní strana LE -> TE

    Důležité:
    lower TE bod musí zůstat. Není duplicitní, pokud má TE nenulovou tloušťku.
    Původní lower_points[1:-1] vyhazovalo lower TE a mohlo vytvářet špatné
    zavření profilu v oblasti odtokové hrany.
    """
    upper_te_to_le = list(reversed(upper_points))
    lower_le_to_te = lower_points[1:]  # bez duplikace LE, ale ponechat lower TE

    return upper_te_to_le + lower_le_to_te


def distance_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def resample_closed_loop_by_arclength(
    points: List[Tuple[float, float]],
    n_points: int,
) -> List[Tuple[float, float]]:
    """
    Převzorkuje uzavřenou smyčku na přibližně rovnoměrné rozestupy po oblouku.

    Důvod:
    cosine spacing vytváří hodně krátké segmenty u LE/TE.
    Pro snappyHexMesh to může znamenat tenké STL fasety a špatný snap.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points for closed loop resampling.")

    closed = points + [points[0]]

    segment_lengths: List[float] = []
    total_length = 0.0

    for i in range(len(closed) - 1):
        length = distance_2d(closed[i], closed[i + 1])
        segment_lengths.append(length)
        total_length += length

    if total_length <= 0:
        raise ValueError("Invalid loop length.")

    resampled: List[Tuple[float, float]] = []

    target_spacing = total_length / n_points
    current_segment = 0
    accumulated = 0.0

    for k in range(n_points):
        target = k * target_spacing

        while (
            current_segment < len(segment_lengths) - 1
            and accumulated + segment_lengths[current_segment] < target
        ):
            accumulated += segment_lengths[current_segment]
            current_segment += 1

        a = closed[current_segment]
        b = closed[current_segment + 1]
        seg_len = segment_lengths[current_segment]

        if seg_len == 0:
            resampled.append(a)
            continue

        local_t = (target - accumulated) / seg_len

        x = a[0] + local_t * (b[0] - a[0])
        y = a[1] + local_t * (b[1] - a[1])

        resampled.append((x, y))

    return resampled


def rotate_points(
    points: List[Tuple[float, float]],
    angle_deg: float,
) -> List[Tuple[float, float]]:
    angle_rad = math.radians(angle_deg)
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)

    rotated = []
    for x, y in points:
        xr = ca * x - sa * y
        yr = sa * x + ca * y
        rotated.append((xr, yr))
    return rotated


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


def write_ascii_stl(
    path: Path,
    case: AirfoilCase,
    loop_points: List[Tuple[float, float]],
    z_min: float,
    z_max: float,
) -> None:
    """
    Extrudovaný ASCII STL.

    Opravy proti původní verzi:
    - smyčka zachovává oba TE body
    - cap triangulace jde z centroidu, ne z jednoho TE bodu
    """
    front = [(x, y, z_min) for x, y in loop_points]
    back = [(x, y, z_max) for x, y in loop_points]
    n = len(loop_points)

    solid_name = f"airfoil_naca_{case.naca_code()}"

    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {solid_name}\n")

        # Boční plášť
        for i in range(n):
            j = (i + 1) % n

            a = front[i]
            b = front[j]
            c = back[j]
            d = back[i]

            write_facet(f, a, b, c)
            write_facet(f, a, c, d)

        # Víka triangulovaná z centroidu.
        # Stabilnější než fan z TE bodu.
        #cx = sum(x for x, _ in loop_points) / n
        #cy = sum(y for _, y in loop_points) / n

        #front_center = (cx, cy, z_min)
        #back_center = (cx, cy, z_max)

        #for i in range(n):
        #    j = (i + 1) % n
        #    write_facet(f, front_center, front[j], front[i])

        #for i in range(n):
        #    j = (i + 1) % n
       #     write_facet(f, back_center, back[i], back[j])

        f.write(f"endsolid {solid_name}\n")


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
    return [
        AirfoilCase(camber_percent=0, camber_position_tenths=0, thickness_percent=12),
        AirfoilCase(camber_percent=2, camber_position_tenths=4, thickness_percent=12),
        AirfoilCase(camber_percent=4, camber_position_tenths=4, thickness_percent=15),
        AirfoilCase(camber_percent=2, camber_position_tenths=2, thickness_percent=8),
    ]


def get_real_run_cases() -> List[AirfoilCase]:
    cases: List[AirfoilCase] = []

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

                if not any(existing.naca_code() == case.naca_code() for existing in cases):
                    cases.append(case)

    return cases


# ============================================================
# HLAVNÍ GENERAČNÍ LOGIKA
# ============================================================

def generate_airfoil_files(cases: List[AirfoilCase], output_dir: Path) -> None:
    ensure_output_dir(output_dir)

    for case in cases:
        upper, lower = generate_naca4_coordinates(case, n_points=N_POINTS)

        loop_raw = build_closed_profile_loop(upper, lower)

        loop = resample_closed_loop_by_arclength(
            loop_raw,
            n_points=STL_LOOP_POINTS,
        )

        dat_path = output_dir / f"naca{case.naca_code()}.dat"
        stl_path = output_dir / f"naca{case.naca_code()}.stl"

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
    print(f"Total generated airfoils: {len(cases)}")


if __name__ == "__main__":
    main()