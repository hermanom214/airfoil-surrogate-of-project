from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


# ============================================================
# KONFIGURACE
# ============================================================
# Tohle jsou základní cesty a přepínače.
# Pro první test nech TEST_MODE = True.
# Později můžeš přepnout na False a použít REAL_RUN_CASES.
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\martina\airfoil_surrogate_OF_project")
OUTPUT_DIR = PROJECT_ROOT / "geometry" / "generated_profiles"
CSV_PATH = OUTPUT_DIR / "airfoil_sampling_table.csv"

TEST_MODE = True

# Počet bodů na jedné straně profilu.
# 121-201 je na začátek rozumné.
N_POINTS = 161

# Tenká extruze pro STL v ose z.
Z_MIN = -0.05
Z_MAX = 0.05

# Chord délka. Pro CFD template jsme si zvolili chord = 1.
CHORD = 1.0


# ============================================================
# DATOVÝ MODEL
# ============================================================
# Tato třída drží parametry jednoho airfoilu.
# Později sem můžeš přidat i AoA, velocity, Reynolds atd.
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
        """
        Vrátí název profilu ve formátu NACA 4-digit.
        Např. 2412 nebo 0012.
        """
        return f"{self.camber_percent}{self.camber_position_tenths}{self.thickness_percent:02d}"


# ============================================================
# POMOCNÉ FUNKCE
# ============================================================

def ensure_output_dir(path: Path) -> None:
    """
    Vytvoří výstupní složku, pokud ještě neexistuje.
    """
    path.mkdir(parents=True, exist_ok=True)


def cosine_spacing(n_points: int) -> List[float]:
    """
    Vrátí x-ové souřadnice v intervalu <0, 1> s cosine spacingem.

    Proč cosine spacing:
    - dává víc bodů u náběžné a odtokové hrany
    - je to běžná volba pro airfoil geometrii
    - na CFD i geometrii je to lepší než rovnoměrné rozložení
    """
    xs = []
    for i in range(n_points):
        beta = math.pi * i / (n_points - 1)
        x = 0.5 * (1.0 - math.cos(beta))
        xs.append(x)
    return xs


def thickness_distribution(x: float, thickness_fraction: float) -> float:
    """
    Spočítá poloviční tloušťku profilu y_t podle klasického vzorce NACA 4-digit.

    thickness_fraction = tloušťka / chord, např. 0.12 pro 12%
    """
    # Klasický NACA vzorec.
    # Poslední koeficient -0.1015 dává téměř uzavřenou trailing edge.
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
    camber_pos_fraction: float
) -> Tuple[float, float]:
    """
    Vrátí:
    - y_c  = camber line
    - dyc_dx = derivaci camber line

    Pro symetrický profil (camber = 0) vrací nulu.
    """
    if camber_fraction == 0.0 or camber_pos_fraction == 0.0:
        return 0.0, 0.0

    m = camber_fraction
    p = camber_pos_fraction

    if x < p:
        yc = m / (p**2) * (2 * p * x - x**2)
        dyc_dx = 2 * m / (p**2) * (p - x)
    else:
        yc = m / ((1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2)
        dyc_dx = 2 * m / ((1 - p)**2) * (p - x)

    return yc, dyc_dx


def generate_naca4_coordinates(case: AirfoilCase, n_points: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Vygeneruje souřadnice horní a dolní strany NACA 4-digit profilu.

    Výstup:
    - upper_points: body od náběžné k odtokové hraně
    - lower_points: body od náběžné k odtokové hraně

    Poznámka:
    Pro .dat export potom obvykle skládáme křivku:
    upper reversed + lower bez duplikace LE/TE.
    """
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
    lower_points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Poskládá uzavřenou 2D smyčku profilu.

    Pořadí:
    - horní strana od TE k LE
    - dolní strana od LE k TE

    Tohle je praktické pro export do DAT i STL.
    """
    upper_te_to_le = list(reversed(upper_points))
    lower_le_to_te = lower_points[1:-1]  # bez duplikace LE a TE
    loop = upper_te_to_le + lower_le_to_te
    return loop


def rotate_points(points: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
    """
    Otočí 2D body o zadaný úhel kolem počátku.

    Pro samotnou geometrii to zatím nepotřebujeme nutně.
    Ale nechávám to tu, protože později můžeš řešit:
    - rotaci profilu
    - nebo test geometrií už pootočených o AoA
    """
    angle_rad = math.radians(angle_deg)
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)

    rotated = []
    for x, y in points:
        xr = ca * x - sa * y
        yr = sa * x + ca * y
        rotated.append((xr, yr))
    return rotated


def write_dat_file(path: Path, case: AirfoilCase, loop_points: List[Tuple[float, float]]) -> None:
    """
    Zapíše profil do .dat formátu.

    To je užitečné:
    - pro rychlou kontrolu geometrie
    - pro další nástroje
    - pro dokumentaci
    """
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NACA {case.naca_code()}\n")
        for x, y in loop_points:
            f.write(f"{x:.8f} {y:.8f}\n")


def triangle_normal(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Spočítá normálu trojúhelníku pro STL facet.
    """
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
    c: Tuple[float, float, float]
) -> None:
    """
    Zapíše jeden trojúhelník do ASCII STL.
    """
    nx, ny, nz = triangle_normal(a, b, c)
    f.write(f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}\n")
    f.write("    outer loop\n")
    f.write(f"      vertex {a[0]:.8e} {a[1]:.8e} {a[2]:.8e}\n")
    f.write(f"      vertex {b[0]:.8e} {b[1]:.8e} {b[2]:.8e}\n")
    f.write(f"      vertex {c[0]:.8e} {c[1]:.8e} {c[2]:.8e}\n")
    f.write("    endloop\n")
    f.write("  endfacet\n")


def write_ascii_stl(path: Path, case: AirfoilCase, loop_points: List[Tuple[float, float]], z_min: float, z_max: float) -> None:
    """
    Vytvoří jednoduchý extrudovaný ASCII STL z 2D profilu.

    Pro náš OpenFOAM template je to přesně to, co potřebujeme:
    - 2D profil
    - tenká extruze v ose z
    - uzavřený STL objekt

    Tohle je testovací a praktická varianta.
    Později, pokud bude třeba, můžeme udělat i robustnější geometrii nebo blunt trailing edge.
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

        # Přední víko (z = z_min)
        for i in range(1, n - 1):
            write_facet(f, front[0], front[i + 1], front[i])

        # Zadní víko (z = z_max)
        for i in range(1, n - 1):
            write_facet(f, back[0], back[i], back[i + 1])

        f.write(f"endsolid {solid_name}\n")


def write_sampling_table(csv_path: Path, cases: List[AirfoilCase]) -> None:
    """
    Zapíše CSV tabulku s parametry všech vygenerovaných profilů.

    Tato tabulka bude důležitá později pro:
    - dataset index
    - mapování case_id -> geometrie
    - propojení s CFD a ML workflow
    """
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
    """
    Vrátí malý testovací seznam profilů.

    Tyto případy jsou jen na ověření:
    - že skript funguje
    - že se generují soubory
    - že geometrii umíme později poslat do OpenFOAM
    """
    return [
        AirfoilCase(camber_percent=0, camber_position_tenths=0, thickness_percent=12),
        AirfoilCase(camber_percent=2, camber_position_tenths=4, thickness_percent=12),
        AirfoilCase(camber_percent=4, camber_position_tenths=4, thickness_percent=15),
        AirfoilCase(camber_percent=2, camber_position_tenths=2, thickness_percent=8),
    ]


def get_real_run_cases() -> List[AirfoilCase]:
    """
    Vrátí širší sadu geometrií pro budoucí reálný běh.

    Zatím sem dávám rozumnou, ale stále malou mřížku parametrů.
    Později můžeme přejít na:
    - Latin Hypercube Sampling
    - náhodný sampling
    - separaci geometry vs operating conditions
    """
    cases: List[AirfoilCase] = []

    camber_values = [0, 2, 4]
    camber_pos_values = [2, 4]
    thickness_values = [8, 12, 16]

    for camber in camber_values:
        for camber_pos in camber_pos_values:
            for thickness in thickness_values:
                # U symetrického profilu musí být camber_pos = 0.
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

                # Nechceme duplicity typu 0012 vícekrát.
                if not any(
                    existing.naca_code() == case.naca_code() for existing in cases
                ):
                    cases.append(case)

    return cases


# ============================================================
# HLAVNÍ GENERAČNÍ LOGIKA
# ============================================================

def generate_airfoil_files(cases: List[AirfoilCase], output_dir: Path) -> None:
    """
    Pro každý profil:
    - vygeneruje geometrii
    - uloží .dat
    - uloží .stl

    Výstupní názvy souborů:
    - naca0012.dat
    - naca0012.stl
    """
    ensure_output_dir(output_dir)

    for case in cases:
        upper, lower = generate_naca4_coordinates(case, n_points=N_POINTS)
        loop = build_closed_profile_loop(upper, lower)

        dat_path = output_dir / f"naca{case.naca_code()}.dat"
        stl_path = output_dir / f"naca{case.naca_code()}.stl"

        write_dat_file(dat_path, case, loop)
        write_ascii_stl(stl_path, case, loop, z_min=Z_MIN, z_max=Z_MAX)

        print(f"Generated: {dat_path.name}, {stl_path.name}")


def main() -> None:
    """
    Hlavní vstupní bod skriptu.

    V test režimu vygeneruje jen pár profilů.
    V real-run režimu připraví širší sadu geometrií.
    """
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