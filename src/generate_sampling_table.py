"""
Generates the airfoil sampling table (CSV) used by build_blockmesh_cases.

Defines the NACA 4-digit parameter sweep and expands it with all CFD conditions
(angle of attack, inlet velocity). No STL or DAT geometry files are produced here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


# ============================================================
# CFD SWEEP CONFIGURATION
# ============================================================

AOA_VALUES: list[float] = [-4.0, 0.0, 4.0]
INLET_VELOCITY_VALUES: list[float] = [15.0, 20.0, 25.0]
CHORD: float = 1.0


# ============================================================
# DATA MODEL
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
# CASE LISTS
# ============================================================

def get_base_cases() -> List[AirfoilCase]:
    """Returns the list of unique NACA profiles to simulate."""
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

    return base_cases


def expand_cases_with_conditions(base_cases: List[AirfoilCase]) -> List[AirfoilCase]:
    """Expands each base profile with all AoA × inlet velocity combinations."""
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


# ============================================================
# CSV WRITE
# ============================================================

def write_sampling_table(csv_path: Path, cases: List[AirfoilCase]) -> None:
    """Writes the full case list to a CSV file."""
    fieldnames = [
        "naca_code",
        "camber_percent",
        "camber_position_tenths",
        "thickness_percent",
        "chord",
        "aoa_deg",
        "inlet_velocity",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            row = asdict(case)
            row["naca_code"] = case.naca_code()
            writer.writerow(row)


# ============================================================
# PUBLIC API
# ============================================================

def generate_sampling_table(csv_path: Path) -> int:
    """
    Generates the sampling table CSV at *csv_path*.

    Returns the number of rows written.
    """
    cases = expand_cases_with_conditions(get_base_cases())
    write_sampling_table(csv_path, cases)
    return len(cases)
