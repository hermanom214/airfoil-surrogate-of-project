"""
Generování sampling tabulky pro blockMesh workflow.

Modul vytváří kombinace NACA 4-digit profilů a provozních podmínek
(AoA, inlet velocity) a ukládá je do CSV. Negeneruje žádnou geometrii
(DAT/STL) – pouze tabulku parametrů pro build krok.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

from src.config import SamplingConfig


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

def get_base_cases(cfg: SamplingConfig) -> list[AirfoilCase]:
    """Return unique NACA profiles used in the sweep."""
    base_cases: list[AirfoilCase] = []
    seen_codes: set[str] = set()

    camber_values = cfg.camber_values
    camber_pos_values = cfg.camber_position_values
    thickness_values = cfg.thickness_values

    for camber in camber_values:
        for camber_pos in camber_pos_values:
            for thickness in thickness_values:
                if camber == 0:
                    case = AirfoilCase(
                        camber_percent=0,
                        camber_position_tenths=0,
                        thickness_percent=thickness,
                        chord=cfg.chord,
                    )
                else:
                    case = AirfoilCase(
                        camber_percent=camber,
                        camber_position_tenths=camber_pos,
                        thickness_percent=thickness,
                        chord=cfg.chord,
                    )

                code = case.naca_code()
                if code not in seen_codes:
                    seen_codes.add(code)
                    base_cases.append(case)

    return base_cases


def expand_cases_with_conditions(
    base_cases: list[AirfoilCase],
    cfg: SamplingConfig,
) -> list[AirfoilCase]:
    """Expand each base profile with all AoA × inlet velocity combinations."""
    expanded: list[AirfoilCase] = []

    for case in base_cases:
        for aoa in cfg.aoa_values:
            for inlet_velocity in cfg.inlet_velocity_values:
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

def write_sampling_table(csv_path: Path, cases: list[AirfoilCase]) -> None:
    """Write the full case list to CSV."""
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

def generate_sampling_table(csv_path: Path, cfg: SamplingConfig) -> int:
    """
    Generates the sampling table CSV at *csv_path*.

    Returns the number of rows written.
    """
    cases = expand_cases_with_conditions(get_base_cases(cfg), cfg)
    write_sampling_table(csv_path, cases)
    return len(cases)
