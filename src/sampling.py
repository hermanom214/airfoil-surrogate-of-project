"""
Načítání sampling tabulky pro build OpenFOAM case.

Modul převádí řádky CSV na typovaný datový model `BuildCaseRow`,
který používají build skripty pro generování blockMesh case složek.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = (
    "camber_percent",
    "camber_position_tenths",
    "thickness_percent",
    "chord",
    "aoa_deg",
    "inlet_velocity",
)


@dataclass
class BuildCaseRow:
    camber_percent: int
    camber_position_tenths: int
    thickness_percent: int
    chord: float
    aoa_deg: float
    inlet_velocity: float

    def naca_code(self) -> str:
        """
        Složí správný NACA 4-digit kód, např. 0012 nebo 2412.
        """
        return f"{self.camber_percent}{self.camber_position_tenths}{self.thickness_percent:02d}"


def load_sampling_table(csv_path: Path) -> list[BuildCaseRow]:
    """
    Načte sampling CSV a vrátí seznam řádků pro stavbu case folderů.
    """
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Sampling table {csv_path} is missing required columns: {missing}"
        )

    rows: list[BuildCaseRow] = []
    for r in df.itertuples(index=False):
        rows.append(
            BuildCaseRow(
                camber_percent=int(r.camber_percent),
                camber_position_tenths=int(r.camber_position_tenths),
                thickness_percent=int(r.thickness_percent),
                chord=float(r.chord),
                aoa_deg=float(r.aoa_deg),
                inlet_velocity=float(r.inlet_velocity),
            )
        )
    return rows