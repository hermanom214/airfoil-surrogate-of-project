from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


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

    rows: list[BuildCaseRow] = []
    for _, r in df.iterrows():
        rows.append(
            BuildCaseRow(
                camber_percent=int(r["camber_percent"]),
                camber_position_tenths=int(r["camber_position_tenths"]),
                thickness_percent=int(r["thickness_percent"]),
                chord=float(r["chord"]),
                aoa_deg=float(r["aoa_deg"]),
                inlet_velocity=float(r["inlet_velocity"]),
            )
        )
    return rows