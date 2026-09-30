"""
Utility funkce pro úpravu OpenFOAM case souborů po zkopírování template.

Modul zajišťuje:
- nastavení inlet/internal rychlosti v souboru 0/U,
- nastavení referenční rychlosti magUInf v system/forceCoeffs.

Používá se při build procesu case folderů ve blockMesh workflow.
"""

from __future__ import annotations

import re
from pathlib import Path


def force_reference_area(chord: float, z_half: float) -> float:
    """Return planform reference area for the actually extruded 2D mesh."""
    if chord <= 0.0:
        raise ValueError("chord must be > 0")
    if z_half <= 0.0:
        raise ValueError("z_half must be > 0")
    return chord * (2.0 * z_half)

def update_u_file(u_file: Path, inlet_velocity: float) -> None:
    """
    Upraví soubor 0/U:
    - internalField
    - inlet value

    Rychlost je vždy pouze ve směru x:
    U = (inlet_velocity, 0, 0)
    """
    if not u_file.exists():
        raise FileNotFoundError(f"Missing U file: {u_file}")

    text = u_file.read_text(encoding="utf-8")

    vec = f"({inlet_velocity:.6f} {0.0:.6f} 0)"

    text, count_internal = re.subn(
        r"internalField\s+uniform\s+\([^)]+\);",
        f"internalField   uniform {vec};",
        text,
    )

    text, count_inlet = re.subn(
        r"(inlet\s*\{.*?value\s+uniform\s+)\([^)]+\)(;)",
        rf"\g<1>{vec}\2",
        text,
        flags=re.DOTALL,
    )

    if count_internal == 0:
        raise RuntimeError(f"Could not find internalField in {u_file}")

    if count_inlet == 0:
        raise RuntimeError(f"Could not find inlet value in {u_file}")

    u_file.write_text(text, encoding="utf-8")


def update_force_coeffs_file(
    force_file: Path,
    inlet_velocity: float,
    chord: float,
    z_half: float,
) -> None:
    """
    Upraví system/forceCoeffs:
    - magUInf
    - lRef from chord
    - Aref from chord * actual extrusion span

    AoA řešíme natočením geometrie,
    proto dragDir/liftDir necháváme fixní.
    """
    if not force_file.exists():
        raise FileNotFoundError(f"Missing forceCoeffs file: {force_file}")

    text = force_file.read_text(encoding="utf-8")

    text, count_u = re.subn(
        r"magUInf\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*;",
        f"magUInf         {inlet_velocity:.6f};",
        text,
    )

    if count_u != 1:
        raise RuntimeError(f"Could not find magUInf entry in {force_file}")

    area = force_reference_area(chord, z_half)
    text, count_lref = re.subn(
        r"lRef\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*;",
        f"lRef            {chord:.12g};",
        text,
    )
    text, count_aref = re.subn(
        r"Aref\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*;",
        f"Aref            {area:.12g};",
        text,
    )
    if count_lref != 1:
        raise RuntimeError(f"Could not find exactly one lRef entry in {force_file}")
    if count_aref != 1:
        raise RuntimeError(f"Could not find exactly one Aref entry in {force_file}")

    force_file.write_text(text, encoding="utf-8")
