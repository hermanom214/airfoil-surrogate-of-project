from __future__ import annotations

import math
import re
from pathlib import Path


def velocity_components(mag_u: float) -> tuple[float, float]:
    """
    Vrací rychlost pouze ve směru x.
    AoA se nebude řešit natočením rychlosti, ale natočením geometrie.
    """
    return mag_u, 0.0


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

    ux, uy = velocity_components(inlet_velocity)
    vec = f"({ux:.6f} {uy:.6f} 0)"

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


def update_force_coeffs_file(force_file: Path, inlet_velocity: float) -> None:
    """
    Upraví system/forceCoeffs:
    - magUInf

    AoA řešíme natočením inlet velocity v 0/U,
    proto dragDir/liftDir necháváme fixní.
    """
    if not force_file.exists():
        raise FileNotFoundError(f"Missing forceCoeffs file: {force_file}")

    text = force_file.read_text(encoding="utf-8")

    text, count = re.subn(
        r"magUInf\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*;",
        f"magUInf         {inlet_velocity:.6f};",
        text,
    )

    if count == 0:
        raise RuntimeError(f"Could not find magUInf entry in {force_file}")

    force_file.write_text(text, encoding="utf-8")