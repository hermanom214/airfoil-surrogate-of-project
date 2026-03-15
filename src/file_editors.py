from __future__ import annotations

import math
import re
from pathlib import Path


def velocity_components(mag_u: float, aoa_deg: float) -> tuple[float, float]:
    """
    Přepočítá velikost rychlosti a angle of attack na složky Ux a Uy.
    """
    angle_rad = math.radians(aoa_deg)
    ux = mag_u * math.cos(angle_rad)
    uy = mag_u * math.sin(angle_rad)
    return ux, uy


def update_u_file(u_file: Path, inlet_velocity: float, aoa_deg: float) -> None:
    """
    Upraví soubor 0/U:
    - internalField
    - inlet value

    Očekává, že soubor obsahuje řádky:
    internalField   uniform (...);
    value           uniform (...);
    """
    text = u_file.read_text(encoding="utf-8")

    ux, uy = velocity_components(inlet_velocity, aoa_deg)
    vec = f"({ux:.6f} {uy:.6f} 0)"

    text = re.sub(
        r"internalField\s+uniform\s+\([^)]+\);",
        f"internalField   uniform {vec};",
        text,
    )

    text = re.sub(
        r"(inlet\s*\{.*?value\s+uniform\s+)\([^)]+\)(;)",
        rf"\g<1>{vec}\2",
        text,
        flags=re.DOTALL,
    )

    u_file.write_text(text, encoding="utf-8")


def update_force_coeffs_file(force_file: Path, inlet_velocity: float) -> None:
    """
    Upraví system/forceCoeffs:
    - magUInf

    Zatím nemění dragDir/liftDir.
    Ty necháme fixní, protože AoA řešíme natočením inlet velocity.
    """
    text = force_file.read_text(encoding="utf-8")

    text = re.sub(
        r"magUInf\s+[-+0-9.eE]+;",
        f"magUInf         {inlet_velocity:.6f};",
        text,
    )

    force_file.write_text(text, encoding="utf-8")