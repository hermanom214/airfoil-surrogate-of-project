from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from src.file_editors import update_force_coeffs_file, update_u_file
from src.sampling import BuildCaseRow


def case_name(index: int, naca_code: str, aoa_deg: float, inlet_velocity: float) -> str:
    """
    Vytvoří jméno case folderu.
    """
    aoa_str = str(aoa_deg).replace(".", "p").replace("-", "m")
    vel_str = str(inlet_velocity).replace(".", "p")
    return f"case_{index:04d}_naca{naca_code}_aoa{aoa_str}_u{vel_str}"


def prepare_case_folder(case_dir: Path, template_case: Path) -> None:
    """
    Smaže starý case folder, pokud existuje, a znovu ho vytvoří z template.
    """
    if case_dir.exists():
        shutil.rmtree(case_dir)

    shutil.copytree(template_case, case_dir)


def copy_stl_to_case(stl_source: Path, case_dir: Path) -> None:
    """
    Zkopíruje STL do case jako constant/triSurface/airfoil.stl
    """
    target = case_dir / "constant" / "triSurface" / "airfoil.stl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stl_source, target)


def write_params_json(case_dir: Path, row: BuildCaseRow, case_id: str, stl_source: Path) -> None:
    """
    Zapíše metadata case do params.json
    """
    payload = asdict(row)
    payload["naca_code"] = row.naca_code()
    payload["case_id"] = case_id
    payload["stl_source"] = str(stl_source)

    out_path = case_dir / "params.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def build_single_case(
    row: BuildCaseRow,
    index: int,
    template_case: Path,
    generated_profiles_dir: Path,
    openfoam_run_root: Path,
) -> Path:
    """
    Postaví jeden konkrétní OpenFOAM case:
    - zkopíruje template
    - vloží STL
    - upraví U
    - upraví forceCoeffs
    - uloží params.json

    Vrací cestu na vytvořený case.
    """
    def format_float_for_name(value: float) -> str:
        return str(value).replace("-", "m").replace(".", "p")

    aoa_name = format_float_for_name(row.aoa_deg)
    stl_source = generated_profiles_dir / f"naca{row.naca_code()}_aoa{aoa_name}.stl"

    if not stl_source.exists():
        raise FileNotFoundError(f"Missing STL file: {stl_source}")

    case_id = case_name(index, row.naca_code(), row.aoa_deg, row.inlet_velocity)
    case_dir = openfoam_run_root / case_id

    prepare_case_folder(case_dir, template_case)
    copy_stl_to_case(stl_source, case_dir)

    update_u_file(case_dir / "0" / "U", row.inlet_velocity)
    update_force_coeffs_file(case_dir / "system" / "forceCoeffs", row.inlet_velocity)

    write_params_json(case_dir, row, case_id, stl_source)

    return case_dir