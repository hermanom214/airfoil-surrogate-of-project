from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from src.blockmesh_generator import CGridBlockMeshConfig, NACA4Params, write_blockmesh_dict
from src.config import BlockMeshBuildConfig
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


def write_case_params_json(
    case_dir: Path,
    row: BuildCaseRow,
    case_id: str,
    template_case: Path,
) -> None:
    """
    Zapíše metadata blockMesh case do params.json.
    """
    payload = asdict(row)
    payload["naca_code"] = row.naca_code()
    payload["case_id"] = case_id
    payload["template_case"] = str(template_case)
    payload["mesh_type"] = "blockmesh"

    out_path = case_dir / "params.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def build_blockmesh_config(cfg: BlockMeshBuildConfig) -> CGridBlockMeshConfig:
    """
    Převede build konfiguraci na CGridBlockMeshConfig pro generátor blockMesh.
    """
    return CGridBlockMeshConfig(
        z_half=cfg.z_half,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        x_far=cfg.x_far,
        y_min=cfg.y_min,
        y_max=cfg.y_max,
        n_airfoil_half=cfg.n_airfoil_half,
        le_cluster_exp=cfg.le_cluster_exp,
        n_streamwise_near=cfg.n_streamwise_near,
        n_wall_normal=cfg.n_wall_normal,
        n_wake_x=cfg.n_wake_x,
        n_z=cfg.n_z,
        grading_to_wall=cfg.grading_to_wall,
        grading_le_tangent=cfg.grading_le_tangent,
        grading_wake_x=cfg.grading_wake_x,
        n_far_wake_x=cfg.n_far_wake_x,
        grading_far_wake_x=cfg.grading_far_wake_x,
        enable_le_cap=cfg.enable_le_cap,
        le_cap_fraction=cfg.le_cap_fraction,
        le_cap_power=cfg.le_cap_power,
        le_topology_fraction=cfg.le_topology_fraction,
        n_le_cap_normal=cfg.n_le_cap_normal,
        te_transition_fraction=cfg.te_transition_fraction,
        wake_cut_length=cfg.wake_cut_length,
    )


def build_single_blockmesh_case(
    row: BuildCaseRow,
    index: int,
    template_case: Path,
    openfoam_run_root: Path,
    blockmesh_cfg: BlockMeshBuildConfig,
) -> Path:
    """
    Postaví jeden konkrétní OpenFOAM case pro blockMesh větev:
    - zkopíruje template yPlus1 case
    - vygeneruje system/blockMeshDict
    - upraví U
    - upraví forceCoeffs
    - uloží params.json

    Vrací cestu na vytvořený case.
    """
    naca = NACA4Params(
        camber_percent=row.camber_percent,
        camber_position_tenths=row.camber_position_tenths,
        thickness_percent=row.thickness_percent,
        chord=row.chord,
    )

    cfg = build_blockmesh_config(blockmesh_cfg)

    case_id = case_name(index, row.naca_code(), row.aoa_deg, row.inlet_velocity)
    case_dir = openfoam_run_root / case_id

    prepare_case_folder(case_dir, template_case)

    blockmesh_path = case_dir / "system" / "blockMeshDict"
    write_blockmesh_dict(
        output_path=blockmesh_path,
        naca=naca,
        aoa_deg=row.aoa_deg,
        cfg=cfg,
    )

    update_u_file(case_dir / "0" / "U", row.inlet_velocity)
    update_force_coeffs_file(case_dir / "system" / "forceCoeffs", row.inlet_velocity)

    write_case_params_json(case_dir, row, case_id, template_case)

    return case_dir