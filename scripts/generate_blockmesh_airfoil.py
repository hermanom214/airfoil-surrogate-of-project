from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.blockmesh_generator import (  # noqa: E402
    CGridBlockMeshConfig,
    NACA4Params,
    write_blockmesh_dict,
)
from src.config import load_paths  # noqa: E402


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    paths = load_paths(config_path)

    case_dir = paths.openfoam_run_root / "blockmesh_test_naca0012_aoa0p0_u20p0"
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    naca = NACA4Params(
        camber_percent=0,
        camber_position_tenths=0,
        thickness_percent=12,
        chord=1.0,
    )

    cfg = CGridBlockMeshConfig(
        z_half=0.05,

        x_min=-5.0,
        x_max=12.0,
        y_min=-5.0,
        y_max=5.0,

        # konzervativní, ale hladký LE sampling
        n_airfoil_half=240,
        le_cluster_exp=2.2,

        n_streamwise_le=120,
        n_streamwise_near=120,
        n_wall_normal=80,
        n_wake_x=160,
        n_z=1,

        # stabilnější grading pro první funkční mesh
        grading_to_wall=250.0,
        grading_from_wall=0.004,
        grading_wake_x=6.0,

        # původní topologie tohle pořád potřebuje
        wake_cut_length=0.03,
    )

    write_blockmesh_dict(
        output_path=system_dir / "blockMeshDict",
        naca=naca,
        aoa_deg=0.0,
        cfg=cfg,
    )

    print(f"Wrote: {system_dir / 'blockMeshDict'}")


if __name__ == "__main__":
    main()