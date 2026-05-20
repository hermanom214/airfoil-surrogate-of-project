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

        # extra smooth LE: dense profile sampling + balanced LE clustering
        n_airfoil_half=520,
        le_cluster_exp=2.8,

        n_streamwise_le=180,
        n_streamwise_near=240,
        n_wall_normal=170,
        n_wake_x=320,
        n_z=1,

        # conservatively shrink first wall cell (towards y+~1) with extra cells,
        # avoiding harsh jumps in the normal direction.
        grading_to_wall=2400.0,
        grading_from_wall=0.006,
        grading_le_tangent=0.65,
        grading_wake_x=1.35,

        # keep geometric LE cap off when using topological LE cap
        enable_le_cap=False,
        le_cap_fraction=0.035,
        le_cap_power=1.2,
        le_topology_fraction=0.028,
        n_le_cap_normal=42,
        te_transition_fraction=0.995,

        # Tiny downstream wake cut keeps the trailing-edge cap short and blunt.
        wake_cut_length=0.0001,
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