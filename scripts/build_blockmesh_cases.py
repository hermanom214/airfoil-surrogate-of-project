from __future__ import annotations

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.case_builder import build_single_blockmesh_case  # noqa: E402
from src.config import load_paths  # noqa: E402
from src.sampling import load_sampling_table  # noqa: E402


def main() -> None:
    """
    Build blockMesh-based OpenFOAM cases from the sampling table.

    Uses the yPlus1 base template, writes a case-specific blockMeshDict and then
    mirrors the built cases to the OpenFOAM simulation directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "paths.yaml"

    paths = load_paths(config_path)
    rows = load_sampling_table(paths.sampling_table)

    template_case = project_root / "templates" / "openfoam_base_case_yPlus1"
    blockmesh_run_root = paths.openfoam_run_root / "blockmesh_cases"
    blockmesh_case_sim_root = paths.openfoam_case_sim / "blockmesh_cases"

    blockmesh_run_root.mkdir(parents=True, exist_ok=True)

    print(f"Template case: {template_case}")
    print(f"Sampling table: {paths.sampling_table}")
    print(f"Output run root: {blockmesh_run_root}")
    print(f"Simulation mirror root: {blockmesh_case_sim_root}")
    print(f"Total rows in sampling table: {len(rows)}")

    built_cases = []

    for i, row in enumerate(rows, start=1):
        case_dir = build_single_blockmesh_case(
            row=row,
            index=i,
            template_case=template_case,
            openfoam_run_root=blockmesh_run_root,
        )
        built_cases.append(case_dir)
        print(f"Built blockMesh case: {case_dir.name}")

    print("\nCopying blockMesh cases to OpenFOAM simulation directory...")
    blockmesh_case_sim_root.mkdir(parents=True, exist_ok=True)

    for case_dir in built_cases:
        target_dir = blockmesh_case_sim_root / case_dir.name

        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.copytree(case_dir, target_dir)
        print(f"Copied: {case_dir.name}")

    print("\nDone.")
    print(f"Total built blockMesh cases: {len(built_cases)}")


if __name__ == "__main__":
    main()