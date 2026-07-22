from __future__ import annotations

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.case_builder import build_single_blockmesh_case  # noqa: E402
from src.config import load_dataset_build_config, load_paths  # noqa: E402
from src.generate_sampling_table import generate_sampling_table  # noqa: E402
from src.sampling import load_sampling_table  # noqa: E402


def remove_case_directories(root: Path) -> int:
    """Remove all case directories directly below a managed output root."""
    if not root.exists():
        return 0

    removed = 0
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            removed += 1

    return removed


def main() -> None:
    """
    Build blockMesh-based OpenFOAM cases from the sampling table.

    Uses the yPlus1 base template, writes a case-specific blockMeshDict and then
    mirrors the built cases to the OpenFOAM simulation directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "paths.yaml"
    dataset_config_path = project_root / "configs" / "dataset_config.yaml"

    paths = load_paths(config_path)
    dataset_cfg = load_dataset_build_config(dataset_config_path)

    # Ensure geometry folders exist even if they were not created beforehand.
    geometry_root = paths.generated_profiles.parent
    geometry_root.mkdir(parents=True, exist_ok=True)
    paths.generated_profiles.mkdir(parents=True, exist_ok=True)

    # Always regenerate the sampling table from the current dataset config.
    n = generate_sampling_table(paths.sampling_table, dataset_cfg.sampling)
    print(f"[INFO] Regenerated sampling table: {paths.sampling_table} ({n} rows)")

    rows = load_sampling_table(paths.sampling_table)

    template_case = project_root / dataset_cfg.template_case_relpath
    blockmesh_run_root = paths.openfoam_run_root / dataset_cfg.blockmesh_cases_subdir
    blockmesh_case_sim_root = paths.openfoam_case_sim / dataset_cfg.blockmesh_cases_subdir

    blockmesh_run_root.mkdir(parents=True, exist_ok=True)
    blockmesh_case_sim_root.mkdir(parents=True, exist_ok=True)

    removed_run = remove_case_directories(blockmesh_run_root)
    print(f"[INFO] Removed old run case directories: {removed_run}")

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
            blockmesh_cfg=dataset_cfg.blockmesh,
        )
        built_cases.append(case_dir)
        print(f"Built blockMesh case: {case_dir.name}")

    print("\nCopying blockMesh cases to OpenFOAM simulation directory...")
    removed_sim = remove_case_directories(blockmesh_case_sim_root)
    print(f"[INFO] Removed old simulation case directories: {removed_sim}")

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
