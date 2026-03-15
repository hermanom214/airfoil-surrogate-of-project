from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.case_builder import build_single_case
from src.config import load_paths
from src.sampling import load_sampling_table


def main() -> None:
    """
    Hlavní build skript:
    - načte config
    - načte sampling table
    - postaví case foldery v OpenFOAM run directory
    """
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "paths.yaml"

    paths = load_paths(config_path)
    rows = load_sampling_table(paths.sampling_table)

    paths.openfoam_run_root.mkdir(parents=True, exist_ok=True)

    print(f"Template case: {paths.template_case}")
    print(f"Sampling table: {paths.sampling_table}")
    print(f"Output run root: {paths.openfoam_run_root}")
    print(f"Total rows in sampling table: {len(rows)}")

    built_cases = []

    for i, row in enumerate(rows, start=1):
        case_dir = build_single_case(
            row=row,
            index=i,
            template_case=paths.template_case,
            generated_profiles_dir=paths.generated_profiles,
            openfoam_run_root=paths.openfoam_run_root,
        )
        built_cases.append(case_dir)
        print(f"Built case: {case_dir.name}")

    print("\nDone.")
    print(f"Total built cases: {len(built_cases)}")


if __name__ == "__main__":
    main()