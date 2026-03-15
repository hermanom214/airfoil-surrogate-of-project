from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ProjectPaths:
    project_root: Path
    template_case: Path
    generated_profiles: Path
    sampling_table: Path
    openfoam_run_root: Path


def load_paths(config_path: Path) -> ProjectPaths:
    """
    Načte YAML config s cestami a vrátí je jako Path objekty.
    """
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ProjectPaths(
        project_root=Path(data["project_root_windows"]),
        template_case=Path(data["template_case_windows"]),
        generated_profiles=Path(data["generated_profiles_windows"]),
        sampling_table=Path(data["sampling_table_windows"]),
        openfoam_run_root=Path(data["openfoam_run_root_windows"]),
    )