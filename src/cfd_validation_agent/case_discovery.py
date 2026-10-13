from __future__ import annotations

from pathlib import Path

from .schemas import FileInventory, relative_source


def discover_cases(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("case_"))


def select_cases(root: Path, case_ids: list[str] | None, limit: int | None) -> list[Path]:
    cases = discover_cases(root)
    by_id = {path.name: path for path in cases}
    if case_ids:
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown case IDs: {', '.join(missing)}")
        selected = [by_id[case_id] for case_id in case_ids]
    else:
        selected = cases[:limit] if limit is not None else cases
    return selected


def inventory_case(case_dir: Path) -> FileInventory:
    log_files = sorted(path for path in case_dir.rglob("*.log") if path.is_file())
    residual_files = sorted(
        path for path in case_dir.rglob("residuals.dat")
        if "postProcessing" in path.parts and path.is_file()
    )
    force_files = sorted(
        path for path in case_dir.rglob("forceCoeffs.dat")
        if "postProcessing" in path.parts and path.is_file()
        and not path.name.endswith(".bak")
    )
    inspected = [*log_files, *residual_files, *force_files]
    return FileInventory(
        case_path=str(case_dir),
        log_files=[relative_source(path, case_dir) for path in log_files],
        residual_files=[relative_source(path, case_dir) for path in residual_files],
        force_coeff_files=[relative_source(path, case_dir) for path in force_files],
        empty_files=[relative_source(path, case_dir) for path in inspected if path.stat().st_size == 0],
    )
