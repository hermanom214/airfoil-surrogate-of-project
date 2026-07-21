from __future__ import annotations

import csv
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from src.config import load_paths, load_solver_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"
SOLVER_CONFIG_PATH = PROJECT_ROOT / "configs" / "solver_config.yaml"
PATHS = load_paths(PATHS_CONFIG_PATH)
SOLVER_CFG = load_solver_config(SOLVER_CONFIG_PATH)
OF_BASH = str(PATHS.openfoam_bash)
N_PROCS = SOLVER_CFG.case_runner.n_procs
REQUIRED_CASE_SUBDIRS = tuple(SOLVER_CFG.case_runner.required_case_subdirs)
FLOW_FIELDS_OUTPUT = PATHS.flow_fields_output

SIMPLEFOAM_MAX_LINEAR_ITERS = 1000
SIMPLEFOAM_RESIDUAL_STOP_THRESHOLD = 1e2


def to_cygwin_path(path: Path) -> str:
    """
    Convert Windows path like C:\\OpenFOAM\\... to Cygwin path like /cygdrive/c/OpenFOAM/...
    """
    p = path.resolve()
    drive = p.drive.rstrip(":").lower()
    rest = p.relative_to(p.anchor).as_posix()
    return f"/cygdrive/{drive}/{rest}"

@dataclass
class CommandResult:
    command: str
    returncode: int
    runtime_sec: float
    log_file: str
    abort_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.abort_reason is None


@dataclass
class CaseRunResult:
    case_id: str
    case_path: str
    blockmesh_status: str
    checkmesh_status: str
    simplefoam_status: str
    foamtovtk_status: str
    archive_status: str
    cleanup_case_status: str
    overall_status: str
    runtime_sec: float


def run_command(
    command: list[str],
    case_dir: Path,
    log_dir: Path,
    log_name: str,
) -> CommandResult:
    """
    Runs an OpenFOAM command inside a case directory and writes stdout/stderr to a log file.
    Can be launched from normal VS Code/PowerShell because the actual command is executed
    through the OpenFOAM Cygwin bash.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_name

    start = time.perf_counter()

    cyg_case_dir = to_cygwin_path(case_dir)

    def wrap_parallel(cmd: list[str]) -> str:
        if "-parallel" in cmd:
            return f"mpiexec -np {N_PROCS} {' '.join(cmd)}"
        return " ".join(cmd)

    foam_command = wrap_parallel(command)

    wrapped_command = [
        OF_BASH,
        "--login",
        "-i",
        "-c",
        f"cd '{cyg_case_dir}' && {foam_command}",
    ]

    abort_reason: str | None = None

    with log_file.open("w", encoding="utf-8") as f:
        should_watch_simplefoam = any(part == "simpleFoam" for part in command)

        if not should_watch_simplefoam:
            process = subprocess.run(
                wrapped_command,
                cwd=case_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                check=False,
            )
            returncode = process.returncode
        else:
            process = subprocess.Popen(
                wrapped_command,
                cwd=case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                f.write(line)

                reason = detect_simplefoam_divergence_reason(line)
                if reason is not None:
                    abort_reason = reason
                    process.terminate()
                    break

            if abort_reason is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            else:
                process.wait()

            returncode = process.returncode

    runtime_sec = time.perf_counter() - start

    return CommandResult(
        command=" ".join(wrapped_command),
        returncode=returncode,
        runtime_sec=runtime_sec,
        log_file=str(log_file),
        abort_reason=abort_reason,
    )


def detect_simplefoam_divergence_reason(log_line: str) -> str | None:
    """
    Detect obviously divergent simpleFoam behavior from a live log line.
    Stop only when linear solver hits max iterations and residual is NaN or huge.
    """
    match = re.search(
        r"Final residual\s*=\s*([^,]+),\s*No Iterations\s*(\d+)",
        log_line,
    )
    if match is None:
        return None

    residual_text = match.group(1).strip().lower()
    try:
        iter_count = int(match.group(2))
    except ValueError:
        return None

    if iter_count != SIMPLEFOAM_MAX_LINEAR_ITERS:
        return None

    if residual_text in {"nan", "-nan", "+nan", "inf", "-inf", "+inf"}:
        return "simpleFoam_diverged_nan"

    try:
        residual_value = abs(float(residual_text))
    except ValueError:
        return None

    if residual_value >= SIMPLEFOAM_RESIDUAL_STOP_THRESHOLD:
        return (
            "simpleFoam_diverged_residual_"
            f"{residual_value:.3e}_at_{SIMPLEFOAM_MAX_LINEAR_ITERS}_iters"
        )

    return None


def discover_case_dirs(cases_root: Path) -> list[Path]:
    """
    Finds case directories. A valid case contains 0, constant, and system folders.
    """
    if not cases_root.exists():
        return []

    case_dirs: list[Path] = []

    for path in sorted(cases_root.iterdir()):
        if not path.is_dir():
            continue

        has_required = all((path / name).is_dir() for name in REQUIRED_CASE_SUBDIRS)

        if has_required:
            case_dirs.append(path)

    return case_dirs


def cleanup_processor_dirs(case_dir: Path) -> int:
    """
    Remove OpenFOAM parallel decomposition folders like processor0, processor1, ...
    after a case run to save disk space.
    """
    removed = 0

    for path in case_dir.glob("processor*"):
        if not path.is_dir():
            continue

        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as e:
            print(f"[WARN] {case_dir.name}: nepodařilo se smazat {path.name} ({e})")

    return removed


def _safe_copytree(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        raise RuntimeError(f"missing directory: {src}")
    shutil.copytree(src, dst)


def archive_case_outputs(case_dir: Path, flow_fields_root: Path) -> tuple[bool, str]:
    """Copy selected case artifacts into data/flow_fields/<case_id>."""
    case_id = case_dir.name
    target_case_dir = flow_fields_root / case_id

    if target_case_dir.exists():
        return False, f"target_exists({target_case_dir})"

    required_dirs = ["logs", "postProcessing", "system", "VTK"]
    optional_files = ["params.json"]

    try:
        target_case_dir.mkdir(parents=True, exist_ok=False)

        for directory_name in required_dirs:
            src = case_dir / directory_name
            dst = target_case_dir / directory_name
            _safe_copytree(src, dst)

        for file_name in optional_files:
            src_file = case_dir / file_name
            if src_file.is_file():
                shutil.copy2(src_file, target_case_dir / file_name)

        latest_time_name = find_latest_numeric_time(case_dir)
        latest_time_src = case_dir / latest_time_name
        latest_time_dst = target_case_dir / latest_time_name
        _safe_copytree(latest_time_src, latest_time_dst)

    except Exception as e:
        if target_case_dir.exists():
            shutil.rmtree(target_case_dir, ignore_errors=True)
        return False, f"copy_failed({e})"

    return True, "ok"


def find_latest_numeric_time(case_dir: Path) -> str:
    """Return latest numeric OpenFOAM time directory name."""
    time_dirs: list[tuple[float, str]] = []
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            numeric_time = float(path.name)
        except ValueError:
            continue
        time_dirs.append((numeric_time, path.name))

    if not time_dirs:
        raise RuntimeError(f"No numeric time directory found in {case_dir}")

    return sorted(time_dirs, key=lambda item: item[0])[-1][1]


def run_single_case(case_dir: Path) -> CaseRunResult:
    """
    Runs the full CFD chain for one case.
    If one step fails, the remaining steps are skipped.
    """
    case_id = case_dir.name
    log_dir = case_dir / "logs"

    overall_start = time.perf_counter()

    statuses = {
        "blockmesh_status": "not_run",
        "checkmesh_status": "not_run",
        "simplefoam_status": "not_run",
        "foamtovtk_status": "not_run",
        "archive_status": "not_run",
        "cleanup_case_status": "not_run",
    }

    for step in SOLVER_CFG.case_runner.run_steps:
        result = run_command(
            command=step.command,
            case_dir=case_dir,
            log_dir=log_dir,
            log_name=step.log_name,
        )

        if result.ok:
            if step.status_key is not None:
                statuses[step.status_key] = "ok"
        else:
            if step.status_key is not None:
                if result.abort_reason is not None:
                    statuses[step.status_key] = f"nOK({result.abort_reason})"
                else:
                    statuses[step.status_key] = f"failed({result.returncode})"
            runtime_sec = time.perf_counter() - overall_start

            removed_count = cleanup_processor_dirs(case_dir)
            if removed_count:
                print(f"[CLEANUP] {case_id}: odstraněno {removed_count}x processor* složka")

            return CaseRunResult(
                case_id=case_id,
                case_path=str(case_dir),
                blockmesh_status=statuses["blockmesh_status"],
                checkmesh_status=statuses["checkmesh_status"],
                simplefoam_status=statuses["simplefoam_status"],
                foamtovtk_status=statuses["foamtovtk_status"],
                archive_status=statuses["archive_status"],
                cleanup_case_status=statuses["cleanup_case_status"],
                overall_status="nOK" if result.abort_reason is not None else "failed",
                runtime_sec=round(runtime_sec, 3),
            )

    foam_to_vtk_result = run_command(
        command=SOLVER_CFG.foam_to_vtk.command,
        case_dir=case_dir,
        log_dir=log_dir,
        log_name=SOLVER_CFG.foam_to_vtk.log_name,
    )

    if foam_to_vtk_result.ok:
        statuses["foamtovtk_status"] = "ok"
    else:
        statuses["foamtovtk_status"] = f"failed({foam_to_vtk_result.returncode})"
        runtime_sec = time.perf_counter() - overall_start
        print(
            f"[WARN] {case_id}: foamToVTK failed, case kept for inspection "
            f"at {case_dir}"
        )
        return CaseRunResult(
            case_id=case_id,
            case_path=str(case_dir),
            blockmesh_status=statuses["blockmesh_status"],
            checkmesh_status=statuses["checkmesh_status"],
            simplefoam_status=statuses["simplefoam_status"],
            foamtovtk_status=statuses["foamtovtk_status"],
            archive_status=statuses["archive_status"],
            cleanup_case_status=statuses["cleanup_case_status"],
            overall_status="failed",
            runtime_sec=round(runtime_sec, 3),
        )

    archived, archive_status = archive_case_outputs(case_dir, FLOW_FIELDS_OUTPUT)
    statuses["archive_status"] = archive_status
    if not archived:
        runtime_sec = time.perf_counter() - overall_start
        print(
            f"[WARN] {case_id}: archive failed ({archive_status}), "
            f"case kept for inspection at {case_dir}"
        )
        return CaseRunResult(
            case_id=case_id,
            case_path=str(case_dir),
            blockmesh_status=statuses["blockmesh_status"],
            checkmesh_status=statuses["checkmesh_status"],
            simplefoam_status=statuses["simplefoam_status"],
            foamtovtk_status=statuses["foamtovtk_status"],
            archive_status=statuses["archive_status"],
            cleanup_case_status=statuses["cleanup_case_status"],
            overall_status="failed",
            runtime_sec=round(runtime_sec, 3),
        )

    try:
        shutil.rmtree(case_dir)
        statuses["cleanup_case_status"] = "ok"
    except OSError as e:
        statuses["cleanup_case_status"] = f"failed({e})"
        print(
            f"[WARN] {case_id}: nepodařilo se smazat výpočetní složku "
            f"{case_dir} ({e})"
        )
        runtime_sec = time.perf_counter() - overall_start
        return CaseRunResult(
            case_id=case_id,
            case_path=str(case_dir),
            blockmesh_status=statuses["blockmesh_status"],
            checkmesh_status=statuses["checkmesh_status"],
            simplefoam_status=statuses["simplefoam_status"],
            foamtovtk_status=statuses["foamtovtk_status"],
            archive_status=statuses["archive_status"],
            cleanup_case_status=statuses["cleanup_case_status"],
            overall_status="failed",
            runtime_sec=round(runtime_sec, 3),
        )

    runtime_sec = time.perf_counter() - overall_start

    return CaseRunResult(
        case_id=case_id,
        case_path=str(case_dir),
        blockmesh_status=statuses["blockmesh_status"],
        checkmesh_status=statuses["checkmesh_status"],
        simplefoam_status=statuses["simplefoam_status"],
        foamtovtk_status=statuses["foamtovtk_status"],
        archive_status=statuses["archive_status"],
        cleanup_case_status=statuses["cleanup_case_status"],
        overall_status="ok",
        runtime_sec=round(runtime_sec, 3),
    )


def write_run_status_csv(results: Iterable[CaseRunResult], output_csv: Path) -> None:
    """
    Writes run results to CSV.
    """
    rows = [asdict(r) for r in results]
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "case_path",
        "blockmesh_status",
        "checkmesh_status",
        "simplefoam_status",
        "foamtovtk_status",
        "archive_status",
        "cleanup_case_status",
        "overall_status",
        "runtime_sec",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if rows:
            writer.writerows(rows)