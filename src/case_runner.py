from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class CommandResult:
    command: str
    returncode: int
    runtime_sec: float
    log_file: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class CaseRunResult:
    case_id: str
    case_path: str
    blockmesh_status: str
    surfacefeatures_status: str
    snappyhexmesh_status: str
    checkmesh_status: str
    simplefoam_status: str
    overall_status: str
    runtime_sec: float


def run_command(
    command: list[str],
    case_dir: Path,
    log_dir: Path,
    log_name: str,
) -> CommandResult:
    """
    Spustí OpenFOAM command v daném case directory a uloží stdout/stderr do logu.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_name

    start = time.perf_counter()

    with log_file.open("w", encoding="utf-8") as f:
        process = subprocess.run(
            command,
            cwd=case_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            check=False,
        )

    runtime_sec = time.perf_counter() - start

    return CommandResult(
        command=" ".join(command),
        returncode=process.returncode,
        runtime_sec=runtime_sec,
        log_file=str(log_file),
    )


def discover_case_dirs(cases_root: Path) -> list[Path]:
    """
    Najde case složky. Jednoduché pravidlo:
    case je adresář obsahující podsložky 0, constant a system.
    """
    case_dirs: list[Path] = []

    for path in sorted(cases_root.iterdir()):
        if not path.is_dir():
            continue

        has_required = (
            (path / "0").is_dir()
            and (path / "constant").is_dir()
            and (path / "system").is_dir()
        )
        if has_required:
            case_dirs.append(path)

    return case_dirs


def run_single_case(case_dir: Path) -> CaseRunResult:
    """
    Spustí celý CFD chain pro jeden case.
    Pokud některý krok failne, další kroky se už nespustí.
    """
    case_id = case_dir.name
    log_dir = case_dir / "logs"

    overall_start = time.perf_counter()

    steps = [
        ("blockmesh_status", ["blockMesh"], "01_blockMesh.log"),
        ("surfacefeatures_status", ["surfaceFeatures"], "02_surfaceFeatures.log"),
        ("snappyhexmesh_status", ["snappyHexMesh", "-overwrite"], "03_snappyHexMesh.log"),
        ("checkmesh_status", ["checkMesh"], "04_checkMesh.log"),
        ("simplefoam_status", ["simpleFoam"], "05_simpleFoam.log"),
    ]

    statuses = {
        "blockmesh_status": "not_run",
        "surfacefeatures_status": "not_run",
        "snappyhexmesh_status": "not_run",
        "checkmesh_status": "not_run",
        "simplefoam_status": "not_run",
    }

    for status_key, command, log_name in steps:
        result = run_command(
            command=command,
            case_dir=case_dir,
            log_dir=log_dir,
            log_name=log_name,
        )

        if result.ok:
            statuses[status_key] = "ok"
        else:
            statuses[status_key] = f"failed({result.returncode})"
            runtime_sec = time.perf_counter() - overall_start
            return CaseRunResult(
                case_id=case_id,
                case_path=str(case_dir),
                blockmesh_status=statuses["blockmesh_status"],
                surfacefeatures_status=statuses["surfacefeatures_status"],
                snappyhexmesh_status=statuses["snappyhexmesh_status"],
                checkmesh_status=statuses["checkmesh_status"],
                simplefoam_status=statuses["simplefoam_status"],
                overall_status="failed",
                runtime_sec=round(runtime_sec, 3),
            )

    runtime_sec = time.perf_counter() - overall_start
    return CaseRunResult(
        case_id=case_id,
        case_path=str(case_dir),
        blockmesh_status=statuses["blockmesh_status"],
        surfacefeatures_status=statuses["surfacefeatures_status"],
        snappyhexmesh_status=statuses["snappyhexmesh_status"],
        checkmesh_status=statuses["checkmesh_status"],
        simplefoam_status=statuses["simplefoam_status"],
        overall_status="ok",
        runtime_sec=round(runtime_sec, 3),
    )


def write_run_status_csv(results: Iterable[CaseRunResult], output_csv: Path) -> None:
    """
    Zapíše výsledky běhů do CSV.
    """
    rows = [asdict(r) for r in results]
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "case_id",
                    "case_path",
                    "blockmesh_status",
                    "surfacefeatures_status",
                    "snappyhexmesh_status",
                    "checkmesh_status",
                    "simplefoam_status",
                    "overall_status",
                    "runtime_sec",
                ]
            )
        return

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)