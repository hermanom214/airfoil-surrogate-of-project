"""Migrate archived force coefficients from Aref=1.0 to the physical Aref=0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


MIGRATION_VERSION = 1
EXPECTED_HEADER = ("Time", "Cm", "Cd", "Cl", "Cl(f)", "Cl(r)")
COEFFICIENT_COLUMNS = ("Cm", "Cd", "Cl", "Cl(f)", "Cl(r)")
DATA_RELPATH = Path("postProcessing/forceCoeffs1/0/forceCoeffs.dat")
BACKUP_SUFFIX = ".pre_aref_1p0.bak"
METADATA_NAME = "forceCoeffs.reference_area_migration.json"


class MigrationError(RuntimeError):
    """Raised when migration state or input data is unsafe or ambiguous."""


@dataclass(frozen=True)
class MigrationPlan:
    case_dir: Path
    data_path: Path
    backup_path: Path
    metadata_path: Path
    before_bytes: bytes
    after_bytes: bytes
    checksum_before: str
    checksum_after: str


@dataclass
class Summary:
    discovered: int = 0
    eligible: int = 0
    migrated: int = 0
    skipped: int = 0
    failed: int = 0


def correction_factor(old_aref: float, new_aref: float) -> float:
    if old_aref <= 0.0 or new_aref <= 0.0:
        raise ValueError("reference areas must be > 0")
    return old_aref / new_aref


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _header_tokens(lines: Sequence[str]) -> tuple[str, ...]:
    headers = [line.lstrip()[1:].strip().split() for line in lines if line.lstrip().startswith("#")]
    matches = [tuple(tokens) for tokens in headers if tokens and tokens[0] == "Time"]
    if len(matches) != 1:
        raise MigrationError(f"expected exactly one '# Time ...' header, found {len(matches)}")
    return matches[0]


def transform_force_coeffs(data: bytes, factor: float) -> bytes:
    """Scale only Aref-dependent coefficient columns; preserve comments and time tokens."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MigrationError("forceCoeffs.dat is not UTF-8 text") from exc
    lines = text.splitlines(keepends=True)
    header = _header_tokens(lines)
    if header != EXPECTED_HEADER:
        raise MigrationError(f"unexpected coefficient header: {header!r}")

    output: list[str] = []
    data_rows = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        tokens = stripped.split()
        if len(tokens) != len(EXPECTED_HEADER):
            raise MigrationError(
                f"line {line_number}: expected {len(EXPECTED_HEADER)} columns, got {len(tokens)}"
            )
        try:
            values = [float(value) for value in tokens]
        except ValueError as exc:
            raise MigrationError(f"line {line_number}: non-numeric data row") from exc
        if not all(math.isfinite(value) for value in values):
            raise MigrationError(f"line {line_number}: non-finite data row")
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        scaled = [values[index] * factor for index in range(1, len(values))]
        output.append(tokens[0] + "\t" + "\t".join(f"{value:.8e}" for value in scaled) + newline)
        data_rows += 1
    if data_rows == 0:
        raise MigrationError("forceCoeffs.dat contains no numeric rows")
    return "".join(output).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _validate_completed_migration(
    data_path: Path, backup_path: Path, metadata_path: Path, old_aref: float, new_aref: float
) -> None:
    if not backup_path.is_file():
        raise MigrationError(f"metadata exists but backup is missing: {backup_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "migration_version": MIGRATION_VERSION,
        "old_Aref": old_aref,
        "new_Aref": new_aref,
        "scale_factor": correction_factor(old_aref, new_aref),
        "affected_file": DATA_RELPATH.as_posix(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise MigrationError(f"existing metadata has unexpected {key}: {metadata.get(key)!r}")
    if _sha256(data_path.read_bytes()) != metadata.get("checksum_after"):
        raise MigrationError("migrated data checksum does not match metadata")
    if _sha256(backup_path.read_bytes()) != metadata.get("checksum_before"):
        raise MigrationError("backup checksum does not match metadata")


def _validate_case_sidecars(case_dir: Path, expected_aref: float) -> None:
    force_dict = case_dir / "system" / "forceCoeffs"
    if not force_dict.is_file():
        raise MigrationError(f"missing archived force dictionary: {force_dict}")
    matches = re.findall(
        r"\bAref\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;",
        force_dict.read_text(encoding="utf-8"),
    )
    if len(matches) != 1 or not math.isclose(
        float(matches[0]), expected_aref, rel_tol=0.0, abs_tol=1e-12
    ):
        raise MigrationError(f"expected exactly one Aref={expected_aref:g} in {force_dict}")
    params_path = case_dir / "params.json"
    if not params_path.is_file():
        raise MigrationError(f"missing case metadata: {params_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    chord = float(params["chord"])
    if chord <= 0.0:
        raise MigrationError(f"invalid chord in {params_path}")
    if "Aref" in params and not math.isclose(
        float(params["Aref"]), expected_aref, rel_tol=0.0, abs_tol=1e-12
    ):
        raise MigrationError(f"unexpected Aref in {params_path}")


def plan_case(case_dir: Path, old_aref: float, new_aref: float) -> MigrationPlan | None:
    data_path = case_dir / DATA_RELPATH
    backup_path = data_path.with_name(data_path.name + BACKUP_SUFFIX)
    metadata_path = data_path.with_name(METADATA_NAME)
    if not data_path.is_file():
        raise MigrationError(f"missing {DATA_RELPATH.as_posix()}")
    if metadata_path.exists():
        _validate_completed_migration(data_path, backup_path, metadata_path, old_aref, new_aref)
        _validate_case_sidecars(case_dir, new_aref)
        return None
    if backup_path.exists():
        raise MigrationError(f"backup exists without migration metadata: {backup_path}")
    _validate_case_sidecars(case_dir, old_aref)
    before = data_path.read_bytes()
    after = transform_force_coeffs(before, correction_factor(old_aref, new_aref))
    return MigrationPlan(
        case_dir, data_path, backup_path, metadata_path, before, after,
        _sha256(before), _sha256(after),
    )


def _update_force_dictionary(case_dir: Path, old_aref: float, new_aref: float) -> None:
    path = case_dir / "system" / "forceCoeffs"
    if not path.is_file():
        raise MigrationError(f"missing archived force dictionary: {path}")
    text = path.read_text(encoding="utf-8")
    match = re.findall(r"\bAref\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;", text)
    if len(match) != 1 or not math.isclose(float(match[0]), old_aref, rel_tol=0.0, abs_tol=1e-12):
        raise MigrationError(f"expected exactly one Aref={old_aref:g} in {path}")
    updated = re.sub(
        r"\bAref\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*;",
        f"Aref            {new_aref:.12g};",
        text,
        count=1,
    )
    _atomic_write(path, updated.encode("utf-8"))


def _update_params(case_dir: Path, new_aref: float) -> None:
    path = case_dir / "params.json"
    if not path.is_file():
        raise MigrationError(f"missing case metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chord = float(payload["chord"])
    span = new_aref / chord
    payload.update({
        "span": span,
        "Aref": new_aref,
        "lRef": chord,
        "force_coefficient_convention": "planform_area_chord_times_actual_span",
    })
    _atomic_write(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def apply_plan(plan: MigrationPlan, old_aref: float, new_aref: float) -> None:
    # The immutable backup is written first; subsequent runs fail closed if interrupted.
    with plan.backup_path.open("xb") as backup:
        backup.write(plan.before_bytes)
        backup.flush()
        os.fsync(backup.fileno())
    _atomic_write(plan.data_path, plan.after_bytes)
    _update_force_dictionary(plan.case_dir, old_aref, new_aref)
    _update_params(plan.case_dir, new_aref)
    metadata = {
        "migration_version": MIGRATION_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "old_Aref": old_aref,
        "new_Aref": new_aref,
        "scale_factor": correction_factor(old_aref, new_aref),
        "coefficient_columns_scaled": list(COEFFICIENT_COLUMNS),
        "affected_file": DATA_RELPATH.as_posix(),
        "backup_file": plan.backup_path.name,
        "checksum_before": plan.checksum_before,
        "checksum_after": plan.checksum_after,
    }
    _atomic_write(plan.metadata_path, (json.dumps(metadata, indent=2) + "\n").encode("utf-8"))


def migrate(root: Path, *, apply: bool, old_aref: float = 1.0, new_aref: float = 0.1) -> Summary:
    summary = Summary()
    plans: list[MigrationPlan] = []
    case_dirs = sorted(path for path in root.glob("case_*") if path.is_dir())
    summary.discovered = len(case_dirs)
    for case_dir in case_dirs:
        try:
            plan = plan_case(case_dir, old_aref, new_aref)
            if plan is None:
                summary.skipped += 1
            else:
                plans.append(plan)
                summary.eligible += 1
        except (MigrationError, KeyError, ValueError, json.JSONDecodeError) as exc:
            summary.failed += 1
            print(f"FAILED {case_dir.name}: {exc}")
    if summary.failed:
        raise MigrationError(f"preflight failed for {summary.failed} case(s); no migration applied")
    if apply:
        for plan in plans:
            apply_plan(plan, old_aref, new_aref)
            summary.migrated += 1
    mode = "APPLY" if apply else "DRY-RUN"
    print(
        f"{mode}: discovered={summary.discovered} eligible={summary.eligible} "
        f"migrated={summary.migrated} skipped={summary.skipped} failed={summary.failed}"
    )
    if not apply:
        print("No files were modified.")
    else:
        print(f"Backups use suffix: {BACKUP_SUFFIX}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("data/flow_fields"))
    parser.add_argument("--old-aref", type=float, default=1.0)
    parser.add_argument("--new-aref", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        migrate(args.root, apply=args.apply, old_aref=args.old_aref, new_aref=args.new_aref)
    except MigrationError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
