from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


INSPECTION_CSV_NAME = "inspect_plausibility.csv"
REQUIRED_INSPECTION_COLUMNS = {"case_name", "overall_status", "overall_reason"}
RECOGNIZED_INSPECTION_STATUSES = {"OK", "nOK"}
CASE_ID_RE = re.compile(r"^case_[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class InspectionMembership:
    csv_path: Path
    case_statuses: dict[str, str]
    exclusion_reasons: dict[str, str]
    ok_case_ids: list[str]
    nok_case_ids: list[str]
    membership_fingerprint: str
    artifact_sha256: str


def inspection_csv_path(data_dir: Path) -> Path:
    return data_dir.parent / "pictures_inspect_flow" / INSPECTION_CSV_NAME


def case_membership_fingerprint(case_ids: list[str]) -> str:
    payload = "\n".join(sorted(case_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_inspection_membership(data_dir: Path) -> InspectionMembership:
    csv_path = inspection_csv_path(data_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Required CFD inspection CSV is missing; refusing to construct an "
            f"unfiltered ML dataset: {csv_path}"
        )

    statuses: dict[str, str] = {}
    excluded: dict[str, str] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_INSPECTION_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"Inspection CSV is missing required columns {sorted(missing)}: {csv_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                case_name = (row.get("case_name") or "").strip()
                status = (row.get("overall_status") or "").strip()
                if not CASE_ID_RE.fullmatch(case_name):
                    raise ValueError(
                        f"Invalid or empty case_name at {csv_path}:{line_number}: {case_name!r}"
                    )
                if status not in RECOGNIZED_INSPECTION_STATUSES:
                    raise ValueError(
                        f"Unrecognized overall_status at {csv_path}:{line_number}: {status!r}"
                    )
                if case_name in statuses:
                    raise ValueError(
                        f"Duplicate inspection record for {case_name!r} at "
                        f"{csv_path}:{line_number}"
                    )
                statuses[case_name] = status
                if status == "nOK":
                    excluded[case_name] = (row.get("overall_reason") or "").strip()
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"Cannot read inspection CSV {csv_path}: {exc}") from exc

    if not statuses:
        raise ValueError(f"Inspection CSV contains no case records: {csv_path}")
    ok_ids = sorted(case_id for case_id, status in statuses.items() if status == "OK")
    nok_ids = sorted(excluded)
    try:
        artifact_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot fingerprint inspection CSV {csv_path}: {exc}") from exc
    return InspectionMembership(
        csv_path=csv_path.resolve(),
        case_statuses=statuses,
        exclusion_reasons=excluded,
        ok_case_ids=ok_ids,
        nok_case_ids=nok_ids,
        membership_fingerprint=case_membership_fingerprint(ok_ids),
        artifact_sha256=artifact_sha256,
    )


def load_excluded_cases(data_dir: Path) -> dict[str, str]:
    return load_inspection_membership(data_dir).exclusion_reasons
