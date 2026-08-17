from __future__ import annotations

import csv
from pathlib import Path


INSPECTION_CSV_NAME = "inspect_plausibility.csv"


def inspection_csv_path(data_dir: Path) -> Path:
    return data_dir.parent / "pictures_inspect_flow" / INSPECTION_CSV_NAME


def load_excluded_cases(data_dir: Path) -> dict[str, str]:
    csv_path = inspection_csv_path(data_dir)
    if not csv_path.is_file():
        print(f"[WARN] Inspection CSV not found; no quality exclusions applied: {csv_path}")
        return {}

    excluded: dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("overall_status") == "nOK":
                case_name = row.get("case_name", "").strip()
                if case_name:
                    excluded[case_name] = row.get("overall_reason", "")
    return excluded
