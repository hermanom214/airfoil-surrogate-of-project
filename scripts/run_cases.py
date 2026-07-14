# run script - start OpenFoam terminal
# 1/cd /cygdrive/c/Users/martina/airfoil_surrogate_OF_project (go to project folder / not OF run folder)
# 2/./.venv/Scripts/python.exe -m scripts.run_cases
# 3/ python -m scripts.run_cases --start-case-id 30 --> run from case_30_ and higher (inclusive)

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_paths, load_solver_config
from src.case_runner import (
    discover_case_dirs,
    run_single_case,
    write_run_status_csv,
)


CASE_ID_RE = re.compile(r"^case_(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenFOAM cases in parallel")
    parser.add_argument(
        "--start-case-id",
        type=int,
        default=1,
        help="Start processing from this numeric case ID (inclusive), e.g. 30",
    )
    return parser.parse_args()


def extract_case_index(case_name: str) -> int | None:
    match = CASE_ID_RE.match(case_name)
    if match is None:
        return None
    return int(match.group(1))

def main() -> None:
    args = parse_args()

    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    solver_config_path = PROJECT_ROOT / "configs" / "solver_config.yaml"
    paths = load_paths(config_path)
    solver_cfg = load_solver_config(solver_config_path)

    max_workers = solver_cfg.run_cases.max_workers
    cases_root = paths.openfoam_case_sim / solver_cfg.run_cases.cases_subdir
    output_csv = cases_root / "run_status.csv"

    discovered_case_dirs = discover_case_dirs(cases_root)

    case_dirs = []
    skipped_without_id = 0
    for case_dir in discovered_case_dirs:
        case_index = extract_case_index(case_dir.name)
        if case_index is None:
            skipped_without_id += 1
            continue
        if case_index >= args.start_case_id:
            case_dirs.append(case_dir)

    if not case_dirs:
        print(
            "[INFO] Nebyly nalezeny žádné case složky "
            f"od ID {args.start_case_id} v: {cases_root}"
        )
        write_run_status_csv([], output_csv)
        return

    print(f"[INFO] Nalezeno {len(case_dirs)} case(s) od ID {args.start_case_id}.")
    if skipped_without_id:
        print(
            "[INFO] Přeskočeno case složek bez patternu case_XXXX_: "
            f"{skipped_without_id}"
        )
    print(f"[INFO] Running with MAX_WORKERS={max_workers}")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(run_single_case, case_dir): case_dir
            for case_dir in case_dirs
        }

        for i, future in enumerate(as_completed(future_to_case), start=1):
            case_dir = future_to_case[future]

            try:
                result = future.result()
                results.append(result)

                print(
                    f"[{i}/{len(case_dirs)}] {case_dir.name} -> "
                    f"{result.overall_status}, runtime_sec={result.runtime_sec}"
                )

            except Exception as e:
                print(f"[ERROR] {case_dir.name}: {e}")

    results = sorted(results, key=lambda r: r.case_id)

    write_run_status_csv(results, output_csv)
    print(f"\n[INFO] Run summary written to: {output_csv}")
    print(f"[INFO] Finished cases: {len(results)} / {len(case_dirs)}")


if __name__ == "__main__":
    main()