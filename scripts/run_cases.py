# run script - start OpenFoam terminal
# 1/cd /cygdrive/c/Users/martina/airfoil_surrogate_OF_project (go to project folder / not OF run folder)
# 2/./.venv/Scripts/python.exe -m scripts.run_cases

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_paths
from src.case_runner import (
    discover_case_dirs,
    run_single_case,
    write_run_status_csv,
)


MAX_WORKERS = 1  # začni opatrně: 2. Později zkus 3 nebo 4.


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    paths = load_paths(config_path)

    cases_root = paths.openfoam_case_sim / "blockmesh_cases"
    output_csv = cases_root / "run_status.csv"

    case_dirs = discover_case_dirs(cases_root)

    if not case_dirs:
        print(f"[INFO] Nebyly nalezeny žádné case složky v: {cases_root}")
        write_run_status_csv([], output_csv)
        return

    print(f"[INFO] Nalezeno {len(case_dirs)} case(s).")
    print(f"[INFO] Running with MAX_WORKERS={MAX_WORKERS}")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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