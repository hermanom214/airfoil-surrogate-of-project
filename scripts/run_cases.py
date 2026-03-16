# run script - start OpenFoam terminal
# 1/cd /cygdrive/c/Users/martina/airfoil_surrogate_OF_project (go to project folder / not OF run folder)
# 2/./.venv/Scripts/python.exe -m scripts.run_cases

from __future__ import annotations

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.case_runner import (
    discover_case_dirs,
    run_single_case,
    write_run_status_csv,
)


def main() -> None:
    # Uprav podle svého prostředí
    cases_root = Path(r"C:\OpenFOAM\20.09\martina-dev\run\airfoil_surrogate_cases")
    output_csv = cases_root / "run_status.csv"

    case_dirs = discover_case_dirs(cases_root)

    if not case_dirs:
        print(f"[INFO] Nebyly nalezeny žádné case složky v: {cases_root}")
        write_run_status_csv([], output_csv)
        return

    print(f"[INFO] Nalezeno {len(case_dirs)} case(s).")
    results = []

    for i, case_dir in enumerate(case_dirs, start=1):
        print(f"\n[{i}/{len(case_dirs)}] Running case: {case_dir.name}")
        result = run_single_case(case_dir)
        results.append(result)

        print(
            f"  -> overall_status={result.overall_status}, "
            f"runtime_sec={result.runtime_sec}"
        )

    write_run_status_csv(results, output_csv)
    print(f"\n[INFO] Run summary written to: {output_csv}")


if __name__ == "__main__":
    main()