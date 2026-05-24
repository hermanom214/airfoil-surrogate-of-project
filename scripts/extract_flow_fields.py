from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_paths
from src.flow_extractor import (
    discover_case_dirs,
    extract_single_case,
    write_flow_index,
)


MAX_WORKERS = 1
EXTRACT_NX = 640
EXTRACT_NY = 320
EXTRACT_X_MIN = -0.75
EXTRACT_X_MAX = 1.75
EXTRACT_Y_MIN = -0.75
EXTRACT_Y_MAX = 0.75


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    paths = load_paths(config_path)

    cases_root = paths.openfoam_case_sim / "blockmesh_cases"
    output_root = paths.flow_fields_output
    index_csv = output_root / "flow_dataset_index.csv"

    case_dirs = discover_case_dirs(cases_root)

    if not case_dirs:
        print(f"[INFO] Nebyly nalezeny žádné case složky v: {cases_root}")
        write_flow_index([], index_csv)
        return

    print(f"[INFO] Nalezeno {len(case_dirs)} case(s).")
    print(f"[INFO] Flow output root: {output_root}")
    print(f"[INFO] Running extraction with MAX_WORKERS={MAX_WORKERS}")
    print(
        "[INFO] Extraction grid: "
        f"{EXTRACT_NX}x{EXTRACT_NY}, "
        f"x=[{EXTRACT_X_MIN}, {EXTRACT_X_MAX}], "
        f"y=[{EXTRACT_Y_MIN}, {EXTRACT_Y_MAX}]"
    )

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_case = {
            executor.submit(
                extract_single_case,
                case_dir,
                output_root,
                EXTRACT_NX,
                EXTRACT_NY,
                EXTRACT_X_MIN,
                EXTRACT_X_MAX,
                EXTRACT_Y_MIN,
                EXTRACT_Y_MAX,
            ): case_dir
            for case_dir in case_dirs
        }

        for i, future in enumerate(as_completed(future_to_case), start=1):
            case_dir = future_to_case[future]

            try:
                result = future.result()
                results.append(result)

                print(
                    f"[{i}/{len(case_dirs)}] {case_dir.name} -> "
                    f"{result.status}"
                )

            except Exception as e:
                print(f"[ERROR] {case_dir.name}: {e}")

    results = sorted(results, key=lambda r: r.case_id)

    write_flow_index(results, index_csv)

    print(f"\n[INFO] Flow dataset index written to: {index_csv}")
    print(f"[INFO] Finished extracted cases: {len(results)} / {len(case_dirs)}")


if __name__ == "__main__":
    main()