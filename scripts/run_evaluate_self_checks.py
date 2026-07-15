from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluate_self_checks import evaluate_run_all_self_checks


def main() -> None:
    results = evaluate_run_all_self_checks()
    print("[SELF CHECKS]")
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        print(f"{status} | {res.name} | {res.details}")


def _entry() -> None:
    main()


if __name__ == "__main__":
    _entry()
