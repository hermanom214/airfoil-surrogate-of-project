from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_environment(project_root: Path = PROJECT_ROOT) -> None:
    """Load project-local settings without overriding the process environment."""
    load_dotenv(dotenv_path=project_root / ".env", override=False)
