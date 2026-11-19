from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=8)
def exam_profile(exam: str) -> dict[str, Any]:
    path = CONFIG_DIR / "exams" / f"{exam.lower()}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown exam profile '{exam}'. Available: {available_exams()}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def available_exams() -> list[str]:
    return sorted(p.stem.upper() for p in (CONFIG_DIR / "exams").glob("*.yaml"))


def data_dir() -> Path:
    d = ROOT / settings()["data_dir"]
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    (d / "projects").mkdir(parents=True, exist_ok=True)
    return d
