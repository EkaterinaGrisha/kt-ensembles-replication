"""Пути внутри репозитория воспроизведения.

Раскладка плоская: ktx/, scripts/, artifacts/, paper/, configs/. Всё
разрешается относительно корня репозитория, поэтому сценарии не зависят от
текущего каталога.
"""
from __future__ import annotations

from pathlib import Path

# ktx/paths.py -> ktx -> корень репозитория
REPO_ROOT = Path(__file__).resolve().parents[1]

RESEARCH_DIR = REPO_ROOT
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PYKT_ROOT = REPO_ROOT
PYKT_CONFIGS = REPO_ROOT / "configs"
PYKT_DATA = REPO_ROOT / "data"
DATA_CONFIG_JSON = PYKT_CONFIGS / "data_config.json"
KT_CONFIG_JSON = PYKT_CONFIGS / "kt_config.json"
MLRUNS_DIR = REPO_ROOT / "mlruns"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"


def ensure_output_dirs() -> None:
    for d in (ARTIFACTS_DIR,):
        d.mkdir(parents=True, exist_ok=True)
