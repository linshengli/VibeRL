from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from src.models.entities import TrainingConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}: expected mapping")
    return data


def load_training_config(path: str | Path) -> TrainingConfig:
    data = load_yaml(path)
    return TrainingConfig(**data)


def save_training_config(path: str | Path, config: TrainingConfig) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False, allow_unicode=True)
