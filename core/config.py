# config.py
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from core.geometry import ArrayGeometry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

PathLike = Union[str, Path]


def load_json(path: PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_array_geometry(path: Optional[PathLike] = None) -> ArrayGeometry:
    return ArrayGeometry.from_dict(load_json(path or CONFIG_DIR / "array_config.json"))


def load_pipeline_config(path: Optional[PathLike] = None) -> Dict[str, Any]:
    return load_json(path or CONFIG_DIR / "pipeline_config.json")
