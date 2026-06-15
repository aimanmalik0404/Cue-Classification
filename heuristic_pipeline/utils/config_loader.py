"""utils/config_loader.py — Load and validate YAML config."""
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get(cfg: dict, *keys, default=None):
    """Safely traverse nested config keys."""
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val
