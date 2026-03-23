from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    with config_path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}
