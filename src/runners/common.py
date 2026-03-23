import argparse
from pathlib import Path

from utils.config import load_yaml_config
from utils.io import ensure_dir


ROOT_DIR = Path(__file__).resolve().parents[2]


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    return parser


def load_config_from_args(description: str):
    parser = build_parser(description)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    return args, config


def prepare_output_dir(path_value: str | None) -> None:
    if path_value:
        ensure_dir(ROOT_DIR / path_value)
