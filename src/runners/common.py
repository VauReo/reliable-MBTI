import argparse
from datetime import datetime
from pathlib import Path

import yaml

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


def prepare_timestamped_run_dir(config: dict) -> str:
    training = config.setdefault('training', {})
    experiment_name = str(config.get('experiment_name', 'experiment')).strip() or 'experiment'

    explicit_root = training.get('output_root_dir')
    explicit_exp = training.get('experiment_dir_name')
    legacy_output_dir = training.get('output_dir')

    if explicit_root is not None or explicit_exp is not None:
        output_root_dir = str(explicit_root or 'outputs')
        experiment_dir_name = str(explicit_exp or experiment_name).strip() or experiment_name
    elif legacy_output_dir:
        legacy_path = Path(str(legacy_output_dir))
        output_root_dir = str(legacy_path.parent) if str(legacy_path.parent) not in ('', '.') else 'outputs'
        experiment_dir_name = legacy_path.name or experiment_name
    else:
        output_root_dir = 'outputs'
        experiment_dir_name = experiment_name

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = Path(output_root_dir) / experiment_dir_name / timestamp

    training['output_root_dir'] = output_root_dir
    training['experiment_dir_name'] = experiment_dir_name
    training['run_start_timestamp'] = timestamp
    training['output_dir'] = str(run_output_dir)
    return str(run_output_dir)


def write_config_copy(config: dict, output_dir: str) -> None:
    output_path = ROOT_DIR / output_dir
    ensure_dir(output_path)
    config_path = output_path / 'config_used.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
