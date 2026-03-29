from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from executors.base import RunContext
from executors.grpo_executor import GRPOExecutor
from executors.sft_executor import SFTExecutor
from utils.config import ROOT_DIR, load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Redraw experiment figures from saved histories and checkpoints.'
    )
    parser.add_argument(
        'exp_path',
        help='Relative path to the experiment run folder, e.g. outputs/foo/20260328_123456',
    )
    return parser


def _resolve_exp_path(exp_path: str) -> Path:
    path = Path(exp_path)
    return path if path.is_absolute() else ROOT_DIR / path


def _load_config(exp_dir: Path) -> tuple[dict[str, Any], str]:
    grpo_path = exp_dir / 'resolved_grpo_config.json'
    sft_path = exp_dir / 'resolved_sft_config.json'
    config_used_path = exp_dir / 'config_used.yaml'

    if grpo_path.exists():
        return json.loads(grpo_path.read_text(encoding='utf-8')), 'grpo'
    if sft_path.exists():
        return json.loads(sft_path.read_text(encoding='utf-8')), 'sft'
    if config_used_path.exists():
        config = load_yaml_config(str(config_used_path))
        training = config.get('training', {})
        if 'grpo' in config or 'reward' in config and 'evaluation' in config and 'model' in config and 'policy_init' in config.get('model', {}):
            return config, 'grpo'
        return config, 'sft'
    raise FileNotFoundError(
        f'Could not find resolved config or config_used.yaml inside {exp_dir}'
    )


def _latest_checkpoint_dir(exp_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in exp_dir.iterdir():
        if not path.is_dir() or not path.name.startswith('checkpoint-'):
            continue
        suffix = path.name.split('checkpoint-', 1)[1]
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _load_log_history(exp_dir: Path) -> list[dict[str, Any]]:
    checkpoint_dir = _latest_checkpoint_dir(exp_dir)
    if checkpoint_dir is None:
        return []
    trainer_state_path = checkpoint_dir / 'trainer_state.json'
    if not trainer_state_path.exists():
        return []
    trainer_state = json.loads(trainer_state_path.read_text(encoding='utf-8'))
    log_history = trainer_state.get('log_history')
    return list(log_history) if isinstance(log_history, list) else []


def main() -> None:
    args = build_parser().parse_args()
    exp_dir = _resolve_exp_path(args.exp_path)
    if not exp_dir.exists():
        raise FileNotFoundError(f'Experiment directory not found: {exp_dir}')

    config, kind = _load_config(exp_dir)
    experiment_name = str(config.get('experiment_name', exp_dir.name))
    executor_cls = GRPOExecutor if kind == 'grpo' else SFTExecutor
    executor = executor_cls(RunContext(experiment_name, str(exp_dir)), config)
    log_history = _load_log_history(exp_dir)
    executor._write_training_graphs(log_history)
    figures_dir = exp_dir / 'figures'
    print(f'[monitor_exp] experiment={kind}')
    print(f'[monitor_exp] exp_dir={exp_dir}')
    print(f'[monitor_exp] log_history_entries={len(log_history)}')
    print(f'[monitor_exp] figures_dir={figures_dir}')
    print('[monitor_exp] figures are redrawn in-place and overwrite previous files')


if __name__ == '__main__':
    main()
