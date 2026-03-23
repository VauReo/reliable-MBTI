from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_yaml_config
from utils.downloads import materialize_hf_snapshot

ROOT_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Download configured model weights.')
    parser.add_argument('--config', required=True, help='Path to downloads YAML config.')
    parser.add_argument('--items', nargs='*', default=None, help='Optional weight keys to process.')
    parser.add_argument('--dry-run', action='store_true', help='Print planned actions only.')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)
    weights = config.get('weights', {})
    selected = set(args.items) if args.items else None

    for name, item in weights.items():
        if selected and name not in selected:
            continue
        if not item.get('enabled', True):
            print(f'[download_weights] skipping disabled weight item: {name}')
            continue
        if not item.get('use_hf_snapshot', True):
            print(f'[download_weights] unsupported non-HF item for {name}; skipping')
            continue

        model_id = item['model_id']
        target_dir = ROOT_DIR / item['local_target_dir']
        revision = item.get('revision', 'main')
        allow_patterns = item.get('allow_patterns') or []
        ignore_patterns = item.get('ignore_patterns') or []

        print(f'[download_weights] processing {name} -> {model_id}')
        materialize_hf_snapshot(
            model_id=model_id,
            local_dir=target_dir,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
