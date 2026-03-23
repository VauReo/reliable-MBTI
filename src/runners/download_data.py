from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_yaml_config
from utils.downloads import infer_filename, materialize_dataset
from utils.io import ensure_dir

ROOT_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Download configured datasets.')
    parser.add_argument('--config', required=True, help='Path to downloads YAML config.')
    parser.add_argument('--items', nargs='*', default=None, help='Optional dataset keys to process.')
    parser.add_argument('--dry-run', action='store_true', help='Print planned actions only.')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)
    datasets = config.get('data', {})
    selected = set(args.items) if args.items else None

    for name, item in datasets.items():
        if selected and name not in selected:
            continue
        if not item.get('enabled', True):
            print(f'[download_data] skipping disabled dataset: {name}')
            continue

        url = item.get('source_url', '').strip()
        target_dir = ROOT_DIR / item['local_target_dir']
        ensure_dir(target_dir)

        if not url:
            print(f'[download_data] no source_url configured for {name}; target_dir={target_dir}')
            continue

        filename = infer_filename(url, item.get('filename'))
        archive_type = item.get('archive_type', 'auto')
        print(f'[download_data] processing {name}')
        materialize_dataset(url, target_dir, filename, archive_type, dry_run=args.dry_run)

        expected_files = item.get('expected_files', [])
        if expected_files:
            print(f'[download_data] expected_files for {name}: {expected_files}')


if __name__ == '__main__':
    main()
