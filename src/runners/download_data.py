from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_yaml_config
from utils.downloads import infer_filename, materialize_dataset, materialize_kagglehub_dataset
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

        source_type = item.get('source_type', 'http')
        target_dir = ROOT_DIR / item['local_target_dir']
        ensure_dir(target_dir)
        print(f'[download_data] processing {name} (source_type={source_type})')

        if source_type == 'kagglehub':
            dataset_handle = item.get('dataset_handle', '').strip()
            if not dataset_handle:
                print(f'[download_data] no dataset_handle configured for {name}; target_dir={target_dir}')
                continue
            sync_mode = item.get('sync_mode', 'copy')
            materialize_kagglehub_dataset(
                dataset_handle=dataset_handle,
                target_dir=target_dir,
                sync_mode=sync_mode,
                dry_run=args.dry_run,
            )
        elif source_type == 'http':
            url = item.get('source_url', '').strip()
            if not url:
                print(f'[download_data] no source_url configured for {name}; target_dir={target_dir}')
                continue
            filename = infer_filename(url, item.get('filename'))
            archive_type = item.get('archive_type', 'auto')
            materialize_dataset(url, target_dir, filename, archive_type, dry_run=args.dry_run)
        else:
            raise ValueError(f'Unsupported source_type for {name}: {source_type}')

        expected_files = item.get('expected_files', [])
        if expected_files:
            print(f'[download_data] expected_files for {name}: {expected_files}')


if __name__ == '__main__':
    main()
