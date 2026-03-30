from __future__ import annotations

import argparse

from dataset.preprocessing import DatasetPreprocessor
from executors.base import RunContext
from executors.eval_executor import EvaluationExecutor
from runners.common import prepare_output_dir
from utils.config import load_yaml_config
from utils.downloads import infer_filename, materialize_dataset, materialize_kagglehub_dataset
from utils.io import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run the full baseline MBTI pipeline: download -> prepare -> evaluate.'
    )
    parser.add_argument('--config', required=True, help='Path to pipeline YAML config.')
    return parser


def run_download_stage(config_path: str) -> None:
    config = load_yaml_config(config_path)
    datasets = config.get('data', {})
    for name, item in datasets.items():
        if not item.get('enabled', True):
            print(f'[pipeline] skip download dataset={name} (disabled)')
            continue
        source_type = item.get('source_type', 'http')
        target_dir = ensure_dir(item['local_target_dir'])
        print(f'[pipeline] download dataset={name} source_type={source_type}')
        if source_type == 'kagglehub':
            dataset_handle = item.get('dataset_handle', '').strip()
            if not dataset_handle:
                raise ValueError(f'Dataset {name} is missing dataset_handle in {config_path}')
            materialize_kagglehub_dataset(
                dataset_handle=dataset_handle,
                target_dir=target_dir,
                sync_mode=item.get('sync_mode', 'copy'),
                dry_run=False,
            )
        elif source_type == 'http':
            url = item.get('source_url', '').strip()
            if not url:
                raise ValueError(f'Dataset {name} is missing source_url in {config_path}')
            materialize_dataset(
                url=url,
                target_dir=target_dir,
                filename=infer_filename(url, item.get('filename')),
                archive_type=item.get('archive_type', 'auto'),
                dry_run=False,
            )
        else:
            raise ValueError(f'Unsupported source_type for {name}: {source_type}')


def main() -> None:
    args = build_parser().parse_args()
    pipeline_cfg = load_yaml_config(args.config)

    download_cfg_path = str(pipeline_cfg['download_config'])
    data_cfg_path = str(pipeline_cfg['data_config'])
    eval_cfg_path = str(pipeline_cfg['eval_config'])

    print('[pipeline] stage=download')
    run_download_stage(download_cfg_path)

    print('[pipeline] stage=prepare_data')
    data_config = load_yaml_config(data_cfg_path)
    DatasetPreprocessor(data_config).run()

    print('[pipeline] stage=evaluate_baseline')
    eval_config = load_yaml_config(eval_cfg_path)
    prepare_output_dir(eval_config['output_dir'])
    executor = EvaluationExecutor(
        RunContext(eval_config['experiment_name'], eval_config['output_dir']),
        eval_config,
    )
    executor.run()
    print('[pipeline] completed all stages')


if __name__ == '__main__':
    main()
