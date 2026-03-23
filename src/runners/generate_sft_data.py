import argparse

from dataset.sft_generation import run_generation
from utils.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate SFT JSONL from a local teacher model.')
    parser.add_argument('--config', required=True, help='Path to YAML (see configs/generate_sft.yaml).')
    parser.add_argument(
        '--max-rows-for-sft-pool',
        type=int,
        default=None,
        help='Override generation.max_rows_for_sft_pool (train rows used for teacher / SFT).',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)
    if args.max_rows_for_sft_pool is not None:
        gen = config.setdefault('generation', {})
        gen['max_rows_for_sft_pool'] = int(args.max_rows_for_sft_pool)

    name = config.get('experiment_name', 'generate_sft')
    print(f'[generate_sft_data] experiment={name}')
    stats = run_generation(config)
    print(
        f'[generate_sft_data] attempted={stats.attempted} accepted={stats.accepted} '
        f'rejected={stats.rejected} grpo_total={stats.grpo_total_rows} '
        f'(reserve={stats.grpo_reserve_rows} + sft_rejects={stats.grpo_from_sft_rejects})'
    )


if __name__ == '__main__':
    main()
