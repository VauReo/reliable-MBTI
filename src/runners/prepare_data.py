from dataset.preprocessing import DatasetPreprocessor
from runners.common import load_config_from_args


def main() -> None:
    _, config = load_config_from_args('Prepare MBTI datasets.')
    preprocessor = DatasetPreprocessor(
        raw_dir=config['raw_data_dir'],
        processed_dir=config['processed_data_dir'],
    )
    preprocessor.run()


if __name__ == '__main__':
    main()
