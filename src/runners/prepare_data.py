from dataset.preprocessing import DatasetPreprocessor
from runners.common import load_config_from_args


def main() -> None:
    _, config = load_config_from_args('Prepare MBTI datasets.')
    DatasetPreprocessor(config).run()


if __name__ == '__main__':
    main()
