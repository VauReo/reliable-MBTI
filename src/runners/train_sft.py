from executors.base import RunContext
from executors.sft_executor import SFTExecutor
from runners.common import load_config_from_args, prepare_output_dir


def main() -> None:
    _, config = load_config_from_args('Train the SFT stage.')
    output_dir = config['training']['output_dir']
    prepare_output_dir(output_dir)
    executor = SFTExecutor(RunContext(config['experiment_name'], output_dir))
    executor.run()


if __name__ == '__main__':
    main()
