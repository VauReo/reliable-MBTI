from executors.base import RunContext
from executors.grpo_executor import GRPOExecutor
from runners.common import load_config_from_args, prepare_output_dir


def main() -> None:
    _, config = load_config_from_args('Train the GRPO stage.')
    output_dir = config['training']['output_dir']
    prepare_output_dir(output_dir)
    executor = GRPOExecutor(RunContext(config['experiment_name'], output_dir))
    executor.run()


if __name__ == '__main__':
    main()
