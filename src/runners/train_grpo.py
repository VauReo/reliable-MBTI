from executors.base import RunContext
from executors.grpo_executor import GRPOExecutor
from runners.common import (
    load_config_from_args,
    prepare_output_dir,
    prepare_timestamped_run_dir,
    write_config_copy,
)


def main() -> None:
    _, config = load_config_from_args('Train the GRPO stage.')
    output_dir = prepare_timestamped_run_dir(config)
    prepare_output_dir(output_dir)
    write_config_copy(config, output_dir)
    executor = GRPOExecutor(
        RunContext(config['experiment_name'], output_dir),
        config,
    )
    executor.run()


if __name__ == '__main__':
    main()
