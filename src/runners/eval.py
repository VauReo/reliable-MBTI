from executors.base import RunContext
from executors.eval_executor import EvaluationExecutor
from runners.common import load_config_from_args, prepare_output_dir


def main() -> None:
    _, config = load_config_from_args('Evaluate MBTI checkpoints.')
    output_dir = config['output_dir']
    prepare_output_dir(output_dir)
    executor = EvaluationExecutor(RunContext(config['experiment_name'], output_dir), config)
    executor.run()


if __name__ == '__main__':
    main()
