from executors.base import RunContext


class EvaluationExecutor:
    def __init__(self, context: RunContext) -> None:
        self.context = context

    def run(self) -> None:
        print(f"[eval] experiment={self.context.experiment_name}")
        print(f"[eval] output_dir={self.context.output_dir}")
        print("[eval] TODO: implement metric computation and checkpoint comparison.")
