from executors.base import RunContext


class GRPOExecutor:
    def __init__(self, context: RunContext) -> None:
        self.context = context

    def run(self) -> None:
        print(f"[train_grpo] experiment={self.context.experiment_name}")
        print(f"[train_grpo] output_dir={self.context.output_dir}")
        print("[train_grpo] TODO: implement GRPO training loop and reward integration.")
