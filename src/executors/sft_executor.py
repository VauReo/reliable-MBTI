from executors.base import RunContext


class SFTExecutor:
    def __init__(self, context: RunContext) -> None:
        self.context = context

    def run(self) -> None:
        print(f"[train_sft] experiment={self.context.experiment_name}")
        print(f"[train_sft] output_dir={self.context.output_dir}")
        print("[train_sft] TODO: implement SFT training loop.")
