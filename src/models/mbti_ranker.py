class MBTIRankerModel:
    """Minimal placeholder interface for future ranking-capable models."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def describe(self) -> None:
        print(f"[model] configured model_id={self.model_id}")
