from pathlib import Path

from utils.io import ensure_dir


class DatasetPreprocessor:
    """Placeholder preprocessing pipeline for MBTI datasets."""

    def __init__(self, raw_dir: str, processed_dir: str) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def run(self) -> None:
        ensure_dir(self.processed_dir)
        print(f"[prepare_data] raw_dir={self.raw_dir}")
        print(f"[prepare_data] processed_dir={self.processed_dir}")
        print("[prepare_data] TODO: implement dataset ingestion, cleaning, and split generation.")
