# Next Steps

- Create the local virtual environment with `uv venv`.
- Activate it and run `uv sync --extra dev --extra train`.
- Review `configs/downloads.yaml` and fill in dataset URLs and any weight filters.
- Test the planned downloads with `bash manage/download_data.sh --dry-run`.
- Test the planned model fetches with `bash manage/download_weights.sh --dry-run`.
- Run the real download helpers or place data and checkpoints manually.
- Update config paths in `configs/*.yaml`.
- Implement dataset preprocessing in `src/dataset/preprocessing.py`.
- Test the baseline pipeline with `bash manage/run_prepare_data.sh` and the runner stubs.
