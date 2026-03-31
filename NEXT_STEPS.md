# Next Steps

- Create the local virtual environment with `uv venv`.
- Run `uv sync --extra dev --extra train`.
- Review `configs/downloads.yaml` and fill in dataset URLs and any weight filters.
- Test the planned downloads with `bash manage/download_data.sh --dry-run`. Activation is optional because wrappers use `uv run`.
- Test the planned model fetches with `bash manage/download_weights.sh --dry-run`.
- Run the real download helpers or place data and checkpoints manually.
- Update config paths in `configs/*.yaml`.
- Implement dataset preprocessing in `src/dataset/preprocessing.py`.
- Run the TF-IDF baseline with `bash manage/run_pipeline.sh --config configs/pipeline.yaml`.
- Run the small Transformer baseline with `bash manage/run_pipeline.sh --config configs/pipeline_transformer.yaml`.
