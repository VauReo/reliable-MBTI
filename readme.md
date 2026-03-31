# reliable-MBTI

Scaffold for reproducing the paper `From Classification to Ranking: Enhancing LLM Reasoning Capabilities for MBTI Personality Detection`.

This repository currently provides a careful local project skeleton: UV-based environment setup, configuration templates, shell helpers, Python runners, and starter modules for data preparation, downloads, SFT, GRPO, and evaluation.

## Project Goal

The planned reproduction pipeline is:

- treat MBTI prediction as ranking over 16 MBTI types,
- generate teacher reasoning traces for SFT,
- fine-tune a student model,
- optimize the ranking behavior with GRPO-style reinforcement learning,
- compare the ranking pipeline against baseline classifiers.

## Repository Structure

```text
reliable-MBTI/
  configs/        YAML templates for data, training, evaluation, and downloads
  data/           Local datasets and processed artifacts
  manage/         Shell scripts for local setup and repeatable workflows
  outputs/        Experiment outputs and run metadata
  src/            Python package code
    dataset/      Dataset schemas and preprocessing entry points
    executors/    Training and evaluation executors
    losses/       Reward and loss placeholders
    models/       Model interface stubs
    runners/      Python entry points called by manage/*.sh
    utils/        Shared helpers
  weights/        Local checkpoints, adapters, and tokenizers
```

## Local Setup With UV

```bash
uv venv
uv sync --extra dev --extra train
```

If you only need the minimal scaffold dependencies:

```bash
uv sync
```

You can also use the helper script:

```bash
bash manage/setup_env.sh
```

All project wrappers in `manage/` use `uv run`, so activation is optional for normal scripted usage.

## Download Workflow

Edit `configs/downloads.yaml` first. Then use the shell wrappers:

```bash
bash manage/download_data.sh
bash manage/download_weights.sh
```

Useful options:

```bash
bash manage/download_data.sh --dry-run
bash manage/download_data.sh --items kaggle_mbti
bash manage/download_weights.sh --dry-run
bash manage/download_weights.sh --items teacher
```

Behavior:

- `kaggle_mbti` uses `kagglehub.dataset_download("datasnaek/mbti-type")`.
- after Kaggle download, the cached files are copied into `data/raw/kaggle_mbti` so the project keeps a stable local dataset layout.
- other datasets can still use direct `source_url` downloads and archive extraction.
- `download_weights.sh` downloads model snapshots from Hugging Face using `model_id` and stores them under `weights/`.
- dry-run mode prints the planned actions without pulling large artifacts.

## Planned Workflow Stages

1. Download or stage raw datasets and model checkpoints.
2. Prepare a unified MBTI dataset format.
3. Build baseline classification and ranking pipelines.
4. Generate SFT traces from a teacher model.
5. Train the student model with SFT.
6. Add GRPO and ranking rewards.
7. Evaluate, compare checkpoints, and run ablations.

## Notes

- This is a reproduction scaffold, not a finished implementation.
- Python entry points live in `src/runners/`.
- Shell automation lives in `manage/`.
- Large data, weights, and outputs should stay out of Git.

