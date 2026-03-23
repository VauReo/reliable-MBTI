#!/usr/bin/env bash
set -euo pipefail

# Generates chat-formatted SFT data with teacher reasoning (configs/generate_sft.yaml).
# Uses generation.max_rows_for_sft_pool train examples for distillation; remaining train rows
# go to grpo_jsonl. SFT rejects (teacher parse / rejection sampling) are appended there too.
#
# GPU selection: visible devices for PyTorch. Default is GPU 0. Override before invoking, e.g.:
#   CUDA_VISIBLE_DEVICES=1 bash manage/run_generate_sft_data.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash manage/run_generate_sft_data.sh
# If CUDA_VISIBLE_DEVICES is already set in the environment, it is preserved.
: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

PYTHONPATH="$ROOT_DIR/src" uv run python -m runners.generate_sft_data --config configs/generate_sft.yaml "$@"
