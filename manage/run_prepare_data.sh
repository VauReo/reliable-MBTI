#!/usr/bin/env bash
set -euo pipefail

# Ingests raw data and writes processed JSONL plus stratified train/val/test splits
# under data/splits/<dataset>/ (ratios and random seed from configs/data.yaml).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

PYTHONPATH="$ROOT_DIR/src" uv run python -m runners.prepare_data --config configs/data.yaml "$@"
