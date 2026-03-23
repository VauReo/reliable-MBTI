#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

uv venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv sync --extra dev --extra train

echo "Environment ready in $ROOT_DIR/.venv"
