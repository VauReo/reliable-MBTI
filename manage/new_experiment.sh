#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NAME="${1:-}"
if [[ -z "$NAME" ]]; then
  echo "Usage: bash manage/new_experiment.sh <experiment_name>"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="outputs/${STAMP}_${NAME}"
mkdir -p "$EXP_DIR"
cat > "$EXP_DIR/metadata.yaml" <<EOF
experiment_name: ${NAME}
created_at: ${STAMP}
status: drafted
notes: Fill in config, seed, and artifact references.
EOF

echo "Created experiment directory: $EXP_DIR"
