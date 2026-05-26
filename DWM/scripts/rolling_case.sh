#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-configs/rolling/rolling_case.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/rolling_case}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --nproc_per_node="$NPROC_PER_NODE" src/dwm/preview.py \
  -c "$CONFIG_PATH" \
  -o "$OUTPUT_PATH"
