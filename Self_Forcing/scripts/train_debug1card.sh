#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/externals/waymo-open-dataset/src:$PROJECT_ROOT/externals/TATS/tats/fvd:${PYTHONPATH:-}"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-29503}" src/dwm/train.py \
  -c "${CONFIG_PATH:-configs/debug/rolling_ref+cleanN.json}" \
  -o "${OUTPUT_PATH:-outputs/debug}"
