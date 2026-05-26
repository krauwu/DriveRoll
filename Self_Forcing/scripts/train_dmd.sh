#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

NUM_GPUS="${1:-4}"
CONFIG_FILE="${CONFIG_FILE:-configs/debug/train_dmd.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dmd}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/externals/waymo-open-dataset/src:$PROJECT_ROOT/externals/TATS/tats/fvd:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

GENERATED_CONFIG="$PROJECT_ROOT/configs/debug/train_dmd_${NUM_GPUS}gpu.json"
export CONFIG_FILE GENERATED_CONFIG NUM_GPUS
python3 - <<'PYINNER'
import json
import os
from pathlib import Path

config_file = Path(os.environ.get("CONFIG_FILE", "configs/debug/train_dmd.json"))
generated_config = Path(os.environ["GENERATED_CONFIG"])
num_gpus = int(os.environ["NUM_GPUS"])

with config_file.open("r") as f:
    config = json.load(f)

if "global_state" in config and "device_mesh" in config["global_state"]:
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, num_gpus]

generated_config.parent.mkdir(parents=True, exist_ok=True)
with generated_config.open("w") as f:
    json.dump(config, f, indent=4)

print(f"Generated config: {generated_config}")
PYINNER

echo "DMD training"
echo "Project root: $PROJECT_ROOT"
echo "Config:       $GENERATED_CONFIG"
echo "Output:       $OUTPUT_DIR"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"

torchrun --nproc_per_node="$NUM_GPUS" src/dwm/train_dmd.py \
  -c "$GENERATED_CONFIG" \
  -o "$OUTPUT_DIR"
