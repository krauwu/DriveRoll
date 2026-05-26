#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# UniAD Stage2 12Hz test script
# Usage: ./uniad_dist_eval_12hz.sh <CONFIG> <CKPT> <GPUS>
# Example: ./uniad_dist_eval_12hz.sh ./configs/stage2_e2e/base_e2e_12hz.py ./ckpts/uniad_base_e2e.pth 4
# -------------------------------------------------- #

# Usually only need to customize these variables
CFG=${1:-"./adzoo/uniad/configs/stage2_e2e/base_e2e_12hz.py"}
CKPT=${2:-"./ckpts/uniad_base_e2e.pth"}
GPUS=${3:-4}

GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-12345}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

echo "=========================================="
echo "UniAD Stage2 12Hz Evaluation"
echo "=========================================="
echo "Config: $CFG"
echo "Checkpoint: $CKPT"
echo "GPUS: $GPUS"
echo "Work Dir: $WORK_DIR"
echo "=========================================="

CUDA_VISIBLE_DEVICES='0' PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval_12hz.$T
