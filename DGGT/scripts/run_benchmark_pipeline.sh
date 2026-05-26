#!/bin/bash
# ============================================================================
# run_benchmark_pipeline.sh
# 一次性执行 convert_sf_to_benchmark.py 的 4 个步骤:
#   Step 1: prepare   - 准备公共元数据 (token + pose)
#   Step 2: convert   - 从 npy 提取图像，按 token 对齐
#   Step 3: benchmark - 调用 benchmark.py 评测
#   Step 4: aggregate - 聚合多场景结果，画平均曲线
#
# 用法:
#   bash scripts/run_benchmark_pipeline.sh \
#       --sf_preview_dir /path/to/sf_output \
#       --pose_base_dir /path/to/dmd_poses \
#       --ckpt_path pretrained/model_latest_waymo.pt \
#       [其他可选参数]
# ============================================================================

set -euo pipefail

# ==================== 默认参数 ====================
SF_PREVIEW_DIR=""
POSE_BASE_DIR="./output/dmd_ode_pretrained_ref3_seq12/vis_validation_generator"
CKPT_PATH="pretrained/model_latest_waymo.pt"
OUTPUT_DIR="benchmark_input"
NUM_SCENES=10
CAMS="1"
STRIDE=2
SEQUENCE_LENGTH=4
START_IDX=0
FRAME_INTERVAL=4
NUM_FRAMES=100
SKIP_PREPARE=true

# ==================== 解析参数 ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sf_preview_dir)  SF_PREVIEW_DIR="$2";  shift 2 ;;
        --pose_base_dir)   POSE_BASE_DIR="$2";   shift 2 ;;
        --ckpt_path)       CKPT_PATH="$2";        shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";       shift 2 ;;
        --num_scenes)      NUM_SCENES="$2";       shift 2 ;;
        --cams)            CAMS="$2";             shift 2 ;;
        --stride)          STRIDE="$2";           shift 2 ;;
        --sequence_length) SEQUENCE_LENGTH="$2";  shift 2 ;;
        --start_idx)       START_IDX="$2";        shift 2 ;;
        --frame_interval)  FRAME_INTERVAL="$2";   shift 2 ;;
        --num_frames)      NUM_FRAMES="$2";       shift 2 ;;
        --skip_prepare)    SKIP_PREPARE=true;     shift   ;;
        -h|--help)
            echo "Usage: bash scripts/run_benchmark_pipeline.sh \\"
            echo "    --sf_preview_dir /path/to/sf_output \\"
            echo "    --pose_base_dir /path/to/dmd_poses \\"
            echo "    --ckpt_path pretrained/model_latest_waymo.pt \\"
            echo "    [--output_dir benchmark_input] \\"
            echo "    [--num_scenes 10] \\"
            echo "    [--cams 0] \\"
            echo "    [--stride 1] \\"
            echo "    [--sequence_length 4] \\"
            echo "    [--start_idx 4] \\"
            echo "    [--frame_interval 4] \\"
            echo "    [--num_frames 100] \\"
            echo "    [--skip_prepare]  # 跳过 Step 1 (metadata 已准备好时使用)"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ==================== 参数校验 ====================
if [ -z "$SF_PREVIEW_DIR" ]; then
    echo "[ERROR] 必须指定 --sf_preview_dir"
    exit 1
fi
if [ -z "$POSE_BASE_DIR" ] && [ "$SKIP_PREPARE" = false ]; then
    echo "[ERROR] 必须指定 --pose_base_dir (或使用 --skip_prepare 跳过 Step 1)"
    exit 1
fi
if [ ! -f "$CKPT_PATH" ]; then
    echo "[WARN] checkpoint 不存在: $CKPT_PATH"
fi

# source_name 取自 sf_preview_dir 最后一级目录名
SOURCE_NAME=$(basename "$(echo "$SF_PREVIEW_DIR" | sed 's:/*$::')")

# 将 CAMS 从逗号/空格分隔转为 Python 列表格式
# 支持 "0" "0 1 2" "0,1,2" 等格式
CAMS_PY=$(echo "$CAMS" | tr ',' ' ' | xargs | sed 's/ / /g')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/convert_sf_to_benchmark.py"

echo "============================================================"
echo "DGGT Benchmark Pipeline"
echo "============================================================"
echo "  sf_preview_dir:  $SF_PREVIEW_DIR"
echo "  pose_base_dir:   $POSE_BASE_DIR"
echo "  source_name:     $SOURCE_NAME"
echo "  ckpt_path:       $CKPT_PATH"
echo "  output_dir:      $OUTPUT_DIR"
echo "  num_scenes:      $NUM_SCENES"
echo "  cams:            $CAMS"
echo "  stride:          $STRIDE"
echo "  sequence_length: $SEQUENCE_LENGTH"
echo "  start_idx:       $START_IDX"
echo "  frame_interval:  $FRAME_INTERVAL"
echo "  num_frames:      $NUM_FRAMES"
echo "  skip_prepare:    $SKIP_PREPARE"
echo "============================================================"

# ==================== Step 1: Prepare ====================
if [ "$SKIP_PREPARE" = true ]; then
    echo ""
    echo "[SKIP] Step 1: prepare (--skip_prepare 已设置)"
else
    echo ""
    echo "============================================================"
    echo "Step 1/4: 准备公共元数据 (token list + ego pose)"
    echo "============================================================"
    python "$PYTHON" prepare \
        --sf_preview_dir "$SF_PREVIEW_DIR" \
        --pose_base_dir "$POSE_BASE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_scenes "$NUM_SCENES"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Step 1 失败，退出"
        exit 1
    fi
    echo "[OK] Step 1 完成"
fi

# ==================== Step 2: Convert ====================
echo ""
echo "============================================================"
echo "Step 2/4: 从 npy 提取图像，按 token 对齐"
echo "============================================================"
python "$PYTHON" convert \
    --sf_preview_dir "$SF_PREVIEW_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_scenes "$NUM_SCENES" \
    --cams $CAMS_PY

if [ $? -ne 0 ]; then
    echo "[ERROR] Step 2 失败，退出"
    exit 1
fi
echo "[OK] Step 2 完成"

# ==================== Step 3: Benchmark ====================
echo ""
echo "============================================================"
echo "Step 3/4: 调用 benchmark.py 评测 (每个 scene/cam 独立运行)"
echo "============================================================"
python "$PYTHON" benchmark \
    --output_dir "$OUTPUT_DIR" \
    --source_name "$SOURCE_NAME" \
    --ckpt_path "$CKPT_PATH" \
    --num_scenes "$NUM_SCENES" \
    --cams $CAMS_PY \
    --sequence_length "$SEQUENCE_LENGTH" \
    --start_idx "$START_IDX" \
    --frame_interval "$FRAME_INTERVAL" \
    --stride "$STRIDE" \
    --num_frames "$NUM_FRAMES"

if [ $? -ne 0 ]; then
    echo "[ERROR] Step 3 失败，退出"
    exit 1
fi
echo "[OK] Step 3 完成"

# ==================== Step 4: Aggregate ====================
echo ""
echo "============================================================"
echo "Step 4/4: 聚合多场景结果，画平均曲线"
echo "============================================================"
python "$PYTHON" aggregate \
    --output_dir "$OUTPUT_DIR" \
    --source_name "$SOURCE_NAME" \
    --num_scenes "$NUM_SCENES" \
    --cams $CAMS_PY \
    --stride "$STRIDE"

if [ $? -ne 0 ]; then
    echo "[ERROR] Step 4 失败，退出"
    exit 1
fi
echo "[OK] Step 4 完成"

# ==================== 完成 ====================
echo ""
echo "============================================================"
echo "Pipeline 完成!"
echo "============================================================"
echo ""
echo "输出目录结构:"
echo "  $OUTPUT_DIR/"
echo "  ├── metadata/                          (Step 1, 公共)"
echo "  │   ├── val_scene_metadata.json"
echo "  │   └── poses/scene_X/cam_Y/"
echo "  └── $SOURCE_NAME/"
echo "      ├── scene_X/                       (Step 2, per-scene)"
echo "      │   ├── gt_rgb/cam_Y/"
echo "      │   ├── generated_rgb/cam_Y/"
echo "      │   └── gt_camera_params/ego_transforms/cam_Y/"
echo "      └── benchmark_output/              (Step 3+4)"
echo "          ├── scene_X/cam_Y/             (per-scene 结果)"
echo "          │   ├── *.json, *.png"
echo "          │   └── rendered_*/, trajectories/"
echo "          └── aggregate/                 (Step 4, 聚合结果)"
echo "              ├── aggregated_metrics.json"
echo "              ├── aggregated_metrics_comparison.png"
echo "              ├── aggregated_pose_metrics_curve.png"
echo "              └── aggregated_nvs_metrics_curve.png"
