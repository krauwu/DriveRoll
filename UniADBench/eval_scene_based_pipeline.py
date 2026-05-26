#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-based Evaluation Pipeline
Step 1: Split PKL (call generate_uniad_pkls.py)
Step 2: Run UniAD evaluation (call UniADEval)
Step 3: Plot scene score table
"""

import os
import sys
import subprocess
import json
import csv
from pathlib import Path
import yaml

from video_utils import get_video_meta
from run_uniad import UniADEval


# ============ Configuration Area ============
BENCH_ROOT = "<path-to-local-resource>"
MAIN_CFG = f"{BENCH_ROOT}/cfg/nuscenes_scene.yaml"


def load_config(cfg_path: str) -> dict:
    """Load YAML configuration file"""
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


# Dynamically read from config file
CFG = load_config(MAIN_CFG)
TASK_NAME = CFG['task']['name']
TASK_CFG = CFG['task'][TASK_NAME]

# UniAD configuration
UNIAD_ROOT = f"{BENCH_ROOT}/UniAD"
CONFIG_PATH = f"{UNIAD_ROOT}/adzoo/uniad/configs/stage2_e2e/base_e2e.py"
CHECKPOINT = "<path-to-local-resource>"
PKL_DIR = f"{TASK_CFG['pseudo_nuscenes_root']}{TASK_CFG['output']['subdir']}"
DATA_ROOT = TASK_CFG['pseudo_nuscenes_root']
OUTPUT_DIR = '<path-to-local-resource>'
VIDEO_DIR = CFG['input']['video_dir']
GPU_IDS = "0,6"

# Summary configuration
SUMMARY_PATH = f"{OUTPUT_DIR}/all_metrics_summary.json"
# =================================


def get_video_fps(video_dir: str) -> float:
    """Get video FPS"""
    video_path = Path(video_dir)
    video_files = sorted(video_path.rglob("*.mp4")) + sorted(video_path.rglob("*.avi"))

    if not video_files:
        print(f"[Warning] No video files in {video_dir}, using default fps=6.0")
        return 6.0

    video_meta = get_video_meta(str(video_files[0]))
    fps = float(video_meta["fps"])
    print(f"[Video] {video_files[0].name}, fps={fps}")
    return fps


def step1_split_pkl() -> bool:
    """Step 1: Split PKL files"""
    print("\n" + "="*60)
    print("Step 1: Splitting PKL files")
    print("="*60)

    cmd = [sys.executable, f"{BENCH_ROOT}/generate_uniad_pkls.py", "-cfg", MAIN_CFG]
    print(f"[Command] {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=BENCH_ROOT)
    return result.returncode == 0


def step2_run_uniad(fps: float, master_port: int = 29500) -> bool:
    """Step 2: Run UniAD evaluation"""
    print("\n" + "="*60)
    print("Step 2: Running UniAD evaluation")
    print("="*60)

    evaluator = UniADEval(
        uniad_root=UNIAD_ROOT,
        config_path=CONFIG_PATH,
        checkpoint=CHECKPOINT,
        pkl_dir=PKL_DIR,
        output_dir=OUTPUT_DIR,
        gpu_ids=GPU_IDS,
        fps=fps,
        master_port=master_port,
        data_dir=DATA_ROOT,
    )
    return evaluator.run()


def _format_value(v):
    """Format metric values, handling list and None"""
    if v is None:
        return "N/A"
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) > 0 else None
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def step3_output_scene_csv() -> bool:
    """Step 3: Output scene scores CSV"""
    print("\n" + "="*60)
    print("Step 3: Outputting scene scores CSV")
    print("="*60)

    summary_path = Path(SUMMARY_PATH)
    if not summary_path.exists():
        print(f"[Error] Summary file not found: {summary_path}")
        return False

    with open(summary_path, 'r') as f:
        all_summaries = json.load(f)

    if not all_summaries:
        print("[Error] No summary data found")
        return False

    # CSV header
    fieldnames = ['Scene', 'NDS', 'mAP', 'L2', 'OCC_IoU', 'drivable_iou', 'divider_iou', 'crossing_iou', 'contour_iou', 'map_iou_mean']

    # Build table data
    rows = []
    for summary in all_summaries:
        pkl_name = summary.get('pkl_name', 'unknown')
        scene_name = pkl_name.replace('val_', '').replace('.pkl', '')

        detection = summary.get('detection', {}) or {}
        planning = summary.get('planning', {})
        occ = summary.get('occ', {})
        map_metrics = summary.get('map', {}) or {}

        l2_value = None
        if isinstance(planning.get('L2'), dict):
            l2_value = planning.get('L2', {}).get('mean')

        occ_iou_value = occ.get('iou') if occ else None

        drivable_iou = map_metrics.get('drivable_iou')
        divider_iou = map_metrics.get('divider_iou')
        crossing_iou = map_metrics.get('crossing_iou')
        contour_iou = map_metrics.get('contour_iou')

        # Calculate map IoU mean
        map_iou_vals = [v for v in [drivable_iou, divider_iou, crossing_iou, contour_iou] if v is not None]
        map_iou_mean = sum(map_iou_vals) / len(map_iou_vals) if map_iou_vals else None

        row = {
            'Scene': scene_name,
            'NDS': detection.get('nd_score'),
            'mAP': detection.get('mAP'),
            'L2': l2_value,
            'OCC_IoU': occ_iou_value,
            'drivable_iou': drivable_iou,
            'divider_iou': divider_iou,
            'crossing_iou': crossing_iou,
            'contour_iou': contour_iou,
            'map_iou_mean': map_iou_mean,
        }
        rows.append(row)

    # Sort by scene name
    rows.sort(key=lambda x: x['Scene'])

    # Print to terminal
    print(f"{'Scene':<15} {'NDS':>8} {'mAP':>8} {'L2':>8} {'OCC_IoU':>8} {'drivable':>8} {'divider':>8} {'crossing':>8} {'contour':>8} {'map_mean':>8}")
    print("-" * 99)
    for row in rows:
        print(f"{row['Scene']:<15} {_format_value(row['NDS']):>8} {_format_value(row['mAP']):>8} {_format_value(row['L2']):>8} {_format_value(row['OCC_IoU']):>8} {_format_value(row['drivable_iou']):>8} {_format_value(row['divider_iou']):>8} {_format_value(row['crossing_iou']):>8} {_format_value(row['contour_iou']):>8} {_format_value(row['map_iou_mean']):>8}")

    # Save CSV
    csv_output_path = Path(OUTPUT_DIR) / 'scene_scores.csv'
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Save] CSV saved to: {csv_output_path}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scene-based UniAD Evaluation Pipeline")
    parser.add_argument("--skip-split", action="store_true", help="Skip PKL splitting")
    parser.add_argument("--skip-eval", action="store_true", help="Skip UniAD evaluation")
    parser.add_argument("--skip-csv", action="store_true", help="Skip scene CSV output")
    parser.add_argument("--master-port", type=int, default=29500, help="torch.distributed communication port")
    args = parser.parse_args()

    print("="*60)
    print("Scene-based UniAD Evaluation Pipeline")
    print("="*60)
    print(f"PKL Dir:   {PKL_DIR}")
    print(f"Output:    {OUTPUT_DIR}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"GPUs:      {GPU_IDS}")
    print("="*60)

    # Step 1
    if not args.skip_split:
        if not step1_split_pkl():
            print("[Error] Step 1 failed!")
            return
    else:
        print("[Skip] Step 1: Splitting PKL")

    # Step 2
    if not args.skip_eval:
        fps = get_video_fps(VIDEO_DIR)
        if not step2_run_uniad(fps, master_port=args.master_port):
            print("[Error] Step 2 failed!")
            return
    else:
        print("[Skip] Step 2: Running UniAD")

    # Step 3
    if not args.skip_csv:
        if not step3_output_scene_csv():
            print("[Error] Step 3 failed!")
            return
    else:
        print("[Skip] Step 3: Outputting scene CSV")

    print("\n" + "="*60)
    print("Pipeline completed!")
    print(f"Results: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
