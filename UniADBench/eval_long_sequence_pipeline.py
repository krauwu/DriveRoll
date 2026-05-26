#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bench Pipeline - Unified Entry
Step 1: Split PKL (call generate_uniad_pkls.py)
Step 2: Run UniAD evaluation (call UniADEval)
Step 3: Plot evaluation metric curves (call plot_metrics)
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

from video_utils import get_video_meta
from run_uniad import UniADEval
from plot_metrics import load_summary, plot_all


# ============ Configuration Area ============
# Video splitting configuration
BENCH_ROOT = "<path-to-local-resource>"
MAIN_CFG = f"{BENCH_ROOT}/cfg/nuscenes_default.yaml"


def load_config(cfg_path: str) -> dict:
    """Load YAML configuration file"""
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


# Dynamically read from config file
CFG = load_config(MAIN_CFG)
TASK_NAME = CFG['task']['name']  # Get task name (e.g. uniad_window_pkls)
TASK_CFG = CFG['task'][TASK_NAME]  # Get task configuration

# UniAD configuration
UNIAD_ROOT = f"{BENCH_ROOT}/UniAD"
CONFIG_PATH = f"{UNIAD_ROOT}/adzoo/uniad/configs/stage2_e2e/base_e2e.py"
CHECKPOINT = "<path-to-local-resource>"
PKL_DIR = f"{TASK_CFG['pseudo_nuscenes_root']}{TASK_CFG['output']['subdir']}"
DATA_ROOT = TASK_CFG['pseudo_nuscenes_root']
OUTPUT_DIR = '<path-to-local-resource>'
VIDEO_DIR = CFG['input']['video_dir']
GPU_IDS = "5,6,7,2"  # Multi-GPU setting, comma-separated, e.g. "0,1,2,3"

# Plot configuration
SUMMARY_PATH = f"{OUTPUT_DIR}/all_metrics_summary.json"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
# =================================


def get_video_fps(video_dir: str) -> float:
    """Get video FPS"""
    video_path = Path(video_dir)
    video_files = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))

    if not video_files:
        print(f"[Warning] No video files in {video_dir}, using default fps=6")
        return 6

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


def step3_plot_metrics() -> bool:
    """Step 3: Plot metric curves"""
    print("\n" + "="*60)
    print("Step 3: Plotting metrics")
    print("="*60)

    summary_path = Path(SUMMARY_PATH)
    if not summary_path.exists():
        print(f"[Error] Summary file not found: {summary_path}")
        return False

    data = load_summary(str(summary_path))
    print(f"Loaded {len(data)} items from {summary_path}")

    output_dir = Path(FIGURES_DIR)
    plot_all(data, output_dir)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bench Pipeline")
    parser.add_argument("--skip-split", action="store_true", help="Skip PKL splitting")
    parser.add_argument("--skip-eval", action="store_true", help="Skip UniAD evaluation")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--master-port", type=int, default=29500, help="torch.distributed communication port")
    args = parser.parse_args()

    print("="*60)
    print("Bench Pipeline")
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
    if not args.skip_plot:
        if not step3_plot_metrics():
            print("[Error] Step 3 failed!")
            return
    else:
        print("[Skip] Step 3: Plotting metrics")

    print("\n" + "="*60)
    print("Pipeline completed!")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
