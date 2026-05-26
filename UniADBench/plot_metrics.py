#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot evaluation metric curves

Supports plotting:
- NDS vs Window
- mAP vs Window
- OCC IoU vs Window
- Planning L2 vs Time (one curve per window, 6 time points in total)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


# Default configuration
DEFAULT_SUMMARY_PATH = "<path-to-local-resource>"
DEFAULT_OUTPUT_DIR = "<path-to-local-resource>"

# Planning time points (seconds), corresponding to 6 prediction steps
PLANNING_TIMESTEPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_summary(summary_path: str) -> List[Dict[str, Any]]:
    """Load summary JSON"""
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_window_index(pkl_name: str) -> int:
    """Extract window start index from pkl name, e.g. val_window_005_015.pkl -> 5"""
    import re
    match = re.search(r'window_(\d+)_(\d+)', pkl_name)
    if match:
        return int(match.group(1))
    return 0


def extract_window_label(pkl_name: str) -> str:
    """Extract window label from pkl name, e.g. val_window_005_015.pkl -> '5-10'"""
    import re
    match = re.search(r'window_(\d+)_(\d+)', pkl_name)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return f'{start}-{end}'
    return '0-0'


def plot_nds(data: List[Dict], output_dir: Path, figsize=(12, 6)):
    """Plot NDS vs Window"""
    windows = []
    nds_values = []

    for item in data:
        det = item.get('detection', {})
        if det and det.get('nd_score') is not None:
            windows.append(extract_window_index(item['pkl_name']))
            nds_values.append(det['nd_score'])

    if not windows:
        print("[Warning] No NDS data found")
        return

    # Sort by window
    sorted_pairs = sorted(zip(windows, nds_values))
    windows, nds_values = zip(*sorted_pairs)

    # Generate labels
    labels = [extract_window_label(item['pkl_name']) for item in sorted(data, key=lambda x: extract_window_index(x['pkl_name'])) if item.get('detection', {}).get('nd_score') is not None]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = range(len(windows))
    ax.plot(x_pos, nds_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('Window (frames)', fontsize=12)
    ax.set_ylabel('NDS', fontsize=12)
    ax.set_title('NDS vs Window', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Annotate values
    for x, y in zip(x_pos, nds_values):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    output_path = output_dir / 'nds_vs_window.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_map(data: List[Dict], output_dir: Path, figsize=(12, 6)):
    """Plot mAP vs Window"""
    windows = []
    map_values = []

    for item in data:
        det = item.get('detection', {})
        if det and det.get('mAP') is not None:
            windows.append(extract_window_index(item['pkl_name']))
            map_values.append(det['mAP'])

    if not windows:
        print("[Warning] No mAP data found")
        return

    sorted_pairs = sorted(zip(windows, map_values))
    windows, map_values = zip(*sorted_pairs)

    # Generate labels
    labels = [extract_window_label(item['pkl_name']) for item in sorted(data, key=lambda x: extract_window_index(x['pkl_name'])) if item.get('detection', {}).get('mAP') is not None]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = range(len(windows))
    ax.plot(x_pos, map_values, 'go-', linewidth=2, markersize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('Window (frames)', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('mAP vs Window', fontsize=14)
    ax.grid(True, alpha=0.3)

    for x, y in zip(x_pos, map_values):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    output_path = output_dir / 'map_vs_window.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_occ_iou(data: List[Dict], output_dir: Path, figsize=(12, 6)):
    """Plot OCC IoU vs Window"""
    windows = []
    iou_values = []

    for item in data:
        occ = item.get('occ', {})
        if occ and 'iou' in occ:
            windows.append(extract_window_index(item['pkl_name']))
            iou_val = occ['iou']
            # If it is a list, take the first value or average
            if isinstance(iou_val, (list, tuple)):
                iou_val = iou_val[0] if iou_val else 0
            iou_values.append(iou_val)

    if not windows:
        print("[Warning] No OCC IoU data found")
        return

    sorted_pairs = sorted(zip(windows, iou_values))
    windows, iou_values = zip(*sorted_pairs)

    # Generate labels
    labels = [extract_window_label(item['pkl_name']) for item in sorted(data, key=lambda x: extract_window_index(x['pkl_name'])) if item.get('occ', {}).get('iou') is not None]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = range(len(windows))
    ax.plot(x_pos, iou_values, 'ro-', linewidth=2, markersize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('Window (frames)', fontsize=12)
    ax.set_ylabel('OCC IoU (%)', fontsize=12)
    ax.set_title('OCC IoU vs Window', fontsize=14)
    ax.grid(True, alpha=0.3)

    for x, y in zip(x_pos, iou_values):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    output_path = output_dir / 'occ_iou_vs_window.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_planning_metric_by_timestep(
    data: List[Dict],
    output_dir: Path,
    metric_name: str,
    metric_label: str,
    output_filename: str,
    figsize=(14, 10)
):
    """
    Plot a metric vs Window for each time point separately

    Args:
        data: summary data
        output_dir: Output directory
        metric_name: metric name (L2, obj_col, obj_box_col)
        metric_label: Y-axis label
        output_filename: output file name
        figsize: figure size
    """
    # Collect metric value for each timestep
    timestep_data = {i: {'windows': [], 'labels': [], 'values': []} for i in range(6)}

    for item in data:
        planning = item.get('planning', {})
        metric_data = planning.get(metric_name, {})
        values = metric_data.get('values', [])

        if values and len(values) >= 6:
            window_idx = extract_window_index(item['pkl_name'])
            window_label = extract_window_label(item['pkl_name'])
            for t in range(6):
                timestep_data[t]['windows'].append(window_idx)
                timestep_data[t]['labels'].append(window_label)
                timestep_data[t]['values'].append(values[t])

    if not any(d['windows'] for d in timestep_data.values()):
        print(f"[Warning] No {metric_name} data found")
        return

    # 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for t in range(6):
        ax = axes[t]
        d = timestep_data[t]

        if not d['windows']:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue

        # Sort
        sorted_pairs = sorted(zip(d['windows'], d['labels'], d['values']))
        windows, labels, vals = zip(*sorted_pairs)

        x_pos = range(len(windows))
        ax.plot(x_pos, vals, 'mo-', linewidth=1.5, markersize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax.set_xlabel('Window', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f't = {PLANNING_TIMESTEPS[t]}s', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{metric_name} at Each Timestep', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_all(data: List[Dict], output_dir: Path):
    """Plot all charts"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print("Plotting Metrics")
    print("="*50)

    plot_nds(data, output_dir)
    plot_map(data, output_dir)
    plot_occ_iou(data, output_dir)

    # Planning L2
    plot_planning_metric_by_timestep(
        data, output_dir,
        metric_name='L2',
        metric_label='L2 (m)',
        output_filename='planning_l2_by_timestep.png'
    )

    # Planning obj_col (Collision)
    plot_planning_metric_by_timestep(
        data, output_dir,
        metric_name='obj_col',
        metric_label='Collision Rate',
        output_filename='planning_obj_col_by_timestep.png'
    )

    # Planning obj_box_col (Box Collision)
    plot_planning_metric_by_timestep(
        data, output_dir,
        metric_name='obj_box_col',
        metric_label='Box Collision Rate',
        output_filename='planning_obj_box_col_by_timestep.png'
    )

    print("\n" + "="*50)
    print(f"All figures saved to: {output_dir}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation metric curves")
    parser.add_argument(
        "--summary", "-s",
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary JSON path (default: {DEFAULT_SUMMARY_PATH})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    # Load data
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"[Error] Summary file not found: {summary_path}")
        return

    data = load_summary(str(summary_path))
    print(f"Loaded {len(data)} items from {summary_path}")

    # Plot
    output_dir = Path(args.output)
    plot_all(data, output_dir)


if __name__ == "__main__":
    main()
