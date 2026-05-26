#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComparisonPlot: plot metrics from two models (e.g. GT vs Rolling-Forcing) on the same figure

Usage:
    python plot_compare.py \
        --summary1 <path-to-local-resource> \
        --label1 "GT" \
        --summary2 <path-to-local-resource> \
        --label2 "Rolling-Forcing" \
        --output ./compare_figures
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

# Planning time points (seconds)
PLANNING_TIMESTEPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Default colors and markers
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS = ['o', 's', '^', 'D', 'v']


def load_summary(summary_path: str) -> List[Dict[str, Any]]:
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_window_index(pkl_name: str) -> int:
    import re
    match = re.search(r'window_(\d+)_(\d+)', pkl_name)
    if match:
        return int(match.group(1))
    return 0


def extract_window_label(pkl_name: str) -> str:
    import re
    match = re.search(r'window_(\d+)_(\d+)', pkl_name)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return f'{start}-{end}'
    return '0-0'


def _extract_metric(data: List[Dict], extract_fn):
    """General extraction function, returns (windows, labels, values)"""
    windows, labels, values = [], [], []
    for item in data:
        val = extract_fn(item)
        if val is not None:
            windows.append(extract_window_index(item['pkl_name']))
            labels.append(extract_window_label(item['pkl_name']))
            values.append(val)
    if not windows:
        return None, None, None
    sorted_pairs = sorted(zip(windows, labels, values))
    windows, labels, values = zip(*sorted_pairs)
    return windows, labels, values


def plot_compare_nds(datasets: List[List[Dict]], labels: List[str], output_dir: Path, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    has_data = False
    for i, data in enumerate(datasets):
        def extract(item, _i=i):
            det = item.get('detection', {})
            return det.get('nd_score') if det else None

        windows, win_labels, values = _extract_metric(data, extract)
        if windows is None:
            continue
        has_data = True
        x_pos = range(len(windows))
        ax.plot(x_pos, values, color=COLORS[i], marker=MARKERS[i],
                linewidth=2, markersize=8, label=labels[i])
        for x, y in zip(x_pos, values):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

    if not has_data:
        print("[Warning] No NDS data found")
        plt.close()
        return

    # Use the labels of the first dataset with data as the x-axis
    for data in datasets:
        def extract(item):
            det = item.get('detection', {})
            return det.get('nd_score') if det else None
        _, win_labels, _ = _extract_metric(data, extract)
        if win_labels:
            ax.set_xticks(range(len(win_labels)))
            ax.set_xticklabels(win_labels, fontsize=10)
            break

    ax.set_xlabel('Window (frames)', fontsize=12)
    ax.set_ylabel('NDS', fontsize=12)
    ax.set_title('NDS vs Window', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'nds_vs_window_compare.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_compare_map(datasets: List[List[Dict]], labels: List[str], output_dir: Path, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    has_data = False
    for i, data in enumerate(datasets):
        def extract(item, _i=i):
            det = item.get('detection', {})
            return det.get('mAP') if det else None

        windows, win_labels, values = _extract_metric(data, extract)
        if windows is None:
            continue
        has_data = True
        x_pos = range(len(windows))
        ax.plot(x_pos, values, color=COLORS[i], marker=MARKERS[i],
                linewidth=2, markersize=8, label=labels[i])
        for x, y in zip(x_pos, values):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

    if not has_data:
        print("[Warning] No mAP data found")
        plt.close()
        return

    for data in datasets:
        def extract(item):
            det = item.get('detection', {})
            return det.get('mAP') if det else None
        _, win_labels, _ = _extract_metric(data, extract)
        if win_labels:
            ax.set_xticks(range(len(win_labels)))
            ax.set_xticklabels(win_labels, fontsize=10)
            break

    ax.set_xlabel('Window (frames)', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('mAP vs Window', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'map_vs_window_compare.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_compare_occ_iou(datasets: List[List[Dict]], labels: List[str], output_dir: Path, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    has_data = False
    for i, data in enumerate(datasets):
        def extract(item, _i=i):
            occ = item.get('occ', {})
            if occ and 'iou' in occ:
                iou_val = occ['iou']
                if isinstance(iou_val, (list, tuple)):
                    return iou_val[0] if iou_val else None
                return iou_val
            return None

        windows, win_labels, values = _extract_metric(data, extract)
        if windows is None:
            continue
        has_data = True
        x_pos = range(len(windows))
        ax.plot(x_pos, values, color=COLORS[i], marker=MARKERS[i],
                linewidth=2, markersize=8, label=labels[i])
        for x, y in zip(x_pos, values):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

    if not has_data:
        print("[Warning] No OCC IoU data found")
        plt.close()
        return

    for data in datasets:
        def extract(item):
            occ = item.get('occ', {})
            if occ and 'iou' in occ:
                return 1  # just to get labels
            return None
        _, win_labels, _ = _extract_metric(data, extract)
        if win_labels:
            ax.set_xticks(range(len(win_labels)))
            ax.set_xticklabels(win_labels, fontsize=10)
            break

    ax.set_xlabel('Window (frames)', fontsize=12)
    ax.set_ylabel('OCC IoU (%)', fontsize=12)
    ax.set_title('OCC IoU vs Window', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'occ_iou_vs_window_compare.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_compare_planning_metric(
    datasets: List[List[Dict]],
    labels: List[str],
    output_dir: Path,
    metric_name: str,
    metric_label: str,
    output_filename: str,
    figsize=(14, 10)
):
    """ComparisonPlot planning metrics, one subplot per timestep, one line per model"""
    # Collect data for each dataset at each timestep
    all_timestep_data = []
    for d_idx, data in enumerate(datasets):
        timestep_data = {t: {'windows': [], 'labels': [], 'values': []} for t in range(6)}
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
        all_timestep_data.append(timestep_data)

    if not any(any(d[t]['windows'] for t in range(6)) for d in all_timestep_data):
        print(f"[Warning] No {metric_name} data found")
        return

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for t in range(6):
        ax = axes[t]

        for d_idx in range(len(datasets)):
            d = all_timestep_data[d_idx][t]
            if not d['windows']:
                continue
            sorted_pairs = sorted(zip(d['windows'], d['labels'], d['values']))
            windows, win_labels, vals = zip(*sorted_pairs)
            x_pos = range(len(windows))
            ax.plot(x_pos, vals, color=COLORS[d_idx], marker=MARKERS[d_idx],
                    linewidth=1.5, markersize=5, label=labels[d_idx])

        ax.set_xlabel('Window', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f't = {PLANNING_TIMESTEPS[t]}s', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Set x-axis labels
        for d_idx in range(len(datasets)):
            d = all_timestep_data[d_idx][t]
            if d['windows']:
                sorted_pairs = sorted(zip(d['windows'], d['labels'], d['values']))
                _, win_labels, _ = zip(*sorted_pairs)
                ax.set_xticks(range(len(win_labels)))
                ax.set_xticklabels(win_labels, fontsize=7, rotation=45, ha='right')
                break

    fig.suptitle(f'{metric_name} at Each Timestep', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_all_compare(datasets: List[List[Dict]], labels: List[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print(f"Plotting Compare: {' vs '.join(labels)}")
    print("="*50)

    plot_compare_nds(datasets, labels, output_dir)
    plot_compare_map(datasets, labels, output_dir)
    plot_compare_occ_iou(datasets, labels, output_dir)

    plot_compare_planning_metric(
        datasets, labels, output_dir,
        metric_name='L2',
        metric_label='L2 (m)',
        output_filename='planning_l2_by_timestep_compare.png'
    )

    plot_compare_planning_metric(
        datasets, labels, output_dir,
        metric_name='obj_col',
        metric_label='Collision Rate',
        output_filename='planning_obj_col_by_timestep_compare.png'
    )

    plot_compare_planning_metric(
        datasets, labels, output_dir,
        metric_name='obj_box_col',
        metric_label='Box Collision Rate',
        output_filename='planning_obj_box_col_by_timestep_compare.png'
    )

    print("\n" + "="*50)
    print(f"All figures saved to: {output_dir}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="ComparisonPlot metric curves for two models")
    parser.add_argument(
        "--summary1", "-s1", required=True,
        help="Summary JSON path for the first model"
    )
    parser.add_argument(
        "--label1", "-l1", required=True,
        help="Name label for the first model"
    )
    parser.add_argument(
        "--summary2", "-s2", required=True,
        help="Summary JSON path for the second model"
    )
    parser.add_argument(
        "--label2", "-l2", required=True,
        help="Name label for the second model"
    )
    parser.add_argument(
        "--output", "-o",
        default="<path-to-local-resource>",
        help="Output directory"
    )
    args = parser.parse_args()

    data1 = load_summary(args.summary1)
    data2 = load_summary(args.summary2)

    print(f"Loaded {len(data1)} items from {args.summary1}")
    print(f"Loaded {len(data2)} items from {args.summary2}")

    plot_all_compare([data1, data2], [args.label1, args.label2], Path(args.output))


if __name__ == "__main__":
    main()
