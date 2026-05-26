#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
View UniAD result pkl file metrics Args
"""

import pickle
import argparse
import numpy as np
from pathlib import Path


def view_result(pkl_path: str):
    """View result pkl file"""
    print(f"Loading: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\nTop-level keys: {list(data.keys())}")

    # ============ Planning ============
    print("\n" + "="*60)
    print("Planning Results")
    print("="*60)
    planning = data.get('planning_results_computed', {})
    if planning:
        for k, v in planning.items():
            if hasattr(v, 'cpu'):
                v = v.cpu().numpy()
            if isinstance(v, np.ndarray):
                print(f"  {k}: {v}")
                print(f"    mean: {np.mean(v):.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("  (none)")

    # ============ OCC ============
    print("\n" + "="*60)
    print("Occupancy Results")
    print("="*60)
    occ = data.get('occ_results_computed', {})
    if occ:
        for k, v in occ.items():
            if hasattr(v, 'cpu'):
                v = v.cpu().numpy()
            print(f"  {k}: {v}")
    else:
        print("  (none)")

    # ============ Detection ============
    print("\n" + "="*60)
    print("Detection Results (bbox_results)")
    print("="*60)
    bbox = data.get('bbox_results', [])
    if bbox:
        print(f"  Total samples: {len(bbox)}")
        print(f"  First sample keys: {list(bbox[0].keys())}")

        # Count detection boxes
        total_boxes = 0
        for item in bbox:
            if 'boxes_3d' in item:
                boxes = item['boxes_3d']
                if hasattr(boxes, 'tensor'):
                    total_boxes += len(boxes.tensor)
                elif hasattr(boxes, '__len__'):
                    total_boxes += len(boxes)
        print(f"  Total 3D boxes: {total_boxes}")

        # Show first sample details
        print(f"\n  First sample:")
        for k, v in bbox[0].items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}")
            elif isinstance(v, (list, tuple)):
                print(f"    {k}: len={len(v)}")
            else:
                print(f"    {k}: {type(v).__name__}")
    else:
        print("  (none)")


def main():
    # parser = argparse.ArgumentParser(description="View result pkl")
    # parser.add_argument("pkl_path", help="pkl file path")
    # args = parser.parse_args()

    pkl_path = "<path-to-local-resource>"
    view_result(pkl_path)


if __name__ == "__main__":
    main()
