"""
Fix cam_intrinsic format and image path in PKL file
Convert list type to numpy.ndarray type
Correct image path to proper relative path
Convert can_bus to float64 to be compatible with torchvision

Usage:
    python fix_intrinsic_format.py --input /path/to/input.pkl --output /path/to/output.pkl
"""

import argparse
import pickle
import numpy as np
import os
import re


def fix_data_path(data_path, data_root="/data/nuscenes"):
    """
    Fix data path, extract relative path after nuscenes dataset
    e.g.：
    <path-to-local-resource>
    -> /data/nuscenes/samples/CAM_FRONT/xxx.jpg
    """
    if data_path is None:
        return data_path

    # Find nuscenes-related path parts
    patterns = [
        r'(/samples/.+)',      # Match /samples/...
        r'(/sweeps/.+)',       # Match /sweeps/...
        r'(/maps/.+)',         # Match /maps/...
    ]

    for pattern in patterns:
        match = re.search(pattern, data_path)
        if match:
            relative_path = match.group(1)
            return os.path.join(data_root, relative_path.lstrip('/'))

    # If no match, return original path
    return data_path


def fix_pkl_paths(input_path, output_path=None):
    """
    Fix path and format issues in pkl:
    1. cam_intrinsic: list -> numpy.ndarray
    2. Image path correction
    3. can_bus: float32 -> float64 (compatible with torchvision.rotate)
    """
    # If no output path specified, overwrite original file
    if output_path is None:
        output_path = input_path

    print(f"Loaded: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    infos = data['infos']
    print(f"Total frames: {len(infos)}")

    fixed_intrinsic_count = 0
    fixed_path_count = 0
    fixed_can_bus_count = 0

    for info in infos:
        # Fix paths and intrinsic in cams
        if 'cams' in info:
            for cam_name, cam_info in info['cams'].items():
                # Fix cam_intrinsic
                if 'cam_intrinsic' in cam_info:
                    intrinsic = cam_info['cam_intrinsic']
                    if isinstance(intrinsic, list):
                        cam_info['cam_intrinsic'] = np.array(intrinsic, dtype=np.float64)
                        fixed_intrinsic_count += 1

                # Fix data_path
                if 'data_path' in cam_info:
                    old_path = cam_info['data_path']
                    new_path = fix_data_path(old_path)
                    if old_path != new_path:
                        cam_info['data_path'] = new_path
                        fixed_path_count += 1

        # Fix lidar_path
        if 'lidar_path' in info:
            old_path = info['lidar_path']
            new_path = fix_data_path(old_path)
            if old_path != new_path:
                info['lidar_path'] = new_path
                fixed_path_count += 1

        # Fix paths in sweeps
        if 'sweeps' in info:
            for sweep in info['sweeps']:
                if 'data_path' in sweep:
                    old_path = sweep['data_path']
                    new_path = fix_data_path(old_path)
                    if old_path != new_path:
                        sweep['data_path'] = new_path
                        fixed_path_count += 1

        # Fix can_bus type (float32 -> float64)
        if 'can_bus' in info:
            can_bus = info['can_bus']
            if isinstance(can_bus, np.ndarray) and can_bus.dtype == np.float32:
                info['can_bus'] = can_bus.astype(np.float64)
                fixed_can_bus_count += 1

    print(f"Fixed {fixed_intrinsic_count} cam_intrinsic fields")
    print(f"Fixed {fixed_path_count} path fields")
    print(f"Fixed {fixed_can_bus_count} can_bus types")

    # Save
    print(f"Saved to: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Fix path and format in PKL')
    parser.add_argument('--input', type=str, required=True, help='Input PKL file path')
    parser.add_argument('--output', type=str, default=None, help='Output PKL file path (default: overwrite original file)')
    parser.add_argument('--data-root', type=str, default="<path-to-local-resource>",
                        help='nuscenes data root directory')

    args = parser.parse_args()

    # Update data_root
    global fix_data_path
    original_fix_data_path = fix_data_path
    fix_data_path = lambda path: original_fix_data_path(path, args.data_root)

    fix_pkl_paths(args.input, args.output)


if __name__ == '__main__':
    main()
