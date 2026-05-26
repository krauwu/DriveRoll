"""
Dynamic sampling evaluation script
Dynamically sample from 12Hz pkl by fps and frame window, create UniAD dataset for evaluation

Usage:
    python test_dynamic_sampling.py --fps 2 --frame-start 0 --frame-end 50
"""

import argparse
import pickle
import numpy as np
import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmcv.datasets import DATASETS
from mmcv.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset


def load_12hz_pkl(pkl_path):
    """Load 12Hz pkl file"""
    print(f"Loaded: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Keys: {data.keys()}")
    print(f"Total frames: {len(data['infos'])}")
    if 'metadata' in data:
        print(f"Metadata: {data['metadata']}")

    return data


def sample_infos_by_fps(infos, fps=2, base_fps=12):
    """
    Sample infos by fps

    Args:
        infos: original infos list (12Hz)
        fps: Target FPS (2, 6, 12)
        base_fps: base fps (default 12)

    Returns:
        Sampled infos list
    """
    if fps == base_fps:
        return infos

    interval = base_fps // fps
    sampled_infos = infos[::interval]

    print(f"Sampling: {len(infos)} frames ({base_fps}Hz) -> {len(sampled_infos)} frames ({fps}Hz)")
    print(f"Sampling interval: taking 1 frame every {interval} frames")

    return sampled_infos


def filter_infos_by_frame_window(infos, frame_start=0, frame_end=None):
    """
    Filter infos by frame window

    Args:
        infos: infos list
        frame_start: Start frame index
        frame_end: end frame index (None means to the end)

    Returns:
        Filtered infos list
    """
    if frame_end is None:
        frame_end = len(infos)

    filtered_infos = infos[frame_start:frame_end]

    print(f"Frame window filter: [{frame_start}, {frame_end}) -> {len(filtered_infos)} frames")

    return filtered_infos


def filter_infos_by_scene(infos, scene_token=None):
    """
    Filter infos by scene

    Args:
        infos: infos list
        scene_token: scene token (None means all)

    Returns:
        Filtered infos list
    """
    if scene_token is None:
        return infos

    filtered_infos = [info for info in infos if info.get('scene_token') == scene_token]

    print(f"Scene filtering: scene_token={scene_token} -> {len(filtered_infos)} frames")

    return filtered_infos


def create_dataset_with_infos(infos, metadata, config_path, data_root):
    """
    Create dataset with provided infos

    Args:
        infos: Sampled infos list
        metadata: metadata
        config_path: Config file path
        data_root: data root directory

    Returns:
        dataset instance
    """
    from mmcv import Config

    cfg = Config.fromfile(config_path)

    # Get dataset config
    dataset_cfg = cfg.data.test

    # Create dataset, pass infos and metadata
    dataset = NuScenesE2EDataset(
        ann_file=dataset_cfg.ann_file,
        pipeline=dataset_cfg.pipeline,
        data_root=data_root,
        test_mode=True,
        infos=infos,
        metadata=metadata,
        **{k: v for k, v in dataset_cfg.items()
           if k not in ['ann_file', 'pipeline', 'data_root', 'type', 'infos', 'metadata']}
    )

    print(f"Dataset created successfully, {len(dataset)} samples")

    return dataset


def test_basic_sampling():
    """Test basic sampling function"""
    print("="*60)
    print("Test 1: Basic sampling function")
    print("="*60)

    pkl_path = '<path-to-local-resource>'

    if not os.path.exists(pkl_path):
        print(f"Warning: pkl file does not exist: {pkl_path}")
        return

    # Load data
    data = load_12hz_pkl(pkl_path)
    infos = data['infos']
    metadata = data.get('metadata', {'version': 'interp_12Hz_trainval'})

    # Test different fps sampling
    for fps in [2, 6, 12]:
        print(f"\n--- FPS={fps} ---")
        sampled = sample_infos_by_fps(infos, fps=fps, base_fps=12)

    # Test frame window filtering
    print("\n--- Frame window filter test ---")
    filtered = filter_infos_by_frame_window(infos, frame_start=0, frame_end=100)

    # Combined test
    print("\n--- Combined test: fps=2 + frame_window=[0,100) ---")
    sampled = sample_infos_by_fps(infos, fps=2, base_fps=12)
    filtered = filter_infos_by_frame_window(sampled, frame_start=0, frame_end=100)

    print("\nTest 1 completed!")


def test_dataset_creation():
    """Test dataset creation"""
    print("\n" + "="*60)
    print("Test 2: Dataset creation")
    print("="*60)

    pkl_path = '<path-to-local-resource>'
    config_path = '<path-to-local-resource>'
    data_root = '<path-to-local-resource>'

    if not os.path.exists(pkl_path):
        print(f"Warning: pkl file does not exist: {pkl_path}")
        return

    if not os.path.exists(config_path):
        print(f"Warning: config file does not exist: {config_path}")
        return

    # Load data
    data = load_12hz_pkl(pkl_path)
    infos = data['infos']
    metadata = data.get('metadata', {'version': 'interp_12Hz_trainval'})

    # Sample
    sampled_infos = sample_infos_by_fps(infos, fps=2, base_fps=12)
    filtered_infos = filter_infos_by_frame_window(sampled_infos, frame_start=0, frame_end=50)

    # Create dataset
    try:
        dataset = create_dataset_with_infos(filtered_infos, metadata, config_path, data_root)

        # Test retrieving data
        print("\nTest retrieving first sample...")
        sample = dataset[0]
        print(f"Sample keys: {sample.keys() if isinstance(sample, dict) else type(sample)}")

        print("\nTest 2 completed!")

    except Exception as e:
        print(f"Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Dynamic sampling test script')
    parser.add_argument('--fps', type=int, default=2, choices=[2, 6, 12],
                        help='Target FPS')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Start frame index')
    parser.add_argument('--frame-end', type=int, default=None,
                        help='End frame index')
    parser.add_argument('--scene-token', type=str, default=None,
                        help='scene token')
    parser.add_argument('--test', type=str, default='all',
                        choices=['basic', 'dataset', 'all'],
                        help='Test type')

    args = parser.parse_args()

    if args.test in ['basic', 'all']:
        test_basic_sampling()

    if args.test in ['dataset', 'all']:
        test_dataset_creation()


if __name__ == '__main__':
    main()