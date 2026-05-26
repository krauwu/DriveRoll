"""
PKL slicing script - generate temporary pkl files for UniAD testing

Usage:
    # 2Hz sampling, scene 0, first 50 frames
    python slice_pkl.py --fps 2 --scene-idx 0 --frame-start 0 --frame-end 50

    # Full scene, 2Hz sampling
    python slice_pkl.py --fps 2 --scene-idx 0

    # All scenes, 2Hz sampling
    python slice_pkl.py --fps 2

    Then modify ann_file in the config file, or use --config args for automatic modification
"""

import argparse
import pickle
import os
import copy


def main():
    parser = argparse.ArgumentParser(description='Slice PKL')
    parser.add_argument('--pkl-path', type=str,
                        default='./data/infos_12hz_native/nuscenes_12hz_infos_temporal_val.pkl',
                        help='Original PKL file path')
    parser.add_argument('--fps', type=int, default=12, choices=[2, 6, 12],
                        help='Target FPS, default 12 means no sampling')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Start frame index')
    parser.add_argument('--frame-end', type=int, default=None,
                        help='End frame index')
    parser.add_argument('--scene-idx', type=int, default=None,
                        help='Scene index')
    parser.add_argument('--out-dir', type=str, default='./data/infos_sliced_12hz',
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path (optional, auto-modifies ann_file)')

    args = parser.parse_args()

    print("="*60)
    print("Slicing PKL")
    print("="*60)

    # Load original pkl
    print(f"Loaded: {args.pkl_path}")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    infos = data['infos']
    metadata = copy.deepcopy(data.get('metadata', {'version': 'interp_12Hz_trainval'}))
    scene_tokens = data.get('scene_tokens', None)

    print(f"Original frame count: {len(infos)}")

    # 1. Scene filtering
    if args.scene_idx is not None:
        unique_scenes = list(set(info['scene_token'] for info in infos))
        unique_scenes.sort()
        if args.scene_idx >= len(unique_scenes):
            print(f"Warning: scene_idx out of range, there are {len(unique_scenes)} scenes")
            args.scene_idx = 0
        target_scene = unique_scenes[args.scene_idx]
        infos = [info for info in infos if info['scene_token'] == target_scene]
        print(f"Scene filtering: scene_idx={args.scene_idx}, {len(infos)} frames (12Hz)")

    # 2. Frame window filtering
    if args.frame_end is None:
        args.frame_end = len(infos)
    infos = infos[args.frame_start:args.frame_end]
    print(f"Frame window: [{args.frame_start}, {args.frame_end}) -> {len(infos)} frames")

    # 3. FPS sampling
    if args.fps != 12:
        interval = 12 // args.fps
        infos = infos[::interval]
        print(f"FPS sampling: {args.fps}Hz (taking 1 frame every {interval} frames) -> {len(infos)} frames")

    # 4. Update prev/next relationships and frame_idx
    # Official code uses index + t to find future frames, so frame_idx must be contiguous
    for i, info in enumerate(infos):
        # Deep copy to avoid modifying original data
        info = copy.deepcopy(info)
        infos[i] = info

        if i > 0:
            info['prev'] = infos[i-1]['token']
        else:
            info['prev'] = ''
        if i < len(infos) - 1:
            info['next'] = infos[i+1]['token']
        else:
            info['next'] = ''
        info['frame_idx'] = i

    # Generate output file name
    out_name = f"sliced_fps{args.fps}"
    if args.scene_idx is not None:
        out_name += f"_scene{args.scene_idx}"
    if args.frame_end is not None:
        out_name += f"_{args.frame_start}-{args.frame_end}"
    out_name += ".pkl"

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, out_name)

    output_data = {
        'infos': infos,
        'metadata': metadata,
    }
    if scene_tokens:
        output_data['scene_tokens'] = scene_tokens

    with open(out_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\nOutput: {out_path}")
    print(f"Sample count: {len(infos)}")

    # Modify config file
    if args.config:
        print(f"\nModify config file: {args.config}")
        modify_config(args.config, out_path)

    print("\n" + "="*60)
    print("Usage:")
    print("="*60)
    print(f"1. Modify ann_file in config file to:")
    print(f"   ann_file={out_path}")
    print(f"\n2. Run test:")
    print(f"   CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 adzoo/uniad/test.py <config> <checkpoint> --launcher pytorch --eval bbox")


def modify_config(config_path, ann_file_path):
    """Modify ann_file in config file"""
    # Read config file
    with open(config_path, 'r') as f:
        content = f.read()

    # Find and replace ann_file
    import re

    # Match ann_file for test dataset
    # Format may be: ann_file=data_root + 'xxx.pkl' or ann_file='xxx.pkl'
    pattern = r"(data\[.test.\].*?ann_file\s*=\s*)[^\n]+"

    def replace_ann_file(match):
        return match.group(1) + f"'{ann_file_path}'"

    # Simple replacement: find ann_file line in test
    lines = content.split('\n')
    in_test = False
    new_lines = []

    for line in lines:
        if 'test=' in line or "test =" in line:
            in_test = True
        if in_test and 'ann_file' in line and 'test' not in line.split('ann_file')[0]:
            # Find ann_file line in test block
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + f"ann_file='{ann_file_path}',")
            continue
        if in_test and line.strip() and not line.strip().startswith('#') and 'ann_file' not in line:
            # Check if leaving test block (encountering new key=val)
            if '=' in line and not line.strip().startswith('pipeline') and not line.strip().startswith('['):
                in_test = False
        new_lines.append(line)

    new_content = '\n'.join(new_lines)

    # Write back
    with open(config_path, 'w') as f:
        f.write(new_content)

    print(f"Modified ann_file in config file to: {ann_file_path}")


if __name__ == '__main__':
    main()