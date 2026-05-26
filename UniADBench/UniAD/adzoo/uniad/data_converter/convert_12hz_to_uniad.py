#!/usr/bin/env python3
"""
Convert WorldLens 12Hz data to UniAD compatible format
Supplement missing fields: scene_token, prev, next, frame_idx, gt_inds, can_bus
"""

import pickle
import numpy as np
from tqdm import tqdm
import os
from nuscenes.nuscenes import NuScenes

def convert_12hz_to_uniad_format(input_pkl, output_pkl, nusc):
    """
    Convert 12Hz data to UniAD compatible format
    """
    print(f"Loading {input_pkl}...")
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    infos = data['infos']
    scene_tokens_12hz = data['scene_tokens']
    metadata = data.get('metadata', {'version': 'v1.0-trainval'})  # changed to original version

    # Build token -> scene_token mapping from nuScenes
    print("Building token -> scene_token mapping from nuScenes...")
    sample_to_scene = {}
    for sample in nusc.sample:
        sample_to_scene[sample['token']] = sample['scene_token']

    # Build key frame token -> correct scene_token mapping
    # Interpolated frames inherit scene_token from their key frames
    print("Mapping tokens to scene_tokens...")
    token_to_scene = {}
    for scene_tokens_list in scene_tokens_12hz:
        # Find first key frame (token length = 32)
        keyframe_token = None
        for t in scene_tokens_list:
            if len(t) == 32:
                keyframe_token = t
                break

        if keyframe_token and keyframe_token in sample_to_scene:
            real_scene_token = sample_to_scene[keyframe_token]
            # All frames (key frames + interpolated frames) use the same scene_token
            for t in scene_tokens_list:
                token_to_scene[t] = real_scene_token

    # Add missing fields for each info
    print(f"Processing {len(infos)} infos...")
    for idx, info in tqdm(enumerate(infos)):
        token = info['token']

        # 1. scene_token - obtained from nuScenes
        info['scene_token'] = token_to_scene.get(token, '')

        # 2. prev and next - based on scene_tokens_12hz order
        scene_idx = -1
        pos_in_scene = -1
        for si, scene in enumerate(scene_tokens_12hz):
            if token in scene:
                scene_idx = si
                pos_in_scene = scene.index(token)
                break

        if pos_in_scene > 0:
            info['prev'] = scene_tokens_12hz[scene_idx][pos_in_scene - 1]
        else:
            info['prev'] = ''

        if scene_idx >= 0 and pos_in_scene < len(scene_tokens_12hz[scene_idx]) - 1:
            info['next'] = scene_tokens_12hz[scene_idx][pos_in_scene + 1]
        else:
            info['next'] = ''

        # 3. frame_idx - based on position in scene
        info['frame_idx'] = pos_in_scene if pos_in_scene >= 0 else 0

        # 4. gt_inds - assign a unique index to each GT box
        num_gt = len(info.get('gt_boxes', []))
        info['gt_inds'] = np.arange(num_gt, dtype=np.int64)

        # 5. can_bus - computed from ego2global
        ego2global_trans = np.array(info['ego2global_translation'])
        ego2global_rot = np.array(info['ego2global_rotation'])

        # can_bus format: [tx, ty, tz, qw, qx, qy, qz, ...]
        can_bus = np.zeros(18, dtype=np.float64)
        can_bus[:3] = ego2global_trans  # translation
        can_bus[3:7] = ego2global_rot   # rotation (quaternion)
        # Remaining fields set to 0 (velocity, steering angle, etc.)
        info['can_bus'] = can_bus

        # 6. visibility_tokens (optional)
        if 'visibility_tokens' not in info and 'visibility' in info:
            info['visibility_tokens'] = info['visibility']

        # 7. Fix cams field
        for cam_name, cam_info in info['cams'].items():
            # Rename camera_intrinsics -> cam_intrinsic (UniAD field name requirement)
            if 'camera_intrinsics' in cam_info and 'cam_intrinsic' not in cam_info:
                cam_info['cam_intrinsic'] = cam_info.pop('camera_intrinsics')
            # Fix image path: remove "data/nuscenes/" prefix
            if 'data_path' in cam_info:
                cam_info['data_path'] = cam_info['data_path'].replace('data/nuscenes/', '', 1)

    # Save converted data
    output_data = {
        'infos': infos,
        'metadata': metadata,
        'scene_tokens': scene_tokens_12hz
    }

    print(f"Saving to {output_pkl}...")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(output_data, f)

    print("Done!")
    return output_data


if __name__ == '__main__':
    # nuScenes data path
    nusc_data_root = '<path-to-local-resource>'

    print("Loading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_data_root, verbose=True)

    # Output directory
    output_dir = '<path-to-local-resource>'

    # Convert validation set
    convert_12hz_to_uniad_format(
        '<path-to-local-resource>',
        output_dir + 'nuscenes_interp_12Hz_infos_val_uniad.pkl',
        nusc
    )

    # Convert training set
    convert_12hz_to_uniad_format(
        '<path-to-local-resource>',
        output_dir + 'nuscenes_interp_12Hz_infos_train_uniad.pkl',
        nusc
    )
