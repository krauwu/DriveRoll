"""
Convert custom pkl format to UniAD-compatible format
Input: copy/infos.pkl (contains frame_mapping)
Output: data/infos_custom/custom_infos_val.pkl
"""
import pickle
import numpy as np
import os
from pyquaternion import Quaternion

# Category mapping: nuScenes category name -> UniAD category name
CATEGORY_MAP = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'barrier',
    'movable_object.debris': 'barrier',
    'static_object.bicycle_rack': 'bicycle',
}

# UniAD category list
NUS_CATEGORIES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def convert_custom_to_uniad(input_pkl_path, output_dir):
    """Convert custom pkl to UniAD format"""

    print(f"Loaded: {input_pkl_path}")
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)

    frame_mapping = data['frame_mapping']
    print(f"Total frames: {len(frame_mapping)}")

    # Get scene info (assuming all frames belong to same scene)
    scene_token = frame_mapping[0]['scene_token']

    # Get lidar to ego transform (estimated from any camera's sensor info)
    # Note: use CAM_FRONT sensor_translation/rotation as reference here
    # In fact lidar to ego transform is fixed, simplified here
    # In nuScenes lidar is at roof center, cameras have their own extrinsics

    # Simplified: use default lidar2ego transform
    # Actual nuScenes: lidar2ego_translation = [0.0, 0.0, 1.52] (approx)
    lidar2ego_translation = [0.0, 0.0, 1.52]
    lidar2ego_rotation = [1.0, 0.0, 0.0, 0.0]  # no rotation

    uniad_infos = []

    for i, frame in enumerate(frame_mapping):
        info = {}

        # Basic info
        info['token'] = frame['sample_token']
        info['scene_token'] = frame['scene_token']
        info['timestamp'] = frame['timestamp']
        info['frame_idx'] = frame['frame_idx']

        # prev/next (computed from frame indices)
        if i > 0:
            info['prev'] = frame_mapping[i-1]['sample_token']
        else:
            info['prev'] = ''

        if i < len(frame_mapping) - 1:
            info['next'] = frame_mapping[i+1]['sample_token']
        else:
            info['next'] = ''

        # can_bus (set to all zeros, can be obtained from raw data later)
        # can_bus format: [pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, ...] total 18 dimensions
        can_bus = np.zeros(18)
        # Fill position and rotation info
        camera_data = frame['camera_data']
        cam_front = camera_data.get('CAM_FRONT', {})
        if cam_front:
            ego_translation = cam_front.get('ego_translation', [0, 0, 0])
            ego_rotation = cam_front.get('ego_rotation', [1, 0, 0, 0])  # w, x, y, z
            can_bus[:3] = ego_translation
            can_bus[3:7] = ego_rotation
        info['can_bus'] = can_bus

        # lidar path (UniAD needs to check file existence, set a dummy path here)
        info['lidar_path'] = 'virtual_lidar.bin'

        # lidar2ego transform
        info['lidar2ego_translation'] = lidar2ego_translation
        info['lidar2ego_rotation'] = lidar2ego_rotation

        # ego2global transform (obtained from CAM_FRONT)
        if cam_front:
            info['ego2global_translation'] = cam_front.get('ego_translation', [0, 0, 0])
            info['ego2global_rotation'] = cam_front.get('ego_rotation', [1, 0, 0, 0])
        else:
            info['ego2global_translation'] = [0, 0, 0]
            info['ego2global_rotation'] = [1, 0, 0, 0]

        # sweeps (empty list because max_sweeps=0)
        info['sweeps'] = []

        # camera data
        info['cams'] = {}
        for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            if cam_name in camera_data:
                cam = camera_data[cam_name]

                # Compute sensor2lidar transform
                # sensor_translation/rotation is camera to ego transform
                # sensor2lidar = sensor2ego @ ego2lidar = sensor2ego @ inv(lidar2ego)
                # Simplified: directly use sensor_translation/rotation
                # Note: UniAD uses sensor2lidar

                cam_info = {
                    'data_path': cam['filename'],
                    'sample_data_token': cam['sample_data_token'],
                    'cam_intrinsic': np.array(cam['camera_intrinsic']),
                    # Simplified: sensor2lidar uses sensor2ego
                    'sensor2ego_translation': cam['sensor_translation'],
                    'sensor2ego_rotation': cam['sensor_rotation'],
                    'sensor2lidar_translation': cam['sensor_translation'],  # simplified
                    'sensor2lidar_rotation': cam['sensor_rotation'],
                    'ego2global_translation': cam['ego_translation'],
                    'ego2global_rotation': cam['ego_rotation'],
                    'timestamp': cam['timestamp'],
                }
                info['cams'][cam_name] = cam_info
            else:
                print(f"Warning: frame {i} missing camera {cam_name}")

        # GT annotations
        annotations = frame.get('annotations', [])
        if annotations:
            gt_boxes = []
            gt_names = []
            gt_velocity = []
            num_lidar_pts = []
            num_radar_pts = []
            valid_flag = []
            gt_inds = []
            instance_tokens = []

            for idx, ann in enumerate(annotations):
                # Convert category name
                category = ann.get('category_name', '')
                uniad_name = CATEGORY_MAP.get(category, None)
                if uniad_name is None or uniad_name not in NUS_CATEGORIES:
                    continue

                # GT box: [x, y, z, w, l, h, yaw]
                translation = ann['translation']  # [x, y, z]
                size = ann['size']  # [w, l, h]
                rotation = ann['rotation']  # quaternion [w, x, y, z] or [x, y, z, w]

                # Compute yaw
                if len(rotation) == 4:
                    # Assumed to be [w, x, y, z] or [x, y, z, w] format
                    # nuScenes uses [w, x, y, z]
                    try:
                        q = Quaternion(rotation)
                        yaw = q.yaw_pitch_roll[0]
                    except:
                        yaw = 0.0
                else:
                    yaw = 0.0

                # Assemble box: [x, y, z, w, l, h, yaw]
                box = translation + size + [yaw]
                gt_boxes.append(box)
                gt_names.append(uniad_name)
                gt_velocity.append([0.0, 0.0])  # velocity info missing
                num_lidar_pts.append(ann.get('num_lidar_pts', 0))
                num_radar_pts.append(ann.get('num_radar_pts', 0))
                valid_flag.append(ann.get('num_lidar_pts', 0) + ann.get('num_radar_pts', 0) > 0)
                gt_inds.append(idx)
                instance_tokens.append(ann.get('instance_token', ''))

            if gt_boxes:
                info['gt_boxes'] = np.array(gt_boxes, dtype=np.float32)
                info['gt_names'] = np.array(gt_names)
                info['gt_velocity'] = np.array(gt_velocity, dtype=np.float32)
                info['num_lidar_pts'] = np.array(num_lidar_pts)
                info['num_radar_pts'] = np.array(num_radar_pts)
                info['valid_flag'] = np.array(valid_flag, dtype=bool)
                info['gt_inds'] = np.array(gt_inds)
                info['gt_ins_tokens'] = np.array(instance_tokens)
            else:
                # Empty GT
                info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                info['gt_names'] = np.array([])
                info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
                info['num_lidar_pts'] = np.array([])
                info['num_radar_pts'] = np.array([])
                info['valid_flag'] = np.array([], dtype=bool)
                info['gt_inds'] = np.array([])
                info['gt_ins_tokens'] = np.array([])
        else:
            # No annotations
            info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
            info['gt_names'] = np.array([])
            info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
            info['num_lidar_pts'] = np.array([])
            info['num_radar_pts'] = np.array([])
            info['valid_flag'] = np.array([], dtype=bool)
            info['gt_inds'] = np.array([])
            info['gt_ins_tokens'] = np.array([])

        # Future trajectory info (missing, set empty)
        info['fut_traj'] = np.zeros((0, 12, 2), dtype=np.float32)
        info['fut_traj_valid_mask'] = np.zeros((0, 12, 2), dtype=np.float32)

        uniad_infos.append(info)

    # Sort by timestamp
    uniad_infos = sorted(uniad_infos, key=lambda x: x['timestamp'])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save in UniAD format
    output_data = {
        'infos': uniad_infos,
        'metadata': {
            'version': 'custom_v1.0',
            'num_samples': len(uniad_infos),
        }
    }

    output_path = os.path.join(output_dir, 'custom_infos_val.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Conversion completed!")
    print(f"Output path: {output_path}")
    print(f"Sample count: {len(uniad_infos)}")

    return output_path


if __name__ == '__main__':
    input_path = '<path-to-local-resource>'
    output_dir = '<path-to-local-resource>'

    convert_custom_to_uniad(input_path, output_dir)