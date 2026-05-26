from __future__ import annotations

import os
import pickle
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion
from nuscenes.prediction.helper import convert_global_coords_to_local

try:
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
except Exception:
    NuScenesCanBus = None


NAME_MAPPING = {
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
}


def _normalize_path(path: str, root_path: Optional[str] = None) -> str:
    path = os.path.abspath(str(path))

    if root_path is not None:
        root_path = os.path.abspath(str(root_path))
        try:
            return os.path.relpath(path, root_path)
        except Exception:
            pass

    cwd = os.path.abspath(os.getcwd())
    if path.startswith(cwd + os.sep):
        return path[len(cwd) + 1:]

    return path


def _safe_box_velocity(nusc, ann_token: str) -> np.ndarray:
    v = nusc.box_velocity(ann_token)
    if v is None:
        return np.zeros(2, dtype=np.float32)
    v = np.asarray(v[:2], dtype=np.float32)
    if np.any(np.isnan(v)):
        return np.zeros(2, dtype=np.float32)
    return v


def _build_instance_index_map(nusc) -> Dict[str, int]:
    return {rec["token"]: i for i, rec in enumerate(nusc.instance)}


def _get_can_bus_info(nusc, nusc_can_bus, sample: Dict) -> np.ndarray:
    if nusc_can_bus is None:
        return np.zeros(18, dtype=np.float64)

    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    sample_timestamp = sample["timestamp"]

    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except Exception:
        return np.zeros(18, dtype=np.float64)

    if len(pose_list) == 0:
        return np.zeros(18, dtype=np.float64)

    last_pose = pose_list[0]
    for pose in pose_list:
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose

    last_pose = dict(last_pose)
    last_pose.pop("utime", None)

    pos = last_pose.pop("pos", [0.0, 0.0, 0.0])
    orientation = last_pose.pop("orientation", [1.0, 0.0, 0.0, 0.0])

    can_bus = []
    can_bus.extend(pos)
    can_bus.extend(orientation)
    for key in last_pose.keys():
        can_bus.extend(last_pose[key])
    can_bus.extend([0.0, 0.0])

    return np.asarray(can_bus, dtype=np.float64)


def _obtain_sensor2top(
    nusc,
    sensor_token: str,
    l2e_t: np.ndarray,
    l2e_r_mat: np.ndarray,
    e2g_t: np.ndarray,
    e2g_r_mat: np.ndarray,
    sensor_type: str,
) -> Dict:
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = _normalize_path(
        nusc.get_sample_data_path(sd_rec["token"]),
        getattr(nusc, "dataroot", None),
    )

    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s = Quaternion(sweep["sensor2ego_rotation"]).rotation_matrix
    l2e_t_s = np.asarray(sweep["sensor2ego_translation"], dtype=np.float64)
    e2g_r_s = Quaternion(sweep["ego2global_rotation"]).rotation_matrix
    e2g_t_s = np.asarray(sweep["ego2global_translation"], dtype=np.float64)

    R = (l2e_r_s.T @ e2g_r_s.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= l2e_t @ np.linalg.inv(l2e_r_mat).T

    sweep["sensor2lidar_rotation"] = R.T.astype(np.float32)
    sweep["sensor2lidar_translation"] = T.astype(np.float32)
    return sweep


def _is_original_2hz_key_sample_token(sample_token: str, original_token_len: int) -> bool:
    return len(str(sample_token)) == int(original_token_len)


def _canonicalize_sample_by_token_len(nusc, sample: Dict, original_token_len: int) -> Dict:
    sample_token = str(sample.get("token", ""))
    if _is_original_2hz_key_sample_token(sample_token, original_token_len):
        return nusc.get("sample", sample_token)
    return sample


def _get_instance_key_ann_timeline(
    nusc,
    instance_token: str,
    instance_key_ann_cache: Dict[str, List[Dict]],
    original_token_len: int,
) -> List[Dict]:
    if instance_token in instance_key_ann_cache:
        return instance_key_ann_cache[instance_token]

    instance = nusc.get("instance", instance_token)
    ann_token = instance["first_annotation_token"]

    timeline = []
    visited = set()
    while ann_token:
        if ann_token in visited:
            break
        visited.add(ann_token)

        ann = nusc.get("sample_annotation", ann_token)
        sample_token = ann["sample_token"]
        if _is_original_2hz_key_sample_token(sample_token, original_token_len):
            sample = nusc.get("sample", sample_token)
            timeline.append(
                {
                    "ann_token": ann["token"],
                    "sample_token": sample_token,
                    "timestamp": int(sample["timestamp"]),
                    "translation": np.asarray(ann["translation"], dtype=np.float32),
                    "rotation": ann["rotation"],
                }
            )

        ann_token = ann["next"]

    timeline.sort(key=lambda x: x["timestamp"])
    instance_key_ann_cache[instance_token] = timeline
    return timeline


def _interpolate_global_xy_from_timeline(
    timeline: List[Dict],
    target_timestamp: int,
) -> Optional[np.ndarray]:
    if not timeline:
        return None

    timestamps = [item["timestamp"] for item in timeline]
    pos = bisect_left(timestamps, target_timestamp)

    if pos < len(timestamps) and timestamps[pos] == target_timestamp:
        return timeline[pos]["translation"][:2].astype(np.float32)

    if pos == 0 or pos == len(timestamps):
        return None

    left = timeline[pos - 1]
    right = timeline[pos]
    t0 = left["timestamp"]
    t1 = right["timestamp"]

    if t1 <= t0:
        return left["translation"][:2].astype(np.float32)

    alpha = float(target_timestamp - t0) / float(t1 - t0)
    xy0 = left["translation"][:2]
    xy1 = right["translation"][:2]
    xy = (1.0 - alpha) * xy0 + alpha * xy1
    return xy.astype(np.float32)


def _global_xy_to_current_agent_local(
    global_xy: np.ndarray,
    current_translation: List[float],
    current_rotation: List[float],
) -> np.ndarray:
    coords = np.asarray(global_xy, dtype=np.float32).reshape(-1, 2)
    local_xy = convert_global_coords_to_local(
        coords,
        current_translation,
        current_rotation,
    )
    return np.asarray(local_xy, dtype=np.float32)


def _get_future_traj_from_key_timeline_exact(
    nusc,
    ann: Dict,
    predict_steps: int,
    instance_key_ann_cache: Dict[str, List[Dict]],
    original_token_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    traj = np.zeros((predict_steps, 2), dtype=np.float32)
    mask = np.zeros((predict_steps, 2), dtype=np.float32)

    instance_token = ann["instance_token"]
    current_sample_token = ann["sample_token"]
    current_translation = ann["translation"]
    current_rotation = ann["rotation"]

    timeline = _get_instance_key_ann_timeline(
        nusc=nusc,
        instance_token=instance_token,
        instance_key_ann_cache=instance_key_ann_cache,
        original_token_len=original_token_len,
    )
    if not timeline:
        return traj, mask

    current_idx = -1
    for idx, item in enumerate(timeline):
        if item["sample_token"] == current_sample_token:
            current_idx = idx
            break

    if current_idx < 0:
        return traj, mask

    future_items = timeline[current_idx + 1: current_idx + 1 + predict_steps]
    if len(future_items) == 0:
        return traj, mask

    future_global_xy = np.asarray(
        [item["translation"][:2] for item in future_items],
        dtype=np.float32,
    )
    future_local_xy = _global_xy_to_current_agent_local(
        global_xy=future_global_xy,
        current_translation=current_translation,
        current_rotation=current_rotation,
    )

    valid_len = min(predict_steps, future_local_xy.shape[0])
    traj[:valid_len, :] = future_local_xy[:valid_len, :]
    mask[:valid_len, :] = 1.0
    return traj, mask


def _get_future_traj_from_interpolated_key_timeline(
    nusc,
    ann: Dict,
    sample_timestamp: int,
    predict_steps: int,
    future_step_time: float,
    instance_key_ann_cache: Dict[str, List[Dict]],
    original_token_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    traj = np.zeros((predict_steps, 2), dtype=np.float32)
    mask = np.zeros((predict_steps, 2), dtype=np.float32)

    instance_token = ann["instance_token"]
    current_translation = ann["translation"]
    current_rotation = ann["rotation"]

    timeline = _get_instance_key_ann_timeline(
        nusc=nusc,
        instance_token=instance_token,
        instance_key_ann_cache=instance_key_ann_cache,
        original_token_len=original_token_len,
    )
    if not timeline:
        return traj, mask

    for step_idx in range(predict_steps):
        target_timestamp = sample_timestamp + int(round((step_idx + 1) * future_step_time * 1e6))
        global_xy = _interpolate_global_xy_from_timeline(
            timeline=timeline,
            target_timestamp=target_timestamp,
        )
        if global_xy is None:
            break

        local_xy = _global_xy_to_current_agent_local(
            global_xy=global_xy,
            current_translation=current_translation,
            current_rotation=current_rotation,
        )[0]

        traj[step_idx] = local_xy
        mask[step_idx] = 1.0

    return traj, mask


def _get_future_traj_info(
    nusc,
    sample: Dict,
    predict_steps: int,
    future_step_time: float,
    instance_key_ann_cache: Dict[str, List[Dict]],
    original_token_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sample = _canonicalize_sample_by_token_len(
        nusc=nusc,
        sample=sample,
        original_token_len=original_token_len,
    )

    ann_tokens = list(sample.get("anns", []))
    if len(ann_tokens) == 0:
        return (
            np.zeros((0, predict_steps, 2), dtype=np.float32),
            np.zeros((0, predict_steps, 2), dtype=np.float32),
        )

    sample_token = sample["token"]
    sample_timestamp = int(sample["timestamp"])

    fut_traj_all = []
    fut_traj_valid_mask_all = []
    for ann_token in ann_tokens:
        ann = nusc.get("sample_annotation", ann_token)

        if _is_original_2hz_key_sample_token(sample_token, original_token_len):
            traj, mask = _get_future_traj_from_key_timeline_exact(
                nusc=nusc,
                ann=ann,
                predict_steps=predict_steps,
                instance_key_ann_cache=instance_key_ann_cache,
                original_token_len=original_token_len,
            )
        else:
            traj, mask = _get_future_traj_from_interpolated_key_timeline(
                nusc=nusc,
                ann=ann,
                sample_timestamp=sample_timestamp,
                predict_steps=predict_steps,
                future_step_time=future_step_time,
                instance_key_ann_cache=instance_key_ann_cache,
                original_token_len=original_token_len,
            )

        fut_traj_all.append(traj)
        fut_traj_valid_mask_all.append(mask)

    return np.stack(fut_traj_all, axis=0), np.stack(fut_traj_valid_mask_all, axis=0)


def build_uniad_info_for_segment(
    dataset,
    segment_frame_mapping: List[Dict],
    can_bus_root_path: Optional[str] = None,
    max_sweeps: int = 0,
    predict_steps: int = 16,
    future_step_time: float = 0.5,
    original_token_len: int = 32,
) -> Dict:
    if not segment_frame_mapping:
        raise ValueError("segment_frame_mapping is empty.")

    nusc = dataset.nusc
    instance_index_map = _build_instance_index_map(nusc)
    instance_key_ann_cache: Dict[str, List[Dict]] = {}

    nusc_can_bus = None
    if can_bus_root_path and NuScenesCanBus is not None:
        try:
            nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
        except Exception:
            nusc_can_bus = None

    infos = []
    num_segment_frames = len(segment_frame_mapping)

    for frame_pos, frame_item in enumerate(segment_frame_mapping):
        sample_token = frame_item["sample_token"]
        sample = dataset._get_sample(sample_token)
        sample = _canonicalize_sample_by_token_len(
            nusc=nusc,
            sample=sample,
            original_token_len=original_token_len,
        )

        seq_prev_token = ""
        seq_next_token = ""
        if frame_pos > 0:
            seq_prev_token = segment_frame_mapping[frame_pos - 1]["sample_token"]
        if frame_pos < num_segment_frames - 1:
            seq_next_token = segment_frame_mapping[frame_pos + 1]["sample_token"]

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = dataset._get_sample_data(lidar_token)
        cs_record = dataset._get_calibrated_sensor(lidar_sd["calibrated_sensor_token"])
        pose_record = dataset._get_ego_pose(lidar_sd["ego_pose_token"])

        lidar_path = _normalize_path(
            nusc.get_sample_data_path(lidar_token),
            getattr(nusc, "dataroot", None),
        )

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "prev": seq_prev_token,
            "next": seq_next_token,
            "nusc_prev": sample["prev"],
            "nusc_next": sample["next"],
            "can_bus": _get_can_bus_info(nusc, nusc_can_bus, sample),
            "frame_idx": int(frame_item["frame_idx"]),
            "sweeps": [],
            "cams": {},
            "scene_token": sample["scene_token"],
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": int(sample["timestamp"]),
        }

        l2e_t = np.asarray(cs_record["translation"], dtype=np.float64)
        l2e_r_mat = Quaternion(cs_record["rotation"]).rotation_matrix
        e2g_t = np.asarray(pose_record["translation"], dtype=np.float64)
        e2g_r_mat = Quaternion(pose_record["rotation"]).rotation_matrix

        for cam in dataset.camera_names:
            cam_token = sample["data"].get(cam)
            if cam_token is None:
                continue

            cam_info = _obtain_sensor2top(
                nusc=nusc,
                sensor_token=cam_token,
                l2e_t=l2e_t,
                l2e_r_mat=l2e_r_mat,
                e2g_t=e2g_t,
                e2g_r_mat=e2g_r_mat,
                sensor_type=cam,
            )
            cam_sd = dataset._get_sample_data(cam_token)
            cam_cs = dataset._get_calibrated_sensor(cam_sd["calibrated_sensor_token"])
            cam_info["cam_intrinsic"] = np.asarray(cam_cs.get("camera_intrinsic"))
            info["cams"][cam] = cam_info

        if max_sweeps > 0:
            sweeps = []
            cur_sd = lidar_sd
            while len(sweeps) < max_sweeps and cur_sd["prev"]:
                prev_token = cur_sd["prev"]
                sweep = _obtain_sensor2top(
                    nusc=nusc,
                    sensor_token=prev_token,
                    l2e_t=l2e_t,
                    l2e_r_mat=l2e_r_mat,
                    e2g_t=e2g_t,
                    e2g_r_mat=e2g_r_mat,
                    sensor_type="lidar",
                )
                sweeps.append(sweep)
                cur_sd = dataset._get_sample_data(prev_token)
            info["sweeps"] = sweeps

        ann_tokens = list(sample.get("anns", []))
        annotations = [nusc.get("sample_annotation", tok) for tok in ann_tokens]

        if len(ann_tokens) == 0:
            info["gt_boxes"] = np.zeros((0, 7), dtype=np.float32)
            info["gt_names"] = np.zeros((0,), dtype="<U1")
            info["gt_velocity"] = np.zeros((0, 2), dtype=np.float32)
            info["num_lidar_pts"] = np.zeros((0,), dtype=np.int64)
            info["num_radar_pts"] = np.zeros((0,), dtype=np.int64)
            info["valid_flag"] = np.zeros((0,), dtype=bool)
            info["gt_inds"] = np.zeros((0,), dtype=np.int64)
            info["gt_ins_tokens"] = np.zeros((0,), dtype="<U1")
            info["fut_traj"] = np.zeros((0, predict_steps, 2), dtype=np.float32)
            info["fut_traj_valid_mask"] = np.zeros((0, predict_steps, 2), dtype=np.float32)
            info["visibility_tokens"] = np.zeros((0,), dtype=np.int64)
            infos.append(info)
            continue

        _, boxes, _ = nusc.get_sample_data(lidar_token, selected_anntokens=ann_tokens)
        locs = np.asarray([b.center for b in boxes], dtype=np.float32).reshape(-1, 3)
        dims = np.asarray([b.wlh for b in boxes], dtype=np.float32).reshape(-1, 3)
        rots = np.asarray(
            [b.orientation.yaw_pitch_roll[0] for b in boxes],
            dtype=np.float32,
        ).reshape(-1, 1)

        velocity = np.asarray(
            [_safe_box_velocity(nusc, tok) for tok in ann_tokens],
            dtype=np.float32,
        )
        for idx in range(len(velocity)):
            velo = np.array([velocity[idx, 0], velocity[idx, 1], 0.0], dtype=np.float32)
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[idx] = velo[:2]

        gt_names = [NAME_MAPPING.get(b.name, b.name) for b in boxes]
        valid_flag = np.asarray(
            [(a["num_lidar_pts"] + a["num_radar_pts"]) > 0 for a in annotations],
            dtype=bool,
        )
        gt_inds = np.asarray(
            [instance_index_map[a["instance_token"]] for a in annotations],
            dtype=np.int64,
        )
        gt_ins_tokens = np.asarray([a["instance_token"] for a in annotations])
        visibility_tokens = np.asarray(
            [int(a["visibility_token"]) for a in annotations],
            dtype=np.int64,
        )

        fut_traj, fut_traj_valid_mask = _get_future_traj_info(
            nusc=nusc,
            sample=sample,
            predict_steps=predict_steps,
            future_step_time=future_step_time,
            instance_key_ann_cache=instance_key_ann_cache,
            original_token_len=original_token_len,
        )

        info["gt_boxes"] = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1).astype(np.float32)
        info["gt_names"] = np.asarray(gt_names)
        info["gt_velocity"] = velocity.astype(np.float32)
        info["num_lidar_pts"] = np.asarray([a["num_lidar_pts"] for a in annotations], dtype=np.int64)
        info["num_radar_pts"] = np.asarray([a["num_radar_pts"] for a in annotations], dtype=np.int64)
        info["valid_flag"] = valid_flag
        info["gt_inds"] = gt_inds
        info["gt_ins_tokens"] = gt_ins_tokens
        info["fut_traj"] = fut_traj.astype(np.float32)
        info["fut_traj_valid_mask"] = fut_traj_valid_mask.astype(np.float32)
        info["visibility_tokens"] = visibility_tokens
        infos.append(info)

    return {
        "infos": infos,
        "metadata": {
            "version": dataset.version,
            "source": "video_segment",
            "future_step_time": future_step_time,
            "original_token_len": original_token_len,
        },
    }


def dump_uniad_info(data: Dict, out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(data, f)
