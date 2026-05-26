from __future__ import annotations

import os
import json
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


# ===============================
# 路径配置
# ===============================
DATAROOT = "<path-to-nuscenes-root>"
VERSION = "interp_12Hz_trainval"

SAVE_DIR = "./data_tools/nuscenes_val_sort"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_JSON_PATH = os.path.join(SAVE_DIR, "classified_val_windows.json")


# ===============================
# 固定 window（12Hz, 7s, 不重叠）
# 84 帧 ≈ 7 秒
# ===============================
WINDOW_SIZE = 84
STEP_SIZE = 84


# ===============================
# 分类阈值
# ===============================
# 怠速放宽
IDLE_DISP_THR = 21.0
IDLE_VMEAN_THR = 3.0

# 转向阈值
TURN_CUMYAW = 15.0

# 车辆数阈值（用于细分 aggressive）
VEHICLE_COUNT_THR = 14

# nuScenes 车辆类别前缀
VEHICLE_CATEGORY_PREFIX = "vehicle."


# ===============================
# 工具函数
# ===============================
def is_vehicle_category(category_name: str) -> bool:
    """判断是否是车辆类别"""
    return category_name.startswith(VEHICLE_CATEGORY_PREFIX)


def quat_to_yaw(q: List[float]) -> float:
    w, x, y, z = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def angle_diff(a: float, b: float) -> float:
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def cumulative_abs_yaw(yaws: List[float]) -> float:
    total = 0.0
    for i in range(len(yaws) - 1):
        total += abs(angle_diff(yaws[i + 1], yaws[i]))
    return math.degrees(total)


def compute_lateral_shift(poses: List[Tuple[float, float]], yaws: List[float]) -> float:
    p0 = np.array(poses[0], dtype=np.float64)
    p1 = np.array(poses[-1], dtype=np.float64)
    theta0 = yaws[0]

    delta = p1 - p0
    lateral_vec = np.array([-np.sin(theta0), np.cos(theta0)], dtype=np.float64)
    lateral_shift = np.dot(delta, lateral_vec)

    return float(abs(lateral_shift))


def collect_scene_samples(nusc: NuScenes, scene: Dict) -> List[Dict]:
    samples: List[Dict] = []
    token = scene["first_sample_token"]

    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]

    samples.sort(key=lambda x: x["timestamp"])
    return samples


def compute_window_metrics(nusc: NuScenes, window_samples: List[Dict]) -> Dict:
    poses: List[Tuple[float, float]] = []
    yaws: List[float] = []
    timestamps: List[int] = []
    sample_tokens: List[str] = []
    vehicle_counts: List[int] = []

    for sample in window_samples:
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        x, y, _ = ego["translation"]
        yaw = quat_to_yaw(ego["rotation"])

        poses.append((float(x), float(y)))
        yaws.append(float(yaw))
        timestamps.append(int(sample["timestamp"]))
        sample_tokens.append(sample["token"])

        # 统计车辆数（过滤 category_name 以 "vehicle." 开头的标注）
        vehicle_count = 0
        for ann_token in sample.get("anns", []):
            ann = nusc.get("sample_annotation", ann_token)
            if is_vehicle_category(ann["category_name"]):
                vehicle_count += 1
        vehicle_counts.append(vehicle_count)

    x0, y0 = poses[0]
    x1, y1 = poses[-1]

    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)

    lateral_shift = compute_lateral_shift(poses, yaws)

    v_mag: List[float] = []
    v_long: List[float] = []

    start_yaw = yaws[0]

    for j in range(len(poses) - 1):
        dt = (timestamps[j + 1] - timestamps[j]) / 1e6
        if dt <= 0:
            continue

        xa, ya = poses[j]
        xb, yb = poses[j + 1]

        vx = (xb - xa) / dt
        vy = (yb - ya) / dt

        v_mag.append(math.hypot(vx, vy))
        v_long.append(vx * math.cos(start_yaw) + vy * math.sin(start_yaw))

    if len(v_mag) == 0:
        return {
            "valid": False,
            "sample_tokens": sample_tokens,
            "timestamps": timestamps,
        }

    acc_long: List[float] = []

    for j in range(len(v_long) - 1):
        dt = (timestamps[j + 1] - timestamps[j]) / 1e6
        if dt <= 0:
            continue

        acc_long.append((v_long[j + 1] - v_long[j]) / dt)

    v_mean = float(np.mean(v_mag))
    max_acc = float(max(acc_long)) if acc_long else 0.0
    max_dec = float(min(acc_long)) if acc_long else 0.0
    cum_yaw = float(cumulative_abs_yaw(yaws))

    return {
        "valid": True,
        "sample_tokens": sample_tokens,
        "timestamps": timestamps,
        "distance_m": float(dist),
        "v_mean_m_s": float(v_mean),
        "max_acc_m_s2": float(max_acc),
        "max_dec_m_s2": float(max_dec),
        "cum_yaw_deg": float(cum_yaw),
        "lateral_shift_m": float(lateral_shift),
        "mean_vehicle_count": float(np.mean(vehicle_counts)),
    }


def classify_window(metrics: Dict) -> str:
    dist = metrics["distance_m"]
    v_mean = metrics["v_mean_m_s"]
    cum_yaw = metrics["cum_yaw_deg"]
    mean_vehicle_count = metrics["mean_vehicle_count"]

    # 只要有明显转向，就算 turning
    if cum_yaw >= TURN_CUMYAW:
        return "turning"

    # 不转向，并且整体运动较小，算 idle
    if dist <= IDLE_DISP_THR and v_mean <= IDLE_VMEAN_THR:
        return "idle"

    # 剩下根据车辆数细分 aggressive
    if mean_vehicle_count >= VEHICLE_COUNT_THR:
        return "aggressive_dense"  # 车多
    else:
        return "aggressive_sparse"  # 车少


def build_clip_name(scene_name: str, start_idx: int) -> str:
    return f"{scene_name}__{start_idx:06d}"


def save_json(data, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ===============================
# 主程序
# ===============================
print("Loading nuScenes...")
nusc = NuScenes(
    version=VERSION,
    dataroot=DATAROOT,
    verbose=False,
)

val_scene_names = set(create_splits_scenes()["val"])
val_scenes = [scene for scene in nusc.scene if scene["name"] in val_scene_names]

print("Version:", VERSION)
print("Val scenes:", len(val_scenes))
print("WINDOW_SIZE:", WINDOW_SIZE)
print("STEP_SIZE:", STEP_SIZE)

classified_windows: List[Dict] = []

for scene in tqdm(val_scenes, desc="Processing val scenes"):
    samples = collect_scene_samples(nusc, scene)

    if len(samples) < WINDOW_SIZE:
        continue

    for start_idx in range(0, len(samples) - WINDOW_SIZE + 1, STEP_SIZE):
        window_samples = samples[start_idx:start_idx + WINDOW_SIZE]
        metrics = compute_window_metrics(nusc, window_samples)

        if not metrics["valid"]:
            continue

        clip_name = build_clip_name(scene["name"], start_idx)

        window_meta = {
            "clip_name": clip_name,
            "scene_name": scene["name"],
            "window_start_index": int(start_idx),
            "window_size": int(WINDOW_SIZE),
            "step_size": int(STEP_SIZE),
            "num_frames": int(len(window_samples)),
            "start_sample_token": window_samples[0]["token"],
            "end_sample_token": window_samples[-1]["token"],
            "sample_tokens": metrics["sample_tokens"],
            "timestamps": metrics["timestamps"],
            "metrics": {
                "distance_m": metrics["distance_m"],
                "v_mean_m_s": metrics["v_mean_m_s"],
                "max_acc_m_s2": metrics["max_acc_m_s2"],
                "max_dec_m_s2": metrics["max_dec_m_s2"],
                "cum_yaw_deg": metrics["cum_yaw_deg"],
                "lateral_shift_m": metrics["lateral_shift_m"],
                "mean_vehicle_count": metrics["mean_vehicle_count"],
            },
            "label": classify_window(metrics),
        }

        classified_windows.append(window_meta)

label_counter = Counter([item["label"] for item in classified_windows])

print("Total windows generated:", len(classified_windows))
print("Label distribution:", label_counter)

save_json(classified_windows, SAVE_JSON_PATH)

print("Saved to:", SAVE_JSON_PATH)