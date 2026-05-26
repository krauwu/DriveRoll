"""
主程序
"""


import os
import sys
import json
import yaml
import torch
import traceback
from pathlib import Path
from omegaconf import OmegaConf
# ============================================================
# [MOVE HERE] 路径加载：提前到所有自定义import之前
# ============================================================
if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ["CUDA_VISIBLE_DEVICES"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_path_env = os.environ.get("NUPLAN_SIM_CONFIG_PATHS", "").strip()
if config_path_env:
    config_path = Path(config_path_env).expanduser().resolve()
else:
    config_path = Path(__file__).parent / "config_paths.yaml"

if not config_path.exists():
    raise FileNotFoundError(f"配置文件不存在: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    path_config = yaml.safe_load(f)

# ---- env nuplan ----
os.environ["NUPLAN_DATA_ROOT"] = path_config["nuplan_data_root"]
os.environ["NUPLAN_MAPS_ROOT"] = path_config["nuplan_maps_root"]
if not os.environ.get("NUPLAN_DB_FILES"):
    os.environ["NUPLAN_DB_FILES"] = path_config["nuplan_db_files"]
os.environ["NUPLAN_MAP_VERSION"] = path_config["nuplan_map_version"]
os.environ["NUPLAN_EXP_ROOT"] = path_config["nuplan_exp_root"]
os.environ["BLOB_PATH"] = path_config["blob_path"]
os.environ["NUPLAN_DATA_STORE"] = path_config.get("nuplan_data_store", "")

# ---- dwm_path：加到sys.path（让你的 prepare_cond / dwm 包可import）----
dwm_path = path_config.get("dwm_path", "")
if dwm_path and dwm_path not in sys.path:
    sys.path.insert(0, dwm_path)

# ---- gen_cfg：从 json 读----
gen_cfg = None
gen_cfg_path = path_config.get("gen_cfg", "")
if gen_cfg_path:
    with open(gen_cfg_path, "r", encoding="utf-8") as f:
        gen_cfg = json.load(f)

import numpy as np
from typing import List
import logging
import av
import time
import ray
import cv2
from dataclasses import dataclass
from PIL import Image as PILImage

# ===== [FIX] Hydra =====
from hydra import compose
from hydra import initialize_config_dir
from hydra.utils import instantiate

# ===== 可视化 =====
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#  tool files
from info_tool_DB import RollingInfoBuffer5Hz, CAM_TYPES
from prepare_cond_gpu import RollingCondPackCache
import dwm.common
from collections import deque
from copy import deepcopy
from generator import StreamingARClient

from common import (
    rotate_round_z_axis,
    get_transmat_for_lidarpc_token_from_db,
    global_trajectory_to_states,
)

# ===== style patch =====
if "seaborn-v0_8-whitegrid" not in plt.style.available:
    available_seaborn_styles = [s for s in plt.style.available if "seaborn" in s and "whitegrid" in s]
    if available_seaborn_styles:
        plt.style.use(available_seaborn_styles[0])
    else:
        plt.style.use("default")

    original_use = plt.style.use

    def patched_use(style):
        if style == "seaborn-v0_8-whitegrid":
            if available_seaborn_styles:
                style = available_seaborn_styles[0]
            else:
                style = "default"
        return original_use(style)

    plt.style.use = patched_use


# ================= NuPlan imports =================
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.observation_builder import build_observations
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.planner_builder import build_planners

from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.callback.multi_callback import MultiCallback

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInput
from nuplan.planning.simulation.runner.runner_report import RunnerReport

from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db

from nuplan.planning.simulation.observation.observation_type import CameraChannel
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


logger = logging.getLogger(__name__)


# ==========================
# 轨迹：预定义（保留你原逻辑）
# ==========================
def smoothstep01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    return u * u * (3.0 - 2.0 * u)


def _get_global_trajectory(local_trajectory: np.ndarray, ego_state):
    """局部轨迹转全局轨迹"""
    origin = ego_state.rear_axle.array
    angle = ego_state.rear_axle.heading

    global_position = (
        rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
        + origin
    )
    global_vec = global_position[1:] - global_position[:-1]
    global_heading = np.arctan2(global_vec[..., 1], global_vec[..., 0])
    global_heading = np.concatenate([np.array([angle]), global_heading])

    global_trajectory = np.concatenate([global_position, global_heading[..., None]], axis=1)
    return global_trajectory
    
def get_traj(ego_state, ego_state_buffer, trajectory_type="right_then_back"):
    """
    生成 future-only 2.2s 轨迹：22个点（10Hz）
    需求：先直行 -> 向右偏一点 -> 再回到车道中心（y回到0）

    速度策略（按你的要求）：
    - v0 = max(current_speed, 3.0)
    - v_max = max(current_speed, 6.0)
    - 继续匀加速 a（保持你原来的 0.6）
    """
    dt = 0.1
    num_future = 22
    horizon_s = num_future * dt  # 2.2s

    # =========================
    # speed profile (updated)
    # =========================
    vel2d = ego_state.dynamic_car_state.rear_axle_velocity_2d.array
    current_speed = float(np.linalg.norm(vel2d))

    v0 = float(max(current_speed, 3.0))
    v_max = float(max(current_speed, 6.0))
    a = 0.6

    t = (np.arange(1, num_future + 1) * dt).astype(np.float32)  # [0.1 ... 2.2]
    s = v0 * t + 0.5 * a * (t ** 2)

    v_t = v0 + a * t
    over = v_t > v_max
    if np.any(over):
        t0 = (v_max - v0) / max(a, 1e-6)
        t0 = float(max(t0, 0.0))
        s0 = v0 * t0 + 0.5 * a * t0 * t0
        s[over] = s0 + v_max * (t[over] - t0)

    x = s.astype(np.float32)

    # =========================
    # lateral profile (keep yours)
    # =========================
    if trajectory_type == "straight":
        y = np.zeros_like(x)

    elif trajectory_type == "right_then_back":
        straight_plan_steps = 6
        y1_max = 0.06

        plan_idx = getattr(ego_state, "_plan_idx_after_warmup", None)

        y1 = 0.0
        if plan_idx is None:
            y1 = -y1_max
        else:
            if plan_idx < straight_plan_steps:
                y1 = 0.0
            else:
                ramp = float(np.clip((plan_idx - straight_plan_steps) / 10.0, 0.0, 1.0))
                y1 = -y1_max * ramp

        x1 = float(max(x[0], 1e-3))
        y = (y1 * (x / x1)).astype(np.float32)

    elif trajectory_type == "cosine":
        lateral_offset = -1.5
        y = lateral_offset * 0.5 * (1 - np.cos(np.pi * t / horizon_s))

    else:
        raise ValueError(f"不支持的轨迹类型: {trajectory_type}")

    local_traj = np.stack([x, y], axis=1)
    global_traj = _get_global_trajectory(local_traj, ego_state)

    traj = InterpolatedTrajectory(
        trajectory=global_trajectory_to_states(
            global_trajectory=global_traj,
            ego_history=ego_state_buffer,
            future_horizon=horizon_s,
            step_interval=dt,
        )
    )
    return traj

# ==========================
# 可视化：BEV + Camera
# ==========================
def _overlay_add_clip(base_rgb, top_img):
    """按像素相加"""
    a = np.asarray(base_rgb.convert("RGB"), dtype=np.uint16)
    b = np.asarray(top_img.convert("RGB"), dtype=np.uint16)
    out = np.minimum(a + b, 255).astype(np.uint8)
    return PILImage.fromarray(out, mode="RGB")

def render_overlay_conditions_8cam(bbox_frames, hdmap_frames, resize_to=(480, 270), t_idx=None):
    """
    输出 2x4 的 cond overlay（BGR），尺寸: (2*h, 4*w, 3)
    bbox_frames/hdmap_frames: [T][V] PIL.Image 或 [V] PIL.Image
    """
    if bbox_frames and isinstance(bbox_frames[0], PILImage.Image):
        bbox_frames = [bbox_frames]
        hdmap_frames = [hdmap_frames]

    T = len(bbox_frames)
    V = len(bbox_frames[0]) if T > 0 else 0
    if T == 0 or V < 8:
        h = resize_to[1] * 2
        w = resize_to[0] * 4
        return np.zeros((h, w, 3), dtype=np.uint8)

    if t_idx is None:
        t_idx = T - 1
    t_idx = int(np.clip(t_idx, 0, T - 1))

    w0, h0 = resize_to

    merged_bgr = []
    for c in range(8):
        base = hdmap_frames[t_idx][c].convert("RGB")
        top = bbox_frames[t_idx][c].convert("RGB")

        base = base.resize((w0, h0), PILImage.BILINEAR)
        top = top.resize((w0, h0), PILImage.BILINEAR)

        merged = _overlay_add_clip(base, top)
        merged_bgr.append(cv2.cvtColor(np.array(merged, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    row1 = np.hstack([merged_bgr[1], merged_bgr[2], merged_bgr[3], merged_bgr[4]])
    row2 = np.hstack([merged_bgr[0], merged_bgr[5], merged_bgr[6], merged_bgr[7]])
    return np.vstack([row1, row2])

def center_crop(img_bgr, crop_w, crop_h):
    h, w = img_bgr.shape[:2]
    crop_w = int(min(crop_w, w))
    crop_h = int(min(crop_h, h))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return img_bgr[y0:y0 + crop_h, x0:x0 + crop_w]

def render_scene_frame(
    ego_state,
    ego_trajectory_log,
    tracked_objects,
    step_count,
    scenario_name,
    planned_trajectory=None,
    map_api=None,
    view_range=50.0,
):
    """渲染BEV视角场景（带地图可选）"""
    fig, ax = plt.subplots(figsize=(16, 12))

    ego_x, ego_y = ego_state.center.x, ego_state.center.y

    # ===== 渲染HD地图（可选）=====
    if map_api is not None:
        try:
            ego_position = ego_state.center.point

            # 1) LANE
            try:
                lanes = map_api.get_proximal_map_objects(
                    ego_position, view_range, [SemanticMapLayer.LANE]
                )
                lane_objects = lanes.get(SemanticMapLayer.LANE, [])
                for lane in lane_objects:
                    if hasattr(lane, "baseline_path"):
                        baseline = lane.baseline_path.discrete_path
                        coords = np.array([[p.x, p.y] for p in baseline])
                        if len(coords) > 1:
                            ax.plot(coords[:, 0], coords[:, 1], "gray", linewidth=1.5, alpha=0.4, zorder=1)
            except Exception as e:
                print(f"  警告: 渲染车道失败: {e}")

            # 2) LANE_CONNECTOR
            try:
                lane_connectors = map_api.get_proximal_map_objects(
                    ego_position, view_range, [SemanticMapLayer.LANE_CONNECTOR]
                )
                connector_objects = lane_connectors.get(SemanticMapLayer.LANE_CONNECTOR, [])
                for connector in connector_objects:
                    if hasattr(connector, "baseline_path"):
                        baseline = connector.baseline_path.discrete_path
                        coords = np.array([[p.x, p.y] for p in baseline])
                        if len(coords) > 1:
                            ax.plot(coords[:, 0], coords[:, 1], "lightblue", linewidth=1.5, alpha=0.5, linestyle="--", zorder=1)
            except Exception as e:
                print(f"  警告: 渲染车道连接器失败: {e}")

        except Exception as e:
            print(f"  警告: 渲染地图元素失败: {e}")

    # ===== ego历史轨迹 =====
    if len(ego_trajectory_log) > 1:
        traj_array = np.array(ego_trajectory_log)
        ax.plot(traj_array[:, 0], traj_array[:, 1], "b-", linewidth=3, alpha=0.7, zorder=5)

    # ===== planned轨迹 =====
    if planned_trajectory is not None:
        try:
            states = planned_trajectory.get_sampled_trajectory()
            if len(states) > 0:
                planned_points = np.array([[s.rear_axle.x, s.rear_axle.y] for s in states])
                ax.plot(planned_points[:, 0], planned_points[:, 1], "g--", linewidth=2, alpha=0.8, zorder=5)
        except Exception:
            pass

    # ===== ego车体 =====
    ego_heading = ego_state.center.heading
    ego_length, ego_width = 5.0, 2.0
    ego_rect = Rectangle(
        (ego_x - ego_length / 2 * np.cos(ego_heading) + ego_width / 2 * np.sin(ego_heading),
         ego_y - ego_length / 2 * np.sin(ego_heading) - ego_width / 2 * np.cos(ego_heading)),
        ego_length,
        ego_width,
        angle=np.degrees(ego_heading),
        facecolor="blue",
        edgecolor="darkblue",
        linewidth=2,
        alpha=0.8,
        zorder=10,
    )
    ax.add_patch(ego_rect)

    # ===== 其他交通参与者 =====
    if tracked_objects is not None:
        for obj in tracked_objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                color, edge_color = "red", "darkred"
            elif obj.tracked_object_type == TrackedObjectType.PEDESTRIAN:
                color, edge_color = "orange", "darkorange"
            else:
                color, edge_color = "gray", "darkgray"

            obj_x, obj_y = obj.center.x, obj.center.y
            obj_heading = obj.center.heading
            obj_length = obj.box.length if obj.box.length > 0.1 else 4.5
            obj_width = obj.box.width if obj.box.width > 0.1 else 2.0

            obj_rect = Rectangle(
                (obj_x - obj_length / 2 * np.cos(obj_heading) + obj_width / 2 * np.sin(obj_heading),
                 obj_y - obj_length / 2 * np.sin(obj_heading) - obj_width / 2 * np.cos(obj_heading)),
                obj_length,
                obj_width,
                angle=np.degrees(obj_heading),
                facecolor=color,
                edgecolor=edge_color,
                linewidth=1.5,
                alpha=0.6,
                zorder=8,
            )
            ax.add_patch(obj_rect)

    ax.set_xlim(ego_x - view_range, ego_x + view_range)
    ax.set_ylim(ego_y - view_range, ego_y + view_range)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")

    velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.array
    speed = np.linalg.norm(velocity)
    ax.set_title(f"{scenario_name} | step={step_count} | speed={speed:.2f}m/s", fontsize=14)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    w, h = canvas.get_width_height()
    img_rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img_rgb = img_rgba[:, :, :3]  # 丢掉 alpha

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img_bgr

def render_camera_views_stage3(sensor_data, camera_channels):
    """渲染8相机拼图"""
    try:
        camera_images = [np.array(img) for img in sensor_data]
        target_size = (480, 270)
        resized_images = []
        for img in camera_images:
            img_resized = cv2.resize(img, target_size)
            if img_resized.ndim == 3 and img_resized.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_resized
            resized_images.append(img_bgr)

        row1 = np.hstack([resized_images[1], resized_images[2], resized_images[3], resized_images[4]])
        row2 = np.hstack([resized_images[0], resized_images[5], resized_images[6], resized_images[7]])
        combined = np.vstack([row1, row2])
        return combined

    except Exception as e:
        print(f"警告: 渲染相机视图失败: {e}")
        return np.zeros((540, 1920, 3), dtype=np.uint8)
def render_combined_view(bev_frame, camera_frame, cur_cond8, step_count, scenario_name, planner_type="Predefined"):
    title_height = 80

    # ===== 1) 拆 2x4 -> 两行 =====
    cam_h, cam_w = camera_frame.shape[:2]
    cam_top = camera_frame[:cam_h // 2]
    cam_bot = camera_frame[cam_h // 2:]

    cond_h, cond_w = cur_cond8.shape[:2]
    cond_top = cur_cond8[:cond_h // 2]
    cond_bot = cur_cond8[cond_h // 2:]

    # ===== 2) 右侧宽度统一 =====
    right_width = max(cam_w, cond_w)
    cam_top  = cv2.resize(cam_top,  (right_width, cam_top.shape[0]))
    cam_bot  = cv2.resize(cam_bot,  (right_width, cam_bot.shape[0]))
    cond_top = cv2.resize(cond_top, (right_width, cond_top.shape[0]))
    cond_bot = cv2.resize(cond_bot, (right_width, cond_bot.shape[0]))

    # ===== 3) 右侧 4 排：cond上 / 生成上 / 生成下 / cond下 =====
    right_panel = np.vstack([cond_top, cam_top, cam_bot, cond_bot])
    right_height = right_panel.shape[0]

    # ===== 4) 左侧 BEV：中心裁剪→放大到 right_height =====
    bev_h, bev_w = bev_frame.shape[:2]
    bev_crop = center_crop(bev_frame, crop_w=int(bev_w * 0.55), crop_h=int(bev_h * 0.55))
    bev_resized = cv2.resize(bev_crop, (int(bev_crop.shape[1] * (right_height / bev_crop.shape[0])), right_height))

    combined = np.hstack([bev_resized, right_panel])

    title_bar = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (40, 40, 40)
    txt = f"Scenario: {scenario_name} | Step: {step_count} | Planner: {planner_type}"
    cv2.putText(title_bar, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return np.vstack([title_bar, combined])


# ==========================
# Runner（Diffusion删光，VAD用pass占位）
# ==========================
class Stage3SimulationRunner(SimulationRunner):
    """
    阶段3 Runner（可运行骨架）：
    - 不含 diffusion
    - VAD保留结构占位（pass）
    - 用预定义轨迹推进仿真
    - 保存 BEV+Camera 视频
    """

    def __init__(
        self,
        cfg,
        simulation: Simulation,
        planner: AbstractPlanner,
        gen_cfg=None,
        use_vad_planner=False,
        vad_config_file=None,
        vad_checkpoint=None,
        output_dir="./stage3_output",
        max_sim_steps=150,
        # ====== 频率 =======
        sim_hz: int = 10,      # nuplan 固定10Hz
        plan_hz: int = 5,      # 你想要的“更新频率”：5Hz -> step=2；2Hz -> step=5
        # ====== 生成 =======
        gen_pipeline=None,          
        gen_device=None,            
    ):
        super().__init__(simulation, planner)
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_sim_steps = max_sim_steps

        self.camera_channels = [
            CameraChannel.CAM_L1,
            CameraChannel.CAM_L0,
            CameraChannel.CAM_F0,
            CameraChannel.CAM_R0,
            CameraChannel.CAM_R1,
            CameraChannel.CAM_R2,
            CameraChannel.CAM_B0,
            CameraChannel.CAM_L2,
        ]

        # ===== VAD配置：保留接口，但不实现 =====
        self.use_vad_planner = use_vad_planner
        self.vad_config_file = vad_config_file
        self.vad_checkpoint = vad_checkpoint
        self.vad_planner_client = None  # 占位

        # ===== 数据库对象 =====================
        self.numap = None
        self.db_record = None

        # ===== 生成器 =====================

        self.sim_hz = int(sim_hz)
        self.plan_hz = int(plan_hz)

        if self.sim_hz <= 0 or self.plan_hz <= 0:
            raise ValueError(f"sim_hz/plan_hz 必须 > 0, got sim_hz={sim_hz}, plan_hz={plan_hz}")
        assert self.sim_hz % self.plan_hz == 0, \
            f"sim_hz={self.sim_hz} 必须整除 plan_hz={self.plan_hz}"

        self.plan_every_steps = self.sim_hz // self.plan_hz
        self.video_fps = self.plan_hz

        self.rollbuf = RollingInfoBuffer5Hz(
                            history_len=9,
                            future_len=11,
                            sim_hz=self.sim_hz,     # 10
                            pack_hz=self.plan_hz,   # 窗口更新频率（规划/生成）
                            lidar_hz=20,
                        )

        # warmup：凑够 history_len 个包，需要 (history_len-1) 个间隔
        self.warmup_steps = (self.rollbuf.history_len - 1) * self.plan_every_steps
        self.gen_cfg = gen_cfg

        self.gen_pipeline = gen_pipeline
        self.gen_device = gen_device

        self.hist_cam8_for_gen = deque(maxlen=self.rollbuf.history_len)  # 9
        self.hist_raw_cam8 = deque(maxlen=self.rollbuf.history_len)
        self.last_gen_cam8 = None

        self.gen_client = None
        if self.gen_pipeline is not None:
            self.gen_client = StreamingARClient(
                pipeline=self.gen_pipeline,
                device=self.gen_device,
                history_len=self.rollbuf.history_len,   # 9
                pack_hz=self.plan_hz,                   # 5
            )
            print("[GEN] StreamingARClient ready")

        self.gen_state = None

        self.overlay_hist_frames = []
        self.overlay_cur_frame = None

        self.traj_straight_plan_steps = 6   # 前 6 次“规划更新点”(global_count%plan_every_steps==0) 用直行
        self.traj_type_after = "right_then_back"

        self.last_cond_pack = None

        self.cond_cache = None
        self.cond_cache_iter_idx = None
        self.committed_ego_states_5hz = deque(maxlen=self.rollbuf.history_len)
        
    def _initialize(self) -> None:
        """初始化：planner + 地图（Diffusion不做）"""
        print("\n[初始化] Stage3SimulationRunner 初始化...")

        self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)
        self.planner.initialize(self._simulation.initialize())
        self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

        # ===== [保留你原来的地图加载逻辑] =====
        try:
            scenario = self._simulation.scenario
            self.db_wrapper = NuPlanDBWrapper(
                scenario._data_root,
                scenario._map_root,
                scenario._log_file_load_path,
                scenario._map_version,
            )
            db_name = Path(scenario._log_file_load_path).stem
            self.db_record = self.db_wrapper.get_log_db(db_name)

            map_factory = NuPlanMapFactory(
                get_maps_db(map_root=scenario._map_root, map_version=scenario._map_version)
            )
            self.numap = map_factory.build_map_from_name(scenario._map_name)
            print("[初始化] 地图加载成功")
        except Exception as e:
            print(f"[警告] 地图加载失败: {e}")
            self.numap = None

        self.lidar2ego = self.db_record.lidar[0].trans_matrix

        self.cam_db_dict = {}
        for cam in CAM_TYPES:
            self.cam_db_dict[cam] = self.db_record.camera.select_one(channel=cam)

        initial_ego2global = get_transmat_for_lidarpc_token_from_db(
            self._simulation.scenario._log_file,
            self._simulation.scenario._initial_lidar_token,
        )
        initial_z = float(initial_ego2global[2, 3])

        self.rollbuf.set_static_metas(self.lidar2ego, self.cam_db_dict, trans_z=initial_z)
        self.rollbuf.db_record = self.db_record
        self.rollbuf.location = getattr(self._simulation.scenario, "_map_name", None)
        self.rollbuf.db_name = Path(scenario._log_file_load_path).stem

        if self.gen_cfg is not None:
            self.cond_cache = RollingCondPackCache(
                self.gen_cfg,
                map_api=self.numap,
                want_proj=False,
                use_gpu=False,
            )
            self.cond_cache_iter_idx = None

        # ===== VAD 预留占位 =====
        if self.use_vad_planner:
            # TODO: 后面你再接VAD初始化
            pass
    
    def _get_db_past_ego_states(self, scenario, iter_idx: int):
        """
        返回长度=history_len 的 ego_state 列表（oldest->newest）
        pack_hz=5 => stride_sim=2 时，iter_idx=16 会得到 [0,2,4,6,8,10,12,14,16]
        """
        H = self.rollbuf.history_len
        stride = self.rollbuf.stride_sim

        idxs = []
        for i in range(H):
            j = iter_idx - (H - 1 - i) * stride
            if j < 0:
                j = 0
            idxs.append(int(j))

        states = []
        for j in idxs:
            states.append(scenario.get_ego_state_at_iteration(j))

        return states

    def _db_obs(self, scenario, it: int):
        return scenario.get_tracked_objects_at_iteration(it)

    def _extract_committed_future0_ego(self, trajectory):
        """
        从当前 prev_trajectory 中取出“下一轮要 commit 的 5Hz future[0] ego”。
        规则必须和 RollingInfoBuffer5Hz.build_infos 里 future 的取样规则一致。

        plan_hz=5 -> 10Hz轨迹里取 index=2
        plan_hz=2 -> 10Hz轨迹里取 index=5
        """
        if trajectory is None:
            return None

        try:
            sampled_states = trajectory.get_sampled_trajectory()  # 10Hz
        except Exception:
            return None

        if sampled_states is None or len(sampled_states) == 0:
            return None

        future_traj_stride = int(10 / self.plan_hz)
        idx = int(future_traj_stride)

        if idx >= len(sampled_states):
            idx = len(sampled_states) - 1
        if idx < 0:
            return None

        return sampled_states[idx]

    def _commit_future0_ego(self, ego_state):
        """
        把这一轮 future[0] ego 提交到 committed 5Hz history 队列。
        下一轮 build_infos 时，这个提交值会成为新的 history[-1]。
        """
        if ego_state is None:
            return

        if len(self.committed_ego_states_5hz) == 0:
            self.committed_ego_states_5hz.append(deepcopy(ego_state))
            return

        self.committed_ego_states_5hz.append(deepcopy(ego_state))


    def run(self) -> RunnerReport:
        start_time = time.perf_counter()

        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
        )

        print(f"\n{'=' * 80}")
        print(f"开始仿真场景: {self._simulation.scenario.scenario_name}")
        print(f"{'=' * 80}")

        self.simulation.callback.on_simulation_start(self.simulation.setup)
        self._initialize()

        ego_trajectory_log = []
        video_frames = []

        prev_sensor_data = None
        prev_trajectory = None
        prev_trajectory_source = "None"

        global_count = 0
        max_steps = self.max_sim_steps
        scenario = self._simulation.scenario

        plan_every_10hz_steps = self.plan_every_steps
        warmup_steps = self.warmup_steps

        print(f"\n开始仿真循环（最大步数：{max_steps}，每{plan_every_10hz_steps}步更新轨迹/生成视频 \
              采用历史帧数{self.rollbuf.history_len}）...")

        while self.simulation.is_simulation_running():
            
            if global_count >= max_steps:
                print(f"\n达到最大步数限制({max_steps})，停止仿真")
                break

            self.simulation.callback.on_step_start(self.simulation.setup, self.planner)
            planner_input = self.simulation.get_planner_input()
            iter_idx = self.simulation._time_controller.get_iteration().index
            # ===== 每plan_every_10hz_steps步规划一次轨迹/生成画面 =====
            if global_count % plan_every_10hz_steps == 0:

                cond_pack = None
                try:
                    sensor_pack = scenario.get_sensors_at_iteration(iter_idx, self.camera_channels)
                    current_sensor_data = [sensor_pack.images[ch].as_pil for ch in self.camera_channels]
                except Exception:
                    current_sensor_data = None

                # ===== hist_cam8_for_gen / prev_sensor_data 更新 =====
                if current_sensor_data is not None:
                    # [NEW] raw history 永远存 nuplan 原图
                    self.hist_raw_cam8.append(current_sensor_data)

                    if global_count < warmup_steps or self.last_gen_cam8 is None:
                        self.hist_cam8_for_gen.append(current_sensor_data)
                    else:
                        self.hist_cam8_for_gen.append(self.last_gen_cam8)

                prev_sensor_data = current_sensor_data  # 这一帧的 nuplan 原图（raw）

                self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

                trajectory_source = "unknown"
                trajectory = None

                if global_count < warmup_steps:
                    traj_states = [scenario.get_ego_state_at_iteration(iter_idx + k) for k in range(0, 22 + 1)]
                    trajectory = InterpolatedTrajectory(trajectory=traj_states)
                    trajectory_source = "DB"
                else:
                    plan_idx_after_warmup = (global_count - warmup_steps) // plan_every_10hz_steps

                    if plan_idx_after_warmup < self.traj_straight_plan_steps:
                        traj_type = "straight"
                    else:
                        traj_type = self.traj_type_after
                        
                    ego = planner_input.history.ego_state_buffer[-1]
                    setattr(ego, "_plan_idx_after_warmup", plan_idx_after_warmup)
                    trajectory = get_traj(
                        ego,
                        planner_input.history.ego_state_buffer,
                        trajectory_type=traj_type,
                    )
                    trajectory_source = f"Predefined({traj_type})"

                prev_trajectory = trajectory

                self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
                print(f" {global_count} ✓ [轨迹来源] {trajectory_source}")

                infos_all = None

                if global_count >= warmup_steps:

                    if global_count == warmup_steps:
                        past_ego_states = self._get_db_past_ego_states(scenario, iter_idx)

                        self.committed_ego_states_5hz.clear()
                        for st in past_ego_states:
                            self.committed_ego_states_5hz.append(deepcopy(st))
                    else:
                        past_ego_states = list(self.committed_ego_states_5hz)

                    self.rollbuf.set_db_context(scenario, iter_idx)

                    infos_all = self.rollbuf.build_infos(
                        planner_input=planner_input,
                        future_trajectory=prev_trajectory,
                        past_ego_states=past_ego_states,
                    )
                    print(f"[infos_all] build ok, len={len(infos_all)}")

                    if self.cond_cache is None:
                        self.cond_cache = RollingCondPackCache(
                            self.gen_cfg,
                            map_api=self.numap,
                            want_proj=False,
                            use_gpu=False,
                        )

                    if self.cond_cache_iter_idx is None:
                        cond_pack = self.cond_cache.init_from_seq(infos_all)
                        appended_count = len(infos_all)
                    else:
                        pack_shift = (iter_idx - self.cond_cache_iter_idx) // self.rollbuf.stride_sim
                        pack_shift = int(max(pack_shift, 1))

                        if pack_shift >= len(infos_all):
                            cond_pack = self.cond_cache.init_from_seq(infos_all)
                            appended_count = len(infos_all)
                        else:
                            cond_pack = self.cond_cache.append_infos(infos_all[-pack_shift:])
                            appended_count = pack_shift

                    self.cond_cache_iter_idx = int(iter_idx)

                    print(
                        f"[cond_pack] cache ok, append={appended_count}, "
                        f"T={cond_pack['camera_intrinsics'].shape[0]}, keys={list(cond_pack.keys())}"
                    )

                    if global_count == warmup_steps:
                        self.overlay_hist_frames = []
                        for t in range(self.rollbuf.history_len):
                            self.overlay_hist_frames.append(
                                render_overlay_conditions_8cam(
                                    bbox_frames=cond_pack["3dbox_images"],
                                    hdmap_frames=cond_pack["hdmap_images"],
                                    resize_to=(480, 270),
                                    t_idx=t,
                                )
                            )


                    t_cur = self.rollbuf.history_len
                    self.overlay_cur_frame = render_overlay_conditions_8cam(
                        bbox_frames=cond_pack["3dbox_images"],
                        hdmap_frames=cond_pack["hdmap_images"],
                        resize_to=(480, 270),
                        t_idx=t_cur,
                    )

                    committed_future0_ego = self._extract_committed_future0_ego(prev_trajectory)

                    if self.gen_client is not None and len(self.hist_cam8_for_gen) == self.rollbuf.history_len:
                        try:
                            gen_cam8, new_state = self.gen_client.generate_next_cam8(
                                cond_pack=cond_pack,
                                hist_cam8_list=list(self.hist_cam8_for_gen),
                                gen_state=self.gen_state,
                            )
                            if new_state is not None:
                                self.gen_state = new_state

                            if gen_cam8 is not None:
                                self.last_gen_cam8 = gen_cam8
                                print("[GEN] next frame generated (8cam)")
                        except Exception:
                            print("[GEN] generate_next_cam8 failed, traceback:")
                            traceback.print_exc()

                    if committed_future0_ego is not None:
                        self._commit_future0_ego(committed_future0_ego)
            else:
                pass

            # ===== 可视化 =====
            try:
                ego_state = planner_input.history.ego_state_buffer[-1]
                obs_db = self._db_obs(scenario, iter_idx)
                tracked_objects = obs_db.tracked_objects.tracked_objects

                bev_frame = render_scene_frame(
                    ego_state=ego_state,
                    ego_trajectory_log=ego_trajectory_log,
                    tracked_objects=tracked_objects,
                    step_count=global_count,
                    scenario_name=self._simulation.scenario.scenario_name,
                    planned_trajectory=prev_trajectory,
                    map_api=self.numap,
                    view_range=100.0,
                )

                vis_cam8 = prev_sensor_data
                if self.last_gen_cam8 is not None and global_count >= warmup_steps:
                    vis_cam8 = self.last_gen_cam8

                if vis_cam8 is not None:
                    cam_frame = render_camera_views_stage3(vis_cam8, self.camera_channels)
                else:
                    cam_frame = np.zeros((540, 1920, 3), dtype=np.uint8)

                # current cond 8cam（底部）
                cur_cond8 = self.overlay_cur_frame
                if cur_cond8 is None:
                    cur_cond8 = np.zeros((540, 1920, 3), dtype=np.uint8)

                planner_type = "VAD" if self.use_vad_planner else "Predefined"
                combined = render_combined_view(
                    bev_frame=bev_frame,
                    camera_frame=cam_frame,
                    cur_cond8=cur_cond8,
                    step_count=global_count,
                    scenario_name=scenario.scenario_name,
                    planner_type=planner_type,
                )
                if global_count % plan_every_10hz_steps == 0:
                    video_frames.append(combined)
                    # 第一次cond出来：把前9帧里“右侧cond两行”从黑图覆盖成真实cond
                    if global_count == warmup_steps and len(self.overlay_hist_frames) == self.rollbuf.history_len:
                        title_height = 60
                        row_h = (combined.shape[0] - title_height) // 4

                        right_w = cur_cond8.shape[1]          # 正常是1920
                        bev_w = combined.shape[1] - right_w   # 左侧BEV宽度

                        # 覆盖最近9帧（对应history_len=9的包）
                        for t in range(self.rollbuf.history_len):
                            dst = video_frames[-self.rollbuf.history_len + t]
                            cond_img = self.overlay_hist_frames[t]

                            # 保底：尺寸不对就resize
                            if cond_img.shape[0] != row_h * 2 or cond_img.shape[1] != right_w:
                                cond_img = cv2.resize(cond_img, (right_w, row_h * 2))

                            top = cond_img[:row_h]
                            bot = cond_img[row_h:row_h * 2]

                            # 右侧4排：0=cond_top, 1=cam_top, 2=cam_bot, 3=cond_bot
                            y0 = title_height + 0 * row_h
                            y3 = title_height + 3 * row_h

                            dst[y0:y0 + row_h, bev_w:bev_w + right_w] = top
                            dst[y3:y3 + row_h, bev_w:bev_w + right_w] = bot

            except Exception as e:
                print(f"  警告: 第{global_count}步可视化失败: {e}")

            # ===== always propagate with remaining trajectory =====
            self.simulation.propagate(prev_trajectory)

            try:
                last_sample = self.simulation.history.last()
                ego_state_new = last_sample.ego_state
                ego_trajectory_log.append(ego_state_new.center.array)
            except Exception:
                # 兜底：至少别崩
                ego_state = planner_input.history.ego_state_buffer[-1]
                ego_trajectory_log.append(ego_state.center.array)

            global_count += 1
            self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())

        self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)

        # ===== 保存视频 =====

        if len(video_frames) > 0:
            db_name = scenario.log_name.replace(".db", "")
            scenario_token = scenario.token
            video_dir = self.output_dir / db_name / scenario_token
            video_dir.mkdir(parents=True, exist_ok=True)
            video_file = video_dir / "bev_camera_visualization.mp4"

            print(f"\n正在保存视频: {video_file} (frames={len(video_frames)})")

            try:
                fps = self.video_fps
                
                # ===== [FIX] 用 PyAV 写视频 =====
                container = av.open(str(video_file), mode="w")
                stream = container.add_stream("libx264", rate=fps)
                stream.pix_fmt = "yuv420p"

                stream.options = {
                    "crf": "20",        # 18 很清晰；23 默认偏糊 16/18/20
                    "preset": "slow",   # 更清晰但更慢；medium/slow/slower 
                }
                # frame 尺寸必须固定
                h, w = video_frames[0].shape[:2]
                stream.width = w
                stream.height = h

                for frame_bgr in video_frames:
                    # OpenCV 是 BGR，PyAV 这里用 bgr24
                    frame_av = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                    for packet in stream.encode(frame_av):
                        container.mux(packet)

                # flush
                for packet in stream.encode():
                    container.mux(packet)

                container.close()
                print("✓ 视频保存成功 (PyAV)")

            except Exception as e:
                print(f"✗ 视频保存失败 (PyAV): {e}")

        end_time = time.perf_counter()
        report.end_time = end_time
        report.planner_report = self.planner.generate_planner_report()

        print(f"\n仿真完成！总耗时: {end_time - start_time:.2f} 秒")
        print(f"共执行 {global_count} 步")
        print(f"{'=' * 80}\n")
        return report


# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    print("\n开始运行NuPlan仿真...\n")

    # ========= 加载路径配置 =========

    sim_cfg = Path(path_config.get("sim_cfg_path", "")).resolve()

    with initialize_config_dir(version_base=None, config_dir=str(sim_cfg)):
        cfg = compose(
            config_name="simulation_config.yaml",
            overrides=[
                "+simulation=closed_loop_nonreactive_agents",
                "planner=simple_planner",
                "scenario_builder=my_nuplan_mini_debug",
                "scenario_filter=test_random14",
                "worker.threads_per_node=4",
                "experiment_uid=test_random14/simple_planner",
                "verbose=true",
            ],
        )

    OmegaConf.set_struct(cfg, False)
    cfg.scenario_builder.db_files = [os.environ["NUPLAN_DB_FILES"]]
    cfg.scenario_builder.map_root = path_config["nuplan_maps_root"]
    cfg.scenario_builder.map_version = path_config["nuplan_map_version"]

    print("[CFG OVERRIDE] scenario_builder.db_files =", cfg.scenario_builder.db_files)
    # ========= Ray init =========
    # ===== per-run override: DB / output / ray temp =====
    run_tag = os.environ.get("RUN_TAG", "run0")

    # 输出目录隔离（避免5个进程写一起）
    stage3_output_dir = os.environ.get("STAGE3_OUTPUT_DIR")

    # ray temp 隔离（避免冲突）
    ray_temp_dir = Path(path_config["ray_temp_dir"]) / run_tag
    ray_temp_dir.mkdir(parents=True, exist_ok=True)

    print("[RUN] DB =", os.environ.get("NUPLAN_DB_FILES"))
    print("[RUN] OUT =", stage3_output_dir)
    print("[RUN] RAY_TMP =", str(ray_temp_dir))
    print("[RUN] generator conditions initialized")

    ray.init(_temp_dir=str(ray_temp_dir))

    # ========= Build scenarios =========
    worker = build_worker(cfg)
    scenario_builder = build_scenario_builder(cfg=cfg)
    scenario_filter = build_scenario_filter(cfg=cfg.scenario_filter)
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

    print(f"找到 {len(scenarios)} 个场景")

    # ========= 输出目录 =========
    if not stage3_output_dir:
        stage3_output_dir = "./stage3_output_stage3_skeleton"
    print(f"✓ 输出目录: {stage3_output_dir}\n")

    # ========= 生成器初始化 =========
    gen_pipeline_cfg = path_config.get("gen_cfg", "")
    gen_output_path = path_config.get("gen_img_log", "./gen_runtime_output")

    if not gen_pipeline_cfg:
        raise ValueError("config_paths.yaml 里缺少 gen_cfg")

    with open(gen_pipeline_cfg, "r", encoding="utf-8") as f:
        gen_pipe_config = json.load(f)

    device = torch.device(gen_pipe_config.get("device", "cuda"))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)

    # if "global_state" in gen_pipe_config:
    #     for key, value in gen_pipe_config["global_state"].items():
    #         dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    # [CHG] 仿真推理禁用分布式
    gen_pipe_config["global_state"] = {}
    common_cfg = gen_pipe_config.get("pipeline", {}).get("common_config", {})
    common_cfg["distribution_framework"] = "none"   
    common_cfg.pop("ddp_wrapper_settings", None)
    common_cfg.pop("t5_fsdp_wrapper_settings", None)
    pipe = gen_pipe_config.get("pipeline", {})
    pipe.pop("metrics", None)

    pipeline = dwm.common.create_instance_from_config(
        gen_pipe_config["pipeline"],
        output_path=gen_output_path,
        config=gen_pipe_config,
        device=device
    )
    print("[GEN] pipeline loaded")

    for idx, scenario in enumerate(scenarios):
        print(f"\n处理场景 {idx+1}/1: {scenario.token}")

        simulation_time_controller = instantiate(cfg.simulation_time_controller, scenario=scenario)
        ego_controller = instantiate(cfg.ego_controller, scenario=scenario)
        observations = build_observations(cfg.observation, scenario=scenario)

        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )

        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([]),
            simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
        )

        planner = build_planners(cfg.planner, scenario)[0]

        # ===== VAD配置占位（你以后接）=====
        vad_config = path_config.get("vad", {})
        use_vad = False  # 现在占位不启用（可改True测试结构）
        runner = Stage3SimulationRunner(
            cfg=cfg,
            simulation=simulation,
            planner=planner,
            gen_cfg=gen_cfg,
            gen_pipeline=pipeline,      
            gen_device=device,          
            use_vad_planner=use_vad,
            vad_config_file=vad_config.get("config_file"),
            vad_checkpoint=vad_config.get("checkpoint"),
            output_dir=stage3_output_dir,
            max_sim_steps=60,
        )

        try:
            report = runner.run()
            print(f"✓ 场景 {scenario.token} 运行成功: succeeded={report.succeeded}")
        except Exception as e:
            print(f"✗ 场景 {scenario.token} 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("阶段3骨架完成！")
    print("=" * 80)
