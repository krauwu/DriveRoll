from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem


def import_obj(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def pil_grid(images: list[Image.Image], cols: int = 3) -> Image.Image:
    if not images:
        return Image.new("RGB", (640, 360), (30, 30, 30))

    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))

    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        canvas.paste(img.convert("RGB"), (x, y))

    return canvas


def to_pil_rgb(image_like) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")

    if torch.is_tensor(image_like):
        x = image_like.detach().cpu()

        if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = x.permute(1, 2, 0).contiguous()

        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if x.ndim != 3:
            raise ValueError(f"unsupported tensor image shape: {tuple(x.shape)}")

        if x.shape[-1] == 1:
            x = x.repeat(1, 1, 3)

        if x.dtype != torch.uint8:
            x = x.float()
            if x.numel() > 0 and float(x.max().item()) <= 1.5:
                x = x * 255.0
            x = x.clamp(0, 255).to(torch.uint8)

        return Image.fromarray(x.numpy(), mode="RGB")

    arr = np.asarray(image_like)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim != 3:
        raise ValueError(f"unsupported ndarray image shape: {tuple(arr.shape)}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size > 0 and float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def fuse_cond_views(hdmap_views, box_views) -> list[Image.Image]:
    out = []
    total = min(len(hdmap_views), len(box_views))

    i = 0
    while i < total:
        hd = to_pil_rgb(hdmap_views[i])
        box = to_pil_rgb(box_views[i])

        if box.size != hd.size:
            box = box.resize(hd.size, Image.BILINEAR)

        hd_arr = np.asarray(hd, dtype=np.uint8)
        box_arr = np.asarray(box, dtype=np.uint8)

        fused = np.maximum(hd_arr, box_arr)
        out.append(Image.fromarray(fused, mode="RGB"))
        i += 1

    return out


def infer_tile_size(
    center_views,
    box_views=None,
    hdmap_views=None,
    fallback_size: tuple[int, int] = (640, 360),
) -> tuple[int, int]:
    groups = [hdmap_views, box_views, center_views]

    group_idx = 0
    while group_idx < len(groups):
        views = groups[group_idx]
        if views is not None and len(views) > 0:
            img = to_pil_rgb(views[0])
            return img.size
        group_idx += 1

    return fallback_size


def compose_debug_panel(
    center_views,
    cond_views=None,
    cols: int = 3,
    tile_size: tuple[int, int] | None = None,
) -> Image.Image:
    if tile_size is None:
        tile_size = infer_tile_size(center_views=center_views)

    tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    canvas = Image.new("RGB", (cols * tile_w, 4 * tile_h), (0, 0, 0))

    center_count = min(len(center_views), cols * 2)
    i = 0
    while i < center_count:
        img = to_pil_rgb(center_views[i])
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.BILINEAR)

        x = (i % cols) * tile_w
        y = tile_h + (i // cols) * tile_h
        canvas.paste(img, (x, y))
        i += 1

    if cond_views is not None:
        cond_count = min(len(cond_views), cols * 2)
        i = 0
        while i < cond_count:
            img = to_pil_rgb(cond_views[i])
            if img.size != (tile_w, tile_h):
                img = img.resize((tile_w, tile_h), Image.BILINEAR)

            x = (i % cols) * tile_w
            if i < cols:
                y = 0
            else:
                y = 3 * tile_h

            canvas.paste(img, (x, y))
            i += 1

    return canvas


class RollingWindowBuffer:
    """
    长 segment 上的 rolling window：
    - history_len: reference history 数
    - window_len: 模型单次 iteration 的窗口长度
    - cond_total_len = 2 * window_len
    - 每次 commit 一帧后，整个 cond window 向前滚 1 帧

    generated_override 现在同时保存：
    - images
    - ego_transform
    """

    def __init__(self, history_len: int, window_len: int):
        self.history_len = int(history_len)
        self.window_len = int(window_len)
        self.cond_total_len = int(window_len * 2)

        self.total_frames = 0
        self.cursor = -1
        self.generated_override: dict[int, dict[str, Any]] = {}

    def reset(self, total_frames: int):
        self.total_frames = int(total_frames)
        self.cursor = self.history_len - 1
        self.generated_override.clear()

    def history_start(self) -> int:
        return self.cursor - self.history_len + 1

    def history_indices(self) -> list[int]:
        s = self.history_start()
        return list(range(s, s + self.history_len))

    def can_build_full_cond_window(self) -> bool:
        s = self.history_start()
        e = s + self.cond_total_len
        return s >= 0 and e <= self.total_frames

    def try_cond_indices(self) -> list[int] | None:
        if not self.can_build_full_cond_window():
            return None
        s = self.history_start()
        e = s + self.cond_total_len
        return list(range(s, e))

    def cond_indices(self) -> list[int]:
        out = self.try_cond_indices()
        if out is None:
            raise StopIteration("这个 segment 剩余帧数不足，无法再构造完整 rolling cond window")
        return out

    def next_target_index(self) -> int:
        idx = self.cursor + 1
        if idx >= self.total_frames:
            raise StopIteration("这个 segment 已经滚到末尾了")
        return idx

    def get_override(self, frame_idx: int):
        return self.generated_override.get(int(frame_idx), None)

    def get_generated_images(self, frame_idx: int):
        item = self.get_override(frame_idx)
        if item is None:
            return None
        return item.get("images", None)

    def get_generated_ego_transform(self, frame_idx: int):
        item = self.get_override(frame_idx)
        if item is None:
            return None
        return item.get("ego_transform", None)

    def commit_generated(
        self,
        frame_idx: int,
        images: list[Image.Image],
        ego_transform,
    ):
        if torch.is_tensor(ego_transform):
            ego_value = ego_transform.detach().cpu().clone()
        else:
            ego_value = torch.as_tensor(ego_transform).detach().cpu().clone()

        self.generated_override[int(frame_idx)] = {
            "images": [img.copy() for img in images],
            "ego_transform": ego_value,
        }
        self.cursor = int(frame_idx)


class NuscRollingDataAgent:
    """
    负责：
    1) 建立 segment-id 列表
    2) 选择一个长 segment
    3) 在长 segment 上维护 rolling window
    4) build_infos_all()
    5) build_cond_from_infos()
    6) 提交生成结果，覆盖 history 图像和 ego pose
    7) 根据用户指令构造未来 ego plan
    """

    def __init__(self, cfg_path: str | Path):
        self.cfg_path = str(cfg_path)
        self.cfg = self._load_json(self.cfg_path)

        self.dataset_cfg = self.cfg["validation_dataset"]["base_dataset"]["datasets"][0]
        self.inference_cfg = self.cfg["pipeline"]["inference_config"]

        self.history_len = int(self.inference_cfg["reference_frame_count"])
        self.window_len = int(self.inference_cfg["sequence_length_per_iteration"])
        self.cond_total_len = self.window_len * 2
        self.fps = int(self.dataset_cfg["fps_stride_tuples"][0][0])
        self.sensor_channels = list(self.dataset_cfg["sensor_channels"])

        self.dataset = self._build_dataset()
        self.ds_cls = self.dataset.__class__
        self.segment_meta = self._build_segment_index()

        self.buffer = RollingWindowBuffer(self.history_len, self.window_len)

        self.current_segment_id: int | None = None
        self.segment_raw: dict[str, Any] | None = None
        self.annotation_cache: dict[str, list[dict[str, Any]]] = {}
        self.gen_state = None
        self.is_initialized = False
        self.crossview_mask = self._load_crossview_mask_from_cfg()
        self.runtime_cond_cache: dict[tuple, tuple[list[Image.Image] | None, list[Image.Image] | None]] = {}

        self.motion_cfg = self.cfg.get("interactive_motion_config", {})
        self.wheelbase_m = float(self.motion_cfg.get("wheelbase_m", 2.8))
        self.forward_accel_mps2 = float(self.motion_cfg.get("forward_accel_mps2", 0.5))
        self.turn_accel_mps2 = float(self.motion_cfg.get("turn_accel_mps2", 0.0))
        self.brake_accel_mps2 = float(self.motion_cfg.get("brake_accel_mps2", -2.5))
        self.turn_steer_rad = float(self.motion_cfg.get("turn_steer_rad", 0.22))
        self.default_speed_mps = float(self.motion_cfg.get("default_speed_mps", 1.0))
        self.min_forward_speed_mps = float(self.motion_cfg.get("min_forward_speed_mps", 5.0))

    # -------------------------
    # cfg / dataset
    # -------------------------
    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_dataset(self):
        fs_root = self.cfg["global_state"]["nuscenes_fs"]["path"]
        fs = DirFileSystem(path=fs_root, fs=LocalFileSystem())

        dataset_cls = import_obj(self.dataset_cfg["_class_name"])
        kwargs = {k: v for k, v in self.dataset_cfg.items() if k != "_class_name"}
        kwargs["fs"] = fs
        return dataset_cls(**kwargs)

    def _build_segment_index(self):
        metas = []
        for idx, item in enumerate(self.dataset.items):
            scene = self.ds_cls.query(self.dataset.tables, self.dataset.indices, "scene", item["scene"])
            first_sd = self.ds_cls.query(self.dataset.tables, self.dataset.indices, "sample_data", item["segment"][0][0])
            last_sd = self.ds_cls.query(self.dataset.tables, self.dataset.indices, "sample_data", item["segment"][-1][0])

            metas.append({
                "segment_id": idx,
                "scene_name": scene.get("name", item["scene"]),
                "scene_token": item["scene"],
                "fps": float(item["fps"]),
                "start_sample_token": item["segment_samples"][0],
                "end_sample_token": item["segment_samples"][-1],
                "start_ts_us": int(first_sd["timestamp"]),
                "end_ts_us": int(last_sd["timestamp"]),
                "display_name": (
                    f"seg-{idx:04d} | scene={scene.get('name', item['scene'])} | "
                    f"fps={float(item['fps']):.1f} | "
                    f"start={item['segment_samples'][0][:8]} | "
                    f"end={item['segment_samples'][-1][:8]}"
                ),
            })

        return metas

    # -------------------------
    # UI helpers
    # -------------------------
    def get_segment_choices(self):
        return [(m["display_name"], m["segment_id"]) for m in self.segment_meta]

    def get_segment_preview(self, segment_id: int):
        sample = self.dataset[int(segment_id)]
        return pil_grid(sample["images"][0], cols=3)

    # -------------------------
    # segment select
    # -------------------------
    def select_segment(self, segment_id: int):
        self.current_segment_id = int(segment_id)
        self.segment_raw = self.dataset[self.current_segment_id]
        self.annotation_cache.clear()
        self.runtime_cond_cache.clear()
        self.gen_state = None
        self.buffer.reset(total_frames=len(self.segment_raw["images"]))
        self.is_initialized = True

        cond_indices = self.buffer.try_cond_indices()
        if cond_indices is None:
            raise ValueError(
                "这个 segment 太短，无法初始化完整 rolling cond window: "
                f"total_frames={len(self.segment_raw['images'])}, "
                f"history_len={self.history_len}, "
                f"cond_total_len={self.cond_total_len}"
            )

        center_views = self.segment_raw["images"][self.history_len - 1]

        box_views = None
        hdmap_views = None
        if "3dbox_images" in self.segment_raw:
            box_views = self.segment_raw["3dbox_images"][self.history_len - 1]
        if "hdmap_images" in self.segment_raw:
            hdmap_views = self.segment_raw["hdmap_images"][self.history_len - 1]

        tile_size = infer_tile_size(
            center_views=center_views,
            box_views=box_views,
            hdmap_views=hdmap_views,
        )

        image = compose_debug_panel(
            center_views=center_views,
            cond_views=None,
            cols=3,
            tile_size=tile_size,
        )

        status = (
            f"已初始化\n"
            f"{self.segment_meta[self.current_segment_id]['display_name']}\n"
            f"history={self.buffer.history_indices()[0]}->{self.buffer.history_indices()[-1]} | "
            f"cond={cond_indices[0]}->{cond_indices[-1]}"
        )
        return image, status

    # -------------------------
    # command / motion
    # -------------------------
    def parse_command(self, command: str):
        cmd = (command or "").strip().lower()
        if not cmd:
            cmd = "w"

        if cmd in {"a", "left"}:
            return {"action": "left", "turn_sign": 1.0, "accel_mps2": self.turn_accel_mps2}

        if cmd in {"d", "right"}:
            return {"action": "right", "turn_sign": -1.0, "accel_mps2": self.turn_accel_mps2}

        if cmd in {"s", "slow"}:
            return {"action": "slow", "turn_sign": 0.0, "accel_mps2": self.brake_accel_mps2 * 0.5}

        if cmd in {"brake", "stop"}:
            return {"action": "brake", "turn_sign": 0.0, "accel_mps2": self.brake_accel_mps2}

        return {"action": "forward", "turn_sign": 0.0, "accel_mps2": self.forward_accel_mps2}

    def _get_sample_annotations(self, sample_token: str):
        if sample_token not in self.annotation_cache:
            anns = self.ds_cls.query_range(
                self.dataset.tables,
                self.dataset.indices,
                "sample_annotation",
                sample_token,
                column_name="sample_token",
            )
            self.annotation_cache[sample_token] = anns
        return self.annotation_cache[sample_token]

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().clone()
        return torch.as_tensor(x).detach().cpu().clone()

    def _canonicalize_transform_4x4(self, transform_value, name: str = "transform"):
        t = self._to_tensor(transform_value).float()

        if t.ndim < 2 or tuple(t.shape[-2:]) != (4, 4):
            raise ValueError(
                f"{name} shape 非法，期望最后两维是 (4, 4)，实际 got={tuple(t.shape)}"
            )

        while t.ndim > 2:
            t = t[0]

        return t.contiguous()

    def _expand_transform_like(self, transform_4x4, template_transform, name: str = "transform"):
        base = self._to_tensor(transform_4x4).float()
        template = self._to_tensor(template_transform)

        if tuple(base.shape) != (4, 4):
            raise ValueError(f"{name} 的 base 不是 4x4，got={tuple(base.shape)}")

        if template.ndim < 2 or tuple(template.shape[-2:]) != (4, 4):
            raise ValueError(
                f"{name} 的 template shape 非法，期望最后两维是 (4, 4)，实际 got={tuple(template.shape)}"
            )

        if template.ndim == 2:
            return base.to(dtype=template.dtype)

        view_shape = (1,) * (template.ndim - 2) + (4, 4)
        repeat_shape = tuple(int(x) for x in template.shape[:-2]) + (1, 1)

        out = base.view(*view_shape).repeat(*repeat_shape)
        return out.to(dtype=template.dtype)

    def _get_images_for_frame(self, frame_idx: int):
        override = self.buffer.get_generated_images(frame_idx)
        if override is not None:
            return override
        return self.segment_raw["images"][frame_idx]

    def _get_ego_transform_for_frame(self, frame_idx: int):
        override = self.buffer.get_generated_ego_transform(frame_idx)
        if override is not None:
            return self._to_tensor(override)
        return self._to_tensor(self.segment_raw["ego_transforms"][frame_idx])

    def _get_camera_transforms_for_frame(self, frame_idx: int):
        return self._to_tensor(self.segment_raw["camera_transforms"][frame_idx])

    def _rotation_matrix_to_quaternion_wxyz(self, rot):
        r = self._to_tensor(rot).float()
        if tuple(r.shape) != (3, 3):
            raise ValueError(f"rotation matrix shape 非法: {tuple(r.shape)}")

        r00 = float(r[0, 0].item())
        r01 = float(r[0, 1].item())
        r02 = float(r[0, 2].item())
        r10 = float(r[1, 0].item())
        r11 = float(r[1, 1].item())
        r12 = float(r[1, 2].item())
        r20 = float(r[2, 0].item())
        r21 = float(r[2, 1].item())
        r22 = float(r[2, 2].item())

        trace = r00 + r11 + r22

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (r21 - r12) / s
            qy = (r02 - r20) / s
            qz = (r10 - r01) / s
        elif r00 > r11 and r00 > r22:
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
            qw = (r21 - r12) / s
            qx = 0.25 * s
            qy = (r01 + r10) / s
            qz = (r02 + r20) / s
        elif r11 > r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
            qw = (r02 - r20) / s
            qx = (r01 + r10) / s
            qy = 0.25 * s
            qz = (r12 + r21) / s
        else:
            s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
            qw = (r10 - r01) / s
            qx = (r02 + r20) / s
            qy = (r12 + r21) / s
            qz = 0.25 * s

        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm < 1e-12:
            return [1.0, 0.0, 0.0, 0.0]

        return [qw / norm, qx / norm, qy / norm, qz / norm]


    def _ego_transform_to_nuscenes_pose(self, ego_transform):
        t = self._canonicalize_transform_4x4(ego_transform, name="ego_transform")
        rot = t[:3, :3]
        trans = [
            float(t[0, 3].item()),
            float(t[1, 3].item()),
            float(t[2, 3].item()),
        ]
        quat = self._rotation_matrix_to_quaternion_wxyz(rot)
        return quat, trans


    def _get_camera_sample_data_list_for_frame(self, frame_idx: int):
        if self.current_segment_id is None:
            raise RuntimeError("segment 尚未初始化")

        item = self.dataset.items[self.current_segment_id]
        frame_tokens = item["segment"][frame_idx]

        out = []
        i = 0
        while i < len(frame_tokens):
            sd = self.ds_cls.query(
                self.dataset.tables,
                self.dataset.indices,
                "sample_data",
                frame_tokens[i],
            )
            if self.ds_cls.check_sensor(
                self.dataset.tables,
                self.dataset.indices,
                sd,
                modality="camera",
            ):
                out.append(sd)
            i += 1

        return out


    def _is_same_as_raw_ego(self, frame_idx: int, ego_transform, atol: float = 1e-4):
        raw = self._canonicalize_transform_4x4(
            self.segment_raw["ego_transforms"][frame_idx],
            name="raw_ego_transform",
        )
        cur = self._canonicalize_transform_4x4(
            ego_transform,
            name="ego_transform",
        )
        return torch.allclose(raw, cur, atol=atol, rtol=0.0)


    def _copy_pil_list(self, images):
        if images is None:
            return None
        return [img.copy() for img in images]


    def _make_runtime_cond_cache_key(self, frame_idx: int, ego_transform):
        t = self._canonicalize_transform_4x4(ego_transform, name="ego_transform")
        x = round(float(t[0, 3].item()), 4)
        y = round(float(t[1, 3].item()), 4)
        z = round(float(t[2, 3].item()), 4)
        yaw = round(self._yaw_from_transform(t), 5)
        return (int(frame_idx), x, y, z, yaw)


    def _render_cond_images_for_frame(self, frame_idx: int, ego_transform):
        if not self.is_initialized or self.segment_raw is None:
            raise RuntimeError("请先初始化 segment")

        if self._is_same_as_raw_ego(frame_idx, ego_transform):
            box_views = None
            hdmap_views = None

            if "3dbox_images" in self.segment_raw:
                box_views = self.segment_raw["3dbox_images"][frame_idx]
            if "hdmap_images" in self.segment_raw:
                hdmap_views = self.segment_raw["hdmap_images"][frame_idx]

            return self._copy_pil_list(box_views), self._copy_pil_list(hdmap_views)

        cache_key = self._make_runtime_cond_cache_key(frame_idx, ego_transform)
        if cache_key in self.runtime_cond_cache:
            cached_box, cached_hd = self.runtime_cond_cache[cache_key]
            return self._copy_pil_list(cached_box), self._copy_pil_list(cached_hd)

        quat, trans = self._ego_transform_to_nuscenes_pose(ego_transform)
        camera_sample_data_list = self._get_camera_sample_data_list_for_frame(frame_idx)

        box_views = None
        if getattr(self.dataset, "_3dbox_image_settings", None) is not None:
            box_views = []
            i = 0
            while i < len(camera_sample_data_list):
                base_sd = camera_sample_data_list[i]
                render_sd = {k: v for k, v in base_sd.items()}
                render_sd["rotation"] = quat
                render_sd["translation"] = trans

                img = self.ds_cls.get_3dbox_image(
                    self.dataset.tables,
                    self.dataset.indices,
                    render_sd,
                    self.dataset._3dbox_image_settings,
                )
                box_views.append(img)
                i += 1

        hdmap_views = None
        if getattr(self.dataset, "hdmap_image_settings", None) is not None:
            hdmap_views = []
            i = 0
            while i < len(camera_sample_data_list):
                base_sd = camera_sample_data_list[i]
                render_sd = {k: v for k, v in base_sd.items()}
                render_sd["rotation"] = quat
                render_sd["translation"] = trans

                img = self.ds_cls.get_hdmap_image(
                    self.dataset.map_expansion,
                    self.dataset.map_expansion_dict,
                    self.dataset.tables,
                    self.dataset.indices,
                    render_sd,
                    self.dataset.hdmap_image_settings,
                )
                hdmap_views.append(img)
                i += 1

        self.runtime_cond_cache[cache_key] = (
            self._copy_pil_list(box_views),
            self._copy_pil_list(hdmap_views),
        )

        return self._copy_pil_list(box_views), self._copy_pil_list(hdmap_views)

    def _xy_from_transform(self, ego_transform):
        t = self._canonicalize_transform_4x4(ego_transform, name="ego_transform")
        return float(t[0, 3].item()), float(t[1, 3].item())

    def _z_from_transform(self, ego_transform):
        t = self._canonicalize_transform_4x4(ego_transform, name="ego_transform")
        return float(t[2, 3].item())

    def _yaw_from_transform(self, ego_transform):
        t = self._canonicalize_transform_4x4(ego_transform, name="ego_transform")
        return math.atan2(float(t[1, 0].item()), float(t[0, 0].item()))

    def _build_transform_from_pose(self, template_transform, x: float, y: float, z: float, yaw: float):
        base = self._canonicalize_transform_4x4(template_transform, name="template_transform").clone()

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        base[0, 0] = cos_yaw
        base[0, 1] = -sin_yaw
        base[0, 2] = 0.0

        base[1, 0] = sin_yaw
        base[1, 1] = cos_yaw
        base[1, 2] = 0.0

        base[2, 0] = 0.0
        base[2, 1] = 0.0
        base[2, 2] = 1.0

        base[0, 3] = x
        base[1, 3] = y
        base[2, 3] = z

        base[3, 0] = 0.0
        base[3, 1] = 0.0
        base[3, 2] = 0.0
        base[3, 3] = 1.0

        return self._expand_transform_like(
            transform_4x4=base,
            template_transform=template_transform,
            name="ego_transform",
        )

    def _average_speed_from_recent_history(self):
        history_indices = self.buffer.history_indices()
        recent = history_indices[-3:]

        if len(recent) < 2:
            return self.default_speed_mps

        dt = 1.0 / max(float(self.fps), 1.0)
        speed_sum = 0.0
        pair_count = 0

        i = 0
        while i + 1 < len(recent):
            idx0 = recent[i]
            idx1 = recent[i + 1]

            ego0 = self._get_ego_transform_for_frame(idx0)
            ego1 = self._get_ego_transform_for_frame(idx1)

            x0, y0 = self._xy_from_transform(ego0)
            x1, y1 = self._xy_from_transform(ego1)

            dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            speed_sum += dist / max(dt, 1e-6)
            pair_count += 1
            i += 1

        if pair_count == 0:
            return self.default_speed_mps

        speed = speed_sum / pair_count
        return max(speed, 0.0)

    def _integrate_future_ego_plan(self, command: str):
        if not self.is_initialized or self.segment_raw is None:
            raise RuntimeError("请先初始化 segment")

        action = self.parse_command(command)
        anchor_idx = self.buffer.cursor
        anchor_ego = self._get_ego_transform_for_frame(anchor_idx)

        x, y = self._xy_from_transform(anchor_ego)
        z = self._z_from_transform(anchor_ego)
        yaw = self._yaw_from_transform(anchor_ego)

        speed_mps = self._average_speed_from_recent_history()
        dt = 1.0 / max(float(self.fps), 1.0)

        accel_mps2 = float(action["accel_mps2"])
        turn_sign = float(action["turn_sign"])
        action_name = action["action"]

        if action_name in {"forward", "left", "right"}:
            speed_mps = max(speed_mps, self.min_forward_speed_mps)

        future_plan = []
        step_idx = 0

        while step_idx < self.window_len:
            speed_mps = max(0.0, speed_mps + accel_mps2 * dt)

            yaw_rate = 0.0
            if abs(turn_sign) > 0.0 and speed_mps > 1e-6:
                yaw_rate = (speed_mps / max(self.wheelbase_m, 1e-3)) * math.tan(self.turn_steer_rad) * turn_sign

            yaw_mid = yaw + 0.5 * yaw_rate * dt
            x = x + speed_mps * math.cos(yaw_mid) * dt
            y = y + speed_mps * math.sin(yaw_mid) * dt
            yaw = yaw + yaw_rate * dt

            ego_t = self._build_transform_from_pose(
                template_transform=anchor_ego,
                x=x,
                y=y,
                z=z,
                yaw=yaw,
            )
            future_plan.append(ego_t)
            step_idx += 1

        return future_plan

    # -------------------------
    # rolling info
    # -------------------------
    def can_continue_generation(self):
        if not self.is_initialized or self.segment_raw is None:
            return False
        return self.buffer.can_build_full_cond_window()

    def build_infos_all(self, command: str):
        if not self.is_initialized or self.segment_raw is None:
            raise RuntimeError("请先初始化 segment")

        action_dict = self.parse_command(command)
        cond_indices = self.buffer.try_cond_indices()
        if cond_indices is None:
            raise StopIteration("这个 segment 剩余帧数不足，无法再构造完整 rolling cond window")

        history_indices = self.buffer.history_indices()
        history_set = set(history_indices)
        future_ego_plan = self._integrate_future_ego_plan(command)

        infos_all = []
        rel_idx = 0

        while rel_idx < len(cond_indices):
            abs_idx = cond_indices[rel_idx]
            sample_token = self.segment_raw["segment_samples"][abs_idx]
            pts_t = self.segment_raw["pts"][abs_idx]

            if torch.is_tensor(pts_t):
                ts = float(pts_t[0].item()) if pts_t.numel() > 0 else float(abs_idx)
            else:
                ts = float(abs_idx)

            if abs_idx in history_set:
                ego_transform = self._get_ego_transform_for_frame(abs_idx)
            else:
                future_offset = rel_idx - self.history_len
                if 0 <= future_offset < len(future_ego_plan):
                    ego_transform = future_ego_plan[future_offset]
                else:
                    ego_transform = self._to_tensor(self.segment_raw["ego_transforms"][abs_idx])

            box_views, hdmap_views = self._render_cond_images_for_frame(
                frame_idx=abs_idx,
                ego_transform=ego_transform,
            )

            info_t = {
                "timestamp": ts,
                "frame_idx": abs_idx,
                "is_history": abs_idx in history_set,
                "image_source": "generated" if self.buffer.get_generated_images(abs_idx) is not None else "gt",
                "images": self._get_images_for_frame(abs_idx),
                "3dbox_images": box_views,
                "hdmap_images": hdmap_views,
                "pts": self._to_tensor(pts_t),
                "sample_annotation": self._get_sample_annotations(sample_token),
                "ego_transforms": ego_transform,
                "camera_transforms": self._get_camera_transforms_for_frame(abs_idx),
                "camera_intrinsics": self._to_tensor(self.segment_raw["camera_intrinsics"][abs_idx]),
                "image_size": self._to_tensor(self.segment_raw["image_size"][abs_idx]) if "image_size" in self.segment_raw else None,
                "segment_sample_token": sample_token,
                "fps": self.segment_raw["fps"],
                "clip_text": self.segment_raw["image_description"][abs_idx] if "image_description" in self.segment_raw else None,
                "action_dict": action_dict,
            }
            infos_all.append(info_t)
            rel_idx += 1

        step_meta = {
            "action_dict": action_dict,
            "future_ego_plan": future_ego_plan,
            "commit_ego_transform": future_ego_plan[0],
        }

        return infos_all, step_meta

    # -------------------------
    # cond pack
    # -------------------------
    def _load_crossview_mask_from_cfg(self):
        stub = self.dataset_cfg.get("stub_key_data_dict", None)
        if not isinstance(stub, dict):
            return None

        item = stub.get("crossview_mask", None)
        if item is None:
            return None

        if not isinstance(item, list) or len(item) < 2:
            return None

        payload = item[1]
        if not isinstance(payload, dict):
            return None

        data = payload.get("data", None)
        if not isinstance(data, dict):
            return None

        if data.get("_class_name", "") != "json.loads":
            return None

        s = data.get("s", None)
        if not isinstance(s, str):
            return None

        return torch.tensor(json.loads(s), dtype=torch.bool)

    def build_cond_from_infos(self, infos_all: list[dict[str, Any]]):
        hist_cam_list = [x["images"] for x in infos_all[:self.history_len]]

        cond_pack = {
            "fps": float(self.fps),
            "pts": torch.stack([torch.as_tensor(x["pts"]) for x in infos_all], dim=0),
            "camera_intrinsics": torch.stack([torch.as_tensor(x["camera_intrinsics"]) for x in infos_all], dim=0),
            "camera_transforms": torch.stack([torch.as_tensor(x["camera_transforms"]) for x in infos_all], dim=0),
            "ego_transforms": torch.stack([torch.as_tensor(x["ego_transforms"]) for x in infos_all], dim=0),
            "3dbox_images": [x["3dbox_images"] for x in infos_all],
            "hdmap_images": [x["hdmap_images"] for x in infos_all],
            "sample_annotation": [x["sample_annotation"] for x in infos_all],
            "segment_samples": [x["segment_sample_token"] for x in infos_all],
            "_infos_all": infos_all,
            "_action_dict": infos_all[self.history_len - 1]["action_dict"],
        }

        if infos_all[0]["image_size"] is not None:
            cond_pack["image_size"] = torch.stack([torch.as_tensor(x["image_size"]) for x in infos_all], dim=0)

        if infos_all[0]["clip_text"] is not None:
            cond_pack["clip_text"] = [x["clip_text"] for x in infos_all]

        if self.crossview_mask is not None:
            cond_pack["crossview_mask"] = self.crossview_mask

        return cond_pack, hist_cam_list

    # -------------------------
    # commit generated frame
    # -------------------------
    def commit_generated_frame(
        self,
        next_views: list[Image.Image],
        command: str,
        commit_ego_transform,
    ):
        if len(next_views) != len(self.sensor_channels):
            raise ValueError(
                f"生成视角数不对: got={len(next_views)}, expect={len(self.sensor_channels)}"
            )

        frame_idx = self.buffer.next_target_index()
        self.buffer.commit_generated(
            frame_idx=frame_idx,
            images=next_views,
            ego_transform=commit_ego_transform,
        )

        cond_indices = self.buffer.try_cond_indices()
        if cond_indices is None:
            cond_text = "END"
        else:
            cond_text = f"{cond_indices[0]}->{cond_indices[-1]}"

        tile_size = to_pil_rgb(next_views[0]).size

        cond_views = None
        box_views, hdmap_views = self._render_cond_images_for_frame(
            frame_idx=frame_idx,
            ego_transform=commit_ego_transform,
        )

        cond_views = None
        if box_views is not None and hdmap_views is not None:
            cond_views = fuse_cond_views(hdmap_views=hdmap_views, box_views=box_views)
            
        image = compose_debug_panel(
            center_views=next_views,
            cond_views=cond_views,
            cols=3,
            tile_size=tile_size,
        )

        status = (
            f"执行完成 | segment={self.current_segment_id} | "
            f"command={self.parse_command(command)['action']} | "
            f"generated_frame={frame_idx}\n"
            f"history={self.buffer.history_indices()[0]}->{self.buffer.history_indices()[-1]} | "
            f"cond={cond_text}"
        )
        return image, status