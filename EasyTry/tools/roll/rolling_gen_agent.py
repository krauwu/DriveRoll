from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage


class RollingGenAgent:
    """
    输入：
      - cond_pack: time-first 条件，总长度 = 2 * window_len
      - hist_cam_list: len = history_len，每个元素是 [V] PIL.Image
      - gen_state: 上一步流式状态（保留 first-roll latent reset 策略）

    输出：
      - next_views: [V] PIL.Image
      - new_state:
          {
              "first_roll_latent": ...,
              "frozen_ref_count": ...
          }
    """

    def __init__(self, cfg_path: str | Path, output_path: str = "./gen_runtime_output"):
        self.cfg_path = str(cfg_path)
        self.output_path = str(output_path)

        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = torch.device(
            self.cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )

        self.pipeline = None
        self.history_len = None
        self.window_len = None

        self.img_h = None
        self.img_w = None
        self._vae_buf = None

        self.image_latents = None
        self.dataset_ref_left = 0
        self.latent_shape = None
        self.step_state = None
        self.target_hw = self._load_target_hw_from_cfg()

    def _load_target_hw_from_cfg(self):
        transform_list = self.cfg.get("validation_dataset", {}).get("transform_list", [])

        for tr in transform_list:
            new_key = tr.get("new_key", "")
            if new_key not in ("3dbox_images", "hdmap_images", "vae_images"):
                continue

            transform = tr.get("transform", {})
            sub = transform.get("transforms", [])
            for t in sub:
                class_name = t.get("_class_name", "")
                if class_name.endswith("Resize"):
                    size = t.get("size", None)
                    if isinstance(size, list) and len(size) == 2:
                        return int(size[0]), int(size[1])

        return 256, 448

    # -------------------------
    # pipeline
    # -------------------------
    def _build_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        import dwm.common

        pipe_cfg = copy.deepcopy(self.cfg)
        pipe_cfg["global_state"] = {}

        common_cfg = pipe_cfg.get("pipeline", {}).get("common_config", {})
        common_cfg["distribution_framework"] = "none"
        common_cfg.pop("ddp_wrapper_settings", None)
        common_cfg.pop("t5_fsdp_wrapper_settings", None)

        pipe_cfg["pipeline"].pop("metrics", None)

        self.pipeline = dwm.common.create_instance_from_config(
            pipe_cfg["pipeline"],
            output_path=self.output_path,
            config=pipe_cfg,
            device=self.device,
        )

        self.history_len = int(self.pipeline.inference_config.get("reference_frame_count", 1))
        self.window_len = int(self.pipeline.inference_config.get("sequence_length_per_iteration", 8))

        seed = self.pipeline.config.get("generator_seed", None) if hasattr(self.pipeline, "config") else None
        if seed is not None:
            self.pipeline.generator.manual_seed(int(seed))

        self.pipeline.model.eval()
        for m in [self.pipeline.vae, self.pipeline.model]:
            if m is not None:
                m.requires_grad_(False)

        return self.pipeline

    def reset(self):
        self.image_latents = None
        self.dataset_ref_left = 0
        self.latent_shape = None
        self.step_state = None
        self.img_h = None
        self.img_w = None
        self._vae_buf = None

    # -------------------------
    # tensor helpers
    # -------------------------
    def _nested_to_tensor(self, x, name="cond"):
        if torch.is_tensor(x):
            return x

        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)

        if isinstance(x, PILImage.Image):
            arr = np.asarray(x.convert("RGB"), dtype=np.uint8)
            return torch.from_numpy(arr)

        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError(f"{name} is empty list")

            if isinstance(x[0], list):
                stacked = [self._nested_to_tensor(e, name=name) for e in x]
                return torch.stack(stacked, dim=0)

            stacked = []
            i = 0
            while i < len(x):
                e = x[i]
                if torch.is_tensor(e):
                    stacked.append(e)
                elif isinstance(e, np.ndarray):
                    stacked.append(torch.from_numpy(e))
                elif isinstance(e, PILImage.Image):
                    arr = np.asarray(e.convert("RGB"), dtype=np.uint8)
                    stacked.append(torch.from_numpy(arr))
                else:
                    raise TypeError(f"{name} element type unsupported: {type(e)}")
                i += 1

            return torch.stack(stacked, dim=0)

        raise TypeError(f"{name} type unsupported: {type(x)}")

    def _ensure_chw(self, t, name="cond"):
        if t.ndim < 3:
            raise ValueError(f"{name} ndim too small: {t.ndim}, shape={tuple(t.shape)}")

        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
            if t.max() > 1.5:
                t = t / 255.0

        if t.shape[-1] == 3 and t.shape[-3] != 3:
            perm = list(range(t.ndim))
            perm = perm[:-3] + [perm[-1], perm[-3], perm[-2]]
            t = t.permute(*perm).contiguous()

        if t.shape[-3] != 3:
            raise ValueError(f"{name} channel dim != 3, got shape={tuple(t.shape)}")

        return t

    def _as_btvc3hw(self, x, B, T, V, H, W, name="cond_img"):
        t = self._nested_to_tensor(x, name=name)
        t = self._ensure_chw(t, name=name)

        if t.ndim == 3:
            t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif t.ndim == 4:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 5:
            t = t.unsqueeze(0)
        elif t.ndim == 6:
            pass
        else:
            raise ValueError(f"{name} ndim invalid: {t.ndim}, shape={tuple(t.shape)}")

        if t.shape[0] != B:
            if not (t.shape[0] == 1 and B == 1):
                raise ValueError(f"{name} B mismatch: got {t.shape[0]} expect {B}")

        if t.shape[1] < T:
            pad = t[:, -1:].repeat(1, T - t.shape[1], 1, 1, 1, 1)
            t = torch.cat([t, pad], dim=1)
        elif t.shape[1] > T:
            t = t[:, :T]

        if t.shape[2] < V:
            pad = t[:, :, -1:].repeat(1, 1, V - t.shape[2], 1, 1, 1)
            t = torch.cat([t, pad], dim=2)
        elif t.shape[2] > V:
            t = t[:, :, :V]

        if t.shape[-2] != H or t.shape[-1] != W:
            flat = t.flatten(0, 2)
            flat = F.interpolate(flat, size=(H, W), mode="bilinear", align_corners=False)
            t = flat.unflatten(0, (B, T, V))

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        return t.to(device=self.device, dtype=dtype)

    def _pil_views_to_tensor_vchw(self, cam_views, out_h, out_w):
        out = []
        i = 0
        while i < len(cam_views):
            im = cam_views[i]
            if im.size != (out_w, out_h):
                im = im.resize((out_w, out_h), PILImage.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            out.append(arr)
            i += 1

        x = np.stack(out, axis=0)
        return torch.from_numpy(x)

    def _tensor_vchw_to_pil_views(self, x_vchw):
        if isinstance(x_vchw, torch.Tensor):
            x = x_vchw.detach().cpu()
            if x.dtype != torch.uint8:
                x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
            x = x.numpy()
        else:
            x = x_vchw

        views = []
        i = 0
        while i < x.shape[0]:
            chw = x[i]
            hwc = np.transpose(chw, (1, 2, 0))
            views.append(PILImage.fromarray(hwc, mode="RGB"))
            i += 1

        return views

    # -------------------------
    # cond cache
    # -------------------------
    def _infer_hw_from_cond(self, cond_pack):
        return self.target_hw

    def _prepare_cond_cache(self, cond_pack):
        B = 1
        T_total = int(cond_pack["camera_intrinsics"].shape[0])
        V = int(cond_pack["camera_intrinsics"].shape[1])

        if self.img_h is None or self.img_w is None:
            self.img_h, self.img_w = self._infer_hw_from_cond(cond_pack)

        H = int(self.img_h)
        W = int(self.img_w)

        camK_cpu = torch.as_tensor(cond_pack["camera_intrinsics"], dtype=torch.float32).unsqueeze(0)
        camT_cpu = torch.as_tensor(cond_pack["camera_transforms"], dtype=torch.float32).unsqueeze(0)
        egoT_cpu = torch.as_tensor(cond_pack["ego_transforms"], dtype=torch.float32).unsqueeze(0)

        fps_cpu = torch.tensor([float(cond_pack.get("fps", 1.0))], dtype=torch.float32)
        pts_cpu = torch.as_tensor(cond_pack["pts"], dtype=torch.float32).unsqueeze(0)

        if "image_size" in cond_pack:
            image_size_cpu = torch.as_tensor(cond_pack["image_size"], dtype=torch.long).unsqueeze(0)
        else:
            image_size_cpu = torch.zeros((B, T_total, V, 2), dtype=torch.long)
            image_size_cpu[..., 0] = 1600
            image_size_cpu[..., 1] = 900

        box_img_gpu = self._as_btvc3hw(cond_pack["3dbox_images"], B, T_total, V, H, W, name="3dbox_images")
        hd_img_gpu = self._as_btvc3hw(cond_pack["hdmap_images"], B, T_total, V, H, W, name="hdmap_images")

        I4 = torch.eye(4, dtype=torch.float32, device=self.device)
        lidarT_gpu = I4.view(1, 1, 1, 4, 4).repeat(B, T_total, 1, 1, 1)

        crossview_mask_gpu = None
        if "crossview_mask" in cond_pack and cond_pack["crossview_mask"] is not None:
            m = torch.as_tensor(cond_pack["crossview_mask"], dtype=torch.bool)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            crossview_mask_gpu = m.to(self.device)

        clip_text = cond_pack.get("clip_text", None)
        if clip_text is None:
            clip_text_TV_full = [["This is a nuscenes video clip"] * V for _ in range(T_total)]
        elif isinstance(clip_text, str):
            clip_text_TV_full = [[clip_text] * V for _ in range(T_total)]
        elif isinstance(clip_text, list) and len(clip_text) == T_total and isinstance(clip_text[0], str):
            clip_text_TV_full = [[clip_text[t]] * V for t in range(T_total)]
        elif isinstance(clip_text, list) and len(clip_text) == T_total and isinstance(clip_text[0], list):
            clip_text_TV_full = clip_text
        else:
            clip_text_TV_full = [["This is a nuscenes video clip"] * V for _ in range(T_total)]

        return {
            "T_total": T_total,
            "V": V,
            "H": H,
            "W": W,
            "camera_intrinsics_cpu": camK_cpu,
            "camera_transforms_cpu": camT_cpu,
            "ego_transforms_cpu": egoT_cpu,
            "fps_cpu": fps_cpu,
            "pts_cpu": pts_cpu,
            "image_size_cpu": image_size_cpu,
            "3dbox_images_gpu": box_img_gpu,
            "hdmap_images_gpu": hd_img_gpu,
            "lidar_transforms_gpu": lidarT_gpu,
            "crossview_mask_gpu": crossview_mask_gpu,
            "clip_text_TV_full": clip_text_TV_full,
        }

    def _randn_with_pipeline_generator(self, shape):
        gen = getattr(self.pipeline, "generator", None)

        if gen is None:
            return torch.randn(shape, device=self.device)

        gen_device = getattr(gen, "device", torch.device("cpu"))
        if not isinstance(gen_device, torch.device):
            gen_device = torch.device(gen_device)

        noise = torch.randn(shape, generator=gen, device=gen_device)

        if gen_device != self.device:
            noise = noise.to(self.device)

        return noise

    def _build_vae_images_once(self, hist_cam_list, H, W, V):
        B = 1
        T = int(self.window_len)
        shape = (B, T, V, 3, H, W)

        if self._vae_buf is None or self._vae_buf.shape != shape or self._vae_buf.device != self.device:
            self._vae_buf = torch.zeros(shape, dtype=torch.float32, device=self.device)
        else:
            self._vae_buf.zero_()

        vae = self._vae_buf

        t = 0
        while t < self.history_len:
            vae[0, t] = self._pil_views_to_tensor_vchw(hist_cam_list[t], H, W).to(self.device)
            t += 1

        return vae

    def _ensure_scheduler_and_shape(self, batch):
        self.pipeline.test_scheduler.set_timesteps(
            self.pipeline.inference_config["inference_steps"], self.device
        )

        if self.latent_shape is not None:
            return

        B, T, V = batch["vae_images"].shape[:3]
        down = 2 ** (len(self.pipeline.vae.config.down_block_types) - 1)
        latent_h = batch["vae_images"].shape[-2] // down
        latent_w = batch["vae_images"].shape[-1] // down

        latent_T = self.pipeline.get_latent_sequence_length(T)
        self.latent_shape = (
            B,
            latent_T,
            V,
            self.pipeline.vae.config.latent_channels,
            latent_h,
            latent_w,
        )

    def _build_batch_from_cache(self, cache, vae_images_win, start_idx):
        T = int(self.window_len)
        T_total = int(cache["T_total"])

        max_start = max(0, T_total - T)
        s = int(min(max(0, start_idx), max_start))

        batch = {
            "vae_images": vae_images_win,
            "3dbox_images": cache["3dbox_images_gpu"].narrow(1, s, T),
            "hdmap_images": cache["hdmap_images_gpu"].narrow(1, s, T),
            "lidar_transforms": cache["lidar_transforms_gpu"].narrow(1, s, T),
            "fps": cache["fps_cpu"],
            "pts": cache["pts_cpu"].narrow(1, s, T),
            "camera_intrinsics": cache["camera_intrinsics_cpu"].narrow(1, s, T),
            "camera_transforms": cache["camera_transforms_cpu"].narrow(1, s, T),
            "ego_transforms": cache["ego_transforms_cpu"].narrow(1, s, T),
            "image_size": cache["image_size_cpu"].narrow(1, s, T),
            "clip_text": [cache["clip_text_TV_full"][s:s + T]],
        }

        if cache["crossview_mask_gpu"] is not None:
            m = cache["crossview_mask_gpu"]
            if torch.is_tensor(m) and m.ndim >= 2 and m.shape[1] == T_total:
                batch["crossview_mask"] = m.narrow(1, s, T)
            else:
                batch["crossview_mask"] = m

        return batch

    # -------------------------
    # latent init / restore
    # -------------------------
    def _init_latents_if_needed(self, batch):
        self._ensure_scheduler_and_shape(batch)

        if self.image_latents is not None:
            return

        B, T, V = batch["vae_images"].shape[:3]

        ref = batch["vae_images"][:, :self.history_len]
        img_tensor = self.pipeline.image_processor.preprocess(
            ref.flatten(0, 2).to(self.device)
        )

        shift_factor = self.pipeline.vae.config.shift_factor or 0
        lat = self.pipeline.vae.encode(
            img_tensor.to(self.pipeline.vae.dtype)
        ).latent_dist.mode()
        lat = (lat - shift_factor) * self.pipeline.vae.config.scaling_factor
        lat = lat.unflatten(0, (B, self.history_len, V))

        self.dataset_ref_left = lat.shape[1]

        latent_T = int(self.latent_shape[1])
        if self.dataset_ref_left < latent_T:
            tail = self._randn_with_pipeline_generator(
                (B, latent_T - self.dataset_ref_left) + lat.shape[2:]
            ) * getattr(self.pipeline.test_scheduler, "init_noise_sigma", 1)
            self.image_latents = torch.cat([lat, tail], dim=1)
        else:
            self.image_latents = lat

    def _make_step_state(self, latents, frozen_ref_count: int):
        return {
            "first_roll_latent": latents.detach().clone(),
            "frozen_ref_count": int(frozen_ref_count),
        }

    def _restore_latents_from_state(self, gen_state):
        if gen_state is None:
            return False

        if "first_roll_latent" in gen_state:
            src = gen_state["first_roll_latent"]
        elif "latent" in gen_state:
            src = gen_state["latent"]
        else:
            return False

        lat = src[:, 1:].to(self.device)
        self.dataset_ref_left = int(gen_state["frozen_ref_count"])

        noise = self._randn_with_pipeline_generator(
            (lat.shape[0], 1) + lat.shape[2:]
        ) * getattr(self.pipeline.test_scheduler, "init_noise_sigma", 1)

        self.image_latents = torch.cat([lat, noise], dim=1)
        self.step_state = gen_state
        return True

    # -------------------------
    # main generate
    # -------------------------
    @torch.no_grad()
    def generate_next_views(self, cond_pack, hist_cam_list, gen_state=None):
        self._build_pipeline()

        if len(hist_cam_list) != self.history_len:
            raise ValueError(
                f"hist_cam_list len mismatch: got={len(hist_cam_list)}, expect={self.history_len}"
            )

        if self.history_len > self.window_len:
            raise ValueError(
                f"history_len > window_len is not supported: "
                f"history_len={self.history_len}, window_len={self.window_len}"
            )

        infer_ctx = torch.inference_mode()
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with infer_ctx:
            with amp_ctx:
                T_steps = int(self.pipeline.inference_config["inference_steps"])
                clear_ref = int(
                    self.pipeline.inference_config.get("clear_reference_frame_count", 0)
                )

                cache = self._prepare_cond_cache(cond_pack)
                H = int(cache["H"])
                W = int(cache["W"])
                V = int(cache["V"])

                vae_images_win = self._build_vae_images_once(hist_cam_list, H, W, V)
                batch0 = self._build_batch_from_cache(cache, vae_images_win, start_idx=0)

                self._ensure_scheduler_and_shape(batch0)

                restored = self._restore_latents_from_state(gen_state)
                if not restored:
                    self._init_latents_if_needed(batch0)

                latent_T = int(self.latent_shape[1])
                steps_per_inf = T_steps // (latent_T - clear_ref)
                num_roll = latent_T - clear_ref

                win_len = int(self.window_len)
                big_len = int(win_len * 2)

                T_total = int(cache["T_total"])
                base0 = max(0, T_total - big_len)
                max_start_abs = max(0, T_total - win_len)

                out = None
                roll_i = 0

                while roll_i < num_roll:
                    start_idx = base0 + roll_i
                    if start_idx > max_start_abs:
                        start_idx = max_start_abs

                    batch = self._build_batch_from_cache(
                        cache=cache,
                        vae_images_win=vae_images_win,
                        start_idx=start_idx,
                    )

                    out = self.pipeline.inference_pipeline(
                        self.latent_shape,
                        batch,
                        output_type="pt",
                        image_latents=self.image_latents,
                        start_timestep=(T_steps - steps_per_inf),
                        stop_timestep=T_steps,
                        frozen_ref_count=self.dataset_ref_left,
                    )

                    if self.dataset_ref_left > clear_ref:
                        self.dataset_ref_left -= 1

                    if roll_i == 0:
                        self.step_state = self._make_step_state(
                            latents=out["latents"],
                            frozen_ref_count=self.dataset_ref_left,
                        )

                    self.image_latents = out["latents"].detach()

                    noise = self._randn_with_pipeline_generator(
                        (self.latent_shape[0], 1) + self.image_latents.shape[2:]
                    ) * getattr(self.pipeline.test_scheduler, "init_noise_sigma", 1)

                    self.image_latents[:, :-1].copy_(self.image_latents[:, 1:].clone())
                    self.image_latents[:, -1:].copy_(noise)

                    roll_i += 1

                out_images = out["images"] if out is not None else None
                if torch.is_tensor(out_images) and out_images.ndim == 5 and out_images.shape[0] == 1:
                    out_images = out_images[0]

                if torch.is_tensor(out_images) and out_images.ndim == 4 and out_images.shape[0] >= V:
                    if out_images.shape[0] == V:
                        return self._tensor_vchw_to_pil_views(out_images), self.step_state
                    return self._tensor_vchw_to_pil_views(out_images[:V]), self.step_state

                return None, self.step_state