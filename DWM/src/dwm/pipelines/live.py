import diffusers
import dwm.functional
import dwm.utils.preview
import einops
import os
import time
import torch
import torchvision

from dwm.pipelines.dfot import CrossviewTemporalSD as _BaseCrossviewTemporalSD


def _safe_dfot_get_action_ids(
    batch, common_config, action_condition_mask=None,
    streaming_mode=False, prev_ego_transforms=None
):
    if streaming_mode:
        assert batch["ego_transforms"].shape[1] == 1
        if prev_ego_transforms is None:
            ego_transforms = batch["ego_transforms"]
        else:
            ego_transforms = prev_ego_transforms
        ego_transforms = torch.cat([ego_transforms, batch["ego_transforms"]], dim=1)
    else:
        ego_transforms = batch["ego_transforms"]

    current_pose = ego_transforms[:, :, common_config["camera_ego_sensor_indices"]]
    pose_device = current_pose.device
    pose_dtype = current_pose.dtype

    uncondition_pose = torch.eye(
        4, device=pose_device, dtype=pose_dtype
    ).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    pose_is_valid = (current_pose - uncondition_pose).sum((1, 2, 3, 4)).abs() > 1e-3
    if action_condition_mask is None:
        is_conditioned = pose_is_valid
    else:
        action_condition_mask = action_condition_mask.to(device=pose_device)
        is_conditioned = torch.logical_and(pose_is_valid, action_condition_mask)

    relative_pose = torch.linalg.solve(current_pose[:, :-1], current_pose[:, 1:])
    relative_pose = torch.cat([relative_pose[:, :1], relative_pose], dim=1)

    fps = batch["fps"].to(device=pose_device, dtype=pose_dtype)
    moving_distance = torch.norm(relative_pose[..., :3, 3], dim=-1, keepdim=True)
    speed = 3.6 * moving_distance * fps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    rotation_angles = torch.atan2(
        relative_pose[..., 1, 0:1] - relative_pose[..., 0, 1:2],
        relative_pose[..., 0, 0:1] + relative_pose[..., 1, 1:2])
    steering = torch.where(
        torch.abs(moving_distance) > 0.01,
        rotation_angles / moving_distance * 2.7 * 14,
        -1000.0 * torch.ones_like(rotation_angles))

    action_ids = torch.cat([speed, steering], dim=-1)
    action_ids = torch.where(
        is_conditioned.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        action_ids,
        -1000.0 * torch.ones_like(action_ids))
    if streaming_mode:
        action_ids = action_ids.chunk(2, dim=1)[-1]
    return action_ids


_BaseCrossviewTemporalSD.get_action_ids = staticmethod(_safe_dfot_get_action_ids)


def _vae_encode_sample_scaled(block, tensor):
    shift_factor = block.config.shift_factor if block.config.shift_factor is not None else 0
    latent_dist = block.encode(tensor).latent_dist
    latent = latent_dist.sample()
    latent = latent - shift_factor
    latent = latent * block.config.scaling_factor
    return latent


def _vae_encode_mode_scaled(block, tensor):
    shift_factor = block.config.shift_factor if block.config.shift_factor is not None else 0
    latent_dist = block.encode(tensor).latent_dist
    latent = latent_dist.mode()
    latent = latent - shift_factor
    latent = latent * block.config.scaling_factor
    return latent


def _vae_decode_scaled(block, tensor):
    shift_factor = block.config.shift_factor if block.config.shift_factor is not None else 0
    tensor = tensor / block.config.scaling_factor
    tensor = tensor + shift_factor
    image = block.decode(tensor, return_dict=False)[0]
    return image


class CrossviewTemporalSD(_BaseCrossviewTemporalSD):
    """DFoT-style LIVE post-training pipe.

    Forward rollout distribution:
        clean history frames + synchronized noisy future frames.

    Default training layout:
        total frames = 20
        forward windows = 0:8 -> 8:12, 4:12 -> 12:16, 8:16 -> 16:20
        reverse windows = 12:20 -> 8:12, 8:16 -> 4:8, 4:12 -> 0:4

    This class intentionally removes the old 50/50 LIVE/DF mixture.
    train_step always runs post-training recovery.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_config = self.config.get("live_training", {})
        if self.should_save:
            print("[DFoT-LIVE] posttrain-only subclass over dwm.pipelines.dfot enabled")
            print("[DFoT-LIVE] train_total_frames=", self.live_config.get("train_total_frames", 20))
            print("[DFoT-LIVE] forward_history=", self.live_config.get("forward_history", 8))
            print("[DFoT-LIVE] forward_future=", self.live_config.get("forward_future", self.inference_config.get("window_stride", self.inference_config.get("generation_stride", 4))))
            print("[DFoT-LIVE] reverse_steps=", self.live_config.get("reverse_steps", 3))

    def _randn_tensor(self, shape, dtype=None):
        tensor_dtype = self.model_dtype if dtype is None else dtype
        try:
            value = torch.randn(
                shape,
                generator=self.generator,
                device=self.device,
                dtype=tensor_dtype)
        except (RuntimeError, TypeError):
            value = torch.randn(
                shape,
                generator=self.generator,
                dtype=tensor_dtype).to(self.device)
        return value

    def _take_batch_window(self, batch, start, stop, reverse=False):
        source_sequence_length = batch["vae_images"].shape[1]
        target_sequence_length = stop - start
        batch_size = batch["vae_images"].shape[0]
        exception_keys = self.inference_config.get(
            "autoregression_data_exception_for_take_sequence", [])
        sequence_keys = {
            "vae_images",
            "3dbox_images",
            "hdmap_images",
            "ego_transforms",
            "camera_transforms",
            "camera_intrinsics",
            "image_size",
            "pts",
            "lidar_points",
            "lidar_transforms",
            "is_uncalibrated",
        }
        result = {}
        for key, value in batch.items():
            if key in exception_keys:
                result[key] = value
            elif torch.is_tensor(value) and value.ndim >= 2:
                should_slice = False
                if key in sequence_keys and value.shape[1] >= stop:
                    should_slice = True
                if value.shape[1] == source_sequence_length:
                    should_slice = True
                if should_slice:
                    sliced = value[:, start:stop]
                    if reverse:
                        sliced = torch.flip(sliced, dims=[1])
                    result[key] = sliced
                else:
                    result[key] = value
            elif key == "clip_text" and isinstance(value, list):
                if len(value) == source_sequence_length:
                    sliced = value[start:stop]
                    if reverse:
                        sliced = list(reversed(sliced))
                    result[key] = sliced
                elif len(value) == batch_size:
                    sliced_batch = []
                    for item in value:
                        if isinstance(item, list) and len(item) == source_sequence_length:
                            sliced_item = item[start:stop]
                            if reverse:
                                sliced_item = list(reversed(sliced_item))
                            sliced_batch.append(sliced_item)
                        else:
                            sliced_batch.append(item)
                    result[key] = sliced_batch
                else:
                    result[key] = value
            else:
                result[key] = value
        if "pts" in result and hasattr(result["pts"], "shape"):
            if result["pts"].shape[1] != target_sequence_length:
                raise RuntimeError(
                    "window slicing failed for pts: "
                    f"expected {target_sequence_length}, got {result['pts'].shape[1]}, "
                    f"window=({start}, {stop})")
        if result["vae_images"].shape[1] != target_sequence_length:
            raise RuntimeError(
                "window slicing failed for vae_images: "
                f"expected {target_sequence_length}, got {result['vae_images'].shape[1]}, "
                f"window=({start}, {stop})")
        return result

    @torch.no_grad()
    def _encode_vae_latents(self, vae_images, use_mode=False):
        batch_size, sequence_length, view_count = vae_images.shape[:3]
        image_tensor = self.image_processor.preprocess(
            vae_images.flatten(0, 2).to(self.device))
        if self.is_temporal_vae:
            image_tensor = einops.rearrange(
                image_tensor,
                "(b t v) c h w -> (b v) c t h w",
                t=sequence_length,
                v=view_count)
        encode_call = _vae_encode_mode_scaled if use_mode else _vae_encode_sample_scaled
        latents = dwm.functional.memory_efficient_split_call(
            self.vae,
            image_tensor,
            encode_call,
            self.common_config.get("memory_efficient_batch", -1))
        if self.is_temporal_vae:
            latents = einops.rearrange(
                latents,
                "(b v) c t h w -> b t v c h w",
                v=view_count)
        else:
            latents = einops.rearrange(
                latents,
                "(b t v) c h w -> b t v c h w",
                t=sequence_length,
                v=view_count)
        return latents

    @torch.no_grad()
    def _decode_latents_to_images(self, latents, output_type):
        if self.is_temporal_vae:
            view_count = latents.shape[2]
            decode_latents = einops.rearrange(
                latents,
                "b t v c h w -> (b v) c t h w")
        else:
            view_count = None
            decode_latents = latents.flatten(0, 2)
        image_tensor = dwm.functional.memory_efficient_split_call(
            self.vae,
            decode_latents.to(dtype=self.vae.dtype),
            _vae_decode_scaled,
            self.common_config.get("memory_efficient_batch", -1))
        if self.is_temporal_vae:
            image_tensor = einops.rearrange(
                image_tensor,
                "(b v) c t h w -> (b t v) c h w",
                v=view_count)
        images = self.image_processor.postprocess(
            image_tensor,
            output_type=output_type)
        return images

    def _sample_condition_masks(self, batch_size):
        text_ratio = self.training_config.get("text_prompt_condition_ratio", 1.0)
        box_ratio = self.training_config.get("3dbox_condition_ratio", 1.0)
        hdmap_ratio = self.training_config.get("hdmap_condition_ratio", 1.0)
        action_ratio = self.training_config.get(
            "action_condition_ratio",
            self.training_config.get("action_condition_mask", 1.0))
        text_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) < text_ratio
        ).tolist()
        box_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) < box_ratio
        ).to(self.device)
        hdmap_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) < hdmap_ratio
        ).to(self.device)
        action_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) < action_ratio
        ).to(self.device)
        if self.common_config.get("explicit_view_modeling", False):
            explicit_ratio = self.training_config.get("explicit_view_modeling_ratio", 1.0)
            explicit_view_modeling_mask = (
                torch.rand((batch_size,), generator=self.generator) < explicit_ratio
            ).to(self.device)
        else:
            explicit_view_modeling_mask = None
        return (
            text_condition_mask,
            box_condition_mask,
            hdmap_condition_mask,
            action_condition_mask,
            explicit_view_modeling_mask)

    def _make_model_conditions(self, batch, latent_shape, latents_shape,
                               do_classifier_free_guidance=False,
                               text_condition_mask=None,
                               box_condition_mask=None,
                               hdmap_condition_mask=None,
                               action_condition_mask=None,
                               explicit_view_modeling_mask=None):
        if "pts" in batch and hasattr(batch["pts"], "shape"):
            if batch["pts"].shape[1] != latent_shape[1]:
                raise RuntimeError(
                    "condition length and latent window length mismatch: "
                    f"pts={batch['pts'].shape[1]}, latent={latent_shape[1]}. "
                    "This usually means the window batch was not sliced before get_conditions.")
        if "vae_images" in batch and batch["vae_images"].shape[1] != latent_shape[1]:
            raise RuntimeError(
                "vae_images length and latent window length mismatch: "
                f"vae_images={batch['vae_images'].shape[1]}, latent={latent_shape[1]}.")
        condition_tensor_keys = {
            "3dbox_images",
            "hdmap_images",
            "ego_transforms",
            "camera_transforms",
            "camera_intrinsics",
            "image_size",
            "fps",
            "crossview_mask",
            "is_uncalibrated",
        }
        condition_batch = {}
        for key, value in batch.items():
            if key in condition_tensor_keys and torch.is_tensor(value):
                condition_batch[key] = value.to(self.device)
            else:
                condition_batch[key] = value

        text_encoder = self.text_encoders if isinstance(
            self.model, diffusers.SD3Transformer2DModel) else self.text_encoder
        tokenizer = self.tokenizers if isinstance(
            self.model, diffusers.SD3Transformer2DModel) else self.tokenizer
        model_conditions = CrossviewTemporalSD.get_conditions(
            self.model,
            text_encoder,
            tokenizer,
            self.common_config,
            latent_shape,
            condition_batch,
            self.device,
            self.model_dtype,
            text_condition_mask,
            box_condition_mask,
            hdmap_condition_mask,
            action_condition_mask,
            explicit_view_modeling_mask,
            do_classifier_free_guidance=do_classifier_free_guidance,
            latents_shape=latents_shape)
        return model_conditions

    @torch.no_grad()
    def dfot_window_denoise_latents(self, window_latents, window_batch, history_length):
        self.model_wrapper.eval()
        latents = window_latents.to(device=self.device, dtype=self.model_dtype)
        latent_shape = tuple(latents.shape)
        do_classifier_free_guidance = "guidance_scale" in self.inference_config
        guidance_scale = self.inference_config.get("guidance_scale", 1.0)
        self.test_scheduler.set_timesteps(
            self.inference_config["inference_steps"],
            self.device)
        model_conditions = self._make_model_conditions(
            window_batch,
            latent_shape,
            latents.shape,
            do_classifier_free_guidance=do_classifier_free_guidance)
        history_mask = torch.arange(
            latents.shape[1],
            device=self.device) < history_length
        history_mask_latent = history_mask.view(1, -1, 1, 1, 1, 1)
        for t in self.test_scheduler.timesteps:
            timesteps = torch.full(
                latents.shape[:3],
                t,
                dtype=t.dtype,
                device=self.device)
            timesteps[:, :history_length] = torch.zeros(
                (),
                dtype=t.dtype,
                device=self.device)
            latent_model_input = latents.to(dtype=self.model_dtype)
            if hasattr(self.test_scheduler, "scale_model_input"):
                latent_model_input = self.test_scheduler.scale_model_input(
                    latent_model_input,
                    timesteps).to(dtype=self.model_dtype)
            if do_classifier_free_guidance:
                latent_model_input = torch.cat(
                    [latent_model_input, latent_model_input], dim=0)
                timesteps_input = torch.cat([timesteps, timesteps], dim=0)
            else:
                timesteps_input = timesteps
            with self.get_autocast_context():
                model_output, _, _ = self.model_wrapper(
                    latent_model_input,
                    timesteps_input,
                    **model_conditions)
            noise_pred = model_output[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond)
            staging_latents = self.test_scheduler.step(
                noise_pred,
                t,
                latents).prev_sample
            latents = torch.where(
                history_mask_latent,
                latents,
                staging_latents)
        return latents

    @torch.no_grad()
    def _dfot_forward_rollout_latents(self, batch, gt_latents, total_frames):
        forward_history = self.live_config.get("forward_history", 8)
        forward_future = self.live_config.get("forward_future", self.inference_config.get("window_stride", self.inference_config.get("generation_stride", 4)))
        generated_latents = gt_latents[:, :forward_history].detach()
        while generated_latents.shape[1] < total_frames:
            current_frame = generated_latents.shape[1]
            history_length = min(forward_history, generated_latents.shape[1])
            future_length = min(forward_future, total_frames - current_frame)
            condition_start = current_frame - history_length
            condition_stop = current_frame + future_length
            window_batch = self._take_batch_window(
                batch,
                condition_start,
                condition_stop,
                reverse=False)
            history_latents = generated_latents[:, -history_length:]
            future_shape = (gt_latents.shape[0], future_length) + tuple(gt_latents.shape[2:])
            future_noise = self._randn_tensor(
                future_shape,
                dtype=self.model_dtype)
            future_noise = future_noise * getattr(
                self.test_scheduler,
                "init_noise_sigma",
                1)
            window_latents = torch.cat(
                [history_latents.to(dtype=self.model_dtype), future_noise],
                dim=1)
            denoised_window = self.dfot_window_denoise_latents(
                window_latents,
                window_batch,
                history_length)
            new_latents = denoised_window[:, history_length:history_length + future_length]
            generated_latents = torch.cat(
                [generated_latents, new_latents.detach()],
                dim=1)
        return generated_latents[:, :total_frames]

    def _make_reverse_noisy_input(self, reference_latents, target_latents):
        batch_size, target_length, view_count = target_latents.shape[:3]
        u = CrossviewTemporalSD.sd3_compute_density_for_timestep_sampling(
            weighting_scheme=self.training_config.get("weighting_scheme", "logit_normal"),
            size=(batch_size, 1, 1),
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29)
        timestep_indices = (
            u * self.train_scheduler.config.num_train_timesteps
        ).long()
        timestep_indices = timestep_indices.repeat(
            1,
            target_length,
            view_count)
        target_timesteps = self.train_scheduler.timesteps[timestep_indices].to(self.device)
        sigmas = CrossviewTemporalSD.sd3_get_sigmas(
            self.train_scheduler,
            timestep_indices,
            n_dim=target_latents.ndim,
            dtype=target_latents.dtype,
            device=target_latents.device)
        noise = self._randn_tensor(
            target_latents.shape,
            dtype=target_latents.dtype)
        noisy_target = sigmas * noise + (1.0 - sigmas) * target_latents
        clean_timesteps = torch.zeros(
            reference_latents.shape[:3],
            dtype=target_timesteps.dtype,
            device=self.device)
        timesteps = torch.cat(
            [clean_timesteps, target_timesteps],
            dim=1)
        noisy_input = torch.cat(
            [reference_latents, noisy_target],
            dim=1)
        return noisy_input, timesteps, sigmas, noisy_target

    def _reverse_live_loss(self, batch, gt_latents, forward_latents):
        reverse_steps = self.live_config.get("reverse_steps", 3)
        reverse_history = self.live_config.get("reverse_history", 8)
        reverse_future = self.live_config.get("reverse_future", self.inference_config.get("window_stride", self.inference_config.get("generation_stride", 4)))
        total_frames = self.live_config.get("train_total_frames", 20)
        working_latents = forward_latents.detach().clone()
        batch_size = gt_latents.shape[0]
        losses = []
        loss_report = {}
        for reverse_id in range(reverse_steps):
            target_start = total_frames - reverse_history - reverse_future * (reverse_id + 1)
            target_stop = target_start + reverse_future
            reference_start = target_stop
            reference_stop = reference_start + reverse_history
            if target_start < 0:
                raise ValueError("reverse_steps/reverse_history/reverse_future exceed train_total_frames")
            reverse_batch = self._take_batch_window(
                batch,
                target_start,
                reference_stop,
                reverse=True)
            reference_latents = torch.flip(
                working_latents[:, reference_start:reference_stop],
                dims=[1]).to(self.device)
            target_latents = torch.flip(
                gt_latents[:, target_start:target_stop],
                dims=[1]).to(self.device)
            noisy_input, timesteps, sigmas, noisy_target = self._make_reverse_noisy_input(
                reference_latents,
                target_latents)
            condition_masks = self._sample_condition_masks(batch_size)
            model_conditions = self._make_model_conditions(
                reverse_batch,
                noisy_input.shape,
                noisy_input.shape,
                do_classifier_free_guidance=False,
                text_condition_mask=condition_masks[0],
                box_condition_mask=condition_masks[1],
                hdmap_condition_mask=condition_masks[2],
                action_condition_mask=condition_masks[3],
                explicit_view_modeling_mask=condition_masks[4])
            with self.get_autocast_context():
                sd_pred, _, _ = self.model_wrapper(
                    noisy_input.to(dtype=self.model_dtype),
                    timesteps,
                    **model_conditions)
            pred_target = sd_pred[0][:, reverse_history:reverse_history + reverse_future]
            pred_target_latents = pred_target * (-sigmas) + noisy_target
            step_loss = torch.nn.functional.mse_loss(
                pred_target_latents.float(),
                target_latents.float(),
                reduction="mean")
            losses.append(step_loss)
            loss_report[f"rev_{reverse_id}_loss"] = step_loss.item()
            predicted_chrono = torch.flip(
                pred_target_latents.detach(),
                dims=[1])
            working_latents[:, target_start:target_stop] = predicted_chrono
        loss = torch.stack(losses).mean() * self.get_loss_coef("live")
        return loss, loss_report

    def _backward_and_step(self, loss, global_step):
        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        should_optimize = (
            "gradient_accumulation_steps" not in self.training_config or
            (global_step + 1) % self.training_config["gradient_accumulation_steps"] == 0)
        if should_optimize:
            if "max_norm_for_grad_clip" in self.training_config:
                if self.training_config.get("enable_grad_scaler", False):
                    self.grad_scaler.unscale_(self.optimizer)
                if torch.distributed.is_initialized() and self.distribution_framework == "fsdp":
                    self.model_wrapper.clip_grad_norm_(
                        self.training_config["max_norm_for_grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config["max_norm_for_grad_clip"])
            if self.training_config.get("enable_grad_scaler", False):
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train_step(self, batch, global_step):
        self.model_wrapper.train()
        t0 = time.time()
        if not isinstance(self.model, diffusers.SD3Transformer2DModel):
            raise NotImplementedError("DFoT-LIVE posttrain pipe currently targets SD3/FlowMatch models.")
        train_total_frames = self.live_config.get("train_total_frames", 20)
        sequence_length = batch["vae_images"].shape[1]
        if sequence_length < train_total_frames:
            raise ValueError(
                f"training sequence_length must be >= {train_total_frames}, got {sequence_length}")
        if sequence_length > train_total_frames:
            batch = self._take_batch_window(
                batch,
                0,
                train_total_frames,
                reverse=False)
        gt_latents = self._encode_vae_latents(
            batch["vae_images"],
            use_mode=False).to(self.device)
        with torch.no_grad():
            forward_latents = self._dfot_forward_rollout_latents(
                batch,
                gt_latents,
                train_total_frames)
        self.model_wrapper.train()
        loss, detail = self._reverse_live_loss(
            batch,
            gt_latents,
            forward_latents)
        loss_report = {
            "loss": loss.item(),
            "live_reverse_loss": loss.item(),
            "forward_history": float(self.live_config.get("forward_history", 8)),
            "forward_future": float(self.live_config.get("forward_future", self.inference_config.get("window_stride", self.inference_config.get("generation_stride", 4)))),
            "reverse_steps": float(self.live_config.get("reverse_steps", 3))}
        loss_report.update(detail)
        self.loss_report_list.append(loss_report)
        self._backward_and_step(loss, global_step)
        self.step_duration += time.time() - t0

    @torch.no_grad()
    def inference_pipeline(self, latent_shape, batch, output_type,
                           image_latents=None, reference_frame_count=0,
                           start_timestep=0, stop_timestep=None,
                           take_time=0, frozen_ref_count=0):
        history_length = reference_frame_count
        if history_length <= 0:
            history_length = self.inference_config.get("reference_frame_count", 8)
        if image_latents is None:
            image_latents = self._randn_tensor(latent_shape, dtype=self.model_dtype)
            image_latents = image_latents * getattr(self.test_scheduler, "init_noise_sigma", 1)
        denoised_latents = self.dfot_window_denoise_latents(
            image_latents,
            batch,
            history_length)
        images = self._decode_latents_to_images(
            denoised_latents,
            output_type)
        return {"images": images, "latents": denoised_latents}

    @torch.no_grad()
    def autoregressive_inference_pipeline(self, latent_shape, batch, output_type):
        total_frame_count = batch["pts"].shape[1] if "pts" in batch else batch["vae_images"].shape[1]
        reference_frame_count = self.inference_config.get("reference_frame_count", 8)
        generation_stride = self.inference_config.get("generation_stride", self.inference_config.get("window_stride", 4))
        prompt_length = min(reference_frame_count, total_frame_count)
        prompt_images = batch["vae_images"][:, :prompt_length]
        generated_latents = self._encode_vae_latents(
            prompt_images,
            use_mode=True).to(self.device)
        frame_indices = []
        while generated_latents.shape[1] < total_frame_count:
            current_frame = generated_latents.shape[1]
            history_length = min(reference_frame_count, generated_latents.shape[1])
            future_length = min(generation_stride, total_frame_count - current_frame)
            condition_start = current_frame - history_length
            condition_stop = current_frame + future_length
            window_batch = self._take_batch_window(
                batch,
                condition_start,
                condition_stop,
                reverse=False)
            history_latents = generated_latents[:, -history_length:]
            future_shape = (latent_shape[0], future_length) + tuple(latent_shape[2:])
            future_noise = self._randn_tensor(
                future_shape,
                dtype=self.model_dtype)
            future_noise = future_noise * getattr(self.test_scheduler, "init_noise_sigma", 1)
            window_latents = torch.cat(
                [history_latents.to(dtype=self.model_dtype), future_noise],
                dim=1)
            denoised_window = self.dfot_window_denoise_latents(
                window_latents,
                window_batch,
                history_length)
            new_latents = denoised_window[:, history_length:history_length + future_length]
            generated_latents = torch.cat(
                [generated_latents, new_latents.detach()],
                dim=1)
            frame_indices.extend(range(current_frame, current_frame + future_length))
        generated_only = generated_latents[:, prompt_length:]
        images = self._decode_latents_to_images(
            generated_only,
            output_type)
        return {
            "images": images,
            "latents": generated_latents,
            "frame_indices": frame_indices,
            "prompt_length": prompt_length}

    @torch.no_grad()
    def preview_pipeline(self, batch, output_path, global_step):
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_height = batch["vae_images"].shape[-2] // (
            2 ** (len(self.vae.config.down_block_types) - 1))
        latent_width = batch["vae_images"].shape[-1] // (
            2 ** (len(self.vae.config.down_block_types) - 1))
        latent_shape = (
            batch_size,
            self.inference_config.get("sequence_length_per_iteration", 12),
            view_count,
            self.vae.config.latent_channels,
            latent_height,
            latent_width)
        pipeline_output = self.autoregressive_inference_pipeline(
            latent_shape,
            batch,
            "pt")
        if self.should_save or (
            torch.distributed.is_initialized() and
            self.inference_config.get("all_rank_preview", False)):
            os.makedirs(os.path.join(output_path, "preview"), exist_ok=True)
            filename = str(global_step)
            preview_tensor = dwm.utils.preview.make_ctsd_preview_tensor(
                pipeline_output["images"],
                batch,
                self.inference_config)
            if sequence_length == 1:
                image_output_path = os.path.join(
                    output_path,
                    "preview",
                    f"{filename}.png")
                torchvision.transforms.functional.to_pil_image(
                    preview_tensor).save(image_output_path)
            else:
                video_output_path = os.path.join(
                    output_path,
                    "preview",
                    f"{filename}.mp4")
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path,
                    "libx264",
                    batch["fps"][0].item(),
                    preview_tensor)

    @torch.no_grad()
    def evaluate_pipeline(self, global_step, dataset_length,
                          validation_dataloader, validation_datasampler=None):
        if torch.distributed.is_initialized():
            validation_datasampler.set_epoch(0)
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        iteration_count = None
        if "evaluation_item_count" in self.inference_config:
            iteration_count = self.inference_config["evaluation_item_count"] // world_size
        for i, batch in enumerate(validation_dataloader):
            batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
            if iteration_count is not None and i * batch_size >= iteration_count:
                break
            latent_height = batch["vae_images"].shape[-2] // (
                2 ** (len(self.vae.config.down_block_types) - 1))
            latent_width = batch["vae_images"].shape[-1] // (
                2 ** (len(self.vae.config.down_block_types) - 1))
            latent_shape = (
                batch_size,
                self.inference_config.get("sequence_length_per_iteration", 12),
                view_count,
                self.vae.config.latent_channels,
                latent_height,
                latent_width)
            pipeline_output = self.autoregressive_inference_pipeline(
                latent_shape,
                batch,
                "pt")
            if len(pipeline_output["frame_indices"]) == 0:
                continue
            fake_images = pipeline_output["images"].unflatten(
                0,
                (batch_size, -1, view_count))
            frame_indices = torch.tensor(
                pipeline_output["frame_indices"],
                dtype=torch.long)
            real_images = batch["vae_images"][:, frame_indices]
            if "fid" in self.metrics:
                self.metrics["fid"].update(
                    real_images.flatten(0, 2).to(self.device),
                    real=True)
                self.metrics["fid"].update(
                    fake_images.flatten(0, 2),
                    real=False)
            if "fvd" in self.metrics:
                self.metrics["fvd"].update(
                    einops.rearrange(
                        real_images.to(self.device),
                        "b t v c h w -> (b v) t c h w"),
                    real=True)
                self.metrics["fvd"].update(
                    einops.rearrange(
                        fake_images,
                        "b t v c h w -> (b v) t c h w"),
                    real=False)
        text = f"Step {global_step},"
        for key, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            text += f" {key}: {value:.3f}"
            if self.should_save:
                self.summary.add_scalar(
                    f"evaluation/{key}",
                    value,
                    global_step)
        if self.should_save:
            print(text)
