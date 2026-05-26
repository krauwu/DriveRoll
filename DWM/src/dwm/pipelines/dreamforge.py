"""DreamForge LMA inference pipeline (single-file, inference-only).

This module consolidates the inference logic that previously lived across
three classes (`CrossviewTemporalSD`, `DreamForgeSimple`,
`DreamForgeSimpleLMA`) into one self-contained class:
:class:`DreamForgeLMAInferencePipeline`.

Scope
-----
* SD3 / DiT backbone only (``DiTCrossviewTemporalConditionModel``).
* Frame-prediction style ``ctsd`` (reference-frame injection during the
  full-sequence denoising loop).  The legacy ``diffusion_forcing`` branch
  has been removed.
* Multi-view autoregressive video generation.
* Optional Local Motion Attention (LMA) controlled by the model JSON.
* FSDP wrapping of both the diffusion model and the T5 text encoder when
  running under ``torch.distributed``.

Everything related to training (optimizer, lr scheduler, grad scaler, FID
/ FVD metrics, depth estimation, etc.) has been deliberately removed.  Use
this pipeline with :mod:`dwm.preview` for video previewing.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional, Sequence

import einops
import safetensors.torch
import torch
import torch.distributed
import torch.distributed.fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torchvision

import diffusers
import diffusers.image_processor
import transformers

import dwm.common
import dwm.functional
import dwm.utils.preview


# ---------------------------------------------------------------------------
# Free functions / helpers
# ---------------------------------------------------------------------------


def sample_reference_indices(total_frames: int, num_references: int) -> List[int]:
    """Pick ``num_references`` reference indices evenly across ``total_frames``.

    Examples
    --------
    >>> sample_reference_indices(9, 3)
    [0, 4, 8]
    >>> sample_reference_indices(17, 3)
    [0, 8, 16]
    """
    if num_references >= total_frames:
        return list(range(total_frames))
    if num_references <= 1:
        return [0]
    return [
        int(i * (total_frames - 1) / (num_references - 1))
        for i in range(num_references)
    ]


def _rank_zero() -> bool:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


def install_sanitize_hooks(
    root_module: torch.nn.Module,
    patterns: Sequence[str] = ("ff_context.net.2",),
    lo: float = -1e4,
    hi: float = 1e4,
):
    """Forward hooks that clip non-finite values during inference.

    The SD3 transformer running in FP16 can occasionally emit NaN / Inf
    activations from the joint feed-forward MLP.  The hook below replaces
    such values with finite ones to keep the diffusion loop stable.  The
    hook is a no-op when ``root_module`` is in training mode.
    """

    def _fix(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x) or torch.isfinite(x).all():
            return x
        finite = x[torch.isfinite(x)]
        mean = (
            finite.mean()
            if finite.numel()
            else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        )
        return torch.nan_to_num(x, nan=float(mean), posinf=hi, neginf=lo).clamp(lo, hi)

    def _hook(_m, _inp, out):
        if root_module.training:
            return out
        if torch.is_tensor(out):
            return _fix(out)
        if isinstance(out, (list, tuple)):
            return type(out)(_fix(o) if torch.is_tensor(o) else o for o in out)
        if isinstance(out, dict):
            return {k: (_fix(v) if torch.is_tensor(v) else v) for k, v in out.items()}
        return out

    handles, matched = [], []
    for name, module in root_module.named_modules():
        if any(p in name for p in patterns):
            handles.append(module.register_forward_hook(_hook))
            matched.append(name)

    if _rank_zero():
        print(f"[sanitize] installed forward hooks on {len(matched)} modules")
    return handles


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class DreamForgeLMAInferencePipeline:
    """Single-file SD3 multi-view video pipeline for inference.

    Parameters
    ----------
    output_path : str
        Directory used for preview artefacts and (optionally) checkpoint
        loading via ``resume_from``.
    config : dict
        The full training/inference JSON. Only a couple of top-level keys
        are read (``generator_seed``, etc.). All pipeline-specific options
        sit under ``config["pipeline"]`` and are passed in explicitly.
    device : torch.device
        Compute device (typically ``cuda:LOCAL_RANK``).
    common_config : dict
        Common knobs shared with training (FSDP wrappers, dtype hints,
        camera-id slicing, etc.).
    inference_config : dict
        Inference-time knobs (``guidance_scale``, ``inference_steps``,
        ``sequence_length_per_iteration``, ``reference_frame_count``,
        ``preview_image_size`` ...).
    pretrained_model_name_or_path : str
        Path to the SD3 base model (provides VAE, scheduler, tokenizers
        and text encoders).
    model : torch.nn.Module
        The pre-instantiated DiT backbone (e.g.
        ``DiTCrossviewTemporalConditionModel``).
    model_dtype : torch.dtype, optional
        Compute dtype for the DiT model. Defaults to ``torch.float32``.
    model_checkpoint_path : str, optional
        Path to ``.pth`` / ``.safetensors`` weights to load into ``model``.
        Overrides the JSON value when set via the CLI.
    model_load_state_args : dict, optional
        Extra kwargs forwarded to ``model.load_state_dict``.
    resume_from : int, optional
        If set, load ``output_path/checkpoints/{resume_from}.pth`` instead
        of ``model_checkpoint_path``.
    total_frames : int
        Sequence length the model was trained on (only used to derive a
        default reference-index pattern).
    num_reference_frames : int
        Number of reference frames sampled per training sequence; kept
        here for parity with the legacy ``DreamForgeSimpleLMA``.
    """

    # ---- static helpers --------------------------------------------------

    @staticmethod
    def load_state(path: str) -> dict:
        if path.endswith(".safetensors"):
            return safetensors.torch.load_file(path, device="cpu")
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def flatten_clip_text(
        clip_text,
        flat: list,
        parsed_shape: list,
        level: int = 0,
        text_condition_mask=None,
        do_classifier_free_guidance: bool = False,
    ) -> None:
        """Flatten nested text prompts and remember the unflatten shape.

        ``clip_text`` is typically a list of shape ``[B][T][V]`` where each
        innermost element is a string.  Both the flat list of strings and
        the per-level lengths are filled in-place so that the caller can
        ``unflatten`` the text embeddings after encoding.
        """
        level_count = 0
        if isinstance(clip_text, list) and len(parsed_shape) <= level:
            parsed_shape.append(0)

        if do_classifier_free_guidance:
            if isinstance(clip_text, str):
                flat.append("")
                level_count += 1
            else:
                for item in clip_text:
                    DreamForgeLMAInferencePipeline.flatten_clip_text(
                        item, flat, parsed_shape, level + 1,
                        text_condition_mask, do_classifier_free_guidance,
                    )
                    level_count += 1

        if level == 0 or not do_classifier_free_guidance:
            if isinstance(clip_text, str):
                if text_condition_mask is None or (
                    isinstance(text_condition_mask, bool) and text_condition_mask
                ):
                    flat.append(clip_text)
                else:
                    flat.append("")
                level_count += 1
            else:
                for i_id, item in enumerate(clip_text):
                    sub_mask = (
                        None
                        if text_condition_mask is None
                        else (
                            text_condition_mask[i_id]
                            if isinstance(text_condition_mask, list)
                            else text_condition_mask
                        )
                    )
                    DreamForgeLMAInferencePipeline.flatten_clip_text(
                        item, flat, parsed_shape, level + 1, sub_mask, False,
                    )
                    level_count += 1

        if isinstance(clip_text, list):
            parsed_shape[level] = level_count

    @staticmethod
    def get_camera_transform_ids(batch: dict, common_config: dict) -> torch.Tensor:
        """Concatenate selected intrinsics + extrinsics into a flat id tensor."""
        return torch.cat(
            [
                batch["camera_intrinsics"].flatten(-2, -1)[
                    ..., common_config["camera_intrinsic_embedding_indices"]
                ]
                / batch["image_size"][
                    ..., common_config["camera_intrinsic_denom_embedding_indices"]
                ],
                batch["camera_transforms"].flatten(-2, -1)[
                    ..., common_config["camera_transform_embedding_indices"]
                ],
            ],
            -1,
        )

    # ---- text encoders ---------------------------------------------------

    @staticmethod
    def _encode_with_clip(
        text_encoder, tokenizer, prompts: List[str], device,
    ):
        text_inputs = tokenizer(
            prompts, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        outputs = text_encoder(
            text_inputs.input_ids.to(device), output_hidden_states=True,
        )
        pooled = outputs[0]
        embeds = outputs.hidden_states[-2].to(
            dtype=text_encoder.dtype, device=device,
        )
        return embeds, pooled

    @staticmethod
    def _encode_with_t5(
        text_encoder, tokenizer, prompts: List[str], device,
        max_sequence_length: int = 77, joint_attention_dim: int = 4096,
    ):
        if text_encoder is None:
            return torch.zeros(
                (len(prompts), max_sequence_length, joint_attention_dim),
                device=device, dtype=torch.float16,
            )
        text_inputs = tokenizer(
            prompts, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        embeds = text_encoder(text_inputs.input_ids.to(device))[0]
        return embeds.to(dtype=text_encoder.dtype, device=device)

    # ---- construction ----------------------------------------------------

    def __init__(
        self,
        output_path: str,
        config: dict,
        device,
        common_config: dict,
        inference_config: dict,
        pretrained_model_name_or_path: str,
        model,
        model_dtype=None,
        model_checkpoint_path: Optional[str] = None,
        model_load_state_args: Optional[dict] = None,
        resume_from: Optional[int] = None,
        # DreamForge Simple LMA bookkeeping (kept for parity / introspection).
        total_frames: int = 9,
        num_reference_frames: int = 3,
        # Anything else (legacy training_config, metrics, use_ref, use_bev,
        # disable_reference_loss, etc.) is silently dropped.
        **_legacy_kwargs,
    ):
        self.config = config
        self.device = device
        self.common_config = common_config
        self.inference_config = inference_config
        self.model_dtype = model_dtype or torch.float32

        self.should_save = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        # Deterministic noise across runs / ranks (matches legacy behaviour).
        self.generator = torch.Generator()
        if "generator_seed" in config:
            self.generator.manual_seed(config["generator_seed"])
        else:
            self.generator.seed()

        # DreamForge Simple LMA bookkeeping.
        self.total_frames = total_frames
        self.num_reference_frames = num_reference_frames
        self.reference_indices = sample_reference_indices(
            total_frames, num_reference_frames,
        )

        # -- Diffusion backbone --------------------------------------------
        self.model = model.to(dtype=self.model_dtype)
        self.model.enable_gradient_checkpointing()

        self.distribution_framework = common_config.get(
            "distribution_framework", "ddp",
        )
        if (
            not torch.distributed.is_initialized()
            or self.distribution_framework == "ddp"
        ):
            self.model.to(self.device)

        # -- VAE + image processor -----------------------------------------
        vae_type = dwm.common.get_class(
            common_config.get("vae", "diffusers.AutoencoderKL"),
        )
        vae_path = common_config.get(
            "vae_pretrained_model_name_or_path", pretrained_model_name_or_path,
        )
        self.vae = vae_type.from_pretrained(vae_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.to(self.device)
        self.image_processor = diffusers.image_processor.VaeImageProcessor(
            vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1),
        )

        # -- Text encoders & tokenizers (SD3 only) -------------------------
        assert isinstance(self.model, diffusers.SD3Transformer2DModel), (
            "DreamForgeLMAInferencePipeline currently only supports the "
            "SD3 DiT backbone."
        )
        self._init_sd3_text_encoders(
            pretrained_model_name_or_path,
            text_encoder_load_args=common_config.get(
                "text_encoder_load_args", {},
            ),
        )

        # -- Scheduler -----------------------------------------------------
        scheduler_type = dwm.common.get_class(
            inference_config.get(
                "scheduler", "diffusers.FlowMatchEulerDiscreteScheduler",
            ),
        )
        self.test_scheduler = scheduler_type.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler",
        )

        # -- Weights -------------------------------------------------------
        self._load_weights(
            output_path=output_path,
            resume_from=resume_from,
            model_checkpoint_path=model_checkpoint_path,
            model_load_state_args=model_load_state_args or {},
        )

        # -- Distributed wrapping + numerical safety hooks -----------------
        self.model_wrapper = self.model
        if torch.distributed.is_initialized():
            if self.distribution_framework != "fsdp":
                raise RuntimeError(
                    "DreamForgeLMAInferencePipeline expects "
                    "distribution_framework='fsdp' under torch.distributed.",
                )
            self.model_wrapper = FSDP(
                self.model,
                device_id=torch.cuda.current_device(),
                **common_config["ddp_wrapper_settings"],
            )
            self._sanitize_handles = install_sanitize_hooks(self.model_wrapper)

        if self.should_save:
            self._print_model_summary(
                model_checkpoint_path=model_checkpoint_path,
            )

    # ---- ctor sub-routines ----------------------------------------------

    def _init_sd3_text_encoders(
        self, pretrained_model_name_or_path: str, text_encoder_load_args: dict,
    ) -> None:
        clip_tok = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer",
        )
        clip_tok_2 = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2",
        )
        t5_tok = transformers.T5TokenizerFast.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_3",
        )
        self.tokenizers = [clip_tok, clip_tok_2, t5_tok]

        text_encoder = transformers.CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            **text_encoder_load_args,
        )
        text_encoder.requires_grad_(False)
        text_encoder.eval().to(self.device)

        text_encoder_2 = transformers.CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            **text_encoder_load_args,
        )
        text_encoder_2.requires_grad_(False)
        text_encoder_2.eval().to(self.device)

        # T5 is heavy; cast to fp16 for FSDP runs.
        text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            **text_encoder_load_args,
        )
        text_encoder_3.requires_grad_(False)
        text_encoder_3.eval()
        if (
            torch.distributed.is_initialized()
            and self.distribution_framework == "fsdp"
        ):
            text_encoder_3.to(dtype=torch.float16)
            if "t5_fsdp_wrapper_settings" in self.common_config:
                text_encoder_3 = FSDP(
                    text_encoder_3,
                    device_id=torch.cuda.current_device(),
                    **self.common_config["t5_fsdp_wrapper_settings"],
                )
            elif text_encoder_3.device.type != self.device.type:
                text_encoder_3.to(self.device)
        else:
            text_encoder_3.to(self.device)

        self.text_encoders = [text_encoder, text_encoder_2, text_encoder_3]

    def _load_weights(
        self,
        output_path: Optional[str],
        resume_from: Optional[int],
        model_checkpoint_path: Optional[str],
        model_load_state_args: dict,
    ) -> None:
        ckpt_path = None
        if resume_from is not None and output_path is not None:
            ckpt_path = os.path.join(
                output_path, "checkpoints", f"{resume_from}.pth",
            )
        elif model_checkpoint_path is not None:
            ckpt_path = model_checkpoint_path

        if ckpt_path is None:
            return

        state_dict = self.load_state(ckpt_path)
        missing, unexpected = self.model.load_state_dict(
            state_dict, **model_load_state_args,
        )
        if self.should_save and self.common_config.get(
            "print_load_state_info", False,
        ):
            print(f"[ckpt] loaded from {ckpt_path}")
            print(f"       missing keys   = {len(missing)}")
            print(f"       unexpected keys = {len(unexpected)}")

    def _print_model_summary(self, model_checkpoint_path: Optional[str]) -> None:
        param_count = sum(p.numel() for p in self.model.parameters())
        print("=" * 60)
        print("DreamForge LMA Inference Pipeline")
        print("=" * 60)
        print(f"  Total frames           : {self.total_frames}")
        print(f"  Num reference frames   : {self.num_reference_frames}")
        print(f"  Reference indices      : {self.reference_indices}")
        print(f"  Model parameters       : {param_count / 1e6:.1f} M")
        print(f"  Model dtype            : {self.model_dtype}")
        print(f"  Distribution framework : {self.distribution_framework}")
        print(f"  Pretrained weights     : {model_checkpoint_path}")
        if hasattr(self.model, "use_local_motion_attention"):
            lma_enabled = bool(getattr(self.model, "use_local_motion_attention"))
            lma_modules = getattr(self.model, "local_motion_attention", None)
            print(f"  Local Motion Attention : {lma_enabled}")
            if lma_enabled and lma_modules is not None:
                print(f"  LMA module count       : {len(lma_modules)}")
        print("=" * 60)

    # ---- latent / sequence helpers --------------------------------------

    def get_latent_sequence_length(self, sequence_length: int) -> int:
        """Map RGB sequence length to the latent (VAE) sequence length."""
        pre = self.inference_config.get("vae_pre", 0)
        stride = self.inference_config.get("vae_stride", 1)
        assert (
            sequence_length % stride == pre or sequence_length == 0
        ), f"{sequence_length} vs pre={pre} stride={stride}"
        return (sequence_length - pre) // stride + (1 if pre > 0 else 0)

    # ---- condition assembly ---------------------------------------------

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: List[str],
        do_classifier_free_guidance: bool,
    ):
        """Return ``(text_embeds, pooled_embeds)`` ready for SD3.

        ``prompts`` is a flat list; the caller is expected to unflatten the
        outputs back to ``[B, T, V, ...]``.
        """
        # CLIP-G + CLIP-L are concatenated along the channel axis; T5 is
        # appended along the token axis after channel padding.
        clip_embeds_list, clip_pooled_list = [], []
        for tok, enc in zip(self.tokenizers[:2], self.text_encoders[:2]):
            embeds, pooled = self._encode_with_clip(
                enc, tok, prompts, enc.device,
            )
            clip_embeds_list.append(embeds)
            clip_pooled_list.append(pooled)

        clip_embeds = torch.cat(clip_embeds_list, dim=-1)
        pooled = torch.cat(clip_pooled_list, dim=-1)

        t5_embeds = self._encode_with_t5(
            self.text_encoders[-1], self.tokenizers[-1], prompts, self.device,
        )
        clip_embeds = torch.nn.functional.pad(
            clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]),
        )
        text_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)
        return text_embeds, pooled

    @torch.no_grad()
    def get_conditions(
        self,
        latent_shape,
        batch: dict,
        do_classifier_free_guidance: bool,
        latents_shape=None,
    ) -> dict:
        """Build the kwargs dict passed to the DiT model."""
        batch_size, _, view_count = latent_shape[:3]
        sequence_length = batch["pts"].shape[1]
        if do_classifier_free_guidance:
            batch_size *= 2

        # ---- text prompt (CLIP + T5) ------------------------------------
        flat_prompts: List[str] = []
        parsed_shape: List[int] = []
        self.flatten_clip_text(
            batch["clip_text"], flat_prompts, parsed_shape,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        text_embeds, pooled_embeds = self.encode_prompts(
            flat_prompts, do_classifier_free_guidance,
        )

        if len(parsed_shape) == 1:
            # All times and views share the same prompt.
            text_embeds = (
                text_embeds.unsqueeze(1).unsqueeze(1)
                .repeat(1, sequence_length, view_count, 1, 1)
                .to(dtype=self.model_dtype)
            )
            pooled_embeds = (
                pooled_embeds.unsqueeze(1).unsqueeze(1)
                .repeat(1, sequence_length, view_count, 1)
                .to(dtype=self.model_dtype)
            )
        else:
            text_embeds = text_embeds.unflatten(0, parsed_shape).to(
                dtype=self.model_dtype,
            )
            pooled_embeds = pooled_embeds.unflatten(0, parsed_shape).to(
                dtype=self.model_dtype,
            )

        # ---- layout conditions (3dbox / hdmap) ---------------------------
        condition_image_list = []
        condition_on_all_frames = self.common_config.get(
            "condition_on_all_frames", False,
        )
        uncond_color = self.common_config.get("uncondition_image_color", 0)
        for key in ("3dbox_images", "hdmap_images"):
            if key not in batch:
                continue
            imgs = batch[key].to(self.device)
            if not condition_on_all_frames:
                imgs = imgs[:, :1]
            if do_classifier_free_guidance:
                imgs = torch.cat(
                    [torch.ones_like(imgs) * uncond_color, imgs],
                )
            condition_image_list.append(imgs)

        condition_image_tensor = (
            torch.cat(condition_image_list, -3)
            if condition_image_list
            else None
        )

        # ---- explicit view modeling (UniMLVG-style) ----------------------
        camera_intrinsics_norm = None
        camera2referego = None
        if self.common_config.get("explicit_view_modeling", False):
            assert "camera_intrinsics" in batch and "camera_transforms" in batch
            if "ego_transforms" not in batch:
                ego_transforms = (
                    torch.eye(4)
                    .to(batch["camera_transforms"])
                    .unsqueeze(0).unsqueeze(1).unsqueeze(2)
                    .expand(
                        batch["camera_transforms"].shape[0],
                        batch["camera_transforms"].shape[1],
                        batch["camera_transforms"].shape[2], -1, -1,
                    )
                )
            else:
                ego_transforms = batch["ego_transforms"][
                    :, :, -batch["camera_transforms"].shape[2]:, ...
                ]

            camera2world = ego_transforms @ batch["camera_transforms"]
            camera2referego = (
                torch.linalg.inv(
                    ego_transforms[:, 0, 0, :, :].unsqueeze(1).unsqueeze(2),
                )
                @ camera2world
            )

            camera_intrinsics_norm = batch["camera_intrinsics"].clone()
            camera_intrinsics_norm[..., 0, 0] /= batch["image_size"][..., 0]
            camera_intrinsics_norm[..., 1, 1] /= batch["image_size"][..., 1]
            camera_intrinsics_norm[..., 0, 2] /= batch["image_size"][..., 0]
            camera_intrinsics_norm[..., 1, 2] /= batch["image_size"][..., 1]

            if "is_uncalibrated" in batch:
                eye3 = torch.eye(3).to(batch["camera_transforms"])
                eye4 = torch.eye(4).to(batch["camera_transforms"])
                camera_intrinsics_norm[batch["is_uncalibrated"]] = eye3
                camera2referego[batch["is_uncalibrated"]] = eye4

            if do_classifier_free_guidance:
                camera_intrinsics_norm = torch.cat(
                    [camera_intrinsics_norm, camera_intrinsics_norm],
                )
                camera2referego = torch.cat([camera2referego, camera2referego])
            camera_intrinsics_norm = camera_intrinsics_norm.to(self.device)
            camera2referego = camera2referego.to(self.device)

        # ---- additional numeric ids (fps + camera transforms) ------------
        added_time_ids = None
        if (
            "added_time_ids" in self.common_config
            and self.common_config["added_time_ids"] == "fps_camera_transforms"
        ):
            added_time_ids = torch.cat(
                [
                    batch["fps"]
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    .repeat(1, sequence_length, view_count, 1),
                    self.get_camera_transform_ids(batch, self.common_config),
                ],
                -1,
            )
            if do_classifier_free_guidance:
                added_time_ids = torch.cat([added_time_ids, added_time_ids], 0)
            added_time_ids = added_time_ids.to(self.device)

        # ---- depth-related camera matrices (used by some model variants) -
        camera_intrinsics = None
        camera_transforms = None
        if "camera_intrinsics" in batch and "camera_transforms" in batch:
            camera_intrinsics = batch["camera_intrinsics"].to(self.device)
            camera_transforms = batch["camera_transforms"].to(self.device)
            if do_classifier_free_guidance:
                camera_intrinsics = torch.cat(
                    [camera_intrinsics, camera_intrinsics],
                )
                camera_transforms = torch.cat(
                    [camera_transforms, camera_transforms],
                )

        result = {
            "encoder_hidden_states": text_embeds,
            "pooled_projections": pooled_embeds,
            "condition_image_tensor": condition_image_tensor,
            "disable_crossview": torch.tensor(
                [self.common_config.get("disable_crossview", False)],
                device=self.device,
            ).repeat(batch_size),
            "disable_temporal": torch.tensor(
                [self.common_config.get("disable_temporal", False)],
                device=self.device,
            ).repeat(batch_size),
            "crossview_attention_mask": (
                torch.cat(
                    [batch["crossview_mask"], batch["crossview_mask"]],
                )
                if do_classifier_free_guidance
                else batch["crossview_mask"]
            ).to(self.device)
            if "crossview_mask" in batch
            else None,
            "camera_intrinsics": camera_intrinsics,
            "camera_transforms": camera_transforms,
            "camera_intrinsics_norm": camera_intrinsics_norm,
            "camera2referego": camera2referego,
            "added_time_ids": added_time_ids,
        }

        # Some VAEs use a temporal compression that lowers sequence_length;
        # broadcast the per-frame conditions accordingly.
        if (
            latents_shape is not None
            and latents_shape[1] != sequence_length
        ):
            pre = 1 if sequence_length % 2 == 1 else 0
            stride = (sequence_length - pre) // (latents_shape[1] - pre)
            for k, v in result.items():
                if (
                    v is not None
                    and v.ndim > 1
                    and v.shape[1] == sequence_length
                ):
                    result[k] = torch.cat(
                        [v[:, :pre], v[:, pre::stride]], dim=1,
                    )
        return result

    # ---- core inference --------------------------------------------------

    @torch.no_grad()
    def inference_pipeline(
        self,
        latent_shape,
        batch: dict,
        output_type: str = "pt",
        image_latents: Optional[torch.Tensor] = None,
        reference_frame_count: int = 0,
    ) -> dict:
        """Single-window denoising loop with optional reference injection."""
        self.model_wrapper.eval()

        do_cfg = "guidance_scale" in self.inference_config
        guidance_scale = self.inference_config.get("guidance_scale", 1)

        shift_factor = (
            self.vae.config.shift_factor
            if self.vae.config.shift_factor is not None
            else 0
        )

        self.test_scheduler.set_timesteps(
            self.inference_config["inference_steps"], self.device,
        )

        latents = (
            torch.randn(latent_shape, generator=self.generator).to(self.device)
            * getattr(self.test_scheduler, "init_noise_sigma", 1)
        )

        model_conditions = self.get_conditions(
            latent_shape, batch,
            do_classifier_free_guidance=do_cfg,
            latents_shape=latents.shape,
        )

        for i in range(self.inference_config["inference_steps"]):
            t = self.test_scheduler.timesteps[i]
            timesteps = t.unsqueeze(-1).unsqueeze(-1).repeat(*latent_shape[:3])

            latent_model_input = latents
            if image_latents is not None:
                # Reference frames stay clean (timestep = 0); the rest is
                # denoised normally.
                latent_model_input = torch.cat(
                    [
                        image_latents[:, :reference_frame_count],
                        latent_model_input[:, reference_frame_count:],
                    ],
                    1,
                )
                timesteps = torch.cat(
                    [
                        torch.zeros(
                            (
                                timesteps.shape[0],
                                reference_frame_count,
                                timesteps.shape[2],
                            ),
                            dtype=timesteps.dtype,
                            device=self.device,
                        ),
                        timesteps[:, reference_frame_count:],
                    ],
                    1,
                )

            latent_model_input = latent_model_input.to(dtype=self.model_dtype)
            if hasattr(self.test_scheduler, "scale_model_input"):
                latent_model_input = self.test_scheduler.scale_model_input(
                    latent_model_input, t,
                ).to(dtype=self.model_dtype)

            if do_cfg:
                latent_model_input = torch.cat(
                    [latent_model_input, latent_model_input],
                )
                timesteps_input = torch.cat([timesteps, timesteps])
            else:
                timesteps_input = timesteps

            model_output, _, _ = self.model_wrapper(
                latent_model_input, timesteps_input, **model_conditions,
            )
            noise_pred = model_output[0]

            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (
                    noise_cond - noise_uncond
                )

            latents = self.test_scheduler.step(
                noise_pred, t, latents,
            ).prev_sample

        # Reinsert clean reference latents before VAE decoding.
        if image_latents is not None:
            latents = torch.cat(
                [
                    image_latents[:, :reference_frame_count],
                    latents[:, reference_frame_count:],
                ],
                1,
            )

        image_tensor = dwm.functional.memory_efficient_split_call(
            self.vae,
            latents.flatten(0, 2).to(dtype=self.vae.dtype),
            lambda block, tensor: block.decode(
                tensor / block.config.scaling_factor + shift_factor,
                return_dict=False,
            )[0],
            self.common_config.get("memory_efficient_batch", -1),
        )

        return {
            "images": self.image_processor.postprocess(
                image_tensor, output_type=output_type,
            ),
            "latents": latents,
        }

    @torch.no_grad()
    def autoregressive_inference_pipeline(
        self, latent_shape, batch: dict, output_type: str = "pt",
    ) -> dict:
        """Slide a fixed-length window across the full input sequence."""
        total_frame_count = batch["pts"].shape[1]
        seq_len_per_iter = self.inference_config["sequence_length_per_iteration"]
        ref_frame_count = self.inference_config.get(
            "reference_frame_count", 1,
        )
        exception_keys = self.inference_config.get(
            "autoregression_data_exception_for_take_sequence", [],
        )

        # ---- optional dataset-conditioned reference encoding --------------
        if self.inference_config.get("generate_frames_for_reference", True):
            image_latents = None
        else:
            raw = batch["vae_images"][:, :ref_frame_count]
            tensor = self.image_processor.preprocess(
                raw.flatten(0, 2).to(self.device),
            )
            shift_factor = (
                self.vae.config.shift_factor
                if self.vae.config.shift_factor is not None
                else 0
            )
            image_latents = dwm.functional.memory_efficient_split_call(
                self.vae,
                tensor,
                lambda block, t: (
                    block.encode(t).latent_dist.mode() - shift_factor
                )
                * block.config.scaling_factor,
                self.common_config.get("memory_efficient_batch", -1),
            )
            image_latents = image_latents.unflatten(0, raw.shape[:3])

        result: dict = {"images": []}
        for i in range(
            0,
            total_frame_count - seq_len_per_iter + 1,
            seq_len_per_iter - ref_frame_count,
        ):
            iteration_batch = {
                k: (
                    v
                    if k in exception_keys
                    else dwm.functional.take_sequence_clip(
                        v, i, i + seq_len_per_iter,
                    )
                )
                for k, v in batch.items()
            }

            this_ref_count = 0 if image_latents is None else ref_frame_count
            iteration_output = self.inference_pipeline(
                latent_shape,
                iteration_batch,
                output_type,
                image_latents=image_latents,
                reference_frame_count=self.get_latent_sequence_length(
                    this_ref_count,
                ),
            )

            # Drop the reference frames from the saved preview.
            ref_image_count = (
                latent_shape[0] * this_ref_count * latent_shape[2]
            )
            result["images"].append(
                iteration_output["images"][ref_image_count:],
            )

            # Carry the last `ref_frame_count` latents into the next window.
            if (
                i + seq_len_per_iter - ref_frame_count
                < total_frame_count - seq_len_per_iter + 1
            ):
                ref_latent_count = self.get_latent_sequence_length(
                    ref_frame_count,
                )
                image_latents = iteration_output["latents"][
                    :, -ref_latent_count:
                ]

        if output_type == "pt":
            result["images"] = torch.cat(result["images"])
        return result

    # ---- entry point used by preview.py ----------------------------------

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int,
    ) -> None:
        """Generate a preview (single image or MP4) for ``batch``."""
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_h = batch["vae_images"].shape[-2] // (
            2 ** (len(self.vae.config.down_block_types) - 1)
        )
        latent_w = batch["vae_images"].shape[-1] // (
            2 ** (len(self.vae.config.down_block_types) - 1)
        )

        if "sequence_length_per_iteration" in self.inference_config:
            latent_shape = (
                batch_size,
                self.get_latent_sequence_length(
                    self.inference_config["sequence_length_per_iteration"],
                ),
                view_count,
                self.vae.config.latent_channels,
                latent_h,
                latent_w,
            )
            output = self.autoregressive_inference_pipeline(
                latent_shape, batch, "pt",
            )
        else:
            latent_shape = (
                batch_size,
                self.get_latent_sequence_length(sequence_length),
                view_count,
                self.vae.config.latent_channels,
                latent_h,
                latent_w,
            )
            output = self.inference_pipeline(latent_shape, batch, "pt")

        if not (
            self.should_save
            or (
                torch.distributed.is_initialized()
                and self.inference_config.get("all_rank_preview", False)
            )
        ):
            return

        preview_dir = os.path.join(output_path, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        preview_tensor = dwm.utils.preview.make_ctsd_preview_tensor(
            output["images"], batch, self.inference_config,
        )
        if sequence_length == 1:
            torchvision.transforms.functional.to_pil_image(
                preview_tensor,
            ).save(os.path.join(preview_dir, f"{global_step}.png"))
        else:
            dwm.utils.preview.save_tensor_to_video(
                os.path.join(preview_dir, f"{global_step}.mp4"),
                "libx264",
                batch["fps"][0].item(),
                preview_tensor,
            )


# Public re-export.
__all__ = [
    "DreamForgeLMAInferencePipeline",
    "sample_reference_indices",
    "install_sanitize_hooks",
]
