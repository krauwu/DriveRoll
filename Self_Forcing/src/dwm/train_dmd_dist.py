import argparse
import dwm.common
import json
import os
from tqdm import tqdm
import debugpy
import torch
import torch.nn.functional as F
from dwm.utils.sampler import VariableVideoBatchSampler
from typing import Optional, Tuple, Dict
from torchvision.utils import save_image
import imageio
import random

def update_config(base_config: dict, change_config: dict) -> dict:
    """
    递归合并两个配置字典，change_config 中的值会覆盖 base_config 中的值

    Args:
        base_config: 基础配置字典
        change_config: 需要覆盖的配置字典

    Returns:
        合并后的配置字典
    """
    result = base_config.copy()
    for key, value in change_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = update_config(result[key], value)
        else:
            result[key] = value
    return result

def save_grid(tensor, save_path='test_grid.png', normalize=True, nrow=6):
    """
    将 (N, 3, H, W) 的 tensor 保存为拼接大图

    Args:
        tensor (Tensor): shape = (N, 3, H, W)
        save_path (str): 保存路径
        normalize (bool): 是否归一化到 [0,1]
        nrow (int): 每行多少张图
    """
    assert tensor.dim() == 4, "Input must be 4D tensor (N, C, H, W)"
    assert tensor.size(1) == 3, "Channel must be 3"

    save_image(
        tensor,
        save_path,
        nrow=nrow,
        padding=2,
        normalize=normalize,
        value_range=(0, 1) if normalize else None
    )


def save_cam_separated_frames(images, save_dir, start_frame_idx=0):
    """
    将图像按相机分离保存，格式为 cam_x/frame_00xx.png

    Args:
        images: Tensor, shape 可以是:
            - (frame_num, view_num, C, H, W) - 单样本
            - (B, frame_num, view_num, C, H, W) - batch 形式，只处理第一个样本
        save_dir: 保存根目录
        start_frame_idx: 起始帧编号
    """
    # 处理不同的输入 shape
    if images.dim() == 6:  # (B, frame_num, view_num, C, H, W)
        images = images[0]  # 取第一个样本

    # 现在 images shape: (frame_num, view_num, C, H, W)
    frame_num, view_num, C, H, W = images.shape

    # 创建每个相机的文件夹
    for cam_idx in range(view_num):
        cam_dir = os.path.join(save_dir, f"cam_{cam_idx}")
        os.makedirs(cam_dir, exist_ok=True)

    # 保存每一帧的每个相机视角
    for frame_idx in range(frame_num):
        actual_frame_idx = start_frame_idx + frame_idx
        for cam_idx in range(view_num):
            img = images[frame_idx, cam_idx]  # (C, H, W)
            # clamp 到 [0, 1]
            img = img.clamp(0, 1)
            save_path = os.path.join(save_dir, f"cam_{cam_idx}", f"frame_{actual_frame_idx:04d}.png")
            save_image(img, save_path)


def save_cam_params(batch, save_dir, keys=['camera_transforms', 'camera_intrinsics', 'ego_transforms']):
    """
    保存相机参数，每个 key 一个文件夹，格式为 {key}/cam_x/frame_00xx.npy

    Args:
        batch: 数据 batch，包含相机参数
        save_dir: 保存根目录
        keys: 需要保存的参数 key 列表
    """
    import numpy as np

    for key in keys:
        if key not in batch:
            print(f"Warning: {key} not in batch, skip")
            continue

        value = batch[key]
        # shape: (B, frame_num, view_num, k, k)
        if value.dim() == 5:
            value = value[0]  # 取第一个样本

        frame_num, view_num = value.shape[:2]

        # 转为 numpy
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        # 为每个 key 创建独立的文件夹
        key_dir = os.path.join(save_dir, key)
        os.makedirs(key_dir, exist_ok=True)

        # 创建每个相机的文件夹
        for cam_idx in range(view_num):
            cam_dir = os.path.join(key_dir, f"cam_{cam_idx}")
            os.makedirs(cam_dir, exist_ok=True)

        # 保存每一帧
        for frame_idx in range(frame_num):
            for cam_idx in range(view_num):
                param = value[frame_idx, cam_idx]  # (k, k)
                save_path = os.path.join(key_dir, f"cam_{cam_idx}", f"frame_{frame_idx:04d}.npy")
                np.save(save_path, param)


class DMDWrapper:
    """
    DMD (Distribution Matching Distillation) 包装类
    基于 Self-Forcing/DMD 论文的实现，适配 OpenDWM 的 pipeline

    核心思想：
    - generator: 正在被训练的模型
    - real_score: 固定的教师模型 (EMA generator)
    - fake_score: 正在被训练的模型 (与 generator 共享参数)
    - 使用 DMD loss 替代传统的 diffusion loss
    """

    def __init__(self, generator_pipeline, real_score_pipeline, fake_score_pipeline, config, device):
        """
        Args:
            generator_pipeline: generator pipeline (正在训练的模型)
            real_score_pipeline: real score pipeline (教师模型，固定参数)
            fake_score_pipeline: fake score pipeline (与 generator 共享参数)
            config: 训练配置
            device: 设备
        """
        self.generator = generator_pipeline
        self.real_score = real_score_pipeline
        self.fake_score = fake_score_pipeline
        self.config = config
        self.device = device

        # DMD 超参数
        training_config = config.get("training_config", {})
        self.dmd_config = training_config.get("dmd_config", {})

        # DMD 参数 (从 dmd_param 读取)
        self.dmd_param = config.get("dmd_param", {})

        # Timestep 设置
        self.num_train_timestep = self.dmd_config.get("num_train_timestep", 1000)
        self.min_step = self.dmd_param.get("min_step", self.dmd_config.get("min_step", int(0.02 * self.num_train_timestep)))
        self.max_step = self.dmd_param.get("max_step", self.dmd_config.get("max_step", int(0.98 * self.num_train_timestep)))

        # Guidance scale
        self.real_guidance_scale = self.dmd_param.get("real_guidance_scale", self.dmd_config.get("real_guidance_scale", training_config.get("guidance_scale", 7.5)))
        self.fake_guidance_scale = self.dmd_config.get("fake_guidance_scale", 0.0)

        # Skip reference noise: 是否在 compute_x0/train_step_critic 中不对 reference 帧加噪
        self.skip_reference_noise = self.dmd_param.get("skip_reference_noise", False)
        print("*"*10)
        print(f"real score cfg: {self.real_guidance_scale}")
        print("*"*10)

        # 其他超参数
        self.timestep_shift = self.dmd_config.get("timestep_shift", 1.0)  ## self forcing set 5,他们的原因是wan用的5，事实上和这个应该没啥关系，用prolificdreamer里的退火比较合适
        self.ts_schedule = self.dmd_config.get("ts_schedule", True)
        self.ts_schedule_max = self.dmd_config.get("ts_schedule_max", False)
        self.min_score_timestep = self.dmd_config.get("min_score_timestep", 0)

        # Generator 相关
        self.num_training_frames = self.dmd_config.get("num_training_frames", 21)
        self.num_frame_per_block = self.dmd_config.get("num_frame_per_block", 1)
        self.same_step_across_blocks = self.dmd_config.get("same_step_across_blocks", True)
        self.independent_first_frame = self.dmd_config.get("independent_first_frame", False)

        # Backward simulation
        self.backward_simulation = self.dmd_config.get("backward_simulation", True)

        # 获取 scheduler
        self.train_scheduler = self.generator.train_scheduler

    def get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        生成 timestep tensor
        - 如果 uniform_timestep，所有帧使用相同的 timestep
        - 否则，每个 block 使用相同的 timestep
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep, max_timestep,
                [batch_size, 1],
                device=self.device, dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep, max_timestep,
                [batch_size, num_frame],
                device=self.device, dtype=torch.long
            )
            # block 内所有帧使用相同的 timestep
            timestep = timestep.reshape(batch_size, -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(batch_size, -1)
            return timestep

    def _get_sigmas_from_timestep(self, timestep: torch.Tensor, latent_shape: tuple):
        """从 timestep 计算 sigmas (SD3)"""
        # 这里需要根据 scheduler 实现来计算
        # 简化实现，返回与 timestep shape 匹配的 sigmas
        batch, seq, view = timestep.shape[:3]
        c, h, w = latent_shape[3:]
        return timestep.float().reshape(batch, seq, view, 1, 1, 1) / self.num_train_timestep

    def _compute_kl_grad(
        self,
        latent: torch.Tensor,
        batch: dict,
        timestep: torch.Tensor,
        normalization: bool = True,
        reference_frame_count: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 KL 梯度 (DMD 论文 eq. 7)

        Args:
            noisy_latent: [batch, seq, view, c, h, w]
            batch
            timestep: [batch, seq, view]
            conditional_dict: 条件字典
            unconditional_dict: 无条件字典
            normalization: 是否归一化梯度
            reference_frame_count: reference 帧数量，>0 时不对前 k 帧加噪且只计算剩余帧的 score

        Returns:
            kl_grad: 梯度
            log_dict: 日志字典
        """
        # Step 1: 计算 fake score (generator)
        pred_fake, other_result_dict = self.fake_score.compute_x0(
            batch, latent, timestep, guidance_scale=-1,
            reference_frame_count=reference_frame_count
        )

        # Step 2: 计算 real score (teacher)
        ### cyr,  在默认的opendwm config中cfg=3.5，包括在生成ode轨迹时也是，但是这里先用1，加速，并且看起来1和3.5相差不大。
        pred_real, other_result_dict = self.real_score.compute_x0(
            batch, latent, timestep, guidance_scale=self.real_guidance_scale,
            reference_frame_count=reference_frame_count
        )

        # Step 3: 计算 DMD 梯度 (eq. 7)
        grad = pred_fake - pred_real

        # Step 4: 梯度归一化 (eq. 8)
        if normalization:
            p_real = latent - pred_real
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer

        grad = torch.nan_to_num(grad)

        other_result_dict['fake_z0'] = pred_fake
        other_result_dict['real_z0'] = pred_real
        other_result_dict['grad'] = grad


        log_dict = {
            "dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
            "vis_latents": other_result_dict
        }

        return grad, log_dict

    def compute_distribution_matching_loss(
        self,
        latent: torch.Tensor,
        batch: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        reference_frame_count: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 DMD loss

        Args:
            latent: [batch, seq, view, c, h, w] clean latent
            batch: 数据 batch
            gradient_mask: 梯度 mask
            reference_frame_count: reference 帧数量，>0 时不对前 k 帧加噪且只计算剩余帧的 DMD loss

        Returns:
            dmd_loss: DMD loss
            dmd_log_dict: 日志
        """
        original_latent = latent
        batch_size, sequence_length, view_count = latent.shape[:3]

        with torch.no_grad():
            # Step 1: 随机采样 timestep
            min_timestep = self.min_score_timestep if self.ts_schedule else 0
            max_timestep = self.num_train_timestep

            timestep = self.get_timestep(
                min_timestep, max_timestep,
                batch_size, sequence_length,
                uniform_timestep=self.same_step_across_blocks
            )

            # Timestep shift
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * (timestep / 1000) / (
                    1 + (self.timestep_shift - 1) * (timestep / 1000)
                ) * 1000
                timestep = 1000 - timestep # for self forcing code xt = t*x0+(1-t)*e, ours xt = (1-t)*x0+t*e
            else:
                timestep = 1000 - timestep
            timestep = timestep.clamp(self.min_step, self.max_step)

            # skip_reference_noise: 对 reference 帧的 timestep 设为 0（不加噪）
            if reference_frame_count > 0:
                timestep[:, :reference_frame_count] = 0

            print("start compute KL")
            grad, kl_log_dict = self._compute_kl_grad(
                latent=latent,
                batch=batch,
                timestep=timestep,
                reference_frame_count=reference_frame_count
            )

        # Step 3: 计算 DMD loss
        if reference_frame_count > 0:
            # 只对非 reference 帧计算 DMD loss
            non_ref_mask = torch.zeros_like(original_latent, dtype=torch.bool)
            non_ref_mask[:, reference_frame_count:] = True
            dmd_loss = 0.5 * F.mse_loss(
                original_latent[non_ref_mask],
                (original_latent - grad).detach()[non_ref_mask],
                reduction="mean"
            )
        elif gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent[gradient_mask],
                (original_latent - grad).detach()[gradient_mask],
                reduction="mean"
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent,
                (original_latent - grad).detach(),
                reduction="mean"
            )

        return dmd_loss, kl_log_dict
    
    def slice_batch(self, batch: dict, start: int = 0, end: int = 6) -> dict:
            """
            将 batch 中所有符合条件的项（Tensor 或 Nested List），
            在第二维度 (sequence_length) 进行切分 [:, start:end]
            """
            # 获取参考形状 [batch_size, sequence_length, view_count]
            ref_shape = batch["vae_images"].shape[:3]
            B_ref, T_ref, V_ref = ref_shape
            
            sliced_batch = {}
            
            for key, value in batch.items():
                # 情况 1: 处理 Tensor
                if isinstance(value, torch.Tensor):
                    if value.ndim >= 3 and value.shape[:3] == ref_shape:
                        sliced_batch[key] = value[:, start:end, ...]
                    else:
                        sliced_batch[key] = value
                
                # 情况 2: 处理 List (如 clip_text)
                elif isinstance(value, list):
                    # 检查列表是否匹配 [B, T, V] 结构
                    # 逻辑：外层长度等于 B，且第一项也是列表且长度等于 T
                    if len(value) == B_ref and isinstance(value[0], list) and len(value[0]) == T_ref:
                        # 对每个 batch 样本中的 sequence 维度进行切片
                        # 切片后结构保持为 [B][sliced_T][V...]
                        sliced_batch[key] = [sample[start:end] for sample in value]
                    else:
                        sliced_batch[key] = value
                
                # 其他类型 (dict, str, int 等)
                else:
                    sliced_batch[key] = value
                    
            return sliced_batch

    def train_step(self, batch: dict, global_step: int, only_fake_score: bool = False, 
                generated_latents: torch.Tensor = None, selected_timesteps: list = [1000, 500, 0],
                only_preview=False, generator=None) -> Dict:
        """
        DMD 训练步骤

        使用来自 dataset 的 latent，计算 DMD loss, generated_latents会作为condition初始帧
        """
        ## 模拟rollout
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_height = batch["vae_images"].shape[-2] // \
            (2 ** (len(self.generator.vae.config.down_block_types) - 1))
        latent_width = batch["vae_images"].shape[-1] // \
            (2 ** (len(self.generator.vae.config.down_block_types) - 1))
        latent_shape = (
            batch_size, sequence_length, view_count,
            self.generator.vae.config.latent_channels, latent_height,
            latent_width
        )
        # assert (generated_latents is None) == only_fake_score # generated_latents is None 意味着使用GT作为历史帧，则不需要DMD loss
        # with torch.set_grad_enabled(not only_fake_score and not only_preview):
        generator = self.generator if generator is None else generator
        with torch.set_grad_enabled(not only_preview):
            reference_frame_count = self.reference_frame_count
            result = generator.inference_pipeline_few_step(latent_shape, batch, "pt", image_latents=generated_latents,
                                                        reference_frame_count = reference_frame_count, selected_timesteps=selected_timesteps,
                                                        # is_train=generated_latents is not None) 
                                                        is_train=True if not only_preview else False)
            print("roll out")

        latents = result['latents']
        images = result['images']

        if only_preview:
            return {}, latents.clone().detach(), images

        # latents = torch.zeros(latent_shape).cuda()
        ## train student distribution, 我们应该先训练fake score，因为fake score初始化是不准的，并且有时候和real score共同初始化
        latents_for_critic = latents.clone().detach()


        ## 计算 DMD loss
        ref_count = self.reference_frame_count if self.skip_reference_noise else 0
        dmd_loss, log_dict = self.compute_distribution_matching_loss(
            latents, batch,
            reference_frame_count=ref_count
        )

        # 反向传播
        self.generator.optimizer.zero_grad()
        dmd_loss.backward()

        # 梯度裁剪
        if "max_norm_for_grad_clip" in self.generator.training_config:
            torch.nn.utils.clip_grad_norm_(
                self.generator.model.parameters(),
                self.generator.training_config["max_norm_for_grad_clip"]
            )

        self.generator.optimizer.step()

        if self.generator.lr_scheduler is not None:
            self.generator.lr_scheduler.step()

        RATIO = 5
        critic_ref_count = self.reference_frame_count if self.skip_reference_noise else 0
        for i in range(RATIO):
            print("start critic loss")
            critic_pred_latent, critic_noisy_latents, critic_target = self.fake_score.train_step_critic(batch, latents_for_critic, global_step*RATIO+i, reference_frame_count=critic_ref_count)
            print("critic loss")

        log_dict['vis_latents']['critic_pred_latent'] = critic_pred_latent
        log_dict['vis_latents']['critic_noisy_latents'] = critic_noisy_latents
        log_dict['vis_latents']['critic_target'] = critic_target

        loss_report = {
            "loss": dmd_loss.item(),
            **log_dict
        }
        # self.generator.loss_report_list.append(loss_report)
        
        print("train dmd")

        return loss_report, latents.clone().detach(), images

    def train_sequence(self, batch: dict, global_step: int):
        """
        对整个序列进行训练，每次取 segment_len 帧片段，步长为 chunk_size

        例如：序列长度 16，片段长度 6，步长 3
        - i=0: slice [0:6]
        - i=3: slice [3:9]
        - i=6: slice [6:12]
        - i=9: slice [9:15]
        - i=12: slice [12:16] (不足 segment_len 帧会自动截断)
        """
        seq_len = batch["vae_images"].shape[1]

        ## 从 config 读取参数
        train_param = self.dmd_param.get("train_sequence", {})
        segment_len = train_param.get("segment_len", 6)  # 每次取的片段长度
        CHUNK_SIZE = train_param.get("chunk_size", 3) ## 每次roll out chunk_size个新的帧，对应的condition帧数为segment_len-chunk_size
        MAX_GENERATED_ITERATION = train_param.get("max_generated_iteration", 4)
        SELECTED_TIMESTEPS = train_param.get("selected_timesteps", [1000, 750, 500, 250])

        total_loss = 0.0
        step_count = 0
        generated_latents = None

        # 创建可视化保存目录
        vis_dir = os.path.join(args.output_path, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis_other_dir = os.path.join(args.output_path, "vis_abc")
        os.makedirs(vis_other_dir, exist_ok=True)

        vis_dir = os.path.join(vis_dir, f"{global_step}")
        os.makedirs(vis_dir, exist_ok=True)
        vis_other_dir = os.path.join(vis_other_dir, f"{global_step}")
        os.makedirs(vis_other_dir, exist_ok=True)

        full_videos = []
        self.reference_frame_count = segment_len - CHUNK_SIZE if "reference_frame_count" not in train_param else train_param["reference_frame_count"]


        # 从 i=0 开始，每次加 3，直到 i + segment_len 超过序列长度
        for i in range(0, seq_len, CHUNK_SIZE):
            # 每过MAX_GENERATED_ITERATION轮，重新使用GT作为condition
            if i % (CHUNK_SIZE * MAX_GENERATED_ITERATION) == 0:
                generated_latents = None

            end = min(i + segment_len, seq_len)
            print(f"training frame {i}--{end}")

            # 如果片段太短（小于 stride），跳过
            if end - i < segment_len:
                continue

            # slice batch
            sliced_batch = self.slice_batch(batch, i, end)

            # 执行 train_step
            ## for select timesteps
            n = len(SELECTED_TIMESTEPS)
            ## TODO 考虑是否需要每个timestamp都要训练，或者只训练最后一个阶段
            # 所有 rank 必须使用相同的 k，否则 FSDP 的 allgather 顺序不一致导致死锁
            if torch.distributed.is_initialized():
                k_tensor = torch.tensor([random.randint(1, n)], device=self.device)
                torch.distributed.broadcast(k_tensor, src=0)
                k = k_tensor.item()
            else:
                k = random.randint(1, n)
            # k = n
            selected_subset = SELECTED_TIMESTEPS[:k] + [0]

            loss_report, generated_latents, images = self.train_step(
                sliced_batch, global_step * (seq_len // CHUNK_SIZE) + step_count,
                generated_latents=generated_latents[:, CHUNK_SIZE: self.reference_frame_count+CHUNK_SIZE] if generated_latents is not None else None,
                only_fake_score=True if generated_latents is None else False, # 若以gt为condition则不需要DMD loss
                selected_timesteps=selected_subset
            )

            # 保存可视化图像
            if images is not None:
                # images shape: (batch, seq, view, c, h, w) 或 (batch*seq*view, c, h, w)
                # 需要flatten成 (segment_len*6, 3, H, W) 的格式给 save_grid
                if images.dim() == 6:  # (batch, seq, view, c, h, w)
                    images_flat = images.flatten(0, 2)  # (batch*seq*view, c, h, w)
                else:
                    images_flat = images

                # 保存 segment_len * 6 张图（segment_len帧 x 6视角）
                expected_count = segment_len * 6
                if images_flat.shape[0] >= expected_count:
                    save_path = os.path.join(vis_dir, f"{global_step}_{i}_{end}.jpg")
                    save_grid(images_flat[:expected_count], save_path=save_path, nrow=6)

                full_videos.append(images.reshape(segment_len, 6, *images.shape[-3:]))
            if "vis_latents" in loss_report:
                vis_dict = loss_report['vis_latents']
                for key, latents in vis_dict.items():
                    images_flat = self.generator.decode_latents(latents)
                    # 保存 latent 可视化，同样使用 segment_len * 6
                    expected_count = segment_len * 6
                    if images_flat.shape[0] >= expected_count:
                        save_path = os.path.join(vis_other_dir, f"{global_step}_{i}_{end}_{key}.jpg")
                        save_grid(images_flat[:expected_count], save_path=save_path, nrow=6)
                loss_report.pop("vis_latents")

            if loss_report:  # 如果有 loss（not only_fake_score）
                total_loss += loss_report.get("loss", 0.0)
                step_count += 1

        # vis full video
        full_video = [full_videos[0]] + [video[-CHUNK_SIZE:] for video in full_videos[1:]]
        full_video_tensor = torch.cat(full_video, dim=0) # frame_num, 6, c, h, w

        # 保存完整视频：每帧 6 张图横向拼接，fps=3
        frame_num, view_num, c, h, w = full_video_tensor.shape
        # (frame_num, 6, c, h, w) -> (frame_num, c, h, w*6)
        video_frames = torch.cat([full_video_tensor[:, i] for i in range(view_num)], dim=-1)
        # 保存视频
        video_path = os.path.join(vis_dir, f"{global_step}_full.mp4")
        video = video_frames.permute(0, 2, 3, 1)
        video = video.clamp(0, 1)
        # 转 uint8
        video = (video * 255).to(torch.uint8)
        # 转 numpy
        video = video.cpu().numpy()
        imageio.mimwrite(
                            video_path,
                            video,
                            fps=3,
                            quality=8
                        )
        # 返回平均 loss 和所有 latents
        avg_loss = total_loss / step_count if step_count > 0 else 0.0
        return {"avg_loss": avg_loss, "step_count": step_count}
    
    def preview_sequence(self, batch: dict, global_step: int, generator=None, suffix='eval'):
        """
        对整个序列进行预览，每次取 segment_len 帧片段，步长为 chunk_size

        例如：序列长度 16，片段长度 6，步长 3
        - i=0: slice [0:6]
        - i=3: slice [3:9]
        - i=6: slice [6:12]
        - i=9: slice [9:15]
        - i=12: slice [12:16] (不足 segment_len 帧会自动截断)
        """
        seq_len = batch["vae_images"].shape[1]

        ## 从 config 读取参数
        preview_param = self.dmd_param.get("preview_sequence", {})
        segment_len = preview_param.get("segment_len", 6)  # 每次取的片段长度
        CHUNK_SIZE = preview_param.get("chunk_size", 3)
        MAX_GENERATED_ITERATION = preview_param.get("max_generated_iteration", 10000)
        SELECTED_TIMESTEPS = preview_param.get("selected_timesteps", [1000, 750, 500, 250])

        total_loss = 0.0
        step_count = 0
        generated_latents = None

        # 创建可视化保存目录
        vis_dir = os.path.join(args.output_path, f"vis_{suffix}")
        os.makedirs(vis_dir, exist_ok=True)

        vis_dir = os.path.join(vis_dir, f"{global_step}")
        os.makedirs(vis_dir, exist_ok=True)

        # 创建 GT 和生成图像的分离保存目录
        gt_rgb_dir = os.path.join(vis_dir, "gt_rgb")
        gen_rgb_dir = os.path.join(vis_dir, "generated_rgb")
        os.makedirs(gt_rgb_dir, exist_ok=True)
        os.makedirs(gen_rgb_dir, exist_ok=True)

        # 保存 GT 图像 (batch['vae_images'] shape: B, frame_num, 6, C, H, W)
        gt_images = batch["vae_images"]  # (B, frame_num, view_num, C, H, W)
        save_cam_separated_frames(gt_images, gt_rgb_dir, start_frame_idx=0)

        # 保存相机参数，添加 gt_ 前缀
        gt_cam_params_dir = os.path.join(vis_dir, "gt_camera_params")
        save_cam_params(batch, gt_cam_params_dir)

        full_videos = []
        self.reference_frame_count = segment_len - CHUNK_SIZE


        # 从 i=0 开始，每次加 3，直到 i + segment_len 超过序列长度
        for i in range(0, seq_len, CHUNK_SIZE):
            # 每过MAX_GENERATED_ITERATION轮，重新使用GT作为condition
            if i % (CHUNK_SIZE * MAX_GENERATED_ITERATION) == 0:
                generated_latents = None

            end = min(i + segment_len, seq_len)
            print(f"training frame {i}--{end}")

            # 如果片段太短（小于 stride），跳过
            if end - i < segment_len:
                continue

            # slice batch
            sliced_batch = self.slice_batch(batch, i, end)

            # 执行 train_step
            selected_subset = SELECTED_TIMESTEPS + [0]

            loss_report, generated_latents, images = self.train_step(
                sliced_batch, global_step * (seq_len // CHUNK_SIZE) + step_count,
                generated_latents=generated_latents[:, CHUNK_SIZE:] if generated_latents is not None else None,
                only_fake_score=True if generated_latents is None else False, # 若以gt为condition则不需要DMD loss
                selected_timesteps=selected_subset, only_preview=True, generator=generator
            )

            # 保存可视化图像
            if images is not None:
                # images shape: (batch, seq, view, c, h, w) 或 (batch*seq*view, c, h, w)
                # 需要flatten成 (segment_len*6, 3, H, W) 的格式给 save_grid
                if images.dim() == 6:  # (batch, seq, view, c, h, w)
                    images_flat = images.flatten(0, 2)  # (batch*seq*view, c, h, w)
                else:
                    images_flat = images

                # 保存 segment_len * 6 张图（segment_len帧 x 6视角）
                expected_count = segment_len * 6
                if images_flat.shape[0] >= expected_count:
                    save_path = os.path.join(vis_dir, f"{global_step}__{i}_{end}.jpg")
                    save_grid(images_flat[:expected_count], save_path=save_path, nrow=6)

                full_videos.append(images.reshape(segment_len, 6, *images.shape[-3:]))

            if loss_report:  # 如果有 loss（not only_fake_score）
                total_loss += loss_report.get("loss", 0.0)
                step_count += 1

        # vis full video
        full_video = [full_videos[0]] + [video[-CHUNK_SIZE:] for video in full_videos[1:]]
        full_video_tensor = torch.cat(full_video, dim=0) # frame_num, 6, c, h, w

        # 保存生成图像到分离的相机文件夹
        save_cam_separated_frames(full_video_tensor, gen_rgb_dir, start_frame_idx=0)

        # 保存完整视频：每帧 6 张图横向拼接，fps=3
        frame_num, view_num, c, h, w = full_video_tensor.shape
        # (frame_num, 6, c, h, w) -> (frame_num, c, h, w*6)
        video_frames = torch.cat([full_video_tensor[:, i] for i in range(view_num)], dim=-1)
        # 保存视频
        video_path = os.path.join(vis_dir, f"{global_step}_full.mp4")
        video = video_frames.permute(0, 2, 3, 1)
        video = video.clamp(0, 1)
        # 转 uint8
        video = (video * 255).to(torch.uint8)
        # 转 numpy
        video = video.cpu().numpy()
        imageio.mimwrite(
                            video_path,
                            video,
                            fps=3,
                            quality=8
                        )


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-d", "--default-config", type=str, default=None,
        help="The default config path (base config). If not specified, uses ./configs/train_dmd/default.json")
    parser.add_argument(
        "-c", "--change-config", type=str, default=None,
        help="The change config path to override default config. If not specified, only use default config.")
    parser.add_argument(
        "-o", "--output-path", type=str, default=None,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "--log-steps", default=100, type=int,
        help="The step count to print log and update the tensorboard.")
    parser.add_argument(
        "--preview-steps", default=1, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--checkpointing-steps", default=100, type=int,
        help="The step count to save the checkpoint.")
    parser.add_argument(
        "--evaluation-steps", default=10000, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--resume-from", default=None, type=int,
        help="The step to resume from")
    parser.add_argument(
        "--ckpt-path", default=None, type=str,
        help="The checkpoint path to load (will override model_checkpoint_path in config)")
    parser.add_argument(
        "--only-eval", action="store_true",
        help="Only run evaluation (use validation dataset, skip training)")
    parser.add_argument(
        "--wandb", action="store_true",
        help="Use wandb to log the training process.")
    parser.add_argument(
        "--wandb-project", type=str, default="dwm",
        help="The wandb project name.")
    parser.add_argument(
        "--wandb-run-name", type=str, default="train",
        help="The wandb run name.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # 确定默认配置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    default_config_path = args.default_config
    if default_config_path is None:
        default_config_path = os.path.join(project_root, "configs/train_dmd/default.json")

    # 加载默认配置
    with open(default_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"Loaded default config from: {default_config_path}")

    # 加载并合并 change 配置（如果指定）
    if args.change_config is not None:
        with open(args.change_config, "r", encoding="utf-8") as f:
            change_config = json.load(f)
        config = update_config(config, change_config)
        print(f"Merged change config from: {args.change_config}")


    torch.manual_seed(config["generator_seed"])

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    # 单卡训练优化配置：仅在真正单卡时启用（关闭 FSDP、启用 AMP/4bit 量化）
    if ddp:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    single_gpu = not ddp or world_size == 1

    if single_gpu:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        # 1. 移除分布式训练相关配置
        if "distribution_framework" in config["pipeline"]["common_config"]:
            del config["pipeline"]["common_config"]["distribution_framework"]
        if "ddp_wrapper_settings" in config["pipeline"]["common_config"]:
            del config["pipeline"]["common_config"]["ddp_wrapper_settings"]
        if "t5_fsdp_wrapper_settings" in config["pipeline"]["common_config"]:
            del config["pipeline"]["common_config"]["t5_fsdp_wrapper_settings"]
        if "device_mesh" in config.get("global_state", {}):
            del config["global_state"]["device_mesh"]

        # 2. 添加 CUDA AMP 配置
        config["pipeline"]["common_config"]["autocast"] = {"device_type": "cuda"}

        # 3. 添加 text encoder 4bit 量化配置
        if "text_encoder_load_args" not in config["pipeline"]["common_config"]:
            config["pipeline"]["common_config"]["text_encoder_load_args"] = {}
        config["pipeline"]["common_config"]["text_encoder_load_args"]["quantization_config"] = {
            "_class_name": "diffusers.quantizers.quantization_config.BitsAndBytesConfig",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": {
                "_class_name": "get_class",
                "class_name": "torch.float16"
            }
        }

        # 4. 修改 optimizer 为 8bit Adam
        config["optimizer"]["_class_name"] = "bitsandbytes.optim.Adam8bit"

        print("Single GPU mode enabled:")
        print("  - FSDP/DDP disabled")
        print("  - CUDA AMP enabled")
        print("  - Text encoder 4bit quantization enabled")
        print("  - Using bitsandbytes.optim.Adam8bit")
    else:
        print(f"Multi-GPU FSDP mode enabled: world_size={world_size}")

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (not single_gpu and local_rank == 0) or single_gpu
    should_save = single_gpu or (
        torch.distributed.is_initialized() and
        torch.distributed.get_rank() == 0
    )

    # load the pipeline including the models
    output_path = config["output_path"] if args.output_path is None else args.output_path


    # 如果指定了 ckpt_path，替换配置中的 model_checkpoint_path，但是只替换generator，teacher总是使用opendwm
    import copy
    generator_config = copy.deepcopy(config)
    if args.ckpt_path is not None:
        generator_config["pipeline"]["model_checkpoint_path"] = args.ckpt_path
        print(f"Overriding generator model_checkpoint_path with: {args.ckpt_path}")

    # 创建三个 pipeline: generator, real_score, fake_score
    # generator 和 fake_score 共享模型参数，real_score 是独立的教师模型
    generator_pipeline = dwm.common.create_instance_from_config(
        generator_config["pipeline"], output_path=output_path, config=generator_config,
        device=device, resume_from=args.resume_from)

    # real_score 和 fake_score: 仅在训练模式下加载
    real_score_pipeline = None
    fake_score_pipeline = None

    if not args.only_eval:
        # real_score: 教师模型，参数冻结
        real_score_config = copy.deepcopy(config)
        real_score_config["pipeline"].pop("optimizer", None)
        real_score_pipeline = dwm.common.create_instance_from_config(
            real_score_config["pipeline"], output_path=output_path, config=real_score_config,
            device=device, resume_from=args.resume_from)
        # 冻结 real_score 的模型参数
        for param in real_score_pipeline.model.parameters():
            param.requires_grad = False

        # fake_score: 学生模型，与 generator 独立初始化
        fake_score_config = copy.deepcopy(config)
        fake_score_pipeline = dwm.common.create_instance_from_config(
            fake_score_config["pipeline"], output_path=output_path, config=fake_score_config,
            device=device, resume_from=args.resume_from)

    if should_log:
        print("The generator pipeline is loaded.")
        if not args.only_eval:
            print("The real_score pipeline is loaded (frozen).")
            print("The fake_score pipeline is initialized (start from real score).")
        else:
            print("Only-eval mode: real_score and fake_score pipelines skipped.")

    if args.wandb and should_save:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    # create DMD wrapper
    dmd_wrapper = DMDWrapper(
        generator_pipeline=generator_pipeline,
        real_score_pipeline=real_score_pipeline,
        fake_score_pipeline=fake_score_pipeline,
        config=config,
        device=device
    )

    if should_log:
        print("DMD wrapper initialized.")

    # load the dataset
    if args.only_eval:
        # only eval mode: 只加载 validation dataset
        training_dataset = None
        validation_dataset = dwm.common.create_instance_from_config(
            config["validation_dataset"])
        if should_log:
            print("Only-eval mode: training dataset skipped.")
    else:
        training_dataset = dwm.common.create_instance_from_config(
            config["training_dataset"])
        validation_dataset = training_dataset

    training_dataloader = None
    training_datasampler = None

    if not single_gpu:
        if not args.only_eval:
            # if "mix_config" in config.keys():
            if 0:
                print("******** mix config ********")
                process_group = torch.distributed.group.WORLD

                training_datasampler = VariableVideoBatchSampler(
                    training_dataset,
                    config["mix_config"],
                    num_replicas=process_group.size(),
                    rank=process_group.rank(),
                    shuffle=config["data_shuffle"],
                    seed=config["generator_seed"]
                )

                training_dataloader = torch.utils.data.DataLoader(
                    training_dataset,
                    **dwm.common.instantiate_config(config["training_dataloader"]),
                    batch_sampler=training_datasampler)

            else:
                training_datasampler = torch.utils.data.distributed.DistributedSampler(
                    training_dataset, shuffle=config["data_shuffle"],
                    seed=config["generator_seed"])
                training_dataloader = torch.utils.data.DataLoader(
                    training_dataset,
                    **dwm.common.instantiate_config(config["training_dataloader"]),
                    sampler=training_datasampler)

        # make equal sample count for each process to simplify the result
        # gathering
        total_batch_size = int(os.environ["WORLD_SIZE"]) * \
            config["validation_dataloader"]["batch_size"]
        dataset_length = len(validation_dataset) // \
            total_batch_size * total_batch_size
        validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(0, dataset_length))
        validation_datasampler = \
            torch.utils.data.distributed.DistributedSampler(
                validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]),
            sampler=validation_datasampler)
    else:
        if not args.only_eval:
            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                shuffle=config["data_shuffle"])
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]))

    preview_dataloader = torch.utils.data\
        .DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"])) if \
        "preview_dataloader" in config else None
    if preview_dataloader is not None:
        preview_data_iterator = iter(preview_dataloader)

    if should_log:
        if args.only_eval:
            print("The validation dataset is loaded with {} items.".format(
                len(validation_dataset)))
        else:
            print("The training dataset is loaded with {} items.".format(
                len(training_dataset)))
            print("The validation dataset is loaded with {} items.".format(
                len(validation_dataset)))

    # train loop
    ## todo batch size小 lr不确定
    '''
        原来
        "optimizer": {
        "_class_name": "torch.optim.AdamW",
        "lr": 6e-5
    }, now "lr": 1e-6
    '''
    global_step = 0 if args.resume_from is None else args.resume_from

    if args.only_eval:
        # only eval mode: 只运行 preview，不训练
        if should_log:
            print("Running evaluation only...")
        for batch in validation_dataloader:
            dmd_wrapper.preview_sequence(batch, global_step, generator=dmd_wrapper.generator, suffix='validation_generator')
            global_step += 1
        if should_log:
            print("Evaluation done.")
    else:
        # normal training loop
        for epoch in range(config["train_epochs"]):

            if not single_gpu:
                # Fixing training data order reduces the accessed objects per rank,
                # therefore reduces the upper-bound of memory usage comsumed by the
                # Python reference counting of objects.
                sampler_epoch = 0 if config.get("fix_training_data_order", False) \
                    else epoch
                training_datasampler.set_epoch(sampler_epoch)


            for batch in training_dataloader:
                # 使用 DMD 训练步骤
                dmd_wrapper.train_sequence(batch, global_step)
                global_step += 1
                # preview: 只在 global_step % args.preview_steps == 0 时执行，且只在 rank 0 执行，在fsdp下不应该preview，容易死锁
                if global_step % args.preview_steps == 0 and should_save:
                    dmd_wrapper.preview_sequence(batch, global_step, generator=dmd_wrapper.fake_score, suffix='eval_fake_score')
                    dmd_wrapper.preview_sequence(batch, global_step, generator=dmd_wrapper.real_score, suffix='eval_real_score')
                    dmd_wrapper.preview_sequence(batch, global_step, generator=dmd_wrapper.generator, suffix='eval_generator')



                # log
                if global_step % args.log_steps == 0 and should_log:
                    generator_pipeline.log(global_step, args.log_steps)

                # save step checkpoint
                if global_step % args.checkpointing_steps == 0 and should_save:
                    ckpt_path = os.path.join(output_path, "ckpt")
                    os.makedirs(ckpt_path, exist_ok=True)
                    generator_pipeline.save_checkpoint(ckpt_path, global_step)

                # # evaluation
                # if (
                #     args.evaluation_steps > 0 and
                #     global_step % args.evaluation_steps == 0
                # ):
                #     generator_pipeline.evaluate_pipeline(
                #         global_step, len(validation_dataset),
                #         validation_dataloader, validation_datasampler)

            if should_log:
                print("Epoch {} done.".format(epoch))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
