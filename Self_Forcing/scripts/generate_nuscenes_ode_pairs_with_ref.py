#!/usr/bin/env python
"""
生成带 reference frame 的 ODE 轨迹数据集的脚本

基于 generate_nuscenes_ode_pairs.py 修改，支持 reference frame 注入。
前 reference_frame_count 帧使用真实图像的 VAE 编码作为 clean latent，
在 ODE 去噪过程中作为条件注入。

使用方法:
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
      scripts/generate_nuscenes_ode_pairs_with_ref.py \
      --config_path configs/generate_ode/seq12_ref3.json \
      --output_folder output/nuscenes_ode_pairs_ref3 \
      --num_samples 1000
"""

import argparse
import glob
import json
import math
import os
import sys
import torch
import torch.distributed as dist
from tqdm import tqdm

# 添加项目路径
project_root = "."
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "externals/waymo-open-dataset/src"))
sys.path.insert(0, os.path.join(project_root, "externals/TATS/tats/fvd"))

import dwm.common
import dwm.functional
import dwm.pipelines.rolling_ref
from dwm.pipelines.ctsd import CrossviewTemporalSD

try:
    from torchvision.utils import save_image
except ImportError:
    save_image = None

class ODEGenerator:
    """ODE 轨迹生成器"""

    def __init__(self, pipeline_config, device="cuda", output_path="/tmp/output_temp"):
        self.device = device
        self.config = pipeline_config

        self.pipeline = dwm.common.create_instance_from_config(
            pipeline_config["pipeline"], output_path=output_path, config=pipeline_config, device=device
        )

        # 冻结参数
        try:
            for param in self.pipeline.model_wrapper.parameters():
                param.requires_grad = False
            self.pipeline.model_wrapper.eval()
        except AttributeError:
            for param in self.pipeline.model.parameters():
                param.requires_grad = False
            self.pipeline.model.eval()
        torch.set_grad_enabled(False)

        self.inference_config = pipeline_config["pipeline"]["inference_config"]
        self.common_config = pipeline_config["pipeline"]["common_config"]
        self.test_scheduler = self.pipeline.test_scheduler

    def sample_random_segment(self, batch, k=6, length_key='camera_transforms'):
        """
        从 batch 中提取长度为 k 的连续片段。
        
        Args:
            batch (dict): 包含 Tensor 的字典
            k (int): 需要截取的连续片段长度
            length_key (str): 用来确定总长度 (length) 的参考 key
        """
        # 1. 获取总长度 length
        total_length = batch[length_key].shape[1]
        
        if k > total_length:
            raise ValueError(f"截取长度 k={k} 不能大于总长度 length={total_length}")

        # 2. 随机生成起始索引 i (范围在 0 到 length - k 之间)
        high_val = total_length - k + 1
        start_idx = torch.randint(0, high_val, (1,)).item()
        end_idx = start_idx + k

        # 3. 遍历字典，对符合条件的 Tensor 进行切片
        new_batch = {}
        for key, value in batch.items():
            # 检查是否为 Tensor，且第二维 (dim 1) 长度是否等于 total_length
            if torch.is_tensor(value) and value.dim() > 1 and value.shape[1] == total_length:
                # 执行切片：[Batch, start_idx:end_idx, ...]
                new_batch[key] = value[:, start_idx:end_idx, ...]
            else:
                # 其余 key（如标量、不含序列维度的参数等）保持不变
                new_batch[key] = value
        
        # 4. special for text
        new_batch['clip_text'] = new_batch['clip_text'][0][start_idx:end_idx]
                
        return new_batch

    @torch.no_grad()
    def generate_ode_pairs(self, batch, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        # 计算 latent shape
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_height = batch["vae_images"].shape[-2] // (
            2 ** (len(self.pipeline.vae.config.down_block_types) - 1)
        )
        latent_width = batch["vae_images"].shape[-1] // (
            2 ** (len(self.pipeline.vae.config.down_block_types) - 1)
        )

        latent_seq_len = self.inference_config["sequence_length_per_iteration"]
        reference_frame_count = self.inference_config.get("reference_frame_count", 0)
        latent_shape = (batch_size, latent_seq_len, view_count,
                       self.pipeline.vae.config.latent_channels, latent_height, latent_width)

        # 禁用 diffusion forcing
        self.pipeline.common_config["frame_prediction_style"] = "standard"

        chunk_batch = self.sample_random_segment(batch, latent_seq_len)
        chunk_batch['clip_text'] = [chunk_batch['clip_text']]

        # 编码 reference frame 为 clean latent
        image_latents = None
        if reference_frame_count > 0:
            shift_factor = self.pipeline.vae.config.shift_factor \
                if self.pipeline.vae.config.shift_factor is not None else 0
            raw_image_tensor = chunk_batch["vae_images"][:, :reference_frame_count]
            image_tensor = self.pipeline.image_processor.preprocess(
                raw_image_tensor.flatten(0, 2).to(self.device))
            if self.pipeline.is_temporal_vae:
                import einops
                image_tensor = einops.rearrange(image_tensor, "(b t v) c h w -> (b v) c t h w",
                                                t=raw_image_tensor.shape[1], v=raw_image_tensor.shape[2])
            image_latents = dwm.functional.memory_efficient_split_call(
                self.pipeline.vae, image_tensor,
                lambda block, tensor: (
                    block.encode(tensor).latent_dist.mode() - shift_factor
                ) * block.config.scaling_factor,
                self.common_config.get("memory_efficient_batch", -1))
            if self.pipeline.is_temporal_vae:
                import einops
                image_latents = einops.rearrange(
                    image_latents, "(b v) c t h w -> b t v c h w", v=raw_image_tensor.shape[2])
            else:
                image_latents = image_latents.unflatten(0, raw_image_tensor.shape[:3])

        # 运行推理，传入 reference frame 的 clean latent
        capture = self.pipeline.inference_pipeline_record_ODE_traj(
            latent_shape, chunk_batch, "pt",
            image_latents=image_latents,
            reference_frame_count=reference_frame_count,
        )

        return {"capture_latents": capture['latents_list'], "timestamp": capture['timestamp_list'], "batch": chunk_batch, "latent_shape": latent_shape}

    @torch.no_grad()
    def visualize_ode_trajectory(self, selected_latents, selected_timestamps, save_path):
        """
        将 selected_latents 解码为图片并拼接成可视化网格保存。

        布局：每行一个去噪时间步（从最噪声到最干净），每列一个时序帧（T）。
        只取其中一个视角（view=0，即 CAM_FRONT）可视化。

        Args:
            selected_latents: list of tensors, 每个 shape [B, T, V, C, H, W]
            selected_timestamps: list of float, 每个时间步的 timestamp
            save_path: 图片保存路径
        """
        shift_factor = self.pipeline.vae.config.shift_factor \
            if self.pipeline.vae.config.shift_factor is not None else 0
        memory_efficient_batch = self.common_config.get("memory_efficient_batch", -1)

        rows = []
        for latent, ts in zip(selected_latents, selected_timestamps):
            # latent: [B, T, V, C, H, W]，取 B=0, V=0（CAM_FRONT）可视化所有 T 帧
            cur_latent = latent[0, :, 0]  # [T, C, H, W]
            # decode
            decoded = dwm.functional.memory_efficient_split_call(
                self.pipeline.vae, cur_latent.to(dtype=self.pipeline.vae.dtype),
                lambda block, tensor: block.decode(
                    tensor / block.config.scaling_factor + shift_factor,
                    return_dict=False
                )[0],
                memory_efficient_batch)
            # decoded: [T, C, H, W]，归一化到 [0, 1]
            decoded = self.pipeline.image_processor.postprocess(decoded, output_type="pt")
            rows.append(decoded)  # [T, C, H, W]

        # 拼成 grid: nrow = seq_len（每行一个时间步的所有 T 帧）
        grid = torch.cat(rows, dim=0)  # [num_timesteps * T, C, H, W]
        seq_len = rows[0].shape[0]
        save_image(grid, save_path, nrow=seq_len, padding=2)

def find_nearest_indices_brute(ts, selected):
    ts = torch.as_tensor(ts)
    selected = torch.as_tensor(selected)
    
    # 广播计算距离矩阵: [len(selected), len(ts)]
    # (selected[:, None] - ts[None, :]) 会生成一个差值矩阵
    diff = (selected.unsqueeze(1) - ts.unsqueeze(0)).abs()
    
    # 在 ts 维度找到最小值对应的索引
    return diff.argmin(dim=1)


def main():
    global torch, dist
    parser = argparse.ArgumentParser(description="Generate ODE trajectory pairs for NuScenes")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_trajectory_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # 分布式相关参数
    parser.add_argument("--world_size", type=int, default=None,
                        help="Total number of processes (detected from CUDA if not specified)")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                        help="Master node address for distributed training")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="Master node port for distributed training")
    args = parser.parse_args()

    # 初始化分布式环境
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size if args.world_size else torch.cuda.device_count()))

    # 设置 CUDA 设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化分布式进程组（使用 env:// 支持 torchrun 多机）
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    if rank == 0:
        print(f"Generating ODE pairs on {world_size} GPUs")
        print(f"Current rank: {rank}, local_rank: {local_rank}")
        if "MASTER_ADDR" in os.environ:
            print(f"Master: {os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '29500')}")
        else:
            print(f"Master: {args.master_addr}:{args.master_port}")

    # 创建 device mesh 并设置到 global_state（支持多机）
    try:
        import torch.distributed.device_mesh as dm
        # 多机模式下，使用全局初始化，会自动感知所有节点
        dwm.common.global_state["device_mesh"] = dm.init_device_mesh(
            "cuda",
            mesh_shape=(world_size,)  # 使用一维 mesh 支持任意数量的 GPU
        )
        if rank == 0:
            print(f"Created device_mesh with shape: {dwm.common.global_state['device_mesh'].shape}")
    except Exception as e:
        if rank == 0:
            print(f"Warning: Failed to init device_mesh: {e}")
        # 创建 mock device mesh 作为 fallback
        class MockDeviceMesh:
            def __init__(self, shape):
                self.shape = shape
            def get_group(self):
                return torch.distributed.group.WORLD if torch.distributed.is_initialized() else None
        dwm.common.global_state["device_mesh"] = MockDeviceMesh((world_size,))

    # 加载配置
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # 移除不需要的配置
    if "optimizer" in config:
        del config["optimizer"]

    # 修改配置使用 DDP
    if "distribution_framework" in config["pipeline"]["common_config"]:
        config["pipeline"]["common_config"]["distribution_framework"] = "ddp"
        if "ddp_wrapper_settings" in config["pipeline"]["common_config"]:
            ddp_settings = config["pipeline"]["common_config"]["ddp_wrapper_settings"].copy()
            for key in ["sharding_strategy", "device_mesh", "auto_wrap_policy", "mixed_precision"]:
                ddp_settings.pop(key, None)
            config["pipeline"]["common_config"]["ddp_wrapper_settings"] = ddp_settings
        for key in ["t5_fsdp_wrapper_settings", "text_encoder_fsdp_wrapper_settings"]:
            if key in config["pipeline"]["common_config"]:
                del config["pipeline"]["common_config"][key]

    # 设置其他全局状态
    if "global_state" in config:
        for k, v in config["global_state"].items():
            if k == "device_mesh":
                continue
            try:
                dwm.common.global_state[k] = dwm.common.create_instance_from_config(v)
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to set {k}: {e}")

    # 初始化
    os.makedirs(args.output_folder, exist_ok=True)
    ode_generator = ODEGenerator(config, device=device)

    # 加载数据集
    training_dataset = dwm.common.create_instance_from_config(config["training_dataset"])
    total_samples = min(len(training_dataset), args.start_idx + args.num_samples) if args.num_samples else len(training_dataset)

    # 同步所有进程，确保 dataset 加载完成
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank == 0:
        print(f"Dataset size: {total_samples}")
        print(f"Samples per GPU: {math.ceil((total_samples - args.start_idx) / world_size)}")

    #### hard code here, few-step config
    SELECT_TIME = [1000, 748, 502, 247]

    # 检查已生成的样本，实现断点续传
    existing_files = glob.glob(os.path.join(args.output_folder, "*.pt"))
    if existing_files:
        existing_indices = [int(os.path.basename(f).replace(".pt", "")) for f in existing_files]
        existing_indices = sorted(existing_indices)
        start_from_idx = max(existing_indices) + 1
        if rank == 0:
            print(f"Found {len(existing_indices)} existing samples, resuming from index {start_from_idx}")
    else:
        start_from_idx = 0
        if rank == 0:
            print("No existing samples found, starting from index 0")

    # 使用 DataLoader 进行数据加载，num_workers=8
    from dwm.datasets.common import CollateFnIgnoring
    ddp = torch.distributed.is_initialized()
    if ddp:
        training_datasampler = torch.utils.data.distributed.DistributedSampler(
            training_dataset, shuffle=True, seed=config.get("generator_seed", 0))
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=1,
            num_workers=8,
            sampler=training_datasampler,
            collate_fn=CollateFnIgnoring(keys=["clip_text"]),
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=1,
            num_workers=8,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.get("generator_seed", 0)),
            collate_fn=CollateFnIgnoring(keys=["clip_text"]),
            persistent_workers=True,
            prefetch_factor=2,
        )

    samples_per_rank = math.ceil((total_samples - args.start_idx) / world_size)

    for idx, batch in enumerate(tqdm(
        training_dataloader,
        desc=f"GPU {rank}/{world_size}",
        disable=rank != 0  # 只在 rank 0 显示进度条
    )):
        global_idx = args.start_idx + idx * world_size + rank

        if global_idx >= total_samples:
            continue

        # 断点续传：跳过已生成的样本
        if global_idx < start_from_idx:
            continue

        try:
            ode_data = ode_generator.generate_ode_pairs(batch, args.seed)
            capture_times = ode_data['timestamp']
            reference_frame_count = ode_generator.inference_config.get("reference_frame_count", 0)
            # 取第一个非 ref 帧位置的 timestep 作为代表（ref 帧的 timestep 被置为 0，不能用来索引）
            ts_t_idx = reference_frame_count
            ts = [t[0][ts_t_idx][0].item() for t in capture_times]
            selected_index = find_nearest_indices_brute(ts, SELECT_TIME).tolist()
            selected_index.append(-1)

            # 采样选中的 latents 和 timestamps
            selected_latents = [ode_data['capture_latents'][idx] for idx in selected_index]
            # selected_timestamps = [ode_data['timestamp'][idx] for idx in selected_index]
            selected_timestamps = [ts[idx] for idx in selected_index]

            # 准备保存的数据
            save_data = {
                "selected_latents": selected_latents,  # [num_selected, batch, seq, view, c, h, w]
                "selected_timestamps": selected_timestamps,
                "selected_index": selected_index,
                "SELECT_TIME": SELECT_TIME,
                "batch": ode_data['batch'],
                "latent_shape": ode_data['latent_shape'],
                "global_idx": global_idx
            }

            torch.save(save_data,
                       os.path.join(args.output_folder, f"{global_idx:06d}.pt"))

            # 可视化 ODE 轨迹
            vis_dir = os.path.join(args.output_folder, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{global_idx:06d}.png")
            try:
                ode_generator.visualize_ode_trajectory(
                    selected_latents, selected_timestamps, vis_path)
            except Exception as vis_e:
                print(f"[Rank {rank}] Visualization error {global_idx}: {vis_e}")

            if rank == 0 and (idx + 1) % 10 == 0:
                print(f"[Rank {rank}] Saved {idx + 1}/{samples_per_rank} samples")

        except Exception as e:
            print(f"[Rank {rank}] Error {global_idx}: {e}")
            import traceback
            if rank == 0:
                traceback.print_exc()

    # 同步所有进程完成
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank == 0:
        print("=" * 50)
        print(f"ODE pair generation complete!")
        print(f"Output saved to: {args.output_folder}")
        print(f"Total samples: {total_samples - args.start_idx}")
        print("=" * 50)


if __name__ == "__main__":
    main()