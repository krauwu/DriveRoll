"""
ODE NuScenes 数据集
用于加载 generate_nuscenes_ode_pairs.py 生成的 ODE 轨迹数据
"""

import os
import torch
import glob
import numpy as np
from collections import defaultdict
import dwm.datasets.common


class ODENuScenesDataset(torch.utils.data.Dataset):
    """
    ODE NuScenes 数据集

    Args:
        data_folder: 包含 .pt 文件的文件夹路径
        split: "train" 或 "val"，用于文件名过滤
        denoising_step_list: 训练时使用的去噪步数列表
        transform_list: 数据变换列表
    """

    def __init__(
        self,
        data_folder: str,
        split: str = "train",
        denoising_step_list=None,
        transform_list=None
    ):
        self.data_folder = data_folder
        self.denoising_step_list = denoising_step_list or [1000, 748, 502, 247, 0]

        # 查找所有 .pt 文件
        all_files = sorted(glob.glob(os.path.join(data_folder, "*.pt")))

        # 根据 split 划分数据集（95/5 划分）
        if split == "train":
            self.file_list = all_files[:int(len(all_files) * 0.95)]
        elif split == "val":
            self.file_list = all_files[int(len(all_files) * 0.95):]
        else:
            self.file_list = all_files

        self.transform_list = transform_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        # 加载 .pt 文件
        data = torch.load(self.file_list[index], map_location="cpu")

        # 提取 ODE latents 和 batch 数据
        selected_latents = data["selected_latents"]  # list of tensors
        batch = data["batch"]
        selected_timestamps = torch.tensor(data["selected_timestamps"])

        # 构建 ode_latent tensor
        ode_latent = torch.cat(selected_latents, dim=0)  # [num_selected, seq, view, c, h, w]
        # batch['clip_text'] = [
        #                 [
        #                     [element[0] for element in row] 
        #                     for row in matrix
        #                 ] 
        #                 for matrix in batch['clip_text']
        #             ] # 1, seq, view
        # 构建 batch 字典，兼容原始训练逻辑
        result = {
            "ode_latent": ode_latent,
            "clip_text": batch["clip_text"][0],
            "camera_intrinsics": batch["camera_intrinsics"][0],
            "camera_transforms": batch["camera_transforms"][0],
            "image_size": batch["image_size"][0],
            "ego_transforms": batch["ego_transforms"][0],
            "pts": batch["pts"][0],
            "fps": batch["fps"][0],
            "timestamps": selected_timestamps,
            "ODE_TIME": torch.tensor(self.denoising_step_list)
        }

        # 添加可选字段
        if "3dbox_images" in batch:
            result["3dbox_images"] = batch["3dbox_images"][0]
        if "hdmap_images" in batch:
            result["hdmap_images"] = batch["hdmap_images"][0]

        # 应用变换
        if self.transform_list is not None:
            for transform in self.transform_list:
                if transform.get("is_dynamic_transform", False):
                    result = transform["transform"](result)
                else:
                    result[transform["new_key"]] = dwm.datasets.common.DatasetAdapter.apply_transform(
                        transform["transform"],
                        result[transform["old_key"]],
                        transform.get("stack", True)
                    )

        return result
