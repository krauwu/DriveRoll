r"""
DGGT Benchmark 脚本
对生成图像集和GT图像集分别进行推理，对比重建指标

用法:
 CUDA_VISIBLE_DEVICES=4 python benchmark.py \
      --generated_dir ./output/for_dggt/vis_validation_generator/1/generated_rgb/cam_1 \
      --gt_dir ./output/for_dggt/vis_validation_generator/1/gt_rgb/cam_1 \
      --gt_extrinsic_dir ./output/for_dggt/vis_validation_generator/1/gt_camera_params/ego_transforms/cam_1 \
      --ckpt_path pretrained/model_latest_waymo.pt \
      --output_path benchmark_output/ \
      --sequence_length 4 \
      --start_idx 0 \
      --frame_interval 4 \
      --num_frames 20
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON encoder"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.rotation import mat_to_quat, quat_to_mat
from gsplat.rendering import rasterization


# ============== 复用 inference_video.py 中的函数 ==============

def load_and_preprocess_frames(frame_list, target_size=518):
    """从帧列表加载并预处理图像"""
    images = []
    to_tensor = T.ToTensor()

    for frame in frame_list:
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0

        img = Image.fromarray((frame * 255).astype(np.uint8))
        img = img.convert("RGB")

        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        images.append(img)

    images = torch.stack(images)
    return images


def compute_metrics(img1, img2, lpips_fn):
    """计算 PSNR, SSIM, LPIPS 指标"""
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    lpips_device = next(lpips_fn.parameters()).device

    psnr_list, ssim_list, lpips_list = [], [], []

    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()

        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)

        img1_lpips = (img1[i].unsqueeze(0) * 2 - 1).to(lpips_device)
        img2_lpips = (img2[i].unsqueeze(0) * 2 - 1).to(lpips_device)
        lpips_val = lpips_fn(img1_lpips, img2_lpips)

        psnr_list.append(float(psnr))
        ssim_list.append(float(ssim))
        lpips_list.append(float(lpips_val.item()))

    psnr_avg = sum(psnr_list) / len(psnr_list)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    lpips_avg = sum(lpips_list) / len(lpips_list)

    per_frame_metrics = [
        {'frame_idx': i, 'psnr': psnr_list[i], 'ssim': ssim_list[i], 'lpips': lpips_list[i]}
        for i in range(len(psnr_list))
    ]

    return psnr_avg, ssim_avg, lpips_avg, per_frame_metrics


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    """时间一致性函数"""
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t)**2)
    alpha_ = alpha * conf
    return alpha_.float()


def quat_slerp(q1, q2, t):
    """四元数球面线性插值"""
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    DOT_THRESHOLD = 0.9995
    if torch.is_tensor(t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t

    linear_mask = (dot > DOT_THRESHOLD).squeeze(-1)

    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta = torch.sin(theta)
    sin_theta = torch.where(sin_theta.abs() < 1e-6, torch.ones_like(sin_theta), sin_theta)

    if torch.is_tensor(t) and t.dim() > 0:
        s1 = torch.sin((1 - t) * theta) / sin_theta
        s2 = torch.sin(t * theta) / sin_theta
    else:
        s1 = torch.sin((1 - t) * theta) / sin_theta
        s2 = torch.sin(t * theta) / sin_theta

    result = s1 * q1 + s2 * q2

    if linear_mask.any():
        linear_result = (1 - t) * q1 + t * q2
        linear_result = linear_result / (linear_result.norm(dim=-1, keepdim=True) + 1e-8)
        if torch.is_tensor(t) and t.dim() > 0:
            result = torch.where(linear_mask.unsqueeze(-1), linear_result, result)
        else:
            result = torch.where(linear_mask.unsqueeze(-1), linear_result, result)

    result = result / (result.norm(dim=-1, keepdim=True) + 1e-8)
    return result


def interpolate_cameras(extrinsics1, intrinsics1, extrinsics2, intrinsics2, t):
    """在两个相机之间进行插值"""
    device = extrinsics1.device

    R1 = extrinsics1[:3, :3]
    R2 = extrinsics2[:3, :3]
    t1 = extrinsics1[:3, 3]
    t2 = extrinsics2[:3, 3]

    q1 = mat_to_quat(R1.unsqueeze(0)).squeeze(0)
    q2 = mat_to_quat(R2.unsqueeze(0)).squeeze(0)
    q_interp = quat_slerp(q1, q2, t)
    R_interp = quat_to_mat(q_interp.unsqueeze(0)).squeeze(0)

    t_interp = (1 - t) * t1 + t * t2

    interp_extrinsic = torch.eye(4, device=device)
    interp_extrinsic[:3, :3] = R_interp
    interp_extrinsic[:3, 3] = t_interp

    interp_intrinsic = (1 - t) * intrinsics1 + t * intrinsics2

    return interp_extrinsic, interp_intrinsic


def generate_interpolated_cameras(extrinsics, intrinsics, num_interpolations=3):
    """在相邻帧之间生成插值相机"""
    S = extrinsics.shape[0]
    device = extrinsics.device

    all_extrinsics = []
    all_intrinsics = []
    frame_indices = []
    is_interpolated = []

    for i in range(S):
        all_extrinsics.append(extrinsics[i])
        all_intrinsics.append(intrinsics[i])
        frame_indices.append(i)
        is_interpolated.append(False)

        if i < S - 1:
            for j in range(1, num_interpolations + 1):
                t = j / (num_interpolations + 1)
                interp_ext, interp_int = interpolate_cameras(
                    extrinsics[i], intrinsics[i],
                    extrinsics[i + 1], intrinsics[i + 1],
                    t
                )
                all_extrinsics.append(interp_ext)
                all_intrinsics.append(interp_int)
                frame_indices.append(i)
                is_interpolated.append(True)

    all_extrinsics = torch.stack(all_extrinsics, dim=0)
    all_intrinsics = torch.stack(all_intrinsics, dim=0)
    is_interpolated = torch.tensor(is_interpolated, device=device)

    return all_extrinsics, all_intrinsics, frame_indices, is_interpolated


def process_sequence(model, images, device, dtype, num_interpolations=3):
    """处理一个图像序列"""
    S = images.shape[0]
    H, W = images.shape[-2:]

    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
        extrinsic = extrinsics[0]
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
        extrinsic = torch.cat([extrinsic, bottom], dim=1)
        intrinsic = intrinsics[0]

        depth_map = predictions["depth"][0]
        point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None, ...]
        point_map = torch.from_numpy(point_map).to(device).float()

        gs_map = predictions["gs_map"]
        gs_conf = predictions["gs_conf"]
        dy_map = predictions["dynamic_conf"].squeeze(-1)

        timestamps = torch.linspace(0, S / 4, S, device=device)

        bg_mask = torch.ones(1, S, H, W, dtype=torch.bool, device=device)

        static_mask = (bg_mask & (dy_map < 0.5))
        static_points = point_map[static_mask].reshape(-1, 3)
        gs_dynamic_list = dy_map[static_mask].sigmoid()
        static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
        static_opacity = static_opacity * (1 - gs_dynamic_list)
        static_gs_conf = gs_conf[static_mask]
        frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
        gs_timestamps = timestamps[frame_idx]

        dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
        for i in range(S):
            point_map_i = point_map[0, i]
            bg_mask_i = bg_mask[0, i]

            dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
            dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[0, i:i+1], bg_mask[0, i:i+1])
            gs_dynamic_list_i = dy_map[0, i][bg_mask_i].sigmoid()
            dynamic_opacity = dynamic_opacity * gs_dynamic_list_i

            dynamic_points.append(dynamic_point)
            dynamic_rgbs.append(dynamic_rgb)
            dynamic_opacitys.append(dynamic_opacity)
            dynamic_scales.append(dynamic_scale)
            dynamic_rotations.append(dynamic_rotation)

        # 生成插值相机
        if num_interpolations > 0:
            all_extrinsics, all_intrinsics, frame_indices_list, is_interpolated = generate_interpolated_cameras(
                extrinsic, intrinsic, num_interpolations
            )
        else:
            all_extrinsics = extrinsic
            all_intrinsics = intrinsic
            frame_indices_list = list(range(S))
            is_interpolated = torch.zeros(S, dtype=torch.bool, device=device)

        total_cameras = all_extrinsics.shape[0]
        chunked_renders, chunked_alphas = [], []

        for cam_idx in range(total_cameras):
            if num_interpolations > 0:
                orig_frame_idx = frame_indices_list[cam_idx]
                if is_interpolated[cam_idx]:
                    interp_position = (cam_idx % (num_interpolations + 1))
                    t_ratio = interp_position / (num_interpolations + 1)
                    t0 = timestamps[orig_frame_idx] * (1 - t_ratio) + timestamps[min(orig_frame_idx + 1, S - 1)] * t_ratio
                else:
                    t0 = timestamps[orig_frame_idx]
            else:
                orig_frame_idx = cam_idx
                t0 = timestamps[cam_idx]

            static_opacity_ = static_opacity  # 静态元素不使用decay
            static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]

            if num_interpolations > 0 and is_interpolated[cam_idx]:
                dynamic_idx = frame_indices_list[cam_idx]
            else:
                dynamic_idx = orig_frame_idx

            if dynamic_points:
                world_points, rgbs, opacity, scales, rotation = concat_list(
                    static_gs_list,
                    [dynamic_points[dynamic_idx], dynamic_rgbs[dynamic_idx], dynamic_opacitys[dynamic_idx], dynamic_scales[dynamic_idx], dynamic_rotations[dynamic_idx]]
                )
            else:
                world_points, rgbs, opacity, scales, rotation = static_gs_list

            renders_chunk, alphas_chunk, _ = rasterization(
                means=world_points,
                quats=rotation,
                scales=scales,
                opacities=opacity,
                colors=rgbs,
                viewmats=all_extrinsics[cam_idx:cam_idx+1],
                Ks=all_intrinsics[cam_idx:cam_idx+1],
                width=W,
                height=H,
                render_mode='RGB+ED',
            )
            chunked_renders.append(renders_chunk)
            chunked_alphas.append(alphas_chunk)

        renders = torch.cat(chunked_renders, dim=0)
        depth_maps = renders[..., -1]
        renders = renders[..., :-1]
        alphas = torch.cat(chunked_alphas, dim=0)

        bg_render = torch.ones_like(renders)
        renders = alphas * renders + (1 - alphas) * bg_render

        rendered_image = renders.permute(0, 3, 1, 2)
        depth_output = depth_maps.unsqueeze(1)

    return rendered_image, depth_output, predictions, extrinsic, intrinsic, is_interpolated, frame_indices_list, point_map, depth_map


# ============== Benchmark 特有函数 ==============

def align_camera_trajectory(camera_positions, camera_directions):
    """
    对齐相机轨迹坐标系，使得：
    1. 轨迹尽量接近于XY平面
    2. 第一帧的相机朝向沿+X方向
    """
    N = camera_positions.shape[0]

    # Step 1: 将轨迹中心移到原点
    center = camera_positions.mean(axis=0)
    positions_centered = camera_positions - center

    # Step 2: 使用PCA找到轨迹的主平面
    U, S, Vt = np.linalg.svd(positions_centered, full_matrices=False)

    R_pca = Vt

    if np.linalg.det(R_pca) < 0:
        R_pca[2] = -R_pca[2]

    positions_rotated = positions_centered @ R_pca.T
    directions_rotated = camera_directions @ R_pca.T

    # Step 3: 使第一帧相机朝向对齐到+X方向
    first_dir = directions_rotated[0]
    x_axis = np.array([1.0, 0.0, 0.0])
    first_dir_norm = first_dir / (np.linalg.norm(first_dir) + 1e-8)

    cross = np.cross(first_dir_norm, x_axis)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-6:
        if np.dot(first_dir_norm, x_axis) > 0:
            R_align = np.eye(3)
        else:
            R_align = np.diag([1.0, -1.0, -1.0])
    else:
        cross_normalized = cross / cross_norm
        cos_angle = np.dot(first_dir_norm, x_axis)
        sin_angle = cross_norm

        K = np.array([
            [0, -cross_normalized[2], cross_normalized[1]],
            [cross_normalized[2], 0, -cross_normalized[0]],
            [-cross_normalized[1], cross_normalized[0], 0]
        ])
        R_align = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)

    R_final = R_align @ R_pca

    aligned_positions = positions_centered @ R_final.T
    aligned_directions = camera_directions @ R_final.T

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_final
    transform_matrix[:3, 3] = -R_final @ center

    return aligned_positions, aligned_directions, transform_matrix


def umeyama_alignment(src_points, dst_points):
    """
    Umeyama算法：求解最优的Sim(3)变换 (s, R, t)
    使得 dst = s * R @ src + t

    Args:
        src_points: 源点云 [N, 3] (预测轨迹，无尺度)
        dst_points: 目标点云 [N, 3] (GT轨迹，有尺度)

    Returns:
        scale: 尺度因子
        R: 旋转矩阵 [3, 3]
        t: 平移向量 [3]
    """
    N = src_points.shape[0]

    # 计算质心
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)

    # 中心化
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    # 计算方差
    src_var = np.sum(src_centered ** 2) / N

    # 计算协方差矩阵
    cov = dst_centered.T @ src_centered / N

    # SVD分解
    U, S, Vt = np.linalg.svd(cov)

    # 计算旋转矩阵
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt

    # 计算尺度因子
    scale = np.sum(S) * d / src_var

    # 计算平移向量
    t = dst_mean - scale * R @ src_mean

    return scale, R, t


def align_rotation_global(pred_rotations, gt_rotations):
    """
    求解全局旋转对齐矩阵 R_global
    使得 gt_rotations ≈ R_global @ pred_rotations

    Args:
        pred_rotations: 预测旋转矩阵 [N, 3, 3]
        gt_rotations: GT旋转矩阵 [N, 3, 3]

    Returns:
        R_global: 全局旋转矩阵 [3, 3]
    """
    N = pred_rotations.shape[0]

    # 计算每帧的 M_i = R_gt @ R_pred^T
    M_sum = np.zeros((3, 3))
    for i in range(N):
        M_i = gt_rotations[i] @ pred_rotations[i].T
        M_sum += M_i

    M_avg = M_sum / N

    # SVD投影到SO(3)
    U, _, Vt = np.linalg.svd(M_avg)
    R_global = U @ Vt

    # 确保是有效旋转矩阵
    if np.linalg.det(R_global) < 0:
        U[:, -1] *= -1
        R_global = U @ Vt

    return R_global


def rotation_error_deg(R1, R2):
    """
    计算两个旋转矩阵之间的角度误差（度）

    Args:
        R1, R2: 旋转矩阵 [3, 3]

    Returns:
        angle: 角度误差（度）
    """
    R_diff = R1 @ R2.T
    # 使用旋转矩阵的trace计算角度
    trace = np.trace(R_diff)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle)


def compute_pose_metrics(pred_extrinsics, gt_extrinsics):
    """
    计算位姿评估指标

    Args:
        pred_extrinsics: 预测的外参矩阵 [N, 4, 4] (world-to-camera)
        gt_extrinsics: GT外参矩阵 [N, 4, 4] (world-to-camera)

    Returns:
        metrics: 包含各项指标的字典
        aligned_pred_positions: 对齐后的预测位置 [N, 3]
        aligned_pred_rotations: 对齐后的预测旋转 [N, 3, 3]
    """
    N = pred_extrinsics.shape[0]

    # 提取位置和旋转
    pred_positions = []
    pred_rotations = []
    gt_positions = []
    gt_rotations = []

    for i in range(N):
        # 预测的camera-to-world
        pred_ext_c2w = np.linalg.inv(pred_extrinsics[i])
        pred_positions.append(pred_ext_c2w[:3, 3])
        pred_rotations.append(pred_ext_c2w[:3, :3])

        # GT的camera-to-world
        gt_ext_c2w = np.linalg.inv(gt_extrinsics[i])
        gt_positions.append(gt_ext_c2w[:3, 3])
        gt_rotations.append(gt_ext_c2w[:3, :3])

    pred_positions = np.array(pred_positions)
    pred_rotations = np.array(pred_rotations)
    gt_positions = np.array(gt_positions)
    gt_rotations = np.array(gt_rotations)

    # ===== 尝试4种组合：对齐前正/反序 × 对齐后正/反序 =====
    def do_alignment(pred_pos, pred_rot, gt_pos, gt_rot):
        """执行对齐并返回结果"""
        scale, R_sim, t = umeyama_alignment(pred_pos, gt_pos)
        aligned_pos = scale * (R_sim @ pred_pos.T).T + t
        aligned_rot = np.array([R_sim @ R for R in pred_rot])
        R_global = align_rotation_global(aligned_rot, gt_rot)
        aligned_rot = np.array([R_global @ R for R in aligned_rot])
        error = np.mean(np.linalg.norm(aligned_pos - gt_pos, axis=1))
        return aligned_pos, aligned_rot, scale, error

    best_error = float('inf')
    best_result = None
    best_desc = ""

    # 组合1: 对齐前正序, 对齐后正序
    aligned_pos, aligned_rot, scale, error = do_alignment(pred_positions, pred_rotations, gt_positions, gt_rotations)
    if error < best_error:
        best_error = error
        best_result = (aligned_pos, aligned_rot, scale)
        best_desc = "正序-正序"

    # 组合2: 对齐前正序, 对齐后反序
    aligned_pos_flip = aligned_pos[::-1].copy()
    aligned_rot_flip = aligned_rot[::-1].copy()
    error = np.mean(np.linalg.norm(aligned_pos_flip - gt_positions, axis=1))
    if error < best_error:
        best_error = error
        best_result = (aligned_pos_flip, aligned_rot_flip, scale)
        best_desc = "正序-反序"

    # 组合3: 对齐前反序, 对齐后正序
    pred_pos_flip = pred_positions[::-1].copy()
    pred_rot_flip = pred_rotations[::-1].copy()
    aligned_pos, aligned_rot, scale, error = do_alignment(pred_pos_flip, pred_rot_flip, gt_positions, gt_rotations)
    if error < best_error:
        best_error = error
        best_result = (aligned_pos, aligned_rot, scale)
        best_desc = "反序-正序"

    # 组合4: 对齐前反序, 对齐后反序
    aligned_pos_flip = aligned_pos[::-1].copy()
    aligned_rot_flip = aligned_rot[::-1].copy()
    error = np.mean(np.linalg.norm(aligned_pos_flip - gt_positions, axis=1))
    if error < best_error:
        best_error = error
        best_result = (aligned_pos_flip, aligned_rot_flip, scale)
        best_desc = "反序-反序"

    aligned_pred_positions, aligned_pred_rotations, scale = best_result
    print(f"  最佳对齐方式: {best_desc}, 误差: {best_error:.4f}")

    # ===== 计算各项指标 =====
    position_errors = np.linalg.norm(aligned_pred_positions - gt_positions, axis=1)
    ate_rmse = np.sqrt(np.mean(position_errors ** 2))
    ate_mean = np.mean(position_errors)
    ate_median = np.median(position_errors)
    ate_std = np.std(position_errors)

    rotation_errors = []
    for i in range(N):
        rot_err = rotation_error_deg(aligned_pred_rotations[i], gt_rotations[i])
        rotation_errors.append(rot_err)
    rotation_errors = np.array(rotation_errors)
    rot_mean = np.mean(rotation_errors)
    rot_median = np.median(rotation_errors)
    rot_std = np.std(rotation_errors)

    trans_errors = []
    rot_rel_errors = []
    for i in range(N - 1):
        pred_rel_pos = aligned_pred_positions[i + 1] - aligned_pred_positions[i]
        pred_rel_rot = aligned_pred_rotations[i + 1] @ aligned_pred_rotations[i].T
        gt_rel_pos = gt_positions[i + 1] - gt_positions[i]
        gt_rel_rot = gt_rotations[i + 1] @ gt_rotations[i].T
        trans_errors.append(np.linalg.norm(pred_rel_pos - gt_rel_pos))
        rot_rel_errors.append(rotation_error_deg(pred_rel_rot, gt_rel_rot))

    rpe_trans = np.sqrt(np.mean(np.array(trans_errors) ** 2)) if trans_errors else 0.0
    rpe_rot = np.mean(rot_rel_errors) if rot_rel_errors else 0.0

    metrics = {
        'ate_rmse': ate_rmse,
        'ate_mean': ate_mean,
        'ate_median': ate_median,
        'ate_std': ate_std,
        'scale': scale,
        'rot_mean': rot_mean,
        'rot_median': rot_median,
        'rot_std': rot_std,
        'rpe_trans': rpe_trans,
        'rpe_rot': rpe_rot,
        'per_frame_position_error': position_errors.tolist(),
        'per_frame_rotation_error': rotation_errors.tolist(),
        'alignment_mode': best_desc,  # 对齐方式
    }

    return metrics, aligned_pred_positions, aligned_pred_rotations, gt_positions, gt_rotations


def plot_pose_metrics_curve(gen_metrics, gt_metrics, output_path, stride=1):
    """
    绘制位姿误差曲线（折线图对比 Generated vs GT）

    Args:
        gen_metrics: Generated 数据集的位姿指标列表
        gt_metrics: GT 数据集的位姿指标列表
        output_path: 输出目录
        stride: 滑动窗口步长
    """
    def extract_data(metrics, stride):
        """提取数据并计算起始帧"""
        if not metrics:
            return None
        start_frames = [i * stride for i in range(len(metrics))]
        return {
            'start_frame': start_frames,
            'ate_rmse': [m['ate_rmse'] for m in metrics],
            'ate_mean': [m['ate_mean'] for m in metrics],
            'scale': [m['scale'] for m in metrics],
            'rot_mean': [m['rot_mean'] for m in metrics],
            'rpe_trans': [m['rpe_trans'] for m in metrics],
            'rpe_rot': [m['rpe_rot'] for m in metrics],
        }

    gen_data = extract_data(gen_metrics, stride)
    gt_data = extract_data(gt_metrics, stride)

    # 创建图表 - 折线图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # ATE RMSE
    ax = axes[0, 0]
    if gt_data:
        ax.plot(gt_data['start_frame'], gt_data['ate_rmse'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gt_data['ate_rmse']), color='b', linestyle='--', alpha=0.5,
                   label=f'GT avg: {np.mean(gt_data["ate_rmse"]):.4f}m')
    if gen_data:
        ax.plot(gen_data['start_frame'], gen_data['ate_rmse'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gen_data['ate_rmse']), color='r', linestyle='--', alpha=0.5,
                   label=f'Gen avg: {np.mean(gen_data["ate_rmse"]):.4f}m')
    ax.set_xlabel('Window Start Frame')
    ax.set_ylabel('ATE RMSE (m)')
    ax.set_title('Absolute Trajectory Error (RMSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ATE Mean
    ax = axes[0, 1]
    if gt_data:
        ax.plot(gt_data['start_frame'], gt_data['ate_mean'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gt_data['ate_mean']), color='b', linestyle='--', alpha=0.5,
                   label=f'GT avg: {np.mean(gt_data["ate_mean"]):.4f}m')
    if gen_data:
        ax.plot(gen_data['start_frame'], gen_data['ate_mean'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gen_data['ate_mean']), color='r', linestyle='--', alpha=0.5,
                   label=f'Gen avg: {np.mean(gen_data["ate_mean"]):.4f}m')
    ax.set_xlabel('Window Start Frame')
    ax.set_ylabel('ATE Mean (m)')
    ax.set_title('Absolute Trajectory Error (Mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scale
    ax = axes[0, 2]
    if gt_data:
        ax.plot(gt_data['start_frame'], gt_data['scale'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gt_data['scale']), color='b', linestyle='--', alpha=0.5,
                   label=f'GT avg: {np.mean(gt_data["scale"]):.4f}')
    if gen_data:
        ax.plot(gen_data['start_frame'], gen_data['scale'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gen_data['scale']), color='r', linestyle='--', alpha=0.5,
                   label=f'Gen avg: {np.mean(gen_data["scale"]):.4f}')
    ax.set_xlabel('Window Start Frame')
    ax.set_ylabel('Scale Factor')
    ax.set_title('Estimated Scale Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotation Error
    ax = axes[1, 0]
    if gt_data:
        ax.plot(gt_data['start_frame'], gt_data['rot_mean'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gt_data['rot_mean']), color='b', linestyle='--', alpha=0.5,
                   label=f'GT avg: {np.mean(gt_data["rot_mean"]):.2f}°')
    if gen_data:
        ax.plot(gen_data['start_frame'], gen_data['rot_mean'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gen_data['rot_mean']), color='r', linestyle='--', alpha=0.5,
                   label=f'Gen avg: {np.mean(gen_data["rot_mean"]):.2f}°')
    ax.set_xlabel('Window Start Frame')
    ax.set_ylabel('Rotation Error (°)')
    ax.set_title('Rotation Error (Mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RPE Translation
    ax = axes[1, 1]
    if gt_data:
        ax.plot(gt_data['start_frame'], gt_data['rpe_trans'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gt_data['rpe_trans']), color='b', linestyle='--', alpha=0.5,
                   label=f'GT avg: {np.mean(gt_data["rpe_trans"]):.4f}m')
    if gen_data:
        ax.plot(gen_data['start_frame'], gen_data['rpe_trans'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax.axhline(y=np.mean(gen_data['rpe_trans']), color='r', linestyle='--', alpha=0.5,
                   label=f'Gen avg: {np.mean(gen_data["rpe_trans"]):.4f}m')
    ax.set_xlabel('Window Start Frame')
    ax.set_ylabel('RPE Translation (m)')
    ax.set_title('Relative Pose Error (Translation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_lines = ["Pose Evaluation Summary", "=" * 40, ""]

    if gen_data:
        summary_lines.extend([
            "Generated:",
            f"  ATE RMSE: {np.mean(gen_data['ate_rmse']):.4f} ± {np.std(gen_data['ate_rmse']):.4f} m",
            f"  Rot Err:  {np.mean(gen_data['rot_mean']):.2f} ± {np.std(gen_data['rot_mean']):.2f} °",
            f"  Scale:    {np.mean(gen_data['scale']):.4f}",
            ""
        ])
    if gt_data:
        summary_lines.extend([
            "GT:",
            f"  ATE RMSE: {np.mean(gt_data['ate_rmse']):.4f} ± {np.std(gt_data['ate_rmse']):.4f} m",
            f"  Rot Err:  {np.mean(gt_data['rot_mean']):.2f} ± {np.std(gt_data['rot_mean']):.2f} °",
            f"  Scale:    {np.mean(gt_data['scale']):.4f}",
        ])

    summary_text = "\n".join(summary_lines)
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(output_path, 'pose_metrics_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"位姿误差曲线已保存: {save_path}")


def visualize_trajectory_comparison_aligned(pred_positions, pred_rotations,
                                             gt_positions, gt_rotations,
                                             output_path, window_idx):
    """
    可视化对齐后的预测轨迹和GT轨迹
    GT轨迹保持原始状态，预测轨迹经过Sim(3)对齐到GT坐标系

    Args:
        pred_positions: 对齐后的预测位置 [N, 3]
        pred_rotations: 对齐后的预测旋转 [N, 3, 3]
        gt_positions: GT位置 [N, 3] (原始，未变换)
        gt_rotations: GT旋转 [N, 3, 3] (原始，未变换)
        output_path: 输出目录
        window_idx: 窗口索引
    """
    N = pred_positions.shape[0]

    # 计算误差
    position_errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    rotation_errors = [rotation_error_deg(pred_rotations[i], gt_rotations[i]) for i in range(N)]

    # 创建可视化
    fig = plt.figure(figsize=(16, 10))

    # 以GT轨迹为基准确定坐标系范围，添加预测轨迹的扩展
    gt_x_range = [gt_positions[:, 0].min() - 0.5, gt_positions[:, 0].max() + 0.5]
    gt_y_range = [gt_positions[:, 1].min() - 0.5, gt_positions[:, 1].max() + 0.5]
    gt_z_range = [gt_positions[:, 2].min() - 0.5, gt_positions[:, 2].max() + 0.5]

    # 如果预测轨迹超出GT范围，适当扩展
    x_range = [min(gt_x_range[0], pred_positions[:, 0].min() - 0.5),
               max(gt_x_range[1], pred_positions[:, 0].max() + 0.5)]
    y_range = [min(gt_y_range[0], pred_positions[:, 1].min() - 0.5),
               max(gt_y_range[1], pred_positions[:, 1].max() + 0.5)]
    z_range = [min(gt_z_range[0], pred_positions[:, 2].min() - 0.5),
               max(gt_z_range[1], pred_positions[:, 2].max() + 0.5)]

    # ===== 子图1: 3D轨迹对比 =====
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    ax1.scatter(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2],
                c='red', s=50, marker='o', label='Predicted (aligned)', alpha=0.8)
    ax1.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2],
             'r-', alpha=0.5, linewidth=2)

    ax1.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                c='blue', s=50, marker='^', label='GT', alpha=0.8)
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
             'b-', alpha=0.5, linewidth=2)

    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Window {window_idx}: 3D Trajectory\n(GT in original coords, Pred aligned to GT)')
    ax1.legend()

    # ===== 子图2: XY平面投影 =====
    ax2 = fig.add_subplot(2, 2, 2)

    ax2.scatter(pred_positions[:, 0], pred_positions[:, 1],
                c='red', s=50, marker='o', label='Predicted')
    ax2.plot(pred_positions[:, 0], pred_positions[:, 1], 'r-', alpha=0.5)

    ax2.scatter(gt_positions[:, 0], gt_positions[:, 1],
                c='blue', s=50, marker='^', label='GT')
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1], 'b-', alpha=0.5)

    # 添加帧编号（GT和Pred都标注）
    for i in range(N):
        # GT标注（蓝色）
        ax2.annotate(f'{i}', (gt_positions[i, 0], gt_positions[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')
        # Pred标注（红色）
        ax2.annotate(f'{i}', (pred_positions[i, 0], pred_positions[i, 1]),
                     textcoords="offset points", xytext=(-15, -10), fontsize=8, color='red')

    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Window {window_idx}: XY Plane (meters)\n(GT coords, blue=GT idx, red=Pred idx)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # ===== 子图3: 位置误差 =====
    ax3 = fig.add_subplot(2, 2, 3)

    frame_indices = np.arange(N)
    ax3.bar(frame_indices, position_errors, color='green', alpha=0.7)
    ax3.axhline(y=np.mean(position_errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(position_errors):.4f}m')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title(f'Window {window_idx}: Position Error per Frame')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== 子图4: 旋转误差 =====
    ax4 = fig.add_subplot(2, 2, 4)

    ax4.bar(frame_indices, rotation_errors, color='purple', alpha=0.7)
    ax4.axhline(y=np.mean(rotation_errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(rotation_errors):.2f}°')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Rotation Error (°)')
    ax4.set_title(f'Window {window_idx}: Rotation Error per Frame')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(output_path, f'trajectory_window_{window_idx:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return np.mean(position_errors), np.mean(rotation_errors)


def extract_camera_poses(extrinsics):
    """
    从外参矩阵提取相机位置和朝向

    Args:
        extrinsics: world-to-camera 外参矩阵 [N, 4, 4]

    Returns:
        positions: 相机位置 [N, 3]
        directions: 相机朝向 [N, 3]
    """
    N = extrinsics.shape[0]
    positions = []
    directions = []

    for i in range(N):
        ext = extrinsics[i]
        # camera-to-world = inverse(world-to-camera)
        ext_c2w = np.linalg.inv(ext)
        cam_pos = ext_c2w[:3, 3]  # 相机位置 = c2w的平移部分
        positions.append(cam_pos)

        # 相机朝向 (Z轴负方向在世界坐标系中)
        cam_dir = -ext_c2w[:3, :3][:, 2]  # c2w旋转矩阵的第三列取负
        directions.append(cam_dir)

    return np.array(positions), np.array(directions)


def visualize_trajectory_comparison(pred_positions, pred_directions,
                                     gt_positions, gt_directions,
                                     output_path, window_idx):
    """
    可视化预测轨迹和GT轨迹的对比

    Args:
        pred_positions: 预测的相机位置 [N, 3]
        pred_directions: 预测的相机朝向 [N, 3]
        gt_positions: GT相机位置 [N, 3]
        gt_directions: GT相机朝向 [N, 3]
        output_path: 输出目录
        window_idx: 窗口索引
    """
    # 对齐预测轨迹
    pred_pos_aligned, pred_dir_aligned, transform = align_camera_trajectory(pred_positions, pred_directions)

    # 使用相同的变换对齐GT轨迹
    R = transform[:3, :3]
    t = transform[:3, 3]
    gt_center = gt_positions.mean(axis=0)
    gt_pos_aligned = (gt_positions - gt_center) @ R.T
    gt_dir_aligned = gt_directions @ R.T

    # 创建可视化
    fig = plt.figure(figsize=(16, 6))

    # ===== 子图1: 3D轨迹对比 =====
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    ax1.scatter(pred_pos_aligned[:, 0], pred_pos_aligned[:, 1], pred_pos_aligned[:, 2],
                c='red', s=50, marker='o', label='Predicted', alpha=0.8)
    ax1.plot(pred_pos_aligned[:, 0], pred_pos_aligned[:, 1], pred_pos_aligned[:, 2],
             'r-', alpha=0.5, linewidth=2)

    ax1.scatter(gt_pos_aligned[:, 0], gt_pos_aligned[:, 1], gt_pos_aligned[:, 2],
                c='blue', s=50, marker='^', label='GT', alpha=0.8)
    ax1.plot(gt_pos_aligned[:, 0], gt_pos_aligned[:, 1], gt_pos_aligned[:, 2],
             'b-', alpha=0.5, linewidth=2)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Window {window_idx}: 3D Trajectory')
    ax1.legend()

    # ===== 子图2: XY平面投影 =====
    ax2 = fig.add_subplot(1, 3, 2)

    ax2.scatter(pred_pos_aligned[:, 0], pred_pos_aligned[:, 1],
                c='red', s=50, marker='o', label='Predicted')
    ax2.plot(pred_pos_aligned[:, 0], pred_pos_aligned[:, 1], 'r-', alpha=0.5)

    ax2.scatter(gt_pos_aligned[:, 0], gt_pos_aligned[:, 1],
                c='blue', s=50, marker='^', label='GT')
    ax2.plot(gt_pos_aligned[:, 0], gt_pos_aligned[:, 1], 'b-', alpha=0.5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Window {window_idx}: XY Plane')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # ===== 子图3: 位置误差 =====
    ax3 = fig.add_subplot(1, 3, 3)

    position_error = np.linalg.norm(pred_pos_aligned - gt_pos_aligned, axis=1)
    frame_indices = np.arange(len(position_error))

    ax3.bar(frame_indices, position_error, color='green', alpha=0.7)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Position Error')
    ax3.set_title(f'Window {window_idx}: Position Error\nAvg: {position_error.mean():.4f}')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(output_path, f'trajectory_window_{window_idx:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return position_error.mean()


def read_gt_extrinsics(gt_extrinsic_dir, frame_indices):
    """
    读取GT相机外参

    Args:
        gt_extrinsic_dir: GT外参目录
        frame_indices: 帧索引列表

    Returns:
        gt_extrinsics: [N, 4, 4] camera-to-world矩阵
    """
    ext_files = sorted([f for f in os.listdir(gt_extrinsic_dir) if f.endswith('.npy')])

    if len(ext_files) == 0:
        raise ValueError(f"目录 {gt_extrinsic_dir} 中没有找到npy文件")

    gt_extrinsics_list = []
    for idx in frame_indices:
        if idx >= len(ext_files):
            print(f"警告: 帧索引 {idx} 超出GT外参文件数量 {len(ext_files)}")
            break
        ext = np.load(os.path.join(gt_extrinsic_dir, ext_files[idx]))
        gt_extrinsics_list.append(ext)

    return np.stack(gt_extrinsics_list, axis=0) if gt_extrinsics_list else None


def read_images_from_dir(image_dir, start_idx, num_frames):
    """
    从目录读取所有图像

    Args:
        image_dir: 图像目录
        start_idx: 起始帧索引
        num_frames: 读取帧数（None表示读取全部）

    Returns:
        frames: 图像列表
        frame_indices: 帧索引列表
    """
    print(f"\n从目录读取图像: {image_dir}")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) == 0:
        raise ValueError(f"目录 {image_dir} 中没有找到图像文件")

    # 确定读取范围
    end_idx = len(image_files)
    if num_frames is not None:
        end_idx = min(start_idx + num_frames, len(image_files))

    frame_indices = list(range(start_idx, end_idx))

    if len(frame_indices) == 0:
        raise ValueError(f"没有有效的帧可读取。start_idx={start_idx}, 总图像数={len(image_files)}")

    print(f"读取帧索引: {frame_indices[0]} ~ {frame_indices[-1]}")
    print(f"共 {len(frame_indices)} 帧")

    # 读取图像
    frames = []
    for idx in frame_indices:
        img_file = image_files[idx]
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)

    return frames, frame_indices


def visualize_depth_map(depth_map, colormap=cv2.COLORMAP_JET):
    """
    将深度图可视化为彩色图像

    Args:
        depth_map: 深度图 [H, W] 或 [1, H, W]
        colormap: OpenCV colormap

    Returns:
        可视化后的RGB图像 [H, W, 3]
    """
    if torch.is_tensor(depth_map):
        depth_np = depth_map.detach().cpu().squeeze().numpy()
    else:
        depth_np = np.squeeze(depth_map)

    # 归一化到0-255
    valid_mask = (depth_np > 0) & (~np.isnan(depth_np)) & (~np.isinf(depth_np))
    if valid_mask.any():
        depth_min = depth_np[valid_mask].min()
        depth_max = depth_np[valid_mask].max()
        if depth_max > depth_min:
            depth_normalized = np.zeros_like(depth_np)
            depth_normalized[valid_mask] = (depth_np[valid_mask] - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_normalized = np.zeros_like(depth_np)
    else:
        depth_normalized = np.zeros_like(depth_np)

    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    return depth_rgb


def visualize_point_cloud(point_map, rgb_image=None):
    """
    将点云投影到图像平面进行可视化

    Args:
        point_map: 点云 [H, W, 3]
        rgb_image: 可选的RGB图像用于叠加 [H, W, 3]

    Returns:
        可视化图像 [H, W, 3]
    """
    if torch.is_tensor(point_map):
        points = point_map.detach().cpu().numpy()
    else:
        points = np.array(point_map)

    # 提取xyz
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # 计算深度作为颜色
    valid_mask = (z > 0) & (~np.isnan(z)) & (~np.isinf(z))

    # 创建可视化图像
    h, w = z.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    if valid_mask.any():
        z_min = z[valid_mask].min()
        z_max = z[valid_mask].max()

        if z_max > z_min:
            z_normalized = np.zeros_like(z)
            z_normalized[valid_mask] = ((z[valid_mask] - z_min) / (z_max - z_min) * 255).astype(np.uint8)
        else:
            z_normalized = np.zeros_like(z, dtype=np.uint8)

        # 使用JET colormap
        z_colored = cv2.applyColorMap(z_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        z_rgb = cv2.cvtColor(z_colored, cv2.COLOR_BGR2RGB)

        # 只填充有效区域
        vis[valid_mask] = z_rgb[valid_mask]

    return vis


def save_point_cloud_ply(points, colors, filepath):
    """
    保存点云为PLY文件

    Args:
        points: 点云坐标 [N, 3]
        colors: 点云颜色 [N, 3] (0-255)
        filepath: 保存路径
    """
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.detach().cpu().numpy()

    # 确保颜色在0-255范围内
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    N = points.shape[0]

    with open(filepath, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 写入点云数据
        for i in range(N):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")


def run_inference_on_dataset(model, frames, args, device, dtype, lpips_fn, dataset_name, save_dir=None, gt_frames=None):
    """
    对一个数据集运行完整推理并收集指标

    Args:
        model: VGGT模型
        frames: 所有帧图像列表
        args: 命令行参数（包含stride, frame_interval, sequence_length）
        device: 设备
        dtype: 数据类型
        lpips_fn: LPIPS函数
        dataset_name: 数据集名称
        save_dir: 渲染图保存目录（None则不保存）
        gt_frames: GT帧图像列表（用于拼接对比）

    Returns:
        all_metrics: 所有主帧的指标列表
        all_nvs_metrics: 所有插值帧的指标列表（novel view synthesis）
        all_extrinsics: 所有窗口的外参（用于轨迹可视化）
    """
    total_frames = len(frames)
    seq_len = args.sequence_length
    stride = args.stride
    frame_interval = args.frame_interval

    if args.num_interpolations is None:
        num_interpolations = max(0, frame_interval - 1)
    else:
        num_interpolations = args.num_interpolations

    # 计算窗口数量：最后一个窗口的起始帧索引 + frame_interval*(seq_len-1) < total_frames
    # 起始帧: i*stride
    # 窗口内帧: i*stride + j*frame_interval (j=0,1,...,seq_len-1)
    # 需要: i*stride + (seq_len-1)*frame_interval < total_frames
    max_start_idx = total_frames - (seq_len - 1) * frame_interval
    num_sequences = max(0, (max_start_idx + stride - 1) // stride) if max_start_idx > 0 else 0

    print(f"\n[{dataset_name}] 开始处理: 总帧数={total_frames}, 序列长度={seq_len}, stride={stride}, frame_interval={frame_interval}")
    print(f"[{dataset_name}] 插值视角数={num_interpolations}, 预计处理 {num_sequences} 个序列\n")

    all_metrics = []  # 每帧指标
    all_nvs_metrics = []  # novel view synthesis指标
    all_extrinsics_list = []  # 用于轨迹可视化
    all_rendered_images = []  # 所有窗口的渲染图（用于返回）

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for seq_idx in tqdm(range(num_sequences), desc=f"处理 {dataset_name}"):
            # 计算当前窗口的帧索引
            start_frame = seq_idx * stride
            window_frame_indices = [start_frame + j * frame_interval for j in range(seq_len)]

            # 检查是否超出范围
            if window_frame_indices[-1] >= total_frames:
                continue

            # 获取窗口内的帧
            seq_frames = [frames[idx] for idx in window_frame_indices]
            images = load_and_preprocess_frames(seq_frames).to(device)

            # 推理
            rendered_image, depth_output, predictions, extrinsic, intrinsic, is_interpolated, frame_indices_list, point_map, depth_map = process_sequence(
                model, images, device, dtype, num_interpolations=num_interpolations
            )

            all_extrinsics_list.append(extrinsic.cpu().numpy())

            # 计算所有帧的指标
            num_renders = rendered_image.shape[0]
            S = images.shape[0]  # 原始帧数

            frame_metrics = {}
            for i in range(num_renders):
                is_interp = is_interpolated[i].item() if torch.is_tensor(is_interpolated[i]) else is_interpolated[i]
                local_frame_idx = frame_indices_list[i]

                if is_interp:
                    # 插值帧的NVS指标
                    interp_position = i % (num_interpolations + 1) if num_interpolations > 0 else 0

                    # 计算对应的GT中间帧索引
                    # local_frame_idx是主帧索引，interp_position是插值位置
                    # 中间帧索引 = 主帧起始索引 + 主帧索引*frame_interval + interp_position
                    main_frame_global_idx = window_frame_indices[local_frame_idx]
                    middle_frame_global_idx = main_frame_global_idx + interp_position

                    if middle_frame_global_idx < len(frames) and gt_frames is not None and middle_frame_global_idx < len(gt_frames):
                        gt_middle_frame = gt_frames[middle_frame_global_idx]
                        gt_middle_tensor = load_and_preprocess_frames([gt_middle_frame]).to(device)

                        rendered = rendered_image[i].unsqueeze(0).clamp(0, 1)
                        psnr_val, ssim_val, lpips_val, _ = compute_metrics(rendered, gt_middle_tensor, lpips_fn)

                        frame_metrics[i] = {
                            'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val,
                            'is_interp': True, 'global_idx': middle_frame_global_idx
                        }

                        all_nvs_metrics.append({
                            'window_idx': seq_idx,
                            'frame_idx': middle_frame_global_idx,
                            'psnr': psnr_val,
                            'ssim': ssim_val,
                            'lpips': lpips_val
                        })
                else:
                    # 原始帧的指标
                    rendered = rendered_image[i].unsqueeze(0).clamp(0, 1)
                    orig_img = images[local_frame_idx].unsqueeze(0)
                    psnr_val, ssim_val, lpips_val, _ = compute_metrics(rendered, orig_img, lpips_fn)

                    global_frame_idx = window_frame_indices[local_frame_idx]
                    frame_metrics[i] = {
                        'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val,
                        'is_interp': False, 'global_idx': global_frame_idx
                    }

                    all_metrics.append({
                        'frame_idx': global_frame_idx,
                        'window_idx': seq_idx,
                        'psnr': psnr_val,
                        'ssim': ssim_val,
                        'lpips': lpips_val
                    })

            # 保存渲染图（与GT拼接，带指标）+ 点云和深度图可视化
            if save_dir is not None:
                window_dir = os.path.join(save_dir, f'window_{seq_idx:03d}')
                os.makedirs(window_dir, exist_ok=True)

                for i in range(num_renders):
                    is_interp = is_interpolated[i].item() if torch.is_tensor(is_interpolated[i]) else is_interpolated[i]
                    local_frame_idx = frame_indices_list[i]

                    rendered = rendered_image[i].detach().cpu().clamp(0, 1)

                    # 获取对应的GT图像
                    gt_img = None
                    if is_interp:
                        interp_position = i % (num_interpolations + 1) if num_interpolations > 0 else 0
                        main_frame_global_idx = window_frame_indices[local_frame_idx]
                        middle_frame_global_idx = main_frame_global_idx + interp_position

                        if gt_frames is not None and middle_frame_global_idx < len(gt_frames):
                            gt_img = gt_frames[middle_frame_global_idx]

                        img_name = f'frame_{local_frame_idx:02d}_interp_{interp_position:02d}.png'
                    else:
                        main_frame_global_idx = window_frame_indices[local_frame_idx]
                        if gt_frames is not None and main_frame_global_idx < len(gt_frames):
                            gt_img = gt_frames[main_frame_global_idx]

                        img_name = f'frame_{local_frame_idx:02d}.png'

                    # 拼接渲染图和GT
                    if gt_img is not None:
                        gt_pil = Image.fromarray(gt_img).convert("RGB")
                        gt_resized = gt_pil.resize((rendered.shape[2], rendered.shape[1]), Image.Resampling.BICUBIC)
                        gt_tensor = T.ToTensor()(gt_resized)

                        # 对于原始帧，添加点云和深度图可视化
                        if not is_interp and local_frame_idx < S:
                            pred_point_vis = visualize_point_cloud(point_map[0, local_frame_idx])
                            pred_depth_vis = visualize_depth_map(depth_map[0, local_frame_idx])
                            render_depth_vis = visualize_depth_map(depth_output[i])

                            h, w = rendered.shape[1], rendered.shape[2]
                            pred_point_vis = cv2.resize(pred_point_vis, (w, h))
                            pred_depth_vis = cv2.resize(pred_depth_vis, (w, h))
                            render_depth_vis = cv2.resize(render_depth_vis, (w, h))

                            pred_point_tensor = T.ToTensor()(pred_point_vis)
                            pred_depth_tensor = T.ToTensor()(pred_depth_vis)
                            render_depth_tensor = T.ToTensor()(render_depth_vis)

                            combined = torch.cat([gt_tensor, rendered, pred_point_tensor, pred_depth_tensor, render_depth_tensor], dim=2)
                        else:
                            combined = torch.cat([gt_tensor, rendered], dim=2)

                        combined_np = (combined.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        combined_bgr = cv2.cvtColor(combined_np, cv2.COLOR_RGB2BGR)

                        h, w = combined_bgr.shape[:2]
                        cv2.putText(combined_bgr, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(combined_bgr, "Rendered", (w // 5 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if not is_interp and local_frame_idx < S:
                            cv2.putText(combined_bgr, "Pred Point", (2 * w // 5 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(combined_bgr, "Pred Depth", (3 * w // 5 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(combined_bgr, "Render Depth", (4 * w // 5 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if i in frame_metrics:
                            m = frame_metrics[i]
                            metrics_text = f"PSNR: {m['psnr']:.2f}  SSIM: {m['ssim']:.4f}  LPIPS: {m['lpips']:.4f}"
                            cv2.putText(combined_bgr, metrics_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        img_path = os.path.join(window_dir, img_name)
                        cv2.imwrite(img_path, combined_bgr)

                        # 保存点云为PLY文件（仅原始帧）
                        if not is_interp and local_frame_idx < S:
                            pts = point_map[0, local_frame_idx]
                            if torch.is_tensor(pts):
                                pts = pts.detach().cpu().numpy()

                            input_img = images[local_frame_idx].detach().cpu()
                            input_img_np = (input_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                            valid_mask = (~np.isnan(pts[..., 0])) & (~np.isnan(pts[..., 1])) & (~np.isnan(pts[..., 2]))
                            valid_mask &= (~np.isinf(pts[..., 0])) & (~np.isinf(pts[..., 1])) & (~np.isinf(pts[..., 2]))
                            valid_mask &= (pts[..., 2] > 0)

                            valid_points = pts[valid_mask]
                            valid_colors = input_img_np[valid_mask]

                            if len(valid_points) > 0:
                                ply_path = os.path.join(window_dir, f'frame_{local_frame_idx:02d}_pointcloud.ply')
                                save_point_cloud_ply(valid_points, valid_colors, ply_path)
                    else:
                        img_path = os.path.join(window_dir, img_name)
                        T.ToPILImage()(rendered).save(img_path)

                # 保存整个窗口的合并点云
                if S > 0:
                    all_points = []
                    all_colors = []

                    for frame_idx in range(S):
                        pts = point_map[0, frame_idx]
                        if torch.is_tensor(pts):
                            pts = pts.detach().cpu().numpy()

                        input_img = images[frame_idx].detach().cpu()
                        input_img_np = (input_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                        valid_mask = (~np.isnan(pts[..., 0])) & (~np.isnan(pts[..., 1])) & (~np.isnan(pts[..., 2]))
                        valid_mask &= (~np.isinf(pts[..., 0])) & (~np.isinf(pts[..., 1])) & (~np.isinf(pts[..., 2]))
                        valid_mask &= (pts[..., 2] > 0)

                        if valid_mask.any():
                            all_points.append(pts[valid_mask])
                            all_colors.append(input_img_np[valid_mask])

                    if len(all_points) > 0:
                        all_points = np.concatenate(all_points, axis=0)
                        all_colors = np.concatenate(all_colors, axis=0)
                        ply_path = os.path.join(window_dir, 'merged_pointcloud.ply')
                        save_point_cloud_ply(all_points, all_colors, ply_path)

                all_rendered_images.append(rendered_image.cpu())

    return all_metrics, all_nvs_metrics, all_extrinsics_list, all_rendered_images


def plot_metrics_comparison(generated_metrics, gt_metrics, output_path, stride=1):
    """
    绘制两组指标的对比曲线
    按滑动窗口计算平均值，以窗口起始帧作为索引
    """
    def group_by_window(metrics, stride):
        """按窗口分组计算平均指标"""
        window_dict = {}
        for m in metrics:
            w = m['window_idx']
            if w not in window_dict:
                window_dict[w] = {'psnr': [], 'ssim': [], 'lpips': []}
            window_dict[w]['psnr'].append(m['psnr'])
            window_dict[w]['ssim'].append(m['ssim'])
            window_dict[w]['lpips'].append(m['lpips'])

        # 计算每个窗口的平均值和起始帧
        result = {'start_frame': [], 'psnr': [], 'ssim': [], 'lpips': []}
        for w in sorted(window_dict.keys()):
            result['start_frame'].append(w * stride)
            result['psnr'].append(np.mean(window_dict[w]['psnr']))
            result['ssim'].append(np.mean(window_dict[w]['ssim']))
            result['lpips'].append(np.mean(window_dict[w]['lpips']))
        return result

    gen_window = group_by_window(generated_metrics, stride) if generated_metrics else None
    gt_window = group_by_window(gt_metrics, stride) if gt_metrics else None

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # PSNR
    ax1 = axes[0]
    if gt_window:
        ax1.plot(gt_window['start_frame'], gt_window['psnr'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax1.axhline(y=np.mean(gt_window['psnr']), color='b', linestyle='--', alpha=0.5,
                    label=f'GT avg: {np.mean(gt_window["psnr"]):.2f}')
    if gen_window:
        ax1.plot(gen_window['start_frame'], gen_window['psnr'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax1.axhline(y=np.mean(gen_window['psnr']), color='r', linestyle='--', alpha=0.5,
                    label=f'Gen avg: {np.mean(gen_window["psnr"]):.2f}')
    ax1.set_xlabel('Window Start Frame')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR per Sliding Window')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SSIM
    ax2 = axes[1]
    if gt_window:
        ax2.plot(gt_window['start_frame'], gt_window['ssim'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax2.axhline(y=np.mean(gt_window['ssim']), color='b', linestyle='--', alpha=0.5,
                    label=f'GT avg: {np.mean(gt_window["ssim"]):.4f}')
    if gen_window:
        ax2.plot(gen_window['start_frame'], gen_window['ssim'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax2.axhline(y=np.mean(gen_window['ssim']), color='r', linestyle='--', alpha=0.5,
                    label=f'Gen avg: {np.mean(gen_window["ssim"]):.4f}')
    ax2.set_xlabel('Window Start Frame')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM per Sliding Window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # LPIPS
    ax3 = axes[2]
    if gt_window:
        ax3.plot(gt_window['start_frame'], gt_window['lpips'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax3.axhline(y=np.mean(gt_window['lpips']), color='b', linestyle='--', alpha=0.5,
                    label=f'GT avg: {np.mean(gt_window["lpips"]):.4f}')
    if gen_window:
        ax3.plot(gen_window['start_frame'], gen_window['lpips'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax3.axhline(y=np.mean(gen_window['lpips']), color='r', linestyle='--', alpha=0.5,
                    label=f'Gen avg: {np.mean(gen_window["lpips"]):.4f}')
    ax3.set_xlabel('Window Start Frame')
    ax3.set_ylabel('LPIPS')
    ax3.set_title('LPIPS per Sliding Window (lower is better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_path, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"指标对比图已保存: {save_path}")


def save_metrics_json(metrics, output_path, filename, config_info=None):
    """保存指标到 JSON 文件"""
    psnr_vals = [m['psnr'] for m in metrics]
    ssim_vals = [m['ssim'] for m in metrics]
    lpips_vals = [m['lpips'] for m in metrics]

    result = {
        'summary': {
            'num_frames': len(metrics),
            'psnr_avg': round(np.mean(psnr_vals), 4),
            'psnr_std': round(np.std(psnr_vals), 4),
            'ssim_avg': round(np.mean(ssim_vals), 4),
            'ssim_std': round(np.std(ssim_vals), 4),
            'lpips_avg': round(np.mean(lpips_vals), 4),
            'lpips_std': round(np.std(lpips_vals), 4),
        },
        'per_frame': metrics,
    }

    if config_info:
        result['config'] = config_info

    json_path = os.path.join(output_path, filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"指标已保存: {json_path}")


def plot_nvs_metrics_curve(gen_nvs_metrics, gt_nvs_metrics, output_path, stride=1):
    """
    绘制Novel View Synthesis指标的滑动窗口曲线（折线图）

    Args:
        gen_nvs_metrics: 生成图像集的NVS指标列表
        gt_nvs_metrics: GT图像集的NVS指标列表
        output_path: 输出目录
        stride: 滑动窗口步长
    """
    if len(gen_nvs_metrics) == 0 and len(gt_nvs_metrics) == 0:
        print("没有NVS指标，跳过绘制")
        return

    # 按窗口分组并计算起始帧
    def group_by_window(metrics, stride):
        window_dict = {}
        for m in metrics:
            w = m['window_idx']
            if w not in window_dict:
                window_dict[w] = {'psnr': [], 'ssim': [], 'lpips': []}
            window_dict[w]['psnr'].append(m['psnr'])
            window_dict[w]['ssim'].append(m['ssim'])
            window_dict[w]['lpips'].append(m['lpips'])

        # 计算每个窗口的平均值和起始帧
        result = {'start_frame': [], 'psnr': [], 'ssim': [], 'lpips': []}
        for w in sorted(window_dict.keys()):
            result['start_frame'].append(w * stride)
            result['psnr'].append(np.mean(window_dict[w]['psnr']))
            result['ssim'].append(np.mean(window_dict[w]['ssim']))
            result['lpips'].append(np.mean(window_dict[w]['lpips']))
        return result

    gen_window_metrics = group_by_window(gen_nvs_metrics, stride) if gen_nvs_metrics else None
    gt_window_metrics = group_by_window(gt_nvs_metrics, stride) if gt_nvs_metrics else None

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # PSNR
    ax1 = axes[0]
    if gt_window_metrics:
        ax1.plot(gt_window_metrics['start_frame'], gt_window_metrics['psnr'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax1.axhline(y=np.mean(gt_window_metrics['psnr']), color='b', linestyle='--', alpha=0.5,
                    label=f'GT avg: {np.mean(gt_window_metrics["psnr"]):.2f}')
    if gen_window_metrics:
        ax1.plot(gen_window_metrics['start_frame'], gen_window_metrics['psnr'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax1.axhline(y=np.mean(gen_window_metrics['psnr']), color='r', linestyle='--', alpha=0.5,
                    label=f'Gen avg: {np.mean(gen_window_metrics["psnr"]):.2f}')

    ax1.set_xlabel('Window Start Frame')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Novel View Synthesis - PSNR per Window')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SSIM
    ax2 = axes[1]
    if gt_window_metrics:
        ax2.plot(gt_window_metrics['start_frame'], gt_window_metrics['ssim'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax2.axhline(y=np.mean(gt_window_metrics['ssim']), color='b', linestyle='--', alpha=0.5,
                    label=f'GT avg: {np.mean(gt_window_metrics["ssim"]):.4f}')
    if gen_window_metrics:
        ax2.plot(gen_window_metrics['start_frame'], gen_window_metrics['ssim'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax2.axhline(y=np.mean(gen_window_metrics['ssim']), color='r', linestyle='--', alpha=0.5,
                    label=f'Gen avg: {np.mean(gen_window_metrics["ssim"]):.4f}')

    ax2.set_xlabel('Window Start Frame')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Novel View Synthesis - SSIM per Window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # LPIPS
    ax3 = axes[2]
    if gt_window_metrics:
        ax3.plot(gt_window_metrics['start_frame'], gt_window_metrics['lpips'], 'b-o', label='GT', markersize=6, linewidth=2)
        ax3.axhline(y=np.mean(gt_window_metrics['lpips']), color='b', linestyle='--', alpha=0.5,
                    label=f'GT avg: {np.mean(gt_window_metrics["lpips"]):.4f}')
    if gen_window_metrics:
        ax3.plot(gen_window_metrics['start_frame'], gen_window_metrics['lpips'], 'r-s', label='Generated', markersize=6, linewidth=2)
        ax3.axhline(y=np.mean(gen_window_metrics['lpips']), color='r', linestyle='--', alpha=0.5,
                    label=f'Gen avg: {np.mean(gen_window_metrics["lpips"]):.4f}')

    ax3.set_xlabel('Window Start Frame')
    ax3.set_ylabel('LPIPS')
    ax3.set_title('Novel View Synthesis - LPIPS per Window (lower is better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_path, 'nvs_metrics_per_window.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"NVS指标曲线已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='DGGT Benchmark 脚本')
    parser.add_argument('--generated_dir', type=str, required=True, help='生成图像目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT图像目录')
    parser.add_argument('--gt_extrinsic_dir', type=str, default=None, help='GT相机外参目录（包含每帧的camera-to-world矩阵npy文件）')
    parser.add_argument('--ckpt_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output_path', type=str, default='./benchmark_output', help='输出目录')
    parser.add_argument('--sequence_length', type=int, default=4, help='每次处理的帧数')
    parser.add_argument('--start_idx', type=int, default=0, help='起始帧索引')
    parser.add_argument('--frame_interval', type=int, default=1, help='帧间隔')
    parser.add_argument('--num_frames', type=int, default=None, help='总帧数')
    parser.add_argument('--stride', type=int, default=1, help='滑动窗口步长')
    parser.add_argument('--num_interpolations', type=int, default=None, help='插值帧数 (None=自动)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    traj_dir = os.path.join(args.output_path, 'trajectories')
    os.makedirs(traj_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {args.ckpt_path}")
    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("模型加载完成")

    # 初始化 LPIPS
    print("初始化 LPIPS 评估模型...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    print("LPIPS 模型加载完成")

    # 读取两个数据集
    gen_frames, gen_indices = read_images_from_dir(
        args.generated_dir, args.start_idx, args.num_frames
    )
    gt_frames, gt_indices = read_images_from_dir(
        args.gt_dir, args.start_idx, args.num_frames
    )

    # 检查帧数是否一致
    if len(gen_frames) != len(gt_frames):
        print(f"警告: 生成图像数({len(gen_frames)})与GT图像数({len(gt_frames)})不一致")

    # 配置信息
    config_info = {
        'checkpoint': args.ckpt_path,
        'sequence_length': args.sequence_length,
        'start_idx': args.start_idx,
        'frame_interval': args.frame_interval,
        'num_frames': args.num_frames,
        'stride': args.stride,
        'num_interpolations': args.num_interpolations,
    }

    # 对生成图像集推理（用生成的图像作为GT进行对比）
    print("\n" + "="*50)
    print("处理生成图像集")
    print("="*50)
    gen_render_dir = os.path.join(args.output_path, 'rendered_generated')
    gen_metrics, gen_nvs_metrics, gen_extrinsics, gen_rendered = run_inference_on_dataset(
        model, gen_frames, args, device, dtype, lpips_fn, "Generated",
        save_dir=gen_render_dir, gt_frames=gen_frames
    )

    # 对GT图像集推理
    print("\n" + "="*50)
    print("处理GT图像集")
    print("="*50)
    gt_render_dir = os.path.join(args.output_path, 'rendered_gt')
    gt_metrics, gt_nvs_metrics, gt_extrinsics, gt_rendered = run_inference_on_dataset(
        model, gt_frames, args, device, dtype, lpips_fn, "GT",
        save_dir=gt_render_dir, gt_frames=gt_frames
    )

    # 保存指标
    save_metrics_json(gen_metrics, args.output_path, 'generated_metrics.json', config_info)
    save_metrics_json(gt_metrics, args.output_path, 'gt_metrics.json', config_info)

    # 保存NVS指标
    if gen_nvs_metrics:
        save_metrics_json(gen_nvs_metrics, args.output_path, 'generated_nvs_metrics.json', config_info)
    if gt_nvs_metrics:
        save_metrics_json(gt_nvs_metrics, args.output_path, 'gt_nvs_metrics.json', config_info)

    # 绘制对比曲线
    plot_metrics_comparison(gen_metrics, gt_metrics, args.output_path, stride=args.stride)

    # 绘制NVS指标曲线
    plot_nvs_metrics_curve(gen_nvs_metrics, gt_nvs_metrics, args.output_path, stride=args.stride)

    # 可视化每个窗口的轨迹并计算位姿评估指标
    gen_pose_metrics_list = []
    gt_pose_metrics_list = []

    if args.gt_extrinsic_dir is not None:
        print("\n" + "="*50)
        print("位姿评估与轨迹可视化")
        print("="*50)

        # 读取所有GT外参文件
        ext_files = sorted([f for f in os.listdir(args.gt_extrinsic_dir) if f.endswith('.npy')])
        if len(ext_files) == 0:
            print(f"警告: 目录 {args.gt_extrinsic_dir} 中没有找到npy文件")
            gt_ext_c2w = None
        else:
            # 加载所有外参
            all_gt_ext = [np.load(os.path.join(args.gt_extrinsic_dir, f)) for f in ext_files]

            # 创建两个子目录
            gen_traj_dir = os.path.join(traj_dir, 'generated_vs_gt')
            gt_traj_dir = os.path.join(traj_dir, 'gt_pred_vs_gt')
            os.makedirs(gen_traj_dir, exist_ok=True)
            os.makedirs(gt_traj_dir, exist_ok=True)

            num_windows = min(len(gen_extrinsics), len(gt_extrinsics))

            for window_idx in range(num_windows):
                seq_len = gen_extrinsics[window_idx].shape[0]
                # 窗口帧索引: [start_frame + j * frame_interval for j in range(seq_len)]
                start_frame = window_idx * args.stride
                window_frame_indices = [start_frame + j * args.frame_interval for j in range(seq_len)]

                # 检查帧索引是否在范围内
                if window_frame_indices[-1] >= len(all_gt_ext):
                    continue

                # 获取GT外参（camera-to-world），转换为world-to-camera
                gt_c2w_window = np.stack([all_gt_ext[idx] for idx in window_frame_indices], axis=0)
                gt_w2c_window = np.linalg.inv(gt_c2w_window)

                # ===== 1. Generated图像预测的轨迹 vs GT外参 =====
                gen_pred_w2c = gen_extrinsics[window_idx]
                gen_metrics_i, gen_aligned_pos, gen_aligned_rot, gt_pos, gt_rot = compute_pose_metrics(
                    gen_pred_w2c, gt_w2c_window
                )
                gen_pose_metrics_list.append(gen_metrics_i)

                # 可视化对齐后的轨迹
                visualize_trajectory_comparison_aligned(
                    gen_aligned_pos, gen_aligned_rot,
                    gt_pos, gt_rot,
                    gen_traj_dir, window_idx
                )

                # ===== 2. GT图像预测的轨迹 vs GT外参 =====
                gt_pred_w2c = gt_extrinsics[window_idx]
                gt_metrics_i, gt_aligned_pos, gt_aligned_rot, gt_pos2, gt_rot2 = compute_pose_metrics(
                    gt_pred_w2c, gt_w2c_window
                )
                gt_pose_metrics_list.append(gt_metrics_i)

                # 可视化对齐后的轨迹
                visualize_trajectory_comparison_aligned(
                    gt_aligned_pos, gt_aligned_rot,
                    gt_pos2, gt_rot2,
                    gt_traj_dir, window_idx
                )

            # 绘制位姿误差曲线（折线图对比）
            plot_pose_metrics_curve(gen_pose_metrics_list, gt_pose_metrics_list, args.output_path, stride=args.stride)

            # 打印位姿评估汇总
            if gen_pose_metrics_list:
                gen_ate = np.mean([m['ate_rmse'] for m in gen_pose_metrics_list])
                gen_scale = np.mean([m['scale'] for m in gen_pose_metrics_list])
                gen_rot = np.mean([m['rot_mean'] for m in gen_pose_metrics_list])
                print(f"\nGenerated图像预测位姿误差:")
                print(f"  ATE RMSE: {gen_ate:.4f} m")
                print(f"  Scale:    {gen_scale:.4f}")
                print(f"  Rot Err:  {gen_rot:.2f}°")
                print(f"  可视化: {gen_traj_dir}")

            if gt_pose_metrics_list:
                gt_ate = np.mean([m['ate_rmse'] for m in gt_pose_metrics_list])
                gt_scale = np.mean([m['scale'] for m in gt_pose_metrics_list])
                gt_rot = np.mean([m['rot_mean'] for m in gt_pose_metrics_list])
                print(f"\nGT图像预测位姿误差:")
                print(f"  ATE RMSE: {gt_ate:.4f} m")
                print(f"  Scale:    {gt_scale:.4f}")
                print(f"  Rot Err:  {gt_rot:.2f}°")
                print(f"  可视化: {gt_traj_dir}")

            # 保存位姿指标到JSON
            pose_result = {
                'generated': {
                    'ate_rmse_avg': np.mean([m['ate_rmse'] for m in gen_pose_metrics_list]) if gen_pose_metrics_list else None,
                    'scale_avg': np.mean([m['scale'] for m in gen_pose_metrics_list]) if gen_pose_metrics_list else None,
                    'rot_mean_avg': np.mean([m['rot_mean'] for m in gen_pose_metrics_list]) if gen_pose_metrics_list else None,
                    'per_window': gen_pose_metrics_list,
                },
                'gt_pred': {
                    'ate_rmse_avg': np.mean([m['ate_rmse'] for m in gt_pose_metrics_list]) if gt_pose_metrics_list else None,
                    'scale_avg': np.mean([m['scale'] for m in gt_pose_metrics_list]) if gt_pose_metrics_list else None,
                    'rot_mean_avg': np.mean([m['rot_mean'] for m in gt_pose_metrics_list]) if gt_pose_metrics_list else None,
                    'per_window': gt_pose_metrics_list,
                }
            }
            with open(os.path.join(args.output_path, 'pose_metrics.json'), 'w') as f:
                json.dump(pose_result, f, indent=2, cls=NumpyEncoder)

    # 打印汇总
    print("\n" + "="*50)
    print("Benchmark 完成!")
    print("="*50)

    gen_psnr_avg = np.mean([m['psnr'] for m in gen_metrics])
    gen_ssim_avg = np.mean([m['ssim'] for m in gen_metrics])
    gen_lpips_avg = np.mean([m['lpips'] for m in gen_metrics])

    gt_psnr_avg = np.mean([m['psnr'] for m in gt_metrics])
    gt_ssim_avg = np.mean([m['ssim'] for m in gt_metrics])
    gt_lpips_avg = np.mean([m['lpips'] for m in gt_metrics])

    print(f"\n生成图像集指标:")
    print(f"  PSNR: {gen_psnr_avg:.4f} dB")
    print(f"  SSIM: {gen_ssim_avg:.4f}")
    print(f"  LPIPS: {gen_lpips_avg:.4f}")

    print(f"\nGT图像集指标:")
    print(f"  PSNR: {gt_psnr_avg:.4f} dB")
    print(f"  SSIM: {gt_ssim_avg:.4f}")
    print(f"  LPIPS: {gt_lpips_avg:.4f}")

    print(f"\n指标差异 (Generated - GT):")
    print(f"  PSNR: {gen_psnr_avg - gt_psnr_avg:.4f} dB")
    print(f"  SSIM: {gen_ssim_avg - gt_ssim_avg:.4f}")
    print(f"  LPIPS: {gen_lpips_avg - gt_lpips_avg:.4f}")

    if gen_pose_metrics_list or gt_pose_metrics_list:
        print(f"\n位姿评估指标:")
        if gen_pose_metrics_list:
            gen_ate = np.mean([m['ate_rmse'] for m in gen_pose_metrics_list])
            gen_scale = np.mean([m['scale'] for m in gen_pose_metrics_list])
            print(f"  Generated预测 ATE: {gen_ate:.4f} m, Scale: {gen_scale:.4f}")
        if gt_pose_metrics_list:
            gt_ate = np.mean([m['ate_rmse'] for m in gt_pose_metrics_list])
            gt_scale = np.mean([m['scale'] for m in gt_pose_metrics_list])
            print(f"  GT预测 ATE: {gt_ate:.4f} m, Scale: {gt_scale:.4f}")

    # 打印NVS指标汇总
    if gen_nvs_metrics or gt_nvs_metrics:
        print(f"\nNovel View Synthesis指标:")
        if gen_nvs_metrics:
            gen_nvs_psnr = np.mean([m['psnr'] for m in gen_nvs_metrics])
            gen_nvs_ssim = np.mean([m['ssim'] for m in gen_nvs_metrics])
            gen_nvs_lpips = np.mean([m['lpips'] for m in gen_nvs_metrics])
            print(f"  Generated NVS - PSNR: {gen_nvs_psnr:.4f} dB, SSIM: {gen_nvs_ssim:.4f}, LPIPS: {gen_nvs_lpips:.4f}")
        if gt_nvs_metrics:
            gt_nvs_psnr = np.mean([m['psnr'] for m in gt_nvs_metrics])
            gt_nvs_ssim = np.mean([m['ssim'] for m in gt_nvs_metrics])
            gt_nvs_lpips = np.mean([m['lpips'] for m in gt_nvs_metrics])
            print(f"  GT NVS - PSNR: {gt_nvs_psnr:.4f} dB, SSIM: {gt_nvs_ssim:.4f}, LPIPS: {gt_nvs_lpips:.4f}")

    print(f"\n输出目录: {args.output_path}")
    print(f"  指标对比图: {os.path.join(args.output_path, 'metrics_comparison.png')}")
    print(f"  指标JSON: {args.output_path}")
    print(f"  渲染图(生成集): {gen_render_dir}")
    print(f"  渲染图(GT集): {gt_render_dir}")
    if args.gt_extrinsic_dir:
        print(f"  位姿误差曲线: {args.output_path}")
        print(f"  轨迹可视化: {traj_dir}")
    if gen_nvs_metrics or gt_nvs_metrics:
        print(f"  NVS指标曲线: {os.path.join(args.output_path, 'nvs_metrics_per_window.png')}")


if __name__ == "__main__":
    main()
