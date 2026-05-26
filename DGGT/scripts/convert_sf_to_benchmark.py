#!/usr/bin/env python3
"""
将公共视频输出格式转换为 benchmark.py 所需的输入格式。

设计思路：
  - prepare  (一劳永逸)：建立公共 val 元数据，包含完整100帧的 token → pose 映射
  - convert  (per-model)：读取模型输出的 token list，与公共 metadata 对齐，
                          只输出模型实际拥有帧的图像 + pose
  - benchmark：调用 benchmark.py 评测
  - aggregate：聚合所有场景的 benchmark 结果，按滑动窗口取平均，画平均曲线

Usage:
    # Step 1: 一次性准备公共元数据 (token list + ego pose)
    python scripts/convert_sf_to_benchmark.py prepare \
        --sf_preview_dir ./output/sf_chunk15_step12 \
        --pose_base_dir ./output/dmd_ode_pretrained_ref3_seq12/vis_validation_generator \
        --output_dir benchmark_input \
        --num_scenes 10

    # Step 2: 从 npy 提取图像，根据 token 对齐后只保留模型有的帧
    python scripts/convert_sf_to_benchmark.py convert \
        --sf_preview_dir ./output/sf_chunk15_step12 \
        --output_dir benchmark_input \
        --num_scenes 10 \
        --cams 0

    # Step 3: 调用 benchmark.py 评测
    python scripts/convert_sf_to_benchmark.py benchmark \
        --output_dir benchmark_input \
        --source_name sf_chunk15_step12 \
        --ckpt_path pretrained/model_latest_waymo.pth \
        --num_scenes 10 \
        --cams 0

    # Step 4: 聚合所有场景结果，画平均曲线
    python scripts/convert_sf_to_benchmark.py aggregate \
        --output_dir benchmark_input \
        --source_name sf_chunk15_step12 \
        --num_scenes 10 \
        --cams 0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================== 常量 ====================

CAM_NAMES = [
    "CAM_FRONT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
]
NUM_ROWS = 4   # 大图行数
NUM_COLS = 6   # 大图列数
GT_ROW = 0     # GT 图像所在行
GEN_ROW = 3    # Generated 图像所在行


# ==================== Step 1: 准备公共元数据 ====================

def prepare_metadata(sf_preview_dir, pose_base_dir, output_dir, num_scenes):
    """
    一次性准备（一劳永逸）：
    1. 从 preview_sampletok 提取每个 scene 完整100帧的 token 列表
    2. 从 dmd 的 pose 目录 copy 完整100帧的 ego_transform npy 文件
    3. 保存 metadata JSON: {scene: {tokens: [...], token_to_frame_idx: {...}, ...}}

    metadata 中的 token_to_frame_idx 映射: token -> 帧序号(0~99)
    之后 convert 步骤用它来做帧对齐。
    """
    tok_dir = os.path.join(sf_preview_dir, "preview_sampletok")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    all_metadata = {}

    for scene_id in range(num_scenes):
        scene_key = f"scene_{scene_id}"

        # --- 1. 读取 token 列表 ---
        tok_path = os.path.join(tok_dir, f"{scene_id}.txt")
        if not os.path.exists(tok_path):
            print(f"[WARN] {tok_path} 不存在，跳过 scene {scene_id}")
            continue

        with open(tok_path, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]

        # 建立 token -> frame_idx 映射
        token_to_idx = {tok: i for i, tok in enumerate(tokens)}
        print(f"scene_{scene_id}: {len(tokens)} tokens")

        # --- 2. Copy ego_transform pose (完整帧) ---
        # 源: pose_base_dir/{scene_id}/gt_camera_params/ego_transforms/cam_X/frame_XXXX.npy
        # 目标: output_dir/metadata/poses/scene_{id}/cam_X/frame_XXXX.npy
        src_pose_base = os.path.join(
            pose_base_dir, str(scene_id), "gt_camera_params", "ego_transforms"
        )
        if not os.path.isdir(src_pose_base):
            print(f"[WARN] {src_pose_base} 不存在，跳过 pose copy")
            continue

        for cam_idx in range(NUM_COLS):
            src_cam_dir = os.path.join(src_pose_base, f"cam_{cam_idx}")
            dst_cam_dir = os.path.join(
                metadata_dir, "poses", scene_key, f"cam_{cam_idx}"
            )

            if not os.path.isdir(src_cam_dir):
                print(f"[WARN] {src_cam_dir} 不存在，跳过")
                continue

            os.makedirs(dst_cam_dir, exist_ok=True)

            # Copy 完整帧（后续 convert 会按 token 筛选）
            for frame_idx in range(len(tokens)):
                src_file = os.path.join(src_cam_dir, f"frame_{frame_idx:04d}.npy")
                dst_file = os.path.join(dst_cam_dir, f"frame_{frame_idx:04d}.npy")
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)

        # --- 3. 记录 metadata ---
        all_metadata[scene_key] = {
            "num_frames": len(tokens),
            "tokens": tokens,
            "token_to_idx": token_to_idx,
            "cam_order": CAM_NAMES,
        }

    # 保存 metadata JSON
    meta_path = os.path.join(metadata_dir, "val_scene_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"\n公共元数据已保存: {meta_path}")
    print(f"共处理 {len(all_metadata)} 个 scene")


# ==================== Step 2: 从 npy 提取图像 ====================

def extract_images_from_npy(sf_preview_dir, output_dir, num_scenes, cam_indices):
    """
    从 preview_numpy 的 npy 大图中提取 GT 和 Generated 图像。

    输出目录结构: {output_dir}/{source_name}/scene_X/...
    source_name 取自 sf_preview_dir 的最后一级目录名，用于区分不同模型的输出。

    关键逻辑：
    1. 读取模型输出的 token list (preview_sampletok/{scene}.txt)
    2. 与公共 metadata 中的 token 对齐，得到帧索引映射
    3. 只保存模型实际拥有的帧的图像和 pose
    4. 重新编号 frame_XXXX 使输出连续（benchmark.py 需要连续帧）
    """
    source_name = os.path.basename(os.path.normpath(sf_preview_dir))
    convert_dir = os.path.join(output_dir, source_name)

    npy_dir = os.path.join(sf_preview_dir, "preview_numpy")
    tok_dir = os.path.join(sf_preview_dir, "preview_sampletok")
    metadata_path = os.path.join(output_dir, "metadata", "val_scene_metadata.json")

    if not os.path.exists(metadata_path):
        print(f"[ERROR] 元数据文件不存在: {metadata_path}")
        print("请先运行 prepare 步骤")
        sys.exit(1)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"模型来源: {source_name}")
    print(f"输出目录: {convert_dir}")

    for scene_id in range(num_scenes):
        scene_key = f"scene_{scene_id}"

        if scene_key not in metadata:
            print(f"[WARN] scene_{scene_id} 不在 metadata 中，跳过")
            continue

        npy_path = os.path.join(npy_dir, f"{scene_id}.npy")
        if not os.path.exists(npy_path):
            print(f"[WARN] {npy_path} 不存在，跳过")
            continue

        # --- 读取模型输出的 token list ---
        model_tok_path = os.path.join(tok_dir, f"{scene_id}.txt")
        if not os.path.exists(model_tok_path):
            print(f"[WARN] {model_tok_path} 不存在，跳过")
            continue

        with open(model_tok_path, "r") as f:
            model_tokens = [line.strip() for line in f if line.strip()]

        # --- Token 对齐：模型 token → metadata 帧索引 → 输出连续帧号 ---
        meta_tokens = metadata[scene_key]["tokens"]
        meta_token_set = set(meta_tokens)

        # 检查模型 token 是否都在 metadata 中
        aligned_frames = []  # [(model_frame_idx, meta_frame_idx, token), ...]
        missing_tokens = []

        for model_idx, tok in enumerate(model_tokens):
            if tok in meta_token_set:
                meta_frame_idx = meta_tokens.index(tok)
                aligned_frames.append((model_idx, meta_frame_idx, tok))
            else:
                missing_tokens.append(tok)

        if missing_tokens:
            print(f"[WARN] scene_{scene_id}: {len(missing_tokens)} 个 token 在 metadata 中找不到")
        if not aligned_frames:
            print(f"[WARN] scene_{scene_id}: 没有可对齐的帧，跳过")
            continue

        print(f"\n处理 scene_{scene_id}: "
              f"模型 {len(model_tokens)} 帧 → 对齐 {len(aligned_frames)} 帧")

        # --- 加载 npy ---
        data = np.load(npy_path, mmap_mode="r")

        # 检测通道维度位置：NCHW vs NHWC
        if data.ndim == 4 and data.shape[1] == 3 and data.shape[1] < data.shape[2]:
            # NCHW 格式，转置为 NHWC
            print(f"  检测到 NCHW 格式 {data.shape}，转换为 NHWC")
            data = np.transpose(data, (0, 2, 3, 1))  # (N, H, W, C)

        h_sub = data.shape[1] // NUM_ROWS   # 252
        w_sub = data.shape[2] // NUM_COLS   # 448

        # 验证 npy 帧数与模型 token 数一致
        if data.shape[0] != len(model_tokens):
            print(f"[WARN] npy 帧数({data.shape[0]}) != token数({len(model_tokens)})，以较少者为准")
            usable_frames = min(data.shape[0], len(model_tokens))
            aligned_frames = [af for af in aligned_frames if af[0] < usable_frames]

        for cam_idx in cam_indices:
            if cam_idx < 0 or cam_idx >= NUM_COLS:
                continue

            # 准备输出目录 (在 source_name 子目录下)
            gt_dir = os.path.join(convert_dir, scene_key, "gt_rgb", f"cam_{cam_idx}")
            gen_dir = os.path.join(convert_dir, scene_key, "generated_rgb", f"cam_{cam_idx}")
            pose_dir = os.path.join(
                convert_dir, scene_key, "gt_camera_params", "ego_transforms", f"cam_{cam_idx}"
            )
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)

            # 公共 pose 目录
            meta_pose_dir = os.path.join(
                output_dir, "metadata", "poses", scene_key, f"cam_{cam_idx}"
            )

            for out_idx, (model_idx, meta_idx, tok) in enumerate(aligned_frames):
                x0 = cam_idx * w_sub
                x1 = x0 + w_sub

                # GT 图像 (行0)
                gt_img = np.array(data[model_idx, GT_ROW * h_sub:(GT_ROW + 1) * h_sub, x0:x1, :])
                if gt_img.dtype != np.uint8:
                    gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)
                gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(gt_dir, f"frame_{out_idx:04d}.png"), gt_bgr)

                # Generated 图像 (行3)
                gen_img = np.array(data[model_idx, GEN_ROW * h_sub:(GEN_ROW + 1) * h_sub, x0:x1, :])
                if gen_img.dtype != np.uint8:
                    gen_img = (gen_img * 255).clip(0, 255).astype(np.uint8)
                gen_bgr = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(gen_dir, f"frame_{out_idx:04d}.png"), gen_bgr)

                # Copy 对应帧的 pose
                src_pose = os.path.join(meta_pose_dir, f"frame_{meta_idx:04d}.npy")
                dst_pose = os.path.join(pose_dir, f"frame_{out_idx:04d}.npy")
                if os.path.exists(src_pose):
                    shutil.copy2(src_pose, dst_pose)

            print(f"  cam_{cam_idx}: {len(aligned_frames)} 帧 -> {gt_dir}")

        # 保存该 scene 的对齐信息（token → 输出帧号映射）
        align_info_path = os.path.join(convert_dir, scene_key, "align_info.json")
        with open(align_info_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": source_name,
                "model_tokens": model_tokens,
                "aligned_frames": [
                    {"out_idx": out_idx, "model_idx": model_idx,
                     "meta_idx": meta_idx, "token": tok}
                    for out_idx, (model_idx, meta_idx, tok) in enumerate(aligned_frames)
                ],
            }, f, indent=2, ensure_ascii=False)

        del data

    print("\n图像提取完成")


# ==================== Step 3: 调用 benchmark.py ====================

def run_benchmark(output_dir, source_name, ckpt_path, num_scenes, cam_indices,
                  sequence_length=4, start_idx=4, frame_interval=4, stride=1,
                  num_frames=100, extra_args=None):
    """
    对每个 scene 和每个 cam 调用 benchmark.py。
    数据目录: {output_dir}/{source_name}/scene_X/...
    """
    convert_dir = os.path.join(output_dir, source_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    benchmark_script = os.path.join(parent_dir, "benchmark.py")

    if not os.path.exists(benchmark_script):
        print(f"[ERROR] benchmark.py 不存在: {benchmark_script}")
        sys.exit(1)

    if not os.path.isdir(convert_dir):
        print(f"[ERROR] 数据目录不存在: {convert_dir}")
        sys.exit(1)

    for scene_id in range(num_scenes):
        scene_key = f"scene_{scene_id}"

        for cam_idx in cam_indices:
            gt_dir = os.path.join(convert_dir, scene_key, "gt_rgb", f"cam_{cam_idx}")
            gen_dir = os.path.join(convert_dir, scene_key, "generated_rgb", f"cam_{cam_idx}")
            pose_dir = os.path.join(
                convert_dir, scene_key, "gt_camera_params", "ego_transforms", f"cam_{cam_idx}"
            )
            bench_output = os.path.join(
                convert_dir, "benchmark_output", scene_key, f"cam_{cam_idx}"
            )

            # 检查目录是否存在且有文件
            if not os.path.isdir(gt_dir) or not os.listdir(gt_dir):
                print(f"[SKIP] {gt_dir} 为空或不存在")
                continue
            if not os.path.isdir(gen_dir) or not os.listdir(gen_dir):
                print(f"[SKIP] {gen_dir} 为空或不存在")
                continue
            if not os.path.isdir(pose_dir) or not os.listdir(pose_dir):
                print(f"[SKIP] {pose_dir} 为空或不存在")
                continue

            cmd = [
                sys.executable, benchmark_script,
                "--generated_dir", gen_dir,
                "--gt_dir", gt_dir,
                "--gt_extrinsic_dir", pose_dir,
                "--ckpt_path", ckpt_path,
                "--output_path", bench_output,
                "--sequence_length", str(sequence_length),
                "--start_idx", str(start_idx),
                "--frame_interval", str(frame_interval),
                "--stride", str(stride),
            ]
            if num_frames is not None:
                cmd.extend(["--num_frames", str(num_frames)])
            if extra_args:
                cmd.extend(extra_args)

            print(f"\n{'='*60}")
            print(f"Running: scene_{scene_id} / cam_{cam_idx}")
            print(f"{' '.join(cmd)}")
            print(f"{'='*60}")

            result = subprocess.run(cmd, cwd=parent_dir)
            if result.returncode != 0:
                print(f"[ERROR] benchmark.py 返回非零退出码: {result.returncode}")
            else:
                print(f"[OK] scene_{scene_id} / cam_{cam_idx} 完成")


# ==================== Step 4: 聚合多场景 benchmark 结果 ====================

def aggregate_metrics_by_window(per_frame_list, metric_keys=("psnr", "ssim", "lpips")):
    """
    将 per_frame 指标按 window_idx 聚合。

    Args:
        per_frame_list: list of dicts, 每个 dict 含 window_idx + 指标值
        metric_keys: 要聚合的指标名

    Returns:
        dict: {window_idx: {key: [values...], ...}, ...}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in per_frame_list:
        w = entry.get("window_idx", 0)
        for k in metric_keys:
            if k in entry:
                grouped[w][k].append(entry[k])
    return dict(grouped)


def aggregate_pose_by_window(per_window_list_list, metric_keys=("ate_rmse", "scale", "rot_mean", "rpe_trans", "rpe_rot")):
    """
    将 pose per_window 指标按 window 索引聚合（跨 scene 对齐）。

    Args:
        per_window_list_list: list of per_window lists, 每个 scene 一个
        metric_keys: 要聚合的指标名

    Returns:
        dict: {window_idx: {key: [values...], ...}, ...}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for scene_per_window in per_window_list_list:
        for idx, entry in enumerate(scene_per_window):
            for k in metric_keys:
                if k in entry:
                    grouped[idx][k].append(entry[k])
    return dict(grouped)


def compute_aggregated_stats(grouped_data, metric_keys):
    """
    从分组数据计算 mean ± std。

    Returns:
        dict: {window_idx: {key_mean, key_std, "num_samples": N}, ...}
    """
    stats = {}
    for w in sorted(grouped_data.keys()):
        w_stats = {"num_samples": None}
        for k in metric_keys:
            vals = grouped_data[w].get(k, [])
            if vals:
                w_stats[f"{k}_mean"] = float(np.mean(vals))
                w_stats[f"{k}_std"] = float(np.std(vals))
                w_stats[f"{k}_num"] = len(vals)
        # 取最大的 num 作为 num_samples
        nums = [w_stats[f"{k}_num"] for k in metric_keys if f"{k}_num" in w_stats]
        if nums:
            w_stats["num_samples"] = max(nums)
            for k in metric_keys:
                w_stats.pop(f"{k}_num", None)
            stats[w] = w_stats
    return stats


def plot_aggregated_metrics_comparison(gen_stats, gt_stats, output_path, stride=1):
    """
    绘制聚合后的渲染指标平均曲线 (PSNR/SSIM/LPIPS)。
    mean 线 + std 阴影。
    """
    def extract_plot_data(stats, stride):
        if not stats:
            return None
        windows = sorted(stats.keys())
        return {
            "start_frame": [w * stride for w in windows],
            "psnr_mean": [stats[w]["psnr_mean"] for w in windows],
            "psnr_std": [stats[w]["psnr_std"] for w in windows],
            "ssim_mean": [stats[w]["ssim_mean"] for w in windows],
            "ssim_std": [stats[w]["ssim_std"] for w in windows],
            "lpips_mean": [stats[w]["lpips_mean"] for w in windows],
            "lpips_std": [stats[w]["lpips_std"] for w in windows],
            "num_samples": stats[windows[0]].get("num_samples", "?"),
        }

    gen_data = extract_plot_data(gen_stats, stride)
    gt_data = extract_plot_data(gt_stats, stride)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for ax, metric, ylabel, title in [
        (axes[0], "psnr", "PSNR (dB)", "PSNR per Sliding Window (Aggregated)"),
        (axes[1], "ssim", "SSIM", "SSIM per Sliding Window (Aggregated)"),
        (axes[2], "lpips", "LPIPS", "LPIPS per Sliding Window (Aggregated, lower is better)"),
    ]:
        if gt_data:
            n = gt_data["num_samples"]
            x = gt_data["start_frame"]
            y = gt_data[f"{metric}_mean"]
            std = gt_data[f"{metric}_std"]
            ax.plot(x, y, "b-o", label=f"GT (n={n})", markersize=4, linewidth=2)
            ax.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                            color="b", alpha=0.15)
            ax.axhline(y=np.mean(y), color="b", linestyle="--", alpha=0.5,
                        label=f"GT avg: {np.mean(y):.4f}")
        if gen_data:
            n = gen_data["num_samples"]
            x = gen_data["start_frame"]
            y = gen_data[f"{metric}_mean"]
            std = gen_data[f"{metric}_std"]
            ax.plot(x, y, "r-s", label=f"Generated (n={n})", markersize=4, linewidth=2)
            ax.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                            color="r", alpha=0.15)
            ax.axhline(y=np.mean(y), color="r", linestyle="--", alpha=0.5,
                        label=f"Gen avg: {np.mean(y):.4f}")
        ax.set_xlabel("Window Start Frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_path, "aggregated_metrics_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"聚合渲染指标曲线已保存: {save_path}")


def plot_aggregated_pose_metrics(gen_stats, gt_stats, output_path, stride=1):
    """
    绘制聚合后的 Pose 指标平均曲线。
    2×3 布局: ATE RMSE / ATE Mean / Scale / Rot / RPE Trans / Summary
    """
    pose_keys = ["ate_rmse", "ate_mean", "scale", "rot_mean", "rpe_trans"]
    pose_labels = [
        ("ate_rmse", "ATE RMSE (m)", "Absolute Trajectory Error (RMSE)"),
        ("ate_mean", "ATE Mean (m)", "Absolute Trajectory Error (Mean)"),
        ("scale", "Scale Factor", "Estimated Scale Factor"),
        ("rot_mean", "Rotation Error (°)", "Rotation Error (Mean)"),
        ("rpe_trans", "RPE Translation (m)", "Relative Pose Error (Translation)"),
    ]

    def extract_plot_data(stats, stride):
        if not stats:
            return None
        windows = sorted(stats.keys())
        result = {"start_frame": [w * stride for w in windows]}
        for k in pose_keys:
            result[f"{k}_mean"] = [stats[w][f"{k}_mean"] for w in windows]
            result[f"{k}_std"] = [stats[w][f"{k}_std"] for w in windows]
        result["num_samples"] = stats[windows[0]].get("num_samples", "?")
        return result

    gen_data = extract_plot_data(gen_stats, stride)
    gt_data = extract_plot_data(gt_stats, stride)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, (key, ylabel, title) in enumerate(pose_labels):
        ax = axes[i // 3, i % 3]
        if gt_data:
            n = gt_data["num_samples"]
            x = gt_data["start_frame"]
            y = gt_data[f"{key}_mean"]
            std = gt_data[f"{key}_std"]
            ax.plot(x, y, "b-o", label=f"GT (n={n})", markersize=4, linewidth=2)
            ax.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                            color="b", alpha=0.15)
            ax.axhline(y=np.mean(y), color="b", linestyle="--", alpha=0.5,
                        label=f"GT avg: {np.mean(y):.4f}")
        if gen_data:
            n = gen_data["num_samples"]
            x = gen_data["start_frame"]
            y = gen_data[f"{key}_mean"]
            std = gen_data[f"{key}_std"]
            ax.plot(x, y, "r-s", label=f"Generated (n={n})", markersize=4, linewidth=2)
            ax.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                            color="r", alpha=0.15)
            ax.axhline(y=np.mean(y), color="r", linestyle="--", alpha=0.5,
                        label=f"Gen avg: {np.mean(y):.4f}")
        ax.set_xlabel("Window Start Frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["Pose Evaluation Summary (Aggregated)", "=" * 40, ""]

    for label, data in [("Generated", gen_data), ("GT", gt_data)]:
        if data:
            lines.extend([
                f"{label}:",
                f"  ATE RMSE: {np.mean(data['ate_rmse_mean']):.4f} ± {np.mean(data['ate_rmse_std']):.4f} m",
                f"  Rot Err:  {np.mean(data['rot_mean_mean']):.2f} ± {np.mean(data['rot_mean_std']):.2f}°",
                f"  Scale:    {np.mean(data['scale_mean']):.4f}",
                "",
            ])

    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(output_path, "aggregated_pose_metrics_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"聚合 Pose 指标曲线已保存: {save_path}")


def plot_aggregated_nvs_metrics(gen_stats, gt_stats, output_path, stride=1):
    """
    绘制聚合后的 NVS 指标平均曲线 (PSNR/SSIM/LPIPS)。
    """
    def extract_plot_data(stats, stride):
        if not stats:
            return None
        windows = sorted(stats.keys())
        return {
            "start_frame": [w * stride for w in windows],
            "psnr_mean": [stats[w]["psnr_mean"] for w in windows],
            "psnr_std": [stats[w]["psnr_std"] for w in windows],
            "ssim_mean": [stats[w]["ssim_mean"] for w in windows],
            "ssim_std": [stats[w]["ssim_std"] for w in windows],
            "lpips_mean": [stats[w]["lpips_mean"] for w in windows],
            "lpips_std": [stats[w]["lpips_std"] for w in windows],
            "num_samples": stats[windows[0]].get("num_samples", "?"),
        }

    gen_data = extract_plot_data(gen_stats, stride)
    gt_data = extract_plot_data(gt_stats, stride)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for ax, metric, ylabel, title in [
        (axes[0], "psnr", "PSNR (dB)", "NVS - PSNR per Window (Aggregated)"),
        (axes[1], "ssim", "SSIM", "NVS - SSIM per Window (Aggregated)"),
        (axes[2], "lpips", "LPIPS", "NVS - LPIPS per Window (Aggregated, lower is better)"),
    ]:
        if gt_data:
            n = gt_data["num_samples"]
            x = gt_data["start_frame"]
            y = gt_data[f"{metric}_mean"]
            std = gt_data[f"{metric}_std"]
            ax.plot(x, y, "b-o", label=f"GT (n={n})", markersize=4, linewidth=2)
            ax.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                            color="b", alpha=0.15)
            ax.axhline(y=np.mean(y), color="b", linestyle="--", alpha=0.5,
                        label=f"GT avg: {np.mean(y):.4f}")
        if gen_data:
            n = gen_data["num_samples"]
            x = gen_data["start_frame"]
            y = gen_data[f"{metric}_mean"]
            std = gen_data[f"{metric}_std"]
            ax.plot(x, y, "r-s", label=f"Generated (n={n})", markersize=4, linewidth=2)
            ax.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                            color="r", alpha=0.15)
            ax.axhline(y=np.mean(y), color="r", linestyle="--", alpha=0.5,
                        label=f"Gen avg: {np.mean(y):.4f}")
        ax.set_xlabel("Window Start Frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_path, "aggregated_nvs_metrics_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"聚合 NVS 指标曲线已保存: {save_path}")


def aggregate_benchmark_results(output_dir, source_name, num_scenes, cam_indices, stride=1):
    """
    聚合所有场景的 benchmark 结果：
    1. 读取每个 scene/cam 的 JSON 输出
    2. 按 window 位置对齐后取 mean ± std
    3. 绘制平均曲线
    4. 保存聚合结果 JSON
    """
    convert_dir = os.path.join(output_dir, source_name)
    bench_base = os.path.join(convert_dir, "benchmark_output")
    agg_dir = os.path.join(bench_base, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    # 收集所有数据
    all_gen_render = []    # per_frame from generated_metrics.json
    all_gt_render = []     # per_frame from gt_metrics.json
    all_gen_nvs = []       # per_frame from generated_nvs_metrics.json
    all_gt_nvs = []        # per_frame from gt_nvs_metrics.json
    all_gen_pose_per_scene = []   # list of per_window lists, one per scene
    all_gt_pose_per_scene = []    # list of per_window lists, one per scene

    valid_scenes = 0

    for scene_id in range(num_scenes):
        scene_key = f"scene_{scene_id}"

        for cam_idx in cam_indices:
            scene_bench_dir = os.path.join(bench_base, scene_key, f"cam_{cam_idx}")
            if not os.path.isdir(scene_bench_dir):
                continue

            # 渲染指标
            gen_metrics_path = os.path.join(scene_bench_dir, "generated_metrics.json")
            gt_metrics_path = os.path.join(scene_bench_dir, "gt_metrics.json")

            if os.path.exists(gen_metrics_path):
                with open(gen_metrics_path, "r") as f:
                    d = json.load(f)
                all_gen_render.extend(d.get("per_frame", []))
            if os.path.exists(gt_metrics_path):
                with open(gt_metrics_path, "r") as f:
                    d = json.load(f)
                all_gt_render.extend(d.get("per_frame", []))

            # Pose 指标 (按 scene 收集)
            pose_path = os.path.join(scene_bench_dir, "pose_metrics.json")
            if os.path.exists(pose_path):
                with open(pose_path, "r") as f:
                    d = json.load(f)
                if "generated" in d and "per_window" in d["generated"]:
                    all_gen_pose_per_scene.append(d["generated"]["per_window"])
                if "gt_pred" in d and "per_window" in d["gt_pred"]:
                    all_gt_pose_per_scene.append(d["gt_pred"]["per_window"])

            # NVS 指标
            gen_nvs_path = os.path.join(scene_bench_dir, "generated_nvs_metrics.json")
            gt_nvs_path = os.path.join(scene_bench_dir, "gt_nvs_metrics.json")

            if os.path.exists(gen_nvs_path):
                with open(gen_nvs_path, "r") as f:
                    d = json.load(f)
                all_gen_nvs.extend(d.get("per_frame", []))
            if os.path.exists(gt_nvs_path):
                with open(gt_nvs_path, "r") as f:
                    d = json.load(f)
                all_gt_nvs.extend(d.get("per_frame", []))

            valid_scenes += 1

    if valid_scenes == 0:
        print("[ERROR] 没有找到任何 benchmark 输出数据")
        sys.exit(1)

    print(f"聚合 {valid_scenes} 个 scene/cam 组合的结果")
    print(f"  渲染指标: gen {len(all_gen_render)} 帧, gt {len(all_gt_render)} 帧")
    print(f"  Pose 指标: {len(all_gen_pose_per_scene)} scenes")
    print(f"  NVS 指标: gen {len(all_gen_nvs)} 帧, gt {len(all_gt_nvs)} 帧")

    # 按 window 聚合渲染指标
    render_keys = ("psnr", "ssim", "lpips")
    gen_render_grouped = aggregate_metrics_by_window(all_gen_render, render_keys)
    gt_render_grouped = aggregate_metrics_by_window(all_gt_render, render_keys)
    gen_render_stats = compute_aggregated_stats(gen_render_grouped, render_keys)
    gt_render_stats = compute_aggregated_stats(gt_render_grouped, render_keys)

    # 按 window 聚合 Pose 指标 (跨 scene 对齐)
    pose_keys = ("ate_rmse", "ate_mean", "scale", "rot_mean", "rpe_trans", "rpe_rot")
    gen_pose_grouped = aggregate_pose_by_window(all_gen_pose_per_scene, pose_keys)
    gt_pose_grouped = aggregate_pose_by_window(all_gt_pose_per_scene, pose_keys)
    gen_pose_stats = compute_aggregated_stats(gen_pose_grouped, pose_keys)
    gt_pose_stats = compute_aggregated_stats(gt_pose_grouped, pose_keys)

    # 按 window 聚合 NVS 指标
    gen_nvs_grouped = aggregate_metrics_by_window(all_gen_nvs, render_keys)
    gt_nvs_grouped = aggregate_metrics_by_window(all_gt_nvs, render_keys)
    gen_nvs_stats = compute_aggregated_stats(gen_nvs_grouped, render_keys)
    gt_nvs_stats = compute_aggregated_stats(gt_nvs_grouped, render_keys)

    # 绘图
    if gen_render_stats or gt_render_stats:
        plot_aggregated_metrics_comparison(gen_render_stats, gt_render_stats, agg_dir, stride)
    if gen_pose_stats or gt_pose_stats:
        plot_aggregated_pose_metrics(gen_pose_stats, gt_pose_stats, agg_dir, stride)
    if gen_nvs_stats or gt_nvs_stats:
        plot_aggregated_nvs_metrics(gen_nvs_stats, gt_nvs_stats, agg_dir, stride)

    # 保存聚合结果 JSON
    aggregated_result = {
        "num_scenes": valid_scenes,
        "rendering": {
            "generated": gen_render_stats,
            "gt": gt_render_stats,
        },
        "pose": {
            "generated": gen_pose_stats,
            "gt": gt_pose_stats,
        },
        "nvs": {
            "generated": gen_nvs_stats,
            "gt": gt_nvs_stats,
        },
    }

    agg_json_path = os.path.join(agg_dir, "aggregated_metrics.json")
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_result, f, indent=2, ensure_ascii=False)
    print(f"\n聚合结果已保存: {agg_json_path}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("聚合 Benchmark 结果汇总")
    print("=" * 60)

    if gen_render_stats:
        all_psnr = [gen_render_stats[w]["psnr_mean"] for w in sorted(gen_render_stats)]
        all_ssim = [gen_render_stats[w]["ssim_mean"] for w in sorted(gen_render_stats)]
        all_lpips = [gen_render_stats[w]["lpips_mean"] for w in sorted(gen_render_stats)]
        print(f"\nGenerated 渲染指标 (avg over {valid_scenes} scenes):")
        print(f"  PSNR:  {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f} dB")
        print(f"  SSIM:  {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
        print(f"  LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")

    if gt_render_stats:
        all_psnr = [gt_render_stats[w]["psnr_mean"] for w in sorted(gt_render_stats)]
        all_ssim = [gt_render_stats[w]["ssim_mean"] for w in sorted(gt_render_stats)]
        all_lpips = [gt_render_stats[w]["lpips_mean"] for w in sorted(gt_render_stats)]
        print(f"\nGT 渲染指标 (avg over {valid_scenes} scenes):")
        print(f"  PSNR:  {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f} dB")
        print(f"  SSIM:  {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
        print(f"  LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")

    if gen_pose_stats:
        all_ate = [gen_pose_stats[w]["ate_rmse_mean"] for w in sorted(gen_pose_stats)]
        all_rot = [gen_pose_stats[w]["rot_mean_mean"] for w in sorted(gen_pose_stats)]
        print(f"\nGenerated Pose 指标 (avg over {valid_scenes} scenes):")
        print(f"  ATE RMSE: {np.mean(all_ate):.4f} m")
        print(f"  Rot Err:  {np.mean(all_rot):.2f}°")

    if gen_nvs_stats:
        all_psnr = [gen_nvs_stats[w]["psnr_mean"] for w in sorted(gen_nvs_stats)]
        all_ssim = [gen_nvs_stats[w]["ssim_mean"] for w in sorted(gen_nvs_stats)]
        all_lpips = [gen_nvs_stats[w]["lpips_mean"] for w in sorted(gen_nvs_stats)]
        print(f"\nGenerated NVS 指标 (avg over {valid_scenes} scenes):")
        print(f"  PSNR:  {np.mean(all_psnr):.4f} dB")
        print(f"  SSIM:  {np.mean(all_ssim):.4f}")
        print(f"  LPIPS: {np.mean(all_lpips):.4f}")

    print(f"\n输出目录: {agg_dir}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="将 sf 输出转换为 benchmark 输入格式")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # --- prepare ---
    p_prepare = subparsers.add_parser("prepare", help="准备公共元数据 (token + pose)，一劳永逸")
    p_prepare.add_argument("--sf_preview_dir", type=str, required=True,
                           help="sf 输出目录 (含 preview_sampletok/)")
    p_prepare.add_argument("--pose_base_dir", type=str, required=True,
                           help="dmd 的 pose 基础目录 (含 scene_id/gt_camera_params/)")
    p_prepare.add_argument("--output_dir", type=str, default="benchmark_input",
                           help="输出目录")
    p_prepare.add_argument("--num_scenes", type=int, default=10,
                           help="处理 scene 数量")

    # --- convert ---
    p_convert = subparsers.add_parser("convert", help="从 npy 提取图像，按 token 对齐")
    p_convert.add_argument("--sf_preview_dir", type=str, required=True,
                           help="模型输出目录 (含 preview_numpy/ 和 preview_sampletok/)")
    p_convert.add_argument("--output_dir", type=str, default="benchmark_input",
                           help="输出目录 (需先运行 prepare)")
    p_convert.add_argument("--num_scenes", type=int, default=10,
                           help="处理 scene 数量")
    p_convert.add_argument("--cams", type=int, nargs="+", default=[0],
                           help="要提取的 camera 索引 (0~5)，默认 [0]")

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="调用 benchmark.py 评测")
    p_bench.add_argument("--output_dir", type=str, default="benchmark_input",
                         help="数据根目录")
    p_bench.add_argument("--source_name", type=str, required=True,
                         help="模型来源目录名 (如 sf_chunk15_step12)")
    p_bench.add_argument("--ckpt_path", type=str, required=True,
                         help="模型 checkpoint 路径")
    p_bench.add_argument("--num_scenes", type=int, default=10,
                         help="评测 scene 数量")
    p_bench.add_argument("--cams", type=int, nargs="+", default=[0],
                         help="要评测的 camera 索引 (0~5)，默认 [0]")
    p_bench.add_argument("--sequence_length", type=int, default=4)
    p_bench.add_argument("--start_idx", type=int, default=4)
    p_bench.add_argument("--frame_interval", type=int, default=4)
    p_bench.add_argument("--stride", type=int, default=1)
    p_bench.add_argument("--num_frames", type=int, default=100)

    # --- aggregate ---
    p_agg = subparsers.add_parser("aggregate", help="聚合多场景 benchmark 结果，画平均曲线")
    p_agg.add_argument("--output_dir", type=str, default="benchmark_input",
                        help="数据根目录")
    p_agg.add_argument("--source_name", type=str, required=True,
                        help="模型来源目录名 (如 sf_chunk15_step12)")
    p_agg.add_argument("--num_scenes", type=int, default=10,
                        help="聚合 scene 数量")
    p_agg.add_argument("--cams", type=int, nargs="+", default=[0],
                        help="要聚合的 camera 索引 (0~5)，默认 [0]")
    p_agg.add_argument("--stride", type=int, default=1,
                        help="滑动窗口步长 (需与 benchmark 步骤一致)")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_metadata(
            args.sf_preview_dir, args.pose_base_dir,
            args.output_dir, args.num_scenes,
        )
    elif args.command == "convert":
        extract_images_from_npy(
            args.sf_preview_dir, args.output_dir,
            args.num_scenes, args.cams,
        )
    elif args.command == "benchmark":
        run_benchmark(
            args.output_dir, args.source_name, args.ckpt_path,
            args.num_scenes, args.cams,
            args.sequence_length, args.start_idx, args.frame_interval,
            args.stride, args.num_frames,
        )
    elif args.command == "aggregate":
        aggregate_benchmark_results(
            args.output_dir, args.source_name,
            args.num_scenes, args.cams, args.stride,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
