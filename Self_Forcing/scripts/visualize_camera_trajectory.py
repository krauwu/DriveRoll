#!/usr/bin/env python
"""
可视化保存的相机外参轨迹
显示相机位置和朝向，只可视化 camera_1
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_camera_params(data_dir, cam_idx=1):
    """
    加载指定相机的参数

    Args:
        data_dir: gt_camera_params 目录路径
        cam_idx: 相机索引

    Returns:
        camera_transforms: list of (4, 4) matrices
        ego_transforms: list of (4, 4) matrices
    """
    cam_dir = f"cam_{cam_idx}"

    # 加载 camera_transforms
    camera_transforms_dir = os.path.join(data_dir, "camera_transforms", cam_dir)
    camera_transforms = []
    if os.path.exists(camera_transforms_dir):
        files = sorted([f for f in os.listdir(camera_transforms_dir) if f.endswith('.npy')])
        for f in files:
            transform = np.load(os.path.join(camera_transforms_dir, f))
            camera_transforms.append(transform)

    # 加载 ego_transforms
    ego_transforms_dir = os.path.join(data_dir, "ego_transforms", cam_dir)
    ego_transforms = []
    if os.path.exists(ego_transforms_dir):
        files = sorted([f for f in os.listdir(ego_transforms_dir) if f.endswith('.npy')])
        for f in files:
            transform = np.load(os.path.join(ego_transforms_dir, f))
            ego_transforms.append(transform)

    return camera_transforms, ego_transforms


def get_camera_pose(transform_matrix):
    """
    从变换矩阵提取相机位置和朝向

    假设 transform_matrix 是 world-to-camera 变换 (world -> camera)
    即 P_camera = T @ P_world

    Returns:
        position: (3,) 相机在世界坐标系中的位置
        forward: (3,) 相机朝向 (看向的方向)
        up: (3,) 相机 up 方向
        right: (3,) 相机 right 方向
    """
    T = transform_matrix

    # 如果是 world-to-camera，需要求逆得到 camera-to-world
    # 但通常自动驾驶数据集中，transform 可能已经是 camera-to-world
    # 这里假设是 camera-to-world (即 T 描述相机在世界坐标系中的位姿)

    # 位置：平移部分
    position = T[:3, 3]

    # 旋转矩阵部分
    R = T[:3, :3]

    # 相机坐标系惯例：
    # forward (看向方向) 通常是 -z 或 z
    # up 通常是 y
    # right 通常是 x

    # 假设 OpenCV 惯例: x-right, y-down, z-forward
    # 或者 OpenGL 惯例: x-right, y-up, z-backward

    # 这里假设 forward 是 -z 方向 (看向场景)
    right = R[:, 0]    # x 轴
    up = R[:, 1]       # y 轴
    forward = -R[:, 2] # -z 轴 (看向方向)

    return position, forward, up, right


def visualize_trajectory(camera_transforms, ego_transforms=None, title="Camera Trajectory", save_path=None):
    """
    可视化相机轨迹

    Args:
        camera_transforms: list of (4, 4) camera transform matrices
        ego_transforms: list of (4, 4) ego transforms (可选)
        title: 图标题
        save_path: 保存路径 (可选)
    """
    fig = plt.figure(figsize=(14, 6))

    # 3D 视图
    ax1 = fig.add_subplot(121, projection='3d')

    # 提取所有相机位置
    positions = []
    forwards = []
    ups = []
    rights = []

    for T in camera_transforms:
        pos, fwd, up, right = get_camera_pose(T)
        positions.append(pos)
        forwards.append(fwd)
        ups.append(up)
        rights.append(right)

    positions = np.array(positions)
    forwards = np.array(forwards)

    # 绘制轨迹线
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
             'b-', linewidth=2, label='Camera trajectory')

    # 绘制相机位置点
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                c=range(len(positions)), cmap='viridis', s=30, label='Camera positions')

    # 绘制相机朝向箭头 (每隔几帧绘制一个)
    arrow_step = max(1, len(positions) // 10)
    arrow_length = np.mean(np.linalg.norm(positions[1:] - positions[:-1], axis=1)) * 3 if len(positions) > 1 else 1.0

    for i in range(0, len(positions), arrow_step):
        # Forward direction (蓝色)
        ax1.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                   forwards[i, 0] * arrow_length,
                   forwards[i, 1] * arrow_length,
                   forwards[i, 2] * arrow_length,
                   color='blue', arrow_length_ratio=0.3, linewidth=1.5)

    # 标记起点和终点
    ax1.scatter(*positions[0], color='green', s=100, marker='o', label='Start')
    ax1.scatter(*positions[-1], color='red', s=100, marker='*', label='End')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title}\nCamera 1 Trajectory (3D View)')
    ax1.legend()

    # 设置相等的轴比例
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2D 俯视图 (X-Y 平面)
    ax2 = fig.add_subplot(122)

    # 绘制轨迹
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax2.scatter(positions[:, 0], positions[:, 1], c=range(len(positions)),
                cmap='viridis', s=30)

    # 绘制朝向箭头
    for i in range(0, len(positions), arrow_step):
        ax2.arrow(positions[i, 0], positions[i, 1],
                  forwards[i, 0] * arrow_length * 0.8,
                  forwards[i, 1] * arrow_length * 0.8,
                  head_width=arrow_length * 0.2, head_length=arrow_length * 0.1,
                  fc='blue', ec='blue', alpha=0.7)

    # 标记起点终点
    ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, marker='*', label='End', zorder=5)

    # 添加帧编号标注
    for i in [0, len(positions)//2, len(positions)-1]:
        ax2.annotate(f'f{i}', (positions[i, 0], positions[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{title}\nCamera 1 Trajectory (Top View)')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


def visualize_with_ego(camera_transforms, ego_transforms, title="Camera & Ego Trajectory", save_path=None):
    """
    可视化相机轨迹和 ego 轨迹
    """
    fig = plt.figure(figsize=(14, 6))

    # 提取相机位置
    cam_positions = []
    cam_forwards = []
    for T in camera_transforms:
        pos, fwd, _, _ = get_camera_pose(T)
        cam_positions.append(pos)
        cam_forwards.append(fwd)
    cam_positions = np.array(cam_positions)
    cam_forwards = np.array(cam_forwards)

    # 提取 ego 位置
    ego_positions = []
    ego_forwards = []
    for T in ego_transforms:
        pos, fwd, _, _ = get_camera_pose(T)
        ego_positions.append(pos)
        ego_forwards.append(fwd)
    ego_positions = np.array(ego_positions)
    ego_forwards = np.array(ego_forwards)

    # 3D 视图
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
             'b-', linewidth=2, label='Camera')
    ax1.plot(ego_positions[:, 0], ego_positions[:, 1], ego_positions[:, 2],
             'r--', linewidth=2, label='Ego vehicle')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title}\n(3D View)')
    ax1.legend()

    # 2D 俯视图
    ax2 = fig.add_subplot(122)

    ax2.plot(cam_positions[:, 0], cam_positions[:, 1], 'b-', linewidth=2, label='Camera')
    ax2.plot(ego_positions[:, 0], ego_positions[:, 1], 'r--', linewidth=2, label='Ego vehicle')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{title}\n(Top View)')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize camera trajectory from saved npy files")
    parser.add_argument("-d", "--data-dir", type=str, required=True,
                        help="Path to gt_camera_params directory")
    parser.add_argument("-c", "--cam-idx", type=int, default=1,
                        help="Camera index to visualize (default: 1)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output image path (optional)")
    parser.add_argument("--with-ego", action="store_true",
                        help="Also visualize ego transforms")
    args = parser.parse_args()

    print(f"Loading camera params from: {args.data_dir}")
    print(f"Camera index: {args.cam_idx}")

    camera_transforms, ego_transforms = load_camera_params(args.data_dir, args.cam_idx)

    if not camera_transforms:
        print("Error: No camera_transforms found!")
        return

    print(f"Loaded {len(camera_transforms)} camera transforms")
    if ego_transforms:
        print(f"Loaded {len(ego_transforms)} ego transforms")

    # 可视化
    if args.with_ego and ego_transforms:
        visualize_with_ego(camera_transforms, ego_transforms,
                          title=f"Camera {args.cam_idx}",
                          save_path=args.output)
    else:
        visualize_trajectory(camera_transforms,
                            title=f"Camera {args.cam_idx}",
                            save_path=args.output)


if __name__ == "__main__":
    main()