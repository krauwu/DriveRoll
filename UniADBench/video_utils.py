from __future__ import annotations

import cv2
from typing import Dict


def get_video_meta(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": (frame_count / fps) if fps > 0 else 0.0,
    }


def frame_idx_to_time_sec(frame_idx: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return frame_idx / fps
