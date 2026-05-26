from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List
import csv

import av
import torch
import torchvision
import torchvision.transforms.functional as TF

from torchmetrics.image.fid import FrechetInceptionDistance
from dwm.metrics.fvd import FrechetVideoDistance


def split_row_cols_uint8(img_rgb, row_idx: int, rows: int, cols: int):
    height, width, _ = img_rgb.shape
    cell_h = height // rows
    cell_w = width // cols
    row = img_rgb[row_idx * cell_h:(row_idx + 1) * cell_h]

    patches = []
    for col_idx in range(cols):
        patch = row[:, col_idx * cell_w:(col_idx + 1) * cell_w]
        patches.append(torch.from_numpy(patch).permute(2, 0, 1))
    return torch.stack(patches, dim=0)


def split_row_cols_float01(img_rgb, row_idx: int, rows: int, cols: int):
    return split_row_cols_uint8(img_rgb, row_idx, rows, cols).float().div(255.0)


def save_debug_vis(img_rgb, out_dir: Path, gt_row: int, gen_row: int, rows: int, cols: int, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    TF.to_pil_image(torch.from_numpy(img_rgb).permute(2, 0, 1)).save(out_dir / f"{tag}_FULL.png")

    gt6 = split_row_cols_float01(img_rgb, gt_row, rows, cols)
    gen6 = split_row_cols_float01(img_rgb, gen_row, rows, cols)
    gt_grid = torchvision.utils.make_grid(gt6, nrow=cols, padding=2)
    gen_grid = torchvision.utils.make_grid(gen6, nrow=cols, padding=2)
    TF.to_pil_image(gt_grid).save(out_dir / f"{tag}_GT_row.png")
    TF.to_pil_image(gen_grid).save(out_dir / f"{tag}_GEN_row.png")


def build_video_paths(video_dir: Path) -> List[Path]:
    return sorted(video_dir.glob("*.mp4"))


def run_video_fid_fvd(cfg: Dict, dataset=None) -> Dict:
    input_cfg = cfg["input"]
    task_cfg = cfg["task"]["video_fid_fvd"]

    video_dir = Path(input_cfg["video_dir"])
    out_csv = Path(task_cfg["out_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    videos = build_video_paths(video_dir)
    if not videos:
        raise FileNotFoundError(f"No mp4 found in: {video_dir}")

    device = task_cfg.get("device", "cuda")
    rows = int(task_cfg.get("rows", 4))
    cols = int(task_cfg.get("cols", 6))
    gt_row = int(task_cfg.get("gt_row", 0))
    gen_row = int(task_cfg.get("gen_row", 3))
    start_frame = int(task_cfg.get("start_frame", 16))
    window_len = int(task_cfg.get("window_len", 15))
    stride = int(task_cfg.get("stride", 1))
    i3d_ckpt = str(task_cfg["i3d_ckpt"])
    save_debug = bool(task_cfg.get("save_debug", True))

    containers = [av.open(str(path)) for path in videos]
    decoders = [container.decode(video=0) for container in containers]

    num_videos = len(videos)
    buf_real = [deque(maxlen=window_len) for _ in range(num_videos)]
    buf_fake = [deque(maxlen=window_len) for _ in range(num_videos)]

    fid = FrechetInceptionDistance(normalize=True).to(device)
    fvd = FrechetVideoDistance(
        inception_3d_checkpoint_path=i3d_ckpt,
        sequence_count=window_len,
    ).to(device)

    first_out_t = start_frame + window_len - 1
    debug_saved = False
    rows_out = []

    try:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_idx",
                    "window_start",
                    "window_end",
                    "window_len",
                    "fid",
                    "fvd",
                    "num_videos_used",
                ],
            )
            writer.writeheader()

            frame_idx = 0
            while True:
                frames = []
                try:
                    for decoder in decoders:
                        frames.append(next(decoder))
                except StopIteration:
                    break

                if frame_idx < start_frame:
                    frame_idx += 1
                    continue

                for video_idx, frame in enumerate(frames):
                    img = frame.to_ndarray(format="rgb24")
                    buf_real[video_idx].append(split_row_cols_uint8(img, gt_row, rows, cols))
                    buf_fake[video_idx].append(split_row_cols_uint8(img, gen_row, rows, cols))

                if frame_idx < first_out_t:
                    frame_idx += 1
                    continue

                if (frame_idx - first_out_t) % stride != 0:
                    frame_idx += 1
                    continue

                if save_debug and (not debug_saved) and frame_idx == first_out_t:
                    debug_dir = out_csv.with_suffix("")
                    debug_dir = debug_dir.parent / f"{debug_dir.name}_debug_first_batch"
                    save_debug_vis(
                        img_rgb=frames[0].to_ndarray(format="rgb24"),
                        out_dir=debug_dir,
                        gt_row=gt_row,
                        gen_row=gen_row,
                        rows=rows,
                        cols=cols,
                        tag=f"video0_t{frame_idx:06d}",
                    )
                    debug_saved = True

                fid.reset()
                fvd.reset()

                for video_idx in range(num_videos):
                    real_win = torch.stack(list(buf_real[video_idx]), dim=0)
                    fake_win = torch.stack(list(buf_fake[video_idx]), dim=0)

                    real_imgs = real_win.flatten(0, 1).to(device).float().div_(255.0)
                    fake_imgs = fake_win.flatten(0, 1).to(device).float().div_(255.0)
                    fid.update(real_imgs, real=True)
                    fid.update(fake_imgs, real=False)

                    real_seq = real_win.permute(1, 0, 2, 3, 4).to(device).float().div_(255.0)
                    fake_seq = fake_win.permute(1, 0, 2, 3, 4).to(device).float().div_(255.0)
                    fvd.update(real_seq, real=True)
                    fvd.update(fake_seq, real=False)

                fid_val = float(fid.compute())
                fvd_val = float(fvd.compute())

                row = {
                    "frame_idx": frame_idx,
                    "window_start": frame_idx - window_len + 1,
                    "window_end": frame_idx,
                    "window_len": window_len,
                    "fid": fid_val,
                    "fvd": fvd_val,
                    "num_videos_used": num_videos,
                }
                writer.writerow(row)
                rows_out.append(row)
                print(
                    f"[t={frame_idx}] win=[{row['window_start']},{row['window_end']}] "
                    f"fid={fid_val:.4f} fvd={fvd_val:.4f}"
                )

                frame_idx += 1
    finally:
        for container in containers:
            container.close()

    return {
        "out_csv": str(out_csv),
        "num_rows": len(rows_out),
        "num_videos": num_videos,
    }
