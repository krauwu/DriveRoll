from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

import cv2
import numpy as np
from tqdm.auto import tqdm

from datasets import build_dataset
from video_utils import get_video_meta
from tasks.uniad_info_builder import build_uniad_info_for_segment, dump_uniad_info
from tasks.video_metrics import run_video_fid_fvd
from tasks.video_reconstruction import run_video_reconstruction


GRID_ROWS = 4
GRID_COLS = 6
GENERATED_ROW_INDEX = -1
GENERATED_IMAGE_HW = (900, 1600)


TASKS = {
    "uniad_window_pkls": {"runner": "run_uniad_window_pkls", "needs_dataset": True},
    "uniad_group_pkls": {"runner": "run_uniad_group_pkls", "needs_dataset": True},
    "video_fid_fvd": {"runner": "run_video_fid_fvd", "needs_dataset": False},
    "video_reconstruction": {"runner": "run_video_reconstruction", "needs_dataset": False},
}


RUNNERS = {
    "run_uniad_window_pkls": None,
    "run_uniad_group_pkls": None,
    "run_video_fid_fvd": run_video_fid_fvd,
    "run_video_reconstruction": run_video_reconstruction,
}

def resolve_sidecar_path(
    video_path: Path,
    video_root: Path,
    sidecar_root: Path,
    new_suffix: str,
) -> Path:
    rel_path = video_path.relative_to(video_root)
    sidecar_path = sidecar_root / rel_path.with_suffix(new_suffix)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar file not found:\n"
            f"video_path={video_path}\n"
            f"expected_path={sidecar_path}"
        )
    return sidecar_path

def read_init_token(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        token = f.readline().strip()
    if not token:
        raise ValueError(f"Empty token file: {path}")
    return token


def collect_video_paths(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.rglob("*.mp4") if p.is_file())


def compute_windows(num_frames: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError(f"window_size and stride must be positive, got {window_size}, {stride}")
    if window_size > num_frames:
        raise ValueError(f"window_size={window_size} > num_frames={num_frames}")

    return [
        (start, start + window_size)
        for start in range(0, num_frames - window_size + 1, stride)
    ]


def safe_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    if dst.is_symlink() and dst.resolve() == src.resolve():
        return

    if dst.exists():
        return

    os.symlink(str(src.resolve()), str(dst), target_is_directory=src.is_dir())


def prepare_pseudo_nuscenes_root(origin_root: Path, pseudo_root: Path, camera_names: List[str]) -> None:
    (pseudo_root / "samples").mkdir(parents=True, exist_ok=True)
    (pseudo_root / "sweeps").mkdir(parents=True, exist_ok=True)

    safe_symlink(origin_root / "maps", pseudo_root / "maps")

    # Symlink can_bus directory
    safe_symlink(origin_root / "can_bus", pseudo_root / "can_bus")

    # Symlink interp_12Hz_trainval directory
    safe_symlink(origin_root / "interp_12Hz_trainval", pseudo_root / "interp_12Hz_trainval")

    for child in origin_root.iterdir():
        if child.is_dir() and child.name.startswith("v1.0"):
            safe_symlink(child, pseudo_root / child.name)

    for parent_name in ["samples", "sweeps"]:
        src_parent = origin_root / parent_name
        dst_parent = pseudo_root / parent_name
        if not src_parent.exists():
            continue

        for child in src_parent.iterdir():
            dst = dst_parent / child.name
            if child.name in camera_names:
                dst.mkdir(parents=True, exist_ok=True)
            else:
                safe_symlink(child, dst)


def normalize_row_index(row_index: int, grid_rows: int) -> int:
    if row_index < 0:
        row_index += grid_rows
    if row_index < 0 or row_index >= grid_rows:
        raise ValueError(f"Invalid row_index={row_index}, grid_rows={grid_rows}")
    return row_index


def split_frame_to_cams(frame, camera_names: List[str]) -> Dict[str, np.ndarray]:
    height, width = frame.shape[:2]
    if height % GRID_ROWS != 0 or width % GRID_COLS != 0:
        raise ValueError(
            f"Frame shape {(height, width)} cannot be evenly split by {GRID_ROWS}x{GRID_COLS}"
        )

    cell_h = height // GRID_ROWS
    cell_w = width // GRID_COLS
    row_index = normalize_row_index(GENERATED_ROW_INDEX, GRID_ROWS)

    y0 = row_index * cell_h
    y1 = y0 + cell_h

    cams = {}
    for col_idx, cam_name in enumerate(camera_names):
        x0 = col_idx * cell_w
        x1 = x0 + cell_w
        cams[cam_name] = frame[y0:y1, x0:x1]
    return cams


def load_npy_video(npy_path: Path):
    arr = np.load(str(npy_path), mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D npy video, got {arr.shape}")

    if arr.shape[-1] in (1, 3, 4):
        return arr
    if arr.shape[1] in (1, 3, 4):
        return np.transpose(arr, (0, 2, 3, 1))

    raise ValueError(f"Unsupported npy shape: {arr.shape}")


def to_uint8_rgb(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        min_val = float(image.min())
        max_val = float(image.max())
        if 0.0 <= min_val and max_val <= 1.0:
            image = image * 255.0
        elif -1.0 <= min_val and max_val <= 1.0:
            image = (image + 1.0) * 127.5

    return np.clip(image, 0, 255).astype(np.uint8)


def infer_group_name(video_path: Path, video_dir: Path) -> str:
    rel_path = video_path.relative_to(video_dir)
    if len(rel_path.parts) <= 1:
        return "default"
    return rel_path.parts[0]


def group_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["group_name"]].append(record)
    return dict(grouped)


def save_generated_images(
    npy_path: Path,
    pseudo_root: Path,
    infos: List[Dict[str, Any]],
    camera_names: List[str],
    video_label: str,
) -> Dict[str, Any]:
    video = load_npy_video(npy_path)
    if int(video.shape[0]) < len(infos):
        raise ValueError(
            f"{npy_path.name} only has {video.shape[0]} frames, but {len(infos)} infos are required"
        )

    written_paths = set()
    duplicate_count = 0
    resize_check = None

    for frame_idx in tqdm(range(len(infos)), desc=f"export {video_label}", leave=False):
        frame = to_uint8_rgb(video[frame_idx])
        cam_images = split_frame_to_cams(frame, camera_names)
        info = infos[frame_idx]

        for cam_name in camera_names:
            rel_path = info["cams"][cam_name]["data_path"]
            out_path = pseudo_root / rel_path

            if rel_path in written_paths:
                duplicate_count += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)

            src_image = cam_images[cam_name]
            resized_image = cv2.resize(
                src_image,
                (GENERATED_IMAGE_HW[1], GENERATED_IMAGE_HW[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            if resized_image.shape[:2] != GENERATED_IMAGE_HW:
                raise ValueError(
                    f"Resize failed for {video_label}/{cam_name}: got {resized_image.shape[:2]}, "
                    f"expected {GENERATED_IMAGE_HW}"
                )

            bgr_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(out_path), bgr_image)
            if not ok:
                raise ValueError(f"Failed to save image: {out_path}")

            written_paths.add(rel_path)

            if resize_check is None:
                resize_check = {
                    "video_label": video_label,
                    "cam_name": cam_name,
                    "src_hw": tuple(int(x) for x in src_image.shape[:2]),
                    "dst_hw": tuple(int(x) for x in resized_image.shape[:2]),
                    "target_hw": GENERATED_IMAGE_HW,
                }
                print(
                    f"[resize-check] {video_label} {cam_name}: "
                    f"{resize_check['src_hw']} -> {resize_check['dst_hw']}"
                )

    print(
        f"[exported] {video_label}: infos={len(infos)} unique_images={len(written_paths)} "
        f"duplicates_skipped={duplicate_count}"
    )
    return {
        "resize_check": resize_check,
        "num_infos": len(infos),
        "num_unique_images": len(written_paths),
        "duplicates_skipped": duplicate_count,
    }


def build_runtime(cfg: Dict[str, Any], dataset=None) -> Dict[str, Any]:
    input_cfg = cfg["input"]
    video_dir = Path(input_cfg["video_dir"])
    token_dir = Path(input_cfg["token_dir"])
    preview_npy_dir = Path(input_cfg["preview_npy_dir"])
    num_frames = int(input_cfg["num_frames"])

    video_paths = collect_video_paths(video_dir)
    if not video_paths:
        raise FileNotFoundError(f"No mp4 found in: {video_dir}")

    records = []
    for video_path in tqdm(video_paths, desc="prepare videos"):
        video_name = video_path.stem
        rel_path = video_path.relative_to(video_dir)
        group_name = infer_group_name(video_path, video_dir)

        preview_npy_path = resolve_sidecar_path(
            video_path=video_path,
            video_root=video_dir,
            sidecar_root=preview_npy_dir,
            new_suffix=".npy",
        )

        video_meta = get_video_meta(str(video_path))
        fps = float(video_meta["fps"])
        print(f"[build-runtime] {video_name} fps={fps}")
        frame_count = int(video_meta["frame_count"])

        if frame_count < num_frames:
            raise ValueError(
                f"{video_path.name} only has {frame_count} frames, but num_frames={num_frames} is required"
            )

        record = {
            "video_name": video_name,
            "video_path": str(video_path),
            "relative_video_path": str(rel_path),
            "group_name": group_name,
            "preview_npy_path": str(preview_npy_path),
            "fps": fps,
            "frame_count": frame_count,
        }

        if dataset is not None:
            token_path = resolve_sidecar_path(
                video_path=video_path,
                video_root=video_dir,
                sidecar_root=token_dir,
                new_suffix=".txt",
            )
            init_sample_token = read_init_token(token_path)
            record["token_path"] = str(token_path)
            record["init_sample_token"] = init_sample_token
            record["frame_mapping"] = dataset.build_frame_mapping(
                fps=fps,
                num_frames=num_frames,
                init_sample_token=init_sample_token,
            )

        records.append(record)

    grouped = group_records(records)
    group_summary = {name: len(grouped[name]) for name in sorted(grouped)}
    print(f"[runtime] videos={len(records)} groups={group_summary} target_hw={GENERATED_IMAGE_HW}")

    runtime = {
        "records": records,
        "grouped_records": grouped,
        "group_summary": group_summary,
        "num_frames": num_frames,
    }

    if "window" in cfg:
        runtime["windows"] = compute_windows(
            num_frames=num_frames,
            window_size=int(cfg["window"]["size"]),
            stride=int(cfg["window"]["stride"]),
        )

    return runtime


def build_uniad_infos_for_record(
    cfg: Dict[str, Any],
    dataset,
    record: Dict[str, Any],
    task_cfg: Dict[str, Any],
    pseudo_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    uniad_data = build_uniad_info_for_segment(
        dataset=dataset,
        segment_frame_mapping=record["frame_mapping"],
        can_bus_root_path=cfg["dataset"].get("can_bus_root_path"),
        max_sweeps=int(task_cfg.get("max_sweeps", 0)),
        predict_steps=int(task_cfg.get("future_steps", 16)),
        future_step_time=float(task_cfg.get("future_step_time", 0.5)),
        original_token_len=int(task_cfg.get("original_token_len", 32)),
    )
    infos = uniad_data["infos"]
    if len(infos) != len(record["frame_mapping"]):
        raise ValueError(
            f"{record['video_name']}: expected {len(record['frame_mapping'])} infos, got {len(infos)}"
        )

    image_summary = save_generated_images(
        npy_path=Path(record["preview_npy_path"]),
        pseudo_root=pseudo_root,
        infos=infos,
        camera_names=list(dataset.camera_names),
        video_label=record["relative_video_path"],
    )

    seq_infos = []
    for frame_idx, info in enumerate(infos):
        item = dict(info)
        item["video_name"] = record["video_name"]
        item["video_path"] = record["video_path"]
        item["relative_video_path"] = record["relative_video_path"]
        item["group_name"] = record["group_name"]
        item["init_sample_token"] = record["init_sample_token"]
        item["global_frame_idx"] = frame_idx
        item["pseudo_dataroot"] = str(pseudo_root)
        seq_infos.append(item)

    return seq_infos, image_summary


def run_uniad_window_pkls(cfg: Dict[str, Any], dataset) -> Dict[str, Any]:
    runtime = build_runtime(cfg, dataset)
    task_cfg = cfg["task"]["uniad_window_pkls"]

    pseudo_root = Path(task_cfg["pseudo_nuscenes_root"])
    out_dir = Path(task_cfg["output"]["root_dir"]) / task_cfg["output"]["subdir"]
    out_prefix = task_cfg["output"]["prefix"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_root.mkdir(parents=True, exist_ok=True)

    prepare_pseudo_nuscenes_root(
        origin_root=Path(cfg["dataset"]["dataroot"]),
        pseudo_root=pseudo_root,
        camera_names=list(dataset.camera_names),
    )

    per_video_infos = {}
    image_summaries = []
    for record in tqdm(runtime["records"], desc="build uniad infos"):
        seq_infos, image_summary = build_uniad_infos_for_record(
            cfg=cfg,
            dataset=dataset,
            record=record,
            task_cfg=task_cfg,
            pseudo_root=pseudo_root,
        )
        per_video_infos[record["video_name"]] = seq_infos
        image_summaries.append(image_summary)

    saved_files = []
    for start, end in tqdm(runtime["windows"], desc="dump windows"):
        merged_infos = []
        for record in runtime["records"]:
            seq_infos = per_video_infos[record["video_name"]]
            window_infos = seq_infos[start:end]

            for local_idx, info in enumerate(window_infos):
                item = dict(info)
                item["window_start"] = start
                item["window_end"] = end
                item["window_local_idx"] = local_idx
                merged_infos.append(item)

        out_path = out_dir / f"{out_prefix}_window_{start:03d}_{end:03d}.pkl"
        dump_uniad_info(
            {
                "infos": merged_infos,
                "metadata": {
                    "version": dataset.version,
                    "pseudo_dataroot": str(pseudo_root),
                    "group_summary": runtime["group_summary"],
                    "target_hw": GENERATED_IMAGE_HW,
                },
            },
            str(out_path),
        )
        saved_files.append(str(out_path))

    resize_check = next((x["resize_check"] for x in image_summaries if x["resize_check"] is not None), None)
    return {
        "task": "uniad_window_pkls",
        "out_dir": str(out_dir),
        "num_files": len(saved_files),
        "saved_files": saved_files,
        "group_summary": runtime["group_summary"],
        "resize_check": resize_check,
        "target_hw": GENERATED_IMAGE_HW,
    }


def run_uniad_group_pkls(cfg: Dict[str, Any], dataset) -> Dict[str, Any]:
    runtime = build_runtime(cfg, dataset)
    task_cfg = cfg["task"]["uniad_group_pkls"]

    pseudo_root = Path(task_cfg["pseudo_nuscenes_root"])
    out_dir = Path(task_cfg["output"]["root_dir"]) / task_cfg["output"]["subdir"]
    out_prefix = task_cfg["output"].get("prefix", "val")
    expected_groups = task_cfg.get("expected_groups")
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_root.mkdir(parents=True, exist_ok=True)

    prepare_pseudo_nuscenes_root(
        origin_root=Path(cfg["dataset"]["dataroot"]),
        pseudo_root=pseudo_root,
        camera_names=list(dataset.camera_names),
    )

    group_to_infos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    group_to_videos: Dict[str, List[str]] = defaultdict(list)
    image_summaries = []

    for record in tqdm(runtime["records"], desc="build grouped uniad infos"):
        seq_infos, image_summary = build_uniad_infos_for_record(
            cfg=cfg,
            dataset=dataset,
            record=record,
            task_cfg=task_cfg,
            pseudo_root=pseudo_root,
        )
        group_name = record["group_name"]
        group_to_infos[group_name].extend(seq_infos)
        group_to_videos[group_name].append(record["video_name"])
        image_summaries.append(image_summary)

    if expected_groups is not None:
        actual_groups = sorted(group_to_infos.keys())
        if sorted(expected_groups) != actual_groups:
            raise ValueError(
                f"expected_groups={sorted(expected_groups)} does not match actual_groups={actual_groups}"
            )

    saved_files = []
    for group_name in tqdm(sorted(group_to_infos.keys()), desc="dump group pkls"):
        infos = group_to_infos[group_name]
        out_path = out_dir / f"{out_prefix}_{group_name}.pkl"
        dump_uniad_info(
            {
                "infos": infos,
                "metadata": {
                    "version": dataset.version,
                    "pseudo_dataroot": str(pseudo_root),
                    "group_name": group_name,
                    "num_sequences": len(group_to_videos[group_name]),
                    "num_infos": len(infos),
                    "target_hw": GENERATED_IMAGE_HW,
                },
            },
            str(out_path),
        )
        saved_files.append(str(out_path))
        print(
            f"[group-pkl] {group_name}: videos={len(group_to_videos[group_name])} "
            f"infos={len(infos)} -> {out_path.name}"
        )

    resize_check = next((x["resize_check"] for x in image_summaries if x["resize_check"] is not None), None)
    return {
        "task": "uniad_group_pkls",
        "out_dir": str(out_dir),
        "num_files": len(saved_files),
        "saved_files": saved_files,
        "group_summary": runtime["group_summary"],
        "resize_check": resize_check,
        "target_hw": GENERATED_IMAGE_HW,
    }


RUNNERS["run_uniad_window_pkls"] = run_uniad_window_pkls
RUNNERS["run_uniad_group_pkls"] = run_uniad_group_pkls


def run_task(cfg: Dict[str, Any]) -> Dict[str, Any]:
    task_name = cfg["task"]["name"]
    if task_name not in TASKS:
        raise ValueError(f"Unsupported task: {task_name}")

    task_meta = TASKS[task_name]
    runner = RUNNERS[task_meta["runner"]]
    dataset = build_dataset(cfg) if task_meta["needs_dataset"] else None
    return runner(cfg, dataset)

