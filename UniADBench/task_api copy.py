from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import os

import cv2
import numpy as np
from video_utils import get_video_meta
from tasks.uniad_info_builder import build_uniad_info_for_segment, dump_uniad_info

def resolve_preview_npy_path(video_path: Path, cfg: Dict) -> Path:
    input_cfg = cfg["input"]
    preview_npy_dir = input_cfg.get("preview_npy_dir")

    if preview_npy_dir is not None:
        npy_dir = Path(preview_npy_dir)
    elif video_path.parent.name == "preview":
        npy_dir = video_path.parent.parent / "preview_npy"
    else:
        npy_dir = video_path.parent / "preview_npy"

    npy_path = npy_dir / f"{video_path.stem}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Preview npy file not found: {npy_path}")
    return npy_path


def load_npy_video(npy_path: Path):
    arr = np.load(str(npy_path), mmap_mode="r")

    if arr.ndim != 4:
        raise ValueError(f"Expected 4D npy, got shape={arr.shape}")

    if arr.shape[-1] in (1, 3, 4):
        return arr

    if arr.shape[1] in (1, 3, 4):
        return np.transpose(arr, (0, 2, 3, 1))

    raise ValueError(f"Unsupported npy shape={arr.shape}")


def to_uint8_image(image):
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

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_image_to_hw(image, resize_hw: Tuple[int, int] | None):
    if resize_hw is None:
        return image

    target_h, target_w = resize_hw
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image

    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def convert_image_to_bgr_for_imwrite(image, color_order: str):
    color_order = color_order.lower()

    if image.ndim != 3 or image.shape[2] != 3:
        return image

    if color_order == "bgr":
        return image

    if color_order == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unsupported color_order={color_order}")


def prepare_image_for_imwrite(image, resize_hw: Tuple[int, int] | None, color_order: str):
    image = to_uint8_image(image)
    image = resize_image_to_hw(image, resize_hw)
    image = convert_image_to_bgr_for_imwrite(image, color_order)
    return image


def remove_file_if_exists(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def generated_images_ready_by_info(
    pseudo_root: Path,
    infos: List[Dict],
    camera_names: List[str],
    expected_hw: Tuple[int, int] | None = None,
) -> bool:
    for info in infos:
        cams = info.get("cams", {})
        for cam_name in camera_names:
            if cam_name not in cams:
                continue

            rel_path = cams[cam_name]["data_path"]
            abs_path = pseudo_root / rel_path
            if not abs_path.exists():
                return False

            if expected_hw is not None:
                img = cv2.imread(str(abs_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    return False

                expected_h, expected_w = expected_hw
                if img.shape[0] != expected_h or img.shape[1] != expected_w:
                    return False

    return True

def export_generated_images_from_npy_by_info(
    npy_path: Path,
    pseudo_root: Path,
    infos: List[Dict],
    grid_rows: int,
    grid_cols: int,
    generated_row_index: int,
    camera_order: List[str],
    resize_hw: Tuple[int, int] | None = None,
    color_order: str = "rgb",
    force_rewrite: bool = False,
    ) -> None:
    if not force_rewrite and generated_images_ready_by_info(
        pseudo_root=pseudo_root,
        infos=infos,
        camera_names=camera_order,
        expected_hw=resize_hw,
    ):
        print(f"skip image export for {npy_path.stem}, generated images already exist.")
        return

    video_array = load_npy_video(npy_path)
    num_frames = len(infos)

    if int(video_array.shape[0]) < num_frames:
        raise ValueError(
            f"{npy_path.name} only has {video_array.shape[0]} frames, "
            f"but {num_frames} infos are required."
        )

    written_paths = set()
    duplicate_count = 0

    for frame_idx in range(num_frames):
        frame = to_uint8_image(video_array[frame_idx])

        cam_images = split_frame_to_6cams(
            frame=frame,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            row_index=generated_row_index,
            camera_order=camera_order,
        )

        info = infos[frame_idx]
        cams = info.get("cams", {})

        for cam_name in camera_order:
            if cam_name not in cams:
                continue

            rel_path = cams[cam_name]["data_path"]
            out_path = pseudo_root / rel_path

            if rel_path in written_paths:
                duplicate_count += 1
                continue

            ensure_parent_dir(out_path)
            if out_path.parent.is_symlink():
                raise ValueError(
                    f"Refuse to write into symlinked directory: {out_path.parent}\n"
                    f"target file: {out_path}"
                )

            image = prepare_image_for_imwrite(
                image=cam_images[cam_name],
                resize_hw=resize_hw,
                color_order=color_order,
            )

            if force_rewrite:
                remove_file_if_exists(out_path)

            ok_write = cv2.imwrite(str(out_path), image)
            if not ok_write:
                raise ValueError(f"Failed to save image: {out_path}")

            written_paths.add(rel_path)

    print(
        f"exported generated images from npy for {npy_path.stem}: "
        f"frames={num_frames}, unique_paths={len(written_paths)}, duplicates_skipped={duplicate_count}"
    )

def read_init_token(token_txt_path: Path) -> str:
    if not token_txt_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_txt_path}")
    with token_txt_path.open("r", encoding="utf-8") as f:
        token = f.readline().strip()
    if not token:
        raise ValueError(f"Empty token file: {token_txt_path}")
    return token


def collect_video_paths(video_dir: Path, exts: List[str], recursive: bool = False) -> List[Path]:
    exts = {x.lower() for x in exts}
    if recursive:
        files = [p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def compute_windows(num_frames: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if window_size > num_frames:
        raise ValueError(f"window_size={window_size} > num_frames={num_frames}")

    return [
        (start, start + window_size)
        for start in range(0, num_frames - window_size + 1, stride)
    ]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise ValueError(f"Symlink exists but points to a different path: {dst}")

    if dst.exists():
        return

    os.symlink(str(src.resolve()), str(dst), target_is_directory=src.is_dir())


def resolve_uniad_window_output_dir(cfg: Dict) -> Tuple[Path, str]:
    export_cfg = cfg["export"]["uniad_window_pkls"]["output"]
    root_dir = Path(export_cfg["root_dir"])
    subdir = export_cfg.get("subdir", "uniad_window_pkls")
    prefix = export_cfg.get("prefix", "val")

    out_dir = root_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, prefix


def resolve_pseudo_nuscenes_root(cfg: Dict) -> Path:
    pseudo_root = cfg["export"].get("pseudo_nuscenes_root")
    if pseudo_root is None:
        raise ValueError("Missing export.pseudo_nuscenes_root in cfg")

    path = Path(pseudo_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_pseudo_nuscenes_root(
    origin_nusc_root: Path,
    pseudo_root: Path,
    camera_names: List[str],
) -> None:
    ensure_dir(pseudo_root)
    ensure_dir(pseudo_root / "samples")
    ensure_dir(pseudo_root / "sweeps")

    safe_symlink(origin_nusc_root / "maps", pseudo_root / "maps")

    for child in origin_nusc_root.iterdir():
        if child.is_dir() and child.name.startswith("v1.0"):
            safe_symlink(child, pseudo_root / child.name)

    origin_samples = origin_nusc_root / "samples"
    if origin_samples.exists():
        for child in origin_samples.iterdir():
            if not child.is_dir():
                continue
            dst = pseudo_root / "samples" / child.name
            if child.name in camera_names:
                ensure_dir(dst)
            else:
                safe_symlink(child, dst)

    origin_sweeps = origin_nusc_root / "sweeps"
    if origin_sweeps.exists():
        for child in origin_sweeps.iterdir():
            if not child.is_dir():
                continue
            dst = pseudo_root / "sweeps" / child.name
            if child.name in camera_names:
                ensure_dir(dst)
            else:
                safe_symlink(child, dst)


def normalize_row_index(row_index: int, grid_rows: int) -> int:
    if row_index < 0:
        row_index = grid_rows + row_index
    if row_index < 0 or row_index >= grid_rows:
        raise ValueError(f"Invalid row_index={row_index}, grid_rows={grid_rows}")
    return row_index


def split_frame_to_6cams(
    frame,
    grid_rows: int,
    grid_cols: int,
    row_index: int,
    camera_order: List[str],
) -> Dict[str, object]:
    height, width = frame.shape[:2]

    if height % grid_rows != 0 or width % grid_cols != 0:
        raise ValueError(
            f"Frame shape {(height, width)} cannot be split by grid_rows={grid_rows}, grid_cols={grid_cols}"
        )

    if len(camera_order) > grid_cols:
        raise ValueError(f"len(camera_order)={len(camera_order)} > grid_cols={grid_cols}")

    row_index = normalize_row_index(row_index, grid_rows)

    cell_h = height // grid_rows
    cell_w = width // grid_cols
    y0 = row_index * cell_h
    y1 = (row_index + 1) * cell_h

    result = {}
    for col_idx, cam_name in enumerate(camera_order):
        x0 = col_idx * cell_w
        x1 = (col_idx + 1) * cell_w
        result[cam_name] = frame[y0:y1, x0:x1]

    return result


def generated_images_ready_by_info(
    pseudo_root: Path,
    infos: List[Dict],
    camera_names: List[str],
) -> bool:
    for info in infos:
        cams = info.get("cams", {})
        for cam_name in camera_names:
            if cam_name not in cams:
                continue
            rel_path = cams[cam_name]["data_path"]
            abs_path = pseudo_root / rel_path
            if not abs_path.exists():
                return False
    return True


def build_unique_camera_rel_paths(
    infos: List[Dict],
    camera_names: List[str],
) -> List[str]:
    unique_paths = []
    seen = set()

    for info in infos:
        cams = info.get("cams", {})
        for cam_name in camera_names:
            if cam_name not in cams:
                continue
            rel_path = cams[cam_name]["data_path"]
            if rel_path in seen:
                continue
            seen.add(rel_path)
            unique_paths.append(rel_path)

    return unique_paths


def export_generated_images_from_video_by_info(
    video_path: Path,
    pseudo_root: Path,
    infos: List[Dict],
    grid_rows: int,
    grid_cols: int,
    generated_row_index: int,
    camera_order: List[str],
) -> None:
    if generated_images_ready_by_info(
        pseudo_root=pseudo_root,
        infos=infos,
        camera_names=camera_order,
    ):
        print(f"skip image export for {video_path.stem}, generated images already exist.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    num_frames = len(infos)
    written_paths = set()
    duplicate_count = 0
    frame_idx = 0

    while frame_idx < num_frames:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(
                f"{video_path.name} ended early while exporting images, "
                f"expected at least {num_frames} frames, got {frame_idx}"
            )

        cam_images = split_frame_to_6cams(
            frame=frame,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            row_index=generated_row_index,
            camera_order=camera_order,
        )

        info = infos[frame_idx]
        cams = info.get("cams", {})

        for cam_name in camera_order:
            if cam_name not in cams:
                continue

            rel_path = cams[cam_name]["data_path"]
            out_path = pseudo_root / rel_path

            if rel_path in written_paths:
                duplicate_count += 1
                continue

            ensure_parent_dir(out_path)
            if out_path.parent.is_symlink():
                cap.release()
                raise ValueError(
                    f"Refuse to write into symlinked directory: {out_path.parent}\n"
                    f"target file: {out_path}"
                )

            ok_write = cv2.imwrite(str(out_path), cam_images[cam_name])
            if not ok_write:
                cap.release()
                raise ValueError(f"Failed to save image: {out_path}")

            written_paths.add(rel_path)

        frame_idx += 1

    cap.release()
    print(
        f"exported generated images for {video_path.stem}: "
        f"frames={num_frames}, unique_paths={len(written_paths)}, duplicates_skipped={duplicate_count}"
    )

def build_common_runtime(cfg: Dict, dataset) -> Dict:
    video_dir = Path(cfg["input"]["video_dir"])
    token_dir = Path(cfg["input"]["token_dir"])
    video_exts = cfg["input"].get("video_exts", [".mp4"])
    recursive = bool(cfg["input"].get("recursive", False))
    num_frames = int(cfg["input"].get("num_frames", 100))
    use_preview_npy = bool(cfg["input"].get("use_preview_npy", True))

    video_paths = collect_video_paths(video_dir, video_exts, recursive=recursive)
    if not video_paths:
        raise ValueError(f"No videos found in {video_dir}")

    per_video_records = {}
    video_order = []

    for idx, video_path in enumerate(video_paths, start=1):
        video_name = video_path.stem
        token_txt_path = token_dir / f"{video_name}.txt"

        init_sample_token = read_init_token(token_txt_path)
        video_meta = get_video_meta(str(video_path))

        if int(video_meta["frame_count"]) < num_frames:
            raise ValueError(
                f"{video_path.name} only has {video_meta['frame_count']} frames, "
                f"but num_frames={num_frames} is required."
            )

        mapping_result = dataset.build_frame_mapping(
            fps=video_meta["fps"],
            num_frames=num_frames,
            init_sample_token=init_sample_token,
        )

        record = {
            "video_name": video_name,
            "video_path": str(video_path),
            "video_meta": video_meta,
            "init_sample_token": init_sample_token,
            "frame_mapping": mapping_result["frame_mapping"],
        }

        if use_preview_npy:
            preview_npy_path = resolve_preview_npy_path(video_path=video_path, cfg=cfg)
            preview_npy_video = load_npy_video(preview_npy_path)

            if int(preview_npy_video.shape[0]) < num_frames:
                raise ValueError(
                    f"{preview_npy_path.name} only has {preview_npy_video.shape[0]} frames, "
                    f"but num_frames={num_frames} is required."
                )

            record["preview_npy_path"] = str(preview_npy_path)

        per_video_records[video_name] = record
        video_order.append(video_name)

        print(f"[{idx}/{len(video_paths)}] prepared runtime for {video_name}")

    return {
        "video_dir": str(video_dir),
        "token_dir": str(token_dir),
        "num_frames": num_frames,
        "video_order": video_order,
        "per_video_records": per_video_records,
    }
    
def run_export_uniad_window_pkls(cfg: Dict, dataset) -> Dict:
    runtime = build_common_runtime(cfg=cfg, dataset=dataset)

    uniad_cfg = cfg["export"]["uniad_window_pkls"]
    out_dir, out_prefix = resolve_uniad_window_output_dir(cfg)
    pseudo_root = resolve_pseudo_nuscenes_root(cfg)

    origin_nusc_root = Path(cfg["dataset"]["dataroot"])
    prepare_pseudo_nuscenes_root(
        origin_nusc_root=origin_nusc_root,
        pseudo_root=pseudo_root,
        camera_names=list(dataset.camera_names),
    )

    grid_rows = int(cfg["input"].get("grid_rows", 4))
    grid_cols = int(cfg["input"].get("grid_cols", 6))
    generated_row_index = int(cfg["input"].get("generated_row_index", -1))
    generated_camera_order = cfg["input"].get("generated_camera_order", list(dataset.camera_names))

    use_preview_npy = bool(cfg["input"].get("use_preview_npy", True))
    generated_resize_hw = tuple(cfg["input"].get("generated_resize_hw", [900, 1600]))
    generated_resize_hw = (int(generated_resize_hw[0]), int(generated_resize_hw[1]))
    generated_color_order = str(cfg["input"].get("generated_color_order", "rgb")).lower()
    force_rewrite_generated_images = bool(
        cfg["input"].get("force_rewrite_generated_images", False)
    )

    if set(generated_camera_order) != set(dataset.camera_names):
        raise ValueError(
            f"generated_camera_order must match dataset.camera_names.\n"
            f"generated_camera_order={generated_camera_order}\n"
            f"dataset.camera_names={dataset.camera_names}"
        )

    num_frames = int(runtime["num_frames"])
    window_size = int(cfg["window"]["size"])
    stride = int(cfg["window"]["stride"])
    windows = compute_windows(num_frames=num_frames, window_size=window_size, stride=stride)

    per_video_infos: Dict[str, List[Dict]] = {}
    video_order = runtime["video_order"]

    for idx, video_name in enumerate(video_order, start=1):
        record = runtime["per_video_records"][video_name]
        video_path = Path(record["video_path"])

        segment_result = dataset.get_segment_tokens(
            frame_mapping=record["frame_mapping"],
            start_frame=0,
            end_frame=num_frames - 1,
            deduplicate=False,
        )

        uniad_info = build_uniad_info_for_segment(
            dataset=dataset,
            segment_result=segment_result,
            can_bus_root_path=cfg["dataset"].get("can_bus_root_path"),
            max_sweeps=int(uniad_cfg.get("max_sweeps", 0)),
            predict_steps=int(uniad_cfg.get("future_steps", 16)),
            future_step_time=float(uniad_cfg.get("future_step_time", 0.5)),
            original_token_len=int(uniad_cfg.get("original_token_len", 32)),
        )

        infos = uniad_info["infos"]
        if len(infos) != num_frames:
            raise ValueError(f"{video_name}: expected {num_frames} infos, got {len(infos)}")

        if use_preview_npy:
            export_generated_images_from_npy_by_info(
                npy_path=Path(record["preview_npy_path"]),
                pseudo_root=pseudo_root,
                infos=infos,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                generated_row_index=generated_row_index,
                camera_order=generated_camera_order,
                resize_hw=generated_resize_hw,
                color_order=generated_color_order,
                force_rewrite=force_rewrite_generated_images,
            )
        else:
            export_generated_images_from_video_by_info(
                video_path=video_path,
                pseudo_root=pseudo_root,
                infos=infos,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                generated_row_index=generated_row_index,
                camera_order=generated_camera_order,
            )

        processed_infos = []
        for frame_idx, info in enumerate(infos):
            item = dict(info)
            item["video_name"] = video_name
            item["video_path"] = str(video_path)
            item["init_sample_token"] = record["init_sample_token"]
            item["global_frame_idx"] = frame_idx
            item["pseudo_dataroot"] = str(pseudo_root)
            processed_infos.append(item)

        per_video_infos[video_name] = processed_infos
        print(f"[{idx}/{len(video_order)}] built infos for {video_name}: {len(processed_infos)} frames")
        

EXPORTER_REGISTRY = {
    "uniad_window_pkls": run_export_uniad_window_pkls,
}


def run_task(cfg: Dict, dataset) -> Dict:
    export_cfg = cfg["export"]
    target = export_cfg.get("target")
    if target is None:
        raise ValueError("Missing export.target in cfg")

    runner = EXPORTER_REGISTRY.get(target)
    if runner is None:
        raise ValueError(
            f"Unsupported export.target: {target}. "
            f"Available targets: {list(EXPORTER_REGISTRY.keys())}"
        )

    return runner(cfg=cfg, dataset=dataset)