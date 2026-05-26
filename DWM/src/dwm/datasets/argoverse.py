import bisect
import json
from collections import OrderedDict

import dwm.common
import dwm.datasets.common
import numpy as np
from PIL import Image, ImageDraw
import pyarrow.feather
import torch


class MotionDataset(torch.utils.data.Dataset):
    """Argoverse 2 motion dataset loaded from an fs root and index JSON.

    Expected fs root:
        <sensor_root>/<split>/<scene_id>/...

    Expected index JSON content:
        - one scene per json: {"scene_name": "...", "files": [...]} or
        - list of scene dicts, or
        - {"entries": [...]}.

    The dataset builds temporal segments from index JSON paths only.
    Actual image / lidar / metadata bytes are read lazily in __getitem__.
    """

    point_keys = ["x", "y", "z"]
    shape_keys = ["length_m", "width_m", "height_m"]
    rotation_keys = ["qw", "qx", "qy", "qz"]
    translation_keys = ["tx_m", "ty_m", "tz_m"]
    intrinsic_focal_keys = ["fx_px", "fy_px"]
    intrinsic_center_keys = ["cx_px", "cy_px"]
    intrinsic_size_keys = ["width_px", "height_px"]

    default_3dbox_color_table = {
        "BICYCLIST": (255, 0, 0),
        "MOTORCYCLIST": (255, 0, 0),
        "PEDESTRIAN": (255, 0, 0),
        "BICYCLE": (128, 255, 0),
        "MOTORCYCLE": (0, 255, 128),
        "BOX_TRUCK": (255, 255, 0),
        "BUS": (128, 0, 255),
        "LARGE_VEHICLE": (0, 0, 255),
        "REGULAR_VEHICLE": (0, 0, 255),
        "SCHOOL_BUS": (128, 0, 255),
        "TRUCK": (255, 255, 0),
        "TRUCK_CAB": (255, 255, 0),
        "VEHICULAR_TRAILER": (255, 255, 255),
    }
    default_hdmap_color_table = {
        "drivable_areas": (0, 0, 255),
        "lane_segments": (0, 255, 0),
        "pedestrian_crossings": (255, 0, 0),
    }
    default_3dbox_corner_template = [
        [-0.5, -0.5, -0.5, 1], [-0.5, -0.5, 0.5, 1],
        [-0.5, 0.5, -0.5, 1], [-0.5, 0.5, 0.5, 1],
        [0.5, -0.5, -0.5, 1], [0.5, -0.5, 0.5, 1],
        [0.5, 0.5, -0.5, 1], [0.5, 0.5, 0.5, 1],
    ]
    default_3dbox_edge_indices = [
        (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        (6, 3), (6, 5),
    ]
    default_bev_from_ego_transform = [
        [6.4, 0, 0, 320],
        [0, -6.4, 0, 320],
        [0, 0, -6.4, 0],
        [0, 0, 0, 1],
    ]
    default_bev_3dbox_corner_template = [
        [-0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1],
        [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1],
    ]
    default_bev_3dbox_edge_indices = [(0, 2), (2, 3), (3, 1), (1, 0)]

    @staticmethod
    def normalize_relative_path(path: str):
        if not isinstance(path, str):
            return path

        normalized = path.replace("\\", "/").lstrip("./")
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        if normalized.startswith("Sensor/"):
            normalized = normalized[len("Sensor/"):]
        if normalized.startswith("sensor/"):
            normalized = normalized[len("sensor/"):]
        return normalized

    @staticmethod
    def list_json_paths(fs, json_path: str):
        if fs.isfile(json_path):
            return [json_path]

        if fs.isdir(json_path):
            items = fs.ls(json_path, detail=True)
            result = []
            for item in items:
                if item["type"] == "file" and item["name"].endswith(".json"):
                    result.append(item["name"])
            result.sort()
            return result

        raise FileNotFoundError(json_path)

    @staticmethod
    def load_json(fs, json_path: str):
        with fs.open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_scene_entries(fs, json_path: str):
        payload = MotionDataset.load_json(fs, json_path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "entries" in payload and isinstance(payload["entries"], list):
            return payload["entries"]
        if isinstance(payload, dict) and "files" in payload:
            return [payload]
        raise ValueError("Unsupported index JSON format: {}".format(json_path))

    @staticmethod
    def parse_sensor_file_path(rel_path: str):
        normalized = MotionDataset.normalize_relative_path(rel_path)
        parts = normalized.split("/")
        if len(parts) < 5:
            return None
        if parts[2] != "sensors":
            return None

        split = parts[0]
        scene_id = parts[1]

        if parts[3] == "cameras":
            if len(parts) < 6:
                return None
            sensor = "cameras/{}".format(parts[4])
            filename = parts[5]
            timestamp_text = filename.split(".")[0]
            if not timestamp_text.isdigit():
                return None
            return {
                "split": split,
                "scene_id": scene_id,
                "sensor": sensor,
                "timestamp": int(timestamp_text),
                "path": normalized,
            }

        if parts[3] == "lidar":
            filename = parts[4]
            timestamp_text = filename.split(".")[0]
            if not timestamp_text.isdigit():
                return None
            return {
                "split": split,
                "scene_id": scene_id,
                "sensor": "lidar",
                "timestamp": int(timestamp_text),
                "path": normalized,
            }

        return None

    @staticmethod
    def parse_map_file_path(rel_path: str):
        normalized = MotionDataset.normalize_relative_path(rel_path)
        parts = normalized.split("/")
        if len(parts) != 4:
            return None
        if parts[2] != "map":
            return None
        if not parts[3].startswith("log_map_archive_") or not parts[3].endswith(".json"):
            return None
        return {
            "split": parts[0],
            "scene_id": parts[1],
            "path": normalized,
        }

    @staticmethod
    def enumerate_begin_times(first_timestamp_ns: int, last_timestamp_ns: int, sequence_duration_s: float, stride_s: float):
        end_time_s = last_timestamp_ns / 1000000000 - sequence_duration_s
        current_time_s = first_timestamp_ns / 1000000000
        while current_time_s <= end_time_s:
            yield current_time_s
            current_time_s += stride_s

    @staticmethod
    def enumerate_segments(channel_sample_data_list: list, sequence_length: int, fps, stride, enable_synchronization_check: bool):
        csdl = [sample_list for sample_list in channel_sample_data_list if len(sample_list) > 0]
        if len(csdl) == 0:
            return

        reference_channel = csdl[0]
        channel_timestamp_list = [[item["timestamp"] for item in sample_list] for sample_list in csdl]

        if fps == 0:
            for t in range(0, len(reference_channel), max(1, stride)):
                ct0 = [
                    dwm.datasets.common.find_nearest(timestamps, reference_channel[t]["timestamp"])
                    for timestamps in channel_timestamp_list
                ]
                if all(t0 + sequence_length <= len(sample_list) for t0, sample_list in zip(ct0, csdl)):
                    yield [
                        [sample_list[t0 + offset] for t0, sample_list in zip(ct0, csdl)]
                        for offset in range(sequence_length)
                    ]
            return

        sequence_duration_s = sequence_length / fps
        begin_times = MotionDataset.enumerate_begin_times(
            reference_channel[0]["timestamp"],
            reference_channel[-1]["timestamp"],
            sequence_duration_s,
            stride,
        )
        for begin_time_s in begin_times:
            begin_timestamp_ns = begin_time_s * 1000000000
            channel_expected_times = [
                [begin_timestamp_ns + frame_idx / fps * 1000000000 for frame_idx in range(sequence_length)]
                for _ in csdl
            ]
            channel_candidates = [
                [
                    sample_list[dwm.datasets.common.find_nearest(timestamps, expected_timestamp)]
                    for expected_timestamp in expected_times
                ]
                for sample_list, timestamps, expected_times in zip(csdl, channel_timestamp_list, channel_expected_times)
            ]
            max_time_error = max(
                abs(candidate["timestamp"] - expected_timestamp)
                for candidates, expected_times in zip(channel_candidates, channel_expected_times)
                for candidate, expected_timestamp in zip(candidates, expected_times)
            )
            if (not enable_synchronization_check) or max_time_error <= 500000000 / fps:
                yield [
                    [candidates[frame_idx] for candidates in channel_candidates]
                    for frame_idx in range(sequence_length)
                ]

    @staticmethod
    def feather_query(feather_dict: dict, key_column: str, queried_key, queried_columns: list):
        keys = feather_dict[key_column]
        index = bisect.bisect_left(keys, queried_key)
        if index < 0 or index >= len(keys) or keys[index] != queried_key:
            raise Exception("The key {} is not found".format(queried_key))
        return [feather_dict[column][index] for column in queried_columns]

    @staticmethod
    def get_transform(pose_dict: dict, key_column: str, queried_key, output_type: str = "np"):
        keys = pose_dict[key_column]
        index = bisect.bisect_left(keys, queried_key)
        if index < 0 or index >= len(keys) or keys[index] != queried_key:
            raise Exception("The key {} is not found".format(queried_key))
        return dwm.datasets.common.get_transform(
            [pose_dict[key][index] for key in MotionDataset.rotation_keys],
            [pose_dict[key][index] for key in MotionDataset.translation_keys],
            output_type,
        )

    @staticmethod
    def get_image_description(image_descriptions: dict, time_list_dict: dict, split: str, scene_id: str, sample_data: dict):
        scene_camera = "sensor/{}/{}|{}".format(split, scene_id, sample_data["sensor"])
        time_list = time_list_dict[scene_camera]
        nearest_time = dwm.datasets.common.find_nearest(time_list, sample_data["timestamp"], return_item=True)
        return image_descriptions["{}|{}".format(scene_camera, nearest_time)]

    @staticmethod
    def list_annotation_indices(annotations: dict, ref_timestamp: int):
        start_index = bisect.bisect_left(annotations["timestamp_ns"], ref_timestamp)
        end_index = bisect.bisect_right(annotations["timestamp_ns"], ref_timestamp)
        return range(start_index, end_index)

    @staticmethod
    def get_3dbox_world_transform(annotations: dict, index: int):
        scale = np.diag([annotations[key][index] for key in MotionDataset.shape_keys] + [1])
        ref_from_annotation = dwm.datasets.common.get_transform(
            [annotations[key][index] for key in MotionDataset.rotation_keys],
            [annotations[key][index] for key in MotionDataset.translation_keys],
        )
        return ref_from_annotation @ scale

    @staticmethod
    def get_nearest_annotation_timestamp(annotation_timestamps: list, target_timestamp: int):
        index = bisect.bisect_left(annotation_timestamps, target_timestamp)
        if index == 0:
            return annotation_timestamps[0]
        if index == len(annotation_timestamps):
            return annotation_timestamps[-1]
        prev_delta = target_timestamp - annotation_timestamps[index - 1]
        next_delta = annotation_timestamps[index] - target_timestamp
        if prev_delta < next_delta:
            return annotation_timestamps[index - 1]
        return annotation_timestamps[index]

    @staticmethod
    def make_empty_camera_image(intrinsics: dict, sensor_name: str):
        sensor_index = bisect.bisect_left(intrinsics["sensor_name"], sensor_name)
        width_px = intrinsics["width_px"][sensor_index]
        height_px = intrinsics["height_px"][sensor_index]
        return Image.new("RGB", (width_px, height_px))

    @staticmethod
    def get_3dbox_image(annotations: dict, ref_timestamp: int, extrinsics: dict, intrinsics: dict, poses: dict, sample_data: dict, _3dbox_image_settings: dict):
        pen_width = _3dbox_image_settings.get("pen_width", 10)
        color_table = _3dbox_image_settings.get("color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_image_settings.get("corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get("edge_indices", MotionDataset.default_3dbox_edge_indices)

        sensor_name = sample_data["sensor"][8:]
        intrinsic_index = bisect.bisect_left(intrinsics["sensor_name"], sensor_name)
        image_size = tuple(intrinsics[key][intrinsic_index] for key in MotionDataset.intrinsic_size_keys)

        intrinsic = np.eye(4)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            [intrinsics[key][intrinsic_index] for key in MotionDataset.intrinsic_focal_keys],
            [intrinsics[key][intrinsic_index] for key in MotionDataset.intrinsic_center_keys],
        )

        ego_from_camera = MotionDataset.get_transform(extrinsics, "sensor_name", sensor_name)
        world_from_ref = MotionDataset.get_transform(poses, "timestamp_ns", ref_timestamp)
        world_from_ego = MotionDataset.get_transform(poses, "timestamp_ns", sample_data["timestamp"])
        camera_from_ref = np.linalg.solve(world_from_ego @ ego_from_camera, world_from_ref)
        image_from_ref = intrinsic @ camera_from_ref

        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)
        corner_templates_np = np.array(corner_templates).transpose()

        for annotation_index in MotionDataset.list_annotation_indices(annotations, ref_timestamp):
            category = annotations["category"][annotation_index]
            if category not in color_table:
                continue
            pen_color = tuple(color_table[category])
            world_transform = MotionDataset.get_3dbox_world_transform(annotations, annotation_index)
            projected = image_from_ref @ world_transform @ corner_templates_np
            for corner_a, corner_b in edge_indices:
                xy = dwm.datasets.common.project_line(projected[:, corner_a], projected[:, corner_b])
                if xy is not None:
                    draw.line(xy, fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def get_hdmap_image(map_dict: dict, extrinsics: dict, intrinsics: dict, poses: dict, sample_data: dict, hdmap_image_settings: dict):
        max_distance = hdmap_image_settings.get("max_distance", 65.0)
        pen_width = hdmap_image_settings.get("pen_width", 10)
        color_table = hdmap_image_settings.get("color_table", MotionDataset.default_hdmap_color_table)

        sensor_name = sample_data["sensor"][8:]
        intrinsic_index = bisect.bisect_left(intrinsics["sensor_name"], sensor_name)
        image_size = tuple(intrinsics[key][intrinsic_index] for key in MotionDataset.intrinsic_size_keys)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            [intrinsics[key][intrinsic_index] for key in MotionDataset.intrinsic_focal_keys],
            [intrinsics[key][intrinsic_index] for key in MotionDataset.intrinsic_center_keys],
        )

        ego_from_camera = MotionDataset.get_transform(extrinsics, "sensor_name", sensor_name)
        world_from_ego = MotionDataset.get_transform(poses, "timestamp_ns", sample_data["timestamp"])
        camera_from_world = np.linalg.inv(world_from_ego @ ego_from_camera)
        image_from_world = intrinsic @ camera_from_world

        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)

        if "lane_segments" in color_table and "lane_segments" in map_dict:
            pen_color = tuple(color_table["lane_segments"])
            for lane_segment in map_dict["lane_segments"].values():
                if lane_segment["is_intersection"]:
                    continue
                left_nodes = np.array([
                    [node[key] for key in MotionDataset.point_keys] + [1]
                    for node in lane_segment["left_lane_boundary"]
                ]).transpose()
                projected_left = image_from_world @ left_nodes
                for index in range(1, projected_left.shape[1]):
                    xy = dwm.datasets.common.project_line(projected_left[:, index - 1], projected_left[:, index], far_z=max_distance)
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

                right_nodes = np.array([
                    [node[key] for key in MotionDataset.point_keys] + [1]
                    for node in lane_segment["right_lane_boundary"]
                ]).transpose()
                projected_right = image_from_world @ right_nodes
                for index in range(1, projected_right.shape[1]):
                    xy = dwm.datasets.common.project_line(projected_right[:, index - 1], projected_right[:, index], far_z=max_distance)
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

        if "drivable_areas" in color_table and "drivable_areas" in map_dict:
            pen_color = tuple(color_table["drivable_areas"])
            for drivable_area in map_dict["drivable_areas"].values():
                polygon_nodes = np.array([
                    [node[key] for key in MotionDataset.point_keys] + [1]
                    for node in drivable_area["area_boundary"]
                ]).transpose()
                projected_polygon = image_from_world @ polygon_nodes
                vertex_count = projected_polygon.shape[1]
                for index in range(vertex_count):
                    xy = dwm.datasets.common.project_line(
                        projected_polygon[:, index],
                        projected_polygon[:, (index + 1) % vertex_count],
                        far_z=max_distance,
                    )
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

        if "pedestrian_crossings" in color_table and "pedestrian_crossings" in map_dict:
            pen_color = tuple(color_table["pedestrian_crossings"])
            for crossing in map_dict["pedestrian_crossings"].values():
                edge1 = crossing["edge1"]
                edge2 = crossing["edge2"]
                polygon_nodes = np.array([
                    [edge1[0][key] for key in MotionDataset.point_keys] + [1],
                    [edge1[1][key] for key in MotionDataset.point_keys] + [1],
                    [edge2[1][key] for key in MotionDataset.point_keys] + [1],
                    [edge2[0][key] for key in MotionDataset.point_keys] + [1],
                ]).transpose()
                projected_polygon = image_from_world @ polygon_nodes
                vertex_count = projected_polygon.shape[1]
                for index in range(vertex_count):
                    xy = dwm.datasets.common.project_line(
                        projected_polygon[:, index],
                        projected_polygon[:, (index + 1) % vertex_count],
                        far_z=max_distance,
                    )
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def get_3dbox_bev_image(annotations: dict, sample_data: dict, _3dbox_bev_settings: dict):
        pen_width = _3dbox_bev_settings.get("pen_width", 2)
        bev_size = _3dbox_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = _3dbox_bev_settings.get("bev_from_ego_transform", MotionDataset.default_bev_from_ego_transform)
        fill_box = _3dbox_bev_settings.get("fill_box", False)
        color_table = _3dbox_bev_settings.get("color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_bev_settings.get("corner_templates", MotionDataset.default_bev_3dbox_corner_template)
        edge_indices = _3dbox_bev_settings.get("edge_indices", MotionDataset.default_bev_3dbox_edge_indices)

        bev_from_ego = np.array(bev_from_ego_transform)
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)
        corner_templates_np = np.array(corner_templates).transpose()

        timestamp = sample_data["timestamp"]
        start_index = bisect.bisect_left(annotations["timestamp_ns"], timestamp)
        end_index = bisect.bisect_right(annotations["timestamp_ns"], timestamp)
        for annotation_index in range(start_index, end_index):
            category = annotations["category"][annotation_index]
            if category not in color_table:
                continue
            pen_color = tuple(color_table[category])
            scale = np.diag([annotations[key][annotation_index] for key in MotionDataset.shape_keys] + [1])
            ego_from_annotation = dwm.datasets.common.get_transform(
                [annotations[key][annotation_index] for key in MotionDataset.rotation_keys],
                [annotations[key][annotation_index] for key in MotionDataset.translation_keys],
            )
            projected = bev_from_ego @ ego_from_annotation @ scale @ corner_templates_np
            if fill_box:
                draw.polygon([(projected[0, a], projected[1, a]) for a, _ in edge_indices], fill=pen_color, width=pen_width)
            else:
                for a, b in edge_indices:
                    draw.line((projected[0, a], projected[1, a], projected[0, b], projected[1, b]), fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def get_hdmap_bev_image(map_dict: dict, poses: dict, sample_data: dict, hdmap_bev_settings: dict):
        pen_width = hdmap_bev_settings.get("pen_width", 2)
        bev_size = hdmap_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = hdmap_bev_settings.get("bev_from_ego_transform", MotionDataset.default_bev_from_ego_transform)
        color_table = hdmap_bev_settings.get("color_table", MotionDataset.default_hdmap_color_table)

        world_from_ego = MotionDataset.get_transform(poses, "timestamp_ns", sample_data["timestamp"])
        ego_from_world = np.linalg.inv(world_from_ego)
        bev_from_ego = np.array(bev_from_ego_transform)
        bev_from_world = bev_from_ego @ ego_from_world

        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        if "drivable_areas" in color_table and "drivable_areas" in map_dict:
            pen_color = tuple(color_table["drivable_areas"])
            for drivable_area in map_dict["drivable_areas"].values():
                polygon_nodes = np.array([
                    [node[key] for key in MotionDataset.point_keys] + [1]
                    for node in drivable_area["area_boundary"]
                ]).transpose()
                projected_polygon = bev_from_world @ polygon_nodes
                draw.polygon(
                    [(projected_polygon[0, index], projected_polygon[1, index]) for index in range(projected_polygon.shape[1])],
                    fill=pen_color,
                    width=pen_width,
                )

        if "pedestrian_crossings" in color_table and "pedestrian_crossings" in map_dict:
            pen_color = tuple(color_table["pedestrian_crossings"])
            for crossing in map_dict["pedestrian_crossings"].values():
                edge1 = crossing["edge1"]
                edge2 = crossing["edge2"]
                polygon_nodes = np.array([
                    [edge1[0][key] for key in MotionDataset.point_keys] + [1],
                    [edge1[1][key] for key in MotionDataset.point_keys] + [1],
                    [edge2[1][key] for key in MotionDataset.point_keys] + [1],
                    [edge2[0][key] for key in MotionDataset.point_keys] + [1],
                ]).transpose()
                projected_polygon = bev_from_world @ polygon_nodes
                draw.polygon(
                    [(projected_polygon[0, index], projected_polygon[1, index]) for index in range(projected_polygon.shape[1])],
                    fill=pen_color,
                    width=pen_width,
                )

        if "lane_segments" in color_table and "lane_segments" in map_dict:
            pen_color = tuple(color_table["lane_segments"])
            for lane_segment in map_dict["lane_segments"].values():
                if lane_segment["is_intersection"]:
                    continue
                left_nodes = np.array([
                    [node[key] for key in MotionDataset.point_keys] + [1]
                    for node in lane_segment["left_lane_boundary"]
                ]).transpose()
                projected_left = bev_from_world @ left_nodes
                for index in range(1, projected_left.shape[1]):
                    draw.line(
                        (projected_left[0, index - 1], projected_left[1, index - 1], projected_left[0, index], projected_left[1, index]),
                        fill=pen_color,
                        width=pen_width,
                    )

                right_nodes = np.array([
                    [node[key] for key in MotionDataset.point_keys] + [1]
                    for node in lane_segment["right_lane_boundary"]
                ]).transpose()
                projected_right = bev_from_world @ right_nodes
                for index in range(1, projected_right.shape[1]):
                    draw.line(
                        (projected_right[0, index - 1], projected_right[1, index - 1], projected_right[0, index], projected_right[1, index]),
                        fill=pen_color,
                        width=pen_width,
                    )

        return image

    def __init__(
        self,
        fs,
        sequence_length: int,
        fps_stride_tuples: list,
        sensor_channels: list = ["cameras/ring_front_left"],
        hide_lidar: bool = False,
        enable_synchronization_check: bool = True,
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False,
        _3dbox_image_settings=None,
        hdmap_image_settings=None,
        _3dbox_bev_settings=None,
        hdmap_bev_settings=None,
        image_description_settings=None,
        stub_key_data_dict=None,
        index_json_path: str = None,
        split: str = "train",
        scene_cache_size: int = 8,
    ):
        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.sensor_channels = sensor_channels
        self.hide_lidar = hide_lidar
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.stub_key_data_dict = {} if stub_key_data_dict is None else stub_key_data_dict
        self.index_json_path = index_json_path
        self.split = split
        self.scene_cache_size = max(1, scene_cache_size)
        self.scene_cache = OrderedDict()

        if self.index_json_path is None:
            raise RuntimeError("index_json_path is required")

        filename_dict = {}
        scene_channel_sample_data = {}
        scene_map_dict = {}
        scene_split_dict = {}
        sensor_channel_set = set(sensor_channels)

        json_paths = MotionDataset.list_json_paths(self.fs, self.index_json_path)
        for json_path in json_paths:
            scene_entries = MotionDataset.load_scene_entries(self.fs, json_path)
            for scene_entry in scene_entries:
                scene_name = scene_entry.get("scene_name", scene_entry.get("scene_id"))
                files = scene_entry.get("files", [])
                if scene_name is None:
                    continue
                for file_path in files:
                    normalized = MotionDataset.normalize_relative_path(file_path)
                    if not normalized.startswith(self.split + "/"):
                        continue

                    sensor_info = MotionDataset.parse_sensor_file_path(normalized)
                    if sensor_info is not None:
                        sensor_name = sensor_info["sensor"]
                        if sensor_name not in sensor_channel_set:
                            continue
                        scene_id = sensor_info["scene_id"]
                        scene_split_dict[scene_id] = sensor_info["split"]
                        if scene_id not in scene_channel_sample_data:
                            scene_channel_sample_data[scene_id] = [[] for _ in sensor_channels]
                        filename_dict["{}/{}/{}".format(scene_id, sensor_name, sensor_info["timestamp"])] = sensor_info["path"]
                        sample_data = {"timestamp": sensor_info["timestamp"], "sensor": sensor_name}
                        for channel_index, configured_name in enumerate(sensor_channels):
                            if configured_name == sensor_name:
                                scene_channel_sample_data[scene_id][channel_index].append(sample_data)
                        continue

                    map_info = MotionDataset.parse_map_file_path(normalized)
                    if map_info is not None:
                        scene_map_dict[map_info["scene_id"]] = {
                            "split": map_info["split"],
                            "filename": map_info["path"],
                        }

        for scene_id in scene_channel_sample_data:
            for channel_index in range(len(scene_channel_sample_data[scene_id])):
                scene_channel_sample_data[scene_id][channel_index].sort(key=lambda x: x["timestamp"])

        items = []
        for scene_id, channel_sample_data in scene_channel_sample_data.items():
            for fps, stride in self.fps_stride_tuples:
                segments = MotionDataset.enumerate_segments(
                    channel_sample_data,
                    self.sequence_length,
                    fps,
                    stride,
                    enable_synchronization_check,
                )
                for segment in segments:
                    items.append({
                        "segment": segment,
                        "fps": fps,
                        "scene_id": scene_id,
                        "split": scene_split_dict.get(scene_id, self.split),
                        "angle": 0.0,
                        "dist": 0.0,
                    })

        self.filename_dict = dwm.common.SerializedReadonlyDict(filename_dict)
        self.scene_map_dict = dwm.common.SerializedReadonlyDict(scene_map_dict)
        self.items = dwm.common.SerializedReadonlyList(items)

        if image_description_settings is not None:
            self.image_descriptions = MotionDataset.load_json(self.fs, image_description_settings["path"])
            self.image_desc_rs = np.random.RandomState(
                image_description_settings["seed"] if "seed" in image_description_settings else None
            )
            self.time_list_dict = MotionDataset.load_json(self.fs, image_description_settings["time_list_dict_path"])

    def __len__(self):
        return len(self.items)

    def _scene_cache_key(self, item: dict):
        return "{}/{}".format(item["split"], item["scene_id"])

    def _get_scene_cache(self, item: dict):
        scene_key = self._scene_cache_key(item)
        if scene_key in self.scene_cache:
            cache_item = self.scene_cache.pop(scene_key)
            self.scene_cache[scene_key] = cache_item
            return cache_item

        cache_item = {}
        self.scene_cache[scene_key] = cache_item
        while len(self.scene_cache) > self.scene_cache_size:
            self.scene_cache.popitem(last=False)
        return cache_item

    def _scene_relpath(self, item: dict, suffix: str):
        return "{}/{}/{}".format(item["split"], item["scene_id"], suffix)

    def _load_scene_table_pydict(self, item: dict, cache_key: str, rel_path: str):
        cache_item = self._get_scene_cache(item)
        if cache_key not in cache_item:
            with self.fs.open(rel_path, "rb") as f:
                cache_item[cache_key] = pyarrow.feather.read_table(f).to_pydict()
        return cache_item[cache_key]

    def _load_scene_json_dict(self, item: dict, cache_key: str, rel_path: str):
        cache_item = self._get_scene_cache(item)
        if cache_key not in cache_item:
            with self.fs.open(rel_path, "r", encoding="utf-8") as f:
                cache_item[cache_key] = json.load(f)
        return cache_item[cache_key]

    def _load_extrinsics(self, item: dict):
        return self._load_scene_table_pydict(
            item,
            "extrinsics",
            self._scene_relpath(item, "calibration/egovehicle_SE3_sensor.feather"),
        )

    def _load_intrinsics(self, item: dict):
        return self._load_scene_table_pydict(
            item,
            "intrinsics",
            self._scene_relpath(item, "calibration/intrinsics.feather"),
        )

    def _load_poses(self, item: dict):
        return self._load_scene_table_pydict(
            item,
            "poses",
            self._scene_relpath(item, "city_SE3_egovehicle.feather"),
        )

    def _load_annotations(self, item: dict):
        return self._load_scene_table_pydict(
            item,
            "annotations",
            self._scene_relpath(item, "annotations.feather"),
        )

    def _load_map(self, item: dict):
        map_relpath = self.scene_map_dict[item["scene_id"]]["filename"]
        return self._load_scene_json_dict(item, "map", map_relpath)

    def __getitem__(self, index: int):
        item = self.items[index]
        result = {
            "fps": torch.tensor(item["fps"], dtype=torch.float32),
            "pts": torch.tensor([
                [
                    (sample["timestamp"] - item["segment"][0][0]["timestamp"]) / 1000000
                    for sample in timestep
                    if sample["sensor"].startswith("cameras") or (sample["sensor"] == "lidar" and not self.hide_lidar)
                ]
                for timestep in item["segment"]
            ], dtype=torch.float32),
        }

        images = []
        lidar_points = []
        for timestep in item["segment"]:
            images_i = []
            lidar_points_i = []
            for sample in timestep:
                path = self.filename_dict["{}/{}/{}".format(item["scene_id"], sample["sensor"], sample["timestamp"])]
                if sample["sensor"].startswith("cameras"):
                    with self.fs.open(path, "rb") as f:
                        image = Image.open(f)
                        image.load()
                    images_i.append(image)
                elif sample["sensor"] == "lidar":
                    with self.fs.open(path, "rb") as f:
                        points = pyarrow.feather.read_feather(f)
                    lidar_points_i.append(torch.tensor(points.to_numpy()[:, :3], dtype=torch.float32))
            if len(images_i) > 0:
                images.append(images_i)
            if len(lidar_points_i) > 0:
                lidar_points.append(lidar_points_i[0])

        if len(images) > 0:
            result["images"] = images
        if len(lidar_points) > 0 and not self.hide_lidar:
            result["lidar_points"] = lidar_points

        extrinsics = None
        intrinsics = None
        if self.enable_camera_transforms:
            if "images" in result:
                extrinsics = self._load_extrinsics(item)
                intrinsics = self._load_intrinsics(item)

                result["camera_transforms"] = torch.stack([
                    torch.stack([
                        MotionDataset.get_transform(extrinsics, "sensor_name", sample["sensor"][8:], "pt")
                        for sample in timestep
                        if sample["sensor"].startswith("cameras")
                    ])
                    for timestep in item["segment"]
                ])
                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        dwm.datasets.common.make_intrinsic_matrix(
                            MotionDataset.feather_query(intrinsics, "sensor_name", sample["sensor"][8:], MotionDataset.intrinsic_focal_keys),
                            MotionDataset.feather_query(intrinsics, "sensor_name", sample["sensor"][8:], MotionDataset.intrinsic_center_keys),
                            "pt",
                        )
                        for sample in timestep
                        if sample["sensor"].startswith("cameras")
                    ])
                    for timestep in item["segment"]
                ])
                result["image_size"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            MotionDataset.feather_query(intrinsics, "sensor_name", sample["sensor"][8:], MotionDataset.intrinsic_size_keys),
                            dtype=torch.long,
                        )
                        for sample in timestep
                        if sample["sensor"].startswith("cameras")
                    ])
                    for timestep in item["segment"]
                ])

            if "lidar_points" in result and not self.hide_lidar:
                result["lidar_transforms"] = torch.stack([
                    torch.stack([
                        torch.eye(4)
                        for sample in timestep
                        if sample["sensor"] == "lidar"
                    ])
                    for timestep in item["segment"]
                ])

        poses = None
        if self.enable_ego_transforms:
            poses = self._load_poses(item)
            result["ego_transforms"] = torch.stack([
                torch.stack([
                    MotionDataset.get_transform(poses, "timestamp_ns", sample["timestamp"], "pt")
                    for sample in timestep
                    if sample["sensor"].startswith("cameras") or (sample["sensor"] == "lidar" and not self.hide_lidar)
                ])
                for timestep in item["segment"]
            ])

        annotations = None
        if self._3dbox_image_settings is not None:
            if poses is None:
                poses = self._load_poses(item)
            if extrinsics is None:
                extrinsics = self._load_extrinsics(item)
            if intrinsics is None:
                intrinsics = self._load_intrinsics(item)
            annotations = self._load_annotations(item)

            result["3dbox_images"] = []
            annotation_timestamps = annotations["timestamp_ns"]
            max_allowed_error = 100 * 1000000
            for timestep in item["segment"]:
                current_ref_ts = timestep[0]["timestamp"]
                nearest_anno_ts = MotionDataset.get_nearest_annotation_timestamp(annotation_timestamps, current_ref_ts)
                if abs(nearest_anno_ts - current_ref_ts) > max_allowed_error:
                    empty_images = []
                    for sample in timestep:
                        if sample["sensor"].startswith("cameras"):
                            empty_images.append(MotionDataset.make_empty_camera_image(intrinsics, sample["sensor"][8:]))
                    result["3dbox_images"].append(empty_images)
                    continue

                camera_images = [
                    MotionDataset.get_3dbox_image(
                        annotations,
                        nearest_anno_ts,
                        extrinsics,
                        intrinsics,
                        poses,
                        sample,
                        self._3dbox_image_settings,
                    )
                    for sample in timestep
                    if sample["sensor"].startswith("cameras")
                ]
                result["3dbox_images"].append(camera_images)

        map_dict = None
        if self.hdmap_image_settings is not None:
            if poses is None:
                poses = self._load_poses(item)
            if extrinsics is None:
                extrinsics = self._load_extrinsics(item)
            if intrinsics is None:
                intrinsics = self._load_intrinsics(item)
            map_dict = self._load_map(item)
            result["hdmap_images"] = [
                [
                    MotionDataset.get_hdmap_image(map_dict, extrinsics, intrinsics, poses, sample, self.hdmap_image_settings)
                    for sample in timestep
                    if sample["sensor"].startswith("cameras")
                ]
                for timestep in item["segment"]
            ]

        if self._3dbox_bev_settings is not None:
            if annotations is None:
                annotations = self._load_annotations(item)
            result["3dbox_bev_images"] = [
                MotionDataset.get_3dbox_bev_image(annotations, sample, self._3dbox_bev_settings)
                for timestep in item["segment"]
                for sample in timestep
                if sample["sensor"] == "lidar"
            ]

        if self.hdmap_bev_settings is not None:
            if poses is None:
                poses = self._load_poses(item)
            if map_dict is None:
                map_dict = self._load_map(item)
            result["hdmap_bev_images"] = [
                MotionDataset.get_hdmap_bev_image(map_dict, poses, sample, self.hdmap_bev_settings)
                for timestep in item["segment"]
                for sample in timestep
                if sample["sensor"] == "lidar"
            ]

        if self.image_description_settings is not None:
            image_captions = [
                dwm.datasets.common.align_image_description_crossview([
                    MotionDataset.get_image_description(
                        self.image_descriptions,
                        self.time_list_dict,
                        item["split"],
                        item["scene_id"],
                        sample,
                    )
                    for sample in timestep
                    if sample["sensor"].startswith("cameras")
                ], self.image_description_settings)
                for timestep in item["segment"]
            ]
            result["image_description"] = [
                [
                    dwm.datasets.common.make_image_description_string(caption, self.image_description_settings, self.image_desc_rs)
                    for caption in timestep_captions
                ]
                for timestep_captions in image_captions
            ]

        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)
        return result
