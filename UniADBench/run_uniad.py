#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniAD Batch Evaluation Module
"""

import os
import sys
import re
import json
import pickle
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any


class UniADEval:
    """UniAD Batch Evaluation Class"""

    def __init__(
        self,
        uniad_root: str,
        config_path: str,
        checkpoint: str,
        pkl_dir: str,
        data_dir: str,
        output_dir: str = "./batch_output",
        gpu_ids: str = "0",
        fps: float = 12.0,
        master_port: int = 29500,
    ):
        """
        Args:
            uniad_root: UniAD project root directory
            config_path: UniAD Config file path (base_e2e.py)
            checkpoint: Model checkpoint path
            pkl_dir: pkl file directory
            output_dir: Output directory
            gpu_ids: GPU IDs, comma-separated (e.g. "0,1,2,3")
            fps: Video FPS
            master_port: torch.distributed communication port
        """
        self.uniad_root = Path(uniad_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.pkl_dir = Path(pkl_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids.split(','))
        self.fps = fps
        self.master_port = master_port
        self.data_dir = Path(data_dir).resolve()


    def get_pkl_files(self) -> List[Path]:
        """Get all pkl files in directory"""
        pkl_files = sorted(self.pkl_dir.glob("*.pkl"))
        return pkl_files

    def find_latest_metrics_json(self, temp_config_name: str) -> Optional[Path]:
        """
        Find the latest metrics_summary.json generated after evaluation

        Args:
            temp_config_name: Temporary config file name (e.g. temp_val_window_000_010.py)

        Returns:
            Path to metrics_summary.json, return None if not found
        """
        test_dir_name = Path(temp_config_name).stem

        # Search output_dir/test/{config_name} first
        test_output_root = self.output_dir / "test" / test_dir_name

        if not test_output_root.exists():
            # Compatible with old path: uniad_root/test/{config_name}
            test_output_root = self.uniad_root / "test" / test_dir_name

        if not test_output_root.exists():
            print(f"[Warning] Test output dir not found: {test_output_root}")
            return None

        # Find the latest timestamp directory
        timestamp_dirs = sorted(
            [d for d in test_output_root.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True
        )

        if not timestamp_dirs:
            print(f"[Warning] No timestamp dirs in {test_output_root}")
            return None

        # Find det/metrics_summary.json in the latest directory
        for ts_dir in timestamp_dirs:
            metrics_path = ts_dir / "det" / "metrics_summary.json"
            if metrics_path.exists():
                return metrics_path

        print(f"[Warning] metrics_summary.json not found in {test_output_root}")
        return None

    def extract_result_pkl_metrics(self, result_pkl_path: Path) -> Dict[str, Any]:
        """
        Extract planning and occ metrics from result pkl

        Returns:
            dict: {planning: {...}, occ: {...}}
        """
        with open(result_pkl_path, 'rb') as f:
            data = pickle.load(f)

        metrics = {}

        # Planning metrics
        planning = data.get('planning_results_computed', {})
        if planning:
            import numpy as np
            metrics['planning'] = {}
            for k, v in planning.items():
                if hasattr(v, 'cpu'):
                    v = v.cpu().numpy()
                if isinstance(v, np.ndarray):
                    metrics['planning'][k] = {
                        'values': v.tolist(),
                        'mean': float(np.mean(v))
                    }
                else:
                    metrics['planning'][k] = v

        # OCC metrics
        occ = data.get('occ_results_computed', {})
        if occ:
            metrics['occ'] = {}
            for k, v in occ.items():
                if hasattr(v, 'cpu'):
                    v = v.cpu().numpy()
                metrics['occ'][k] = float(v) if hasattr(v, 'item') else v

        return metrics

    def find_eval_results_json(self, temp_config_name: str) -> Optional[Path]:
        """
        Find eval_results.json generated after evaluation (contains full metrics like map IoU)

        Args:
            temp_config_name: Temporary config file name (e.g. temp_val_window_000_010.py)

        Returns:
            Path to eval_results.json, return None if not found
        """
        test_dir_name = Path(temp_config_name).stem

        # Search output_dir/test/{config_name} first
        eval_json_path = self.output_dir / "test" / test_dir_name / "eval_results.json"
        if eval_json_path.exists():
            return eval_json_path

        # Compatible with old path: uniad_root/test/{config_name}
        eval_json_path = self.uniad_root / "test" / test_dir_name / "eval_results.json"
        if eval_json_path.exists():
            return eval_json_path

        return None

    def extract_map_metrics(self, eval_json_path: Path) -> Dict[str, Any]:
        """
        Extract map IoU metrics from eval_results.json

        Returns:
            dict: {drivable_iou, lanes_iou, divider_iou, crossing_iou, contour_iou}
        """
        with open(eval_json_path, 'r') as f:
            data = json.load(f)

        map_keys = ['drivable_iou', 'lanes_iou', 'divider_iou', 'crossing_iou', 'contour_iou']
        return {k: data[k] for k in map_keys if k in data}

    def extract_det_metrics(self, metrics_json_path: Path) -> Dict[str, Any]:
        """
        Extract detection metrics from metrics_summary.json

        Returns:
            dict: {nd_score, mAP, mean_dist_aps, tp_errors}
        """
        with open(metrics_json_path, 'r') as f:
            data = json.load(f)

        return {
            'nd_score': data.get('nd_score'),
            'mAP': data.get('mean_ap'),
            'mean_dist_aps': data.get('mean_dist_aps'),
            'tp_errors': data.get('tp_errors'),
            'tp_scores': data.get('tp_scores'),
        }

    def collect_all_metrics(self, pkl_name: str, result_pkl_path: Path, temp_config_name: str) -> Dict[str, Any]:
        """
        Summarize all metrics into a dict

        Args:
            pkl_name: Original pkl file name
            result_pkl_path: Result pkl path
            temp_config_name: Temporary config file name

        Returns:
            Complete metrics summary
        """
        summary = {
            'pkl_name': pkl_name,
            'result_pkl': str(result_pkl_path),
        }

        # Extract planning and occ metrics
        pkl_metrics = self.extract_result_pkl_metrics(result_pkl_path)
        summary['planning'] = pkl_metrics.get('planning', {})
        summary['occ'] = pkl_metrics.get('occ', {})

        # Extract detection metrics
        metrics_json = self.find_latest_metrics_json(temp_config_name)
        if metrics_json:
            det_metrics = self.extract_det_metrics(metrics_json)
            summary['detection'] = det_metrics
            summary['metrics_json'] = str(metrics_json)
        else:
            summary['detection'] = None
            summary['metrics_json'] = None

        # Extract map IoU metrics
        eval_json = self.find_eval_results_json(temp_config_name)
        if eval_json:
            map_metrics = self.extract_map_metrics(eval_json)
            summary['map'] = map_metrics
            summary['eval_json'] = str(eval_json)
        else:
            summary['map'] = None
            summary['eval_json'] = None

        return summary

    def generate_temp_config(self, pkl_path: Path) -> str:
        """
        Generate temporary config, replacing ann_file_test, data_root, gt_fps

        Args:
            pkl_path: pkl file path to replace

        Returns:
            Generated temporary config file path
        """
        pkl_path = str(Path(pkl_path).resolve())
        data_root = str(self.data_dir)
        gt_fps = int(self.fps)  # Video frame rate as gt_fps
        print(f"[Config] Setting ann_file_test = {pkl_path}")
        print(f"[Config] Setting data_root = {data_root}")
        print(f"[Config] Setting gt_fps = {gt_fps}")

        # Read base config
        with open(self.config_path, 'r') as f:
            content = f.read()

        # Replace ann_file_test
        pattern = r'ann_file_test\s*=\s*["\'][^"\']*["\']'
        replacement = f'ann_file_test = "{pkl_path}"'
        content = re.sub(pattern, replacement, content)

        # Replace data_root
        pattern = r'data_root\s*=\s*["\'][^"\']*["\']'
        replacement = f'data_root = "{data_root}"'
        content = re.sub(pattern, replacement, content)

        # Replace gt_fps
        pattern = r'gt_fps\s*=\s*\d+'
        replacement = f'gt_fps = {gt_fps}'
        content = re.sub(pattern, replacement, content)

        # Temporary config placed in same directory as config_path (to keep _base_ relative paths correct)
        temp_name = f"temp_{Path(pkl_path).stem}.py"
        temp_path = self.config_path.parent / temp_name

        with open(temp_path, 'w') as f:
            f.write(content)

        return str(temp_path)

    def run_eval(self, config_path: str, output_pkl: str) -> bool:
        """Run UniAD evaluation"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        # Set NCCL timeout (seconds)
        env["NCCL_BLOCKING_WAIT"] = "1"
        env["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        # PyTorch distributed timeout (needs to be set in code; env var here as fallback)
        env["TORCH_DISTRIBUTION_DEBUG"] = "DETAIL"

        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={self.num_gpus}",
            f"--master_port={self.master_port}",
            str(self.uniad_root / "adzoo/uniad/test.py"),
            config_path,
            str(self.checkpoint),
            "--launcher", "pytorch",
            "--out", output_pkl,
            "--eval", "det", "map"
        ]

        print(f"\n{'='*60}")
        print(f"Working dir: {self.uniad_root}")
        print(f"Config: {config_path}")
        print(f"Output: {output_pkl}")
        print(f"GPUs: {self.gpu_ids} ({self.num_gpus} cards)")
        print(f"{'='*60}")

        result = subprocess.run(cmd, env=env, cwd=str(self.uniad_root))
        return result.returncode == 0

    def run(self) -> bool:
        """Run batch evaluation"""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)

        # Get all pkl files
        pkl_files = self.get_pkl_files()
        print(f"Found {len(pkl_files)} pkl files in {self.pkl_dir}")

        if not pkl_files:
            print("[Error] No pkl files found!")
            return False

        all_summaries = []

        for i, pkl in enumerate(pkl_files):
            print(f"\n[{i+1}/{len(pkl_files)}] Processing: {pkl.name}")

            # Generate temporary config
            temp_config = self.generate_temp_config(pkl)
            temp_config_name = Path(temp_config).name

            # Generate output path
            output_pkl = self.output_dir / "results" / f"results_{pkl.stem}.pkl"

            # Run evaluation
            success = self.run_eval(temp_config, str(output_pkl))

            if success:
                # Summarize metrics
                summary = self.collect_all_metrics(
                    pkl_name=pkl.name,
                    result_pkl_path=output_pkl,
                    temp_config_name=temp_config_name
                )
                all_summaries.append(summary)

                # Print brief results
                det = summary.get('detection', {}) or {}
                planning = summary.get('planning', {})
                occ = summary.get('occ', {})

                print(f"\n[Summary] {pkl.name}:")
                print(f"  Detection: NDS={det.get('nd_score', 'N/A'):.4f}, mAP={det.get('mAP', 'N/A'):.4f}" if det else "  Detection: N/A")
                if planning:
                    l2 = planning.get('L2', {})
                    if isinstance(l2, dict):
                        print(f"  Planning: L2_mean={l2.get('mean', 'N/A'):.4f}")
                if occ:
                    iou_val = occ.get('iou')
                    if iou_val is not None:
                        if isinstance(iou_val, (list, tuple)):
                            iou_val = iou_val[0] if iou_val else 0
                        print(f"  OCC: IoU={iou_val:.4f}")
                map_metrics = summary.get('map', {}) or {}
                if map_metrics:
                    map_str = ', '.join(f'{k}={v:.4f}' for k, v in map_metrics.items())
                    print(f"  Map: {map_str}")
            else:
                print(f"[Failed] {pkl.name}")

        # Save summary results
        summary_path = self.output_dir / "all_metrics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"All done!")
        print(f"Results saved to {self.output_dir / 'results'}")
        print(f"Summary saved to {summary_path}")
        print(f"{'='*60}")
        return True


# ============ Command-line entry ============
# Default configuration (for running script directly)
DEFAULT_CONFIG = {
    "uniad_root": "<path-to-local-resource>",
    "config_path": "<path-to-local-resource>",
    "checkpoint": "<path-to-local-resource>",
    "pkl_dir": "<path-to-local-resource>",
    "output_dir": "./batch_output",
    "gpu_ids": "0",
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UniAD Batch Eval")
    parser.add_argument("--uniad-root", help="UniAD root path")
    parser.add_argument("--config-path", help="Config path")
    parser.add_argument("--checkpoint", help="Checkpoint path")
    parser.add_argument("--pkl-dir", help="PKL directory")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--gpu-ids", type=str, help="GPU IDs, comma separated (e.g., '0,1,2,3')")
    args = parser.parse_args()

    # Merge default configuration and command-line args
    config = DEFAULT_CONFIG.copy()
    if args.uniad_root:
        config["uniad_root"] = args.uniad_root
    if args.config_path:
        config["config_path"] = args.config_path
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.pkl_dir:
        config["pkl_dir"] = args.pkl_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.gpu_ids:
        config["gpu_ids"] = args.gpu_ids

    # Create evaluator and run
    evaluator = UniADEval(**config)
    success = evaluator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()






#
#
#
#
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Simple launch of UniAD test.py
# """
#
# import os
# import sys
# import subprocess
# from pathlib import Path
# from typing import List
#
#
# # ============ Configuration Area ============
# UNIAD_ROOT = "<path-to-local-resource>"
# CONFIG_PATH = "<path-to-local-resource>"
# CHECKPOINT = "<path-to-local-resource>"
# PKL_DIR = "<path-to-local-resource>"
# OUTPUT_DIR = "./batch_output"
# GPU_ID = 0
# # =================================
#
#
# def get_pkl_files(pkl_dir: str) -> List[Path]:
#     """Get all pkl files in directory"""
#     pkl_path = Path(pkl_dir)
#     pkl_files = sorted(pkl_path.glob("*.pkl"))
#     return pkl_files
#
#
# def generate_temp_config(base_config: str, pkl_path: Path, output_dir: str = None) -> str:
#     """
#     Read base config, replace ann_file_test with specified pkl path, generate temporary config file
#
#     Args:
#         base_config: Base config file path
#         pkl_path: pkl file path to replace
#         output_dir: temporary config output directory, defaults to same directory as base_config
#
#     Returns:
#         Generated temporary config file path
#     """
#     import re
#     base_config = Path(base_config)
#     pkl_path = str(Path(pkl_path).resolve())
#     print(f"[Config] Setting ann_file_test = {pkl_path}")
#
#     # Read base config
#     with open(base_config, 'r') as f:
#         content = f.read()
#
#     # Replace ann_file_test definition with regex
#     # Match: ann_file_test="xxx" or ann_file_test='xxx'
#     pattern = r'ann_file_test\s*=\s*["\'][^"\']*["\']'
#     replacement = f'ann_file_test = "{pkl_path}"'
#     content = re.sub(pattern, replacement, content)
#
#     # Generate temporary config path (placed in same directory as base_config to keep _base_ relative paths correct)
#     if output_dir is None:
#         output_dir = base_config.parent
#     else:
#         output_dir = Path(output_dir)
#
#     temp_name = f"temp_{Path(pkl_path).stem}.py"
#     temp_path = output_dir / temp_name
#
#     with open(temp_path, 'w') as f:
#         f.write(content)
#
#     return str(temp_path)
#
#
# def run_eval(config_path: str, output_pkl: str):
#     """Run UniAD evaluation"""
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
#
#     cmd = [
#         sys.executable, "-m", "torch.distributed.launch",
#         "--nproc_per_node=1",
#         "--master_port=12345",
#         os.path.join(UNIAD_ROOT, "adzoo/uniad/test.py"),
#         config_path,
#         CHECKPOINT,
#         "--launcher", "pytorch",
#         "--out", output_pkl,
#         "--eval", "det", "map"
#     ]
#
#     print(f"\n{'='*60}")
#     print(f"Working dir: {UNIAD_ROOT}")
#     print(f"Config: {config_path}")
#     print(f"Output: {output_pkl}")
#     print(f"GPU: {GPU_ID}")
#     print(f"{'='*60}")
#
#     subprocess.run(cmd, env=env, cwd=UNIAD_ROOT)
#
#
# def main():
#     # Create output directory
#     output_dir = Path(OUTPUT_DIR)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     (output_dir / "results").mkdir(exist_ok=True)
#
#     # Get all pkl files
#     pkl_files = get_pkl_files(PKL_DIR)
#     print(f"Found {len(pkl_files)} pkl files in {PKL_DIR}")
#
#     for i, pkl in enumerate(pkl_files):
#         print(f"\n[{i+1}/{len(pkl_files)}] Processing: {pkl.name}")
#
#         # Generate temporary config
#         temp_config = generate_temp_config(CONFIG_PATH, pkl)
#
#         # Generate output path
#         output_pkl = str(output_dir / "results" / f"results_{pkl.stem}.pkl")
#
#         # Run evaluation
#         run_eval(temp_config, output_pkl)
#
#     print(f"\n{'='*60}")
#     print(f"All done! Results saved to {output_dir / 'results'}")
#     print(f"{'='*60}")
#
#
# if __name__ == "__main__":
#     main()
