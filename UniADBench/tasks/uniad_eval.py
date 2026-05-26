"""
UniAD评估任务 - 简化封装，复用UniAD的test.py逻辑
支持单卡和多卡分布式推理
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# 添加UniAD路径
def setup_uniad_path(uniad_root: str) -> None:
    """将UniAD路径添加到sys.path"""
    uniad_root = Path(uniad_root)
    paths_to_add = [
        str(uniad_root),
        str(uniad_root / "adzoo"),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


def is_distributed() -> bool:
    """检查是否在分布式环境中"""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def run_uniad_eval(cfg: Dict[str, Any], dataset=None) -> Dict[str, Any]:
    """
    UniAD评估任务入口
    """
    task_cfg = cfg["task"]["uniad_eval"]
    uniad_root = cfg.get("uniad_root") or task_cfg.get("uniad_root")

    # 设置路径
    setup_uniad_path(uniad_root)

    # 配置路径
    uniad_config_path = task_cfg["uniad_config"].replace("${uniad_root}", uniad_root)
    checkpoint_path = task_cfg["checkpoint_path"].replace("${uniad_root}", uniad_root)
    pkl_dir = Path(task_cfg["pkl_dir"])
    pseudo_root = Path(task_cfg["pseudo_root"])
    output_root = Path(task_cfg["output"]["root_dir"])

    eval_mod = task_cfg.get("eval_mod", ["det", "map", "track", "motion"])
    launcher = task_cfg.get("launcher", "none")
    gpu_ids = task_cfg.get("gpu_ids", [0])

    output_root.mkdir(parents=True, exist_ok=True)

    # 查找所有pkl文件
    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in {pkl_dir}")

    # 多卡模式：需要通过torchrun启动
    if launcher == "pytorch" and not is_distributed():
        # 使用torchrun重新启动当前脚本
        gpu_str = ",".join(str(g) for g in gpu_ids)
        num_gpus = len(gpu_ids)

        # 获取当前python脚本路径和参数
        import __main__
        main_file = __main__.__file__

        # 构建torchrun命令
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            "--nproc_per_node", str(num_gpus),
            "--master_port", "29500",
            main_file,
        ]

        # 传递当前的配置参数
        import argparse
        for arg in sys.argv[1:]:
            cmd.append(arg)

        print(f"[UniAD Eval] Launching distributed training with {num_gpus} GPUs: {gpu_str}")
        print(f"[UniAD Eval] Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str

        result = subprocess.run(cmd, env=env)
        return {"task": "uniad_eval", "subprocess_returncode": result.returncode}

    # 以下是在分布式环境中执行的实际逻辑
    print(f"[UniAD Eval] Found {len(pkl_files)} pkl files")
    print(f"[UniAD Eval] Config: {uniad_config_path}")
    print(f"[UniAD Eval] Checkpoint: {checkpoint_path}")
    print(f"[UniAD Eval] Pseudo root: {pseudo_root}")
    print(f"[UniAD Eval] Eval mod: {eval_mod}")
    print(f"[UniAD Eval] Distributed: {is_distributed()}")

    # 导入UniAD模块
    from mmcv.utils import Config, get_dist_info, init_dist, wrap_fp16_model, load_checkpoint, set_random_seed
    from mmcv.datasets import build_dataset, build_dataloader
    from mmcv.models import build_model
    from mmcv.fileio.io import dump
    from torch.nn.parallel.distributed import DistributedDataParallel

    all_results = {}

    for pkl_file in pkl_files:
        print(f"\n{'='*60}")
        print(f"[UniAD Eval] Processing: {pkl_file.name}")
        print(f"{'='*60}")

        # 加载UniAD配置
        uniad_cfg = Config.fromfile(uniad_config_path)

        # 修改配置以适配伪数据
        uniad_cfg.data.test["data_root"] = str(pseudo_root)
        uniad_cfg.data.test["ann_file"] = str(pkl_file)
        uniad_cfg.data.test["test_mode"] = True
        uniad_cfg.data.test["eval_mod"] = eval_mod
        uniad_cfg.model.pretrained = None

        # 设置随机种子
        set_random_seed(0)

        # 判断是否分布式
        if launcher == "pytorch":
            distributed = True
            if not torch.distributed.is_initialized():
                init_dist(launcher)
        else:
            distributed = False

        # 构建数据集
        test_dataset = build_dataset(uniad_cfg.data.test)

        if distributed:
            data_loader = build_dataloader(
                test_dataset,
                samples_per_gpu=1,
                workers_per_gpu=uniad_cfg.data.get("workers_per_gpu", 4),
                dist=True,
                shuffle=False,
                nonshuffler_sampler=uniad_cfg.data.get("nonshuffler_sampler", dict(type="DistributedSampler")),
            )
        else:
            data_loader = build_dataloader(
                test_dataset,
                samples_per_gpu=1,
                workers_per_gpu=uniad_cfg.data.get("workers_per_gpu", 4),
                dist=False,
                shuffle=False,
            )

        # 构建模型
        uniad_cfg.model.train_cfg = None
        model = build_model(uniad_cfg.model, test_cfg=uniad_cfg.get("test_cfg"))

        fp16_cfg = uniad_cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        # 加载权重
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = test_dataset.CLASSES

        # 模型移至GPU
        if distributed:
            model = DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            from adzoo.uniad.test_utils import custom_multi_gpu_test
            outputs = custom_multi_gpu_test(model, data_loader)
        else:
            model = model.cuda()
            from adzoo.uniad.test_utils import custom_single_gpu_test
            outputs = custom_single_gpu_test(model, data_loader)

        # 保存结果和评估（只在rank 0执行）
        if distributed:
            rank, _ = get_dist_info()
            if rank != 0:
                continue

        out_pkl = output_root / f"results_{pkl_file.stem}.pkl"
        dump(outputs, str(out_pkl))
        print(f"[UniAD Eval] Results saved to {out_pkl}")

        # 运行评估
        if eval_mod:
            eval_kwargs = uniad_cfg.get("evaluation", {}).copy()
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)

            jsonfile_prefix = str(output_root / f"eval_{pkl_file.stem}")
            eval_kwargs["jsonfile_prefix"] = jsonfile_prefix
            eval_kwargs["metric"] = eval_mod

            print(f"\n[UniAD Eval] Running evaluation for {pkl_file.name}...")
            eval_result = test_dataset.evaluate(outputs, **eval_kwargs)
            all_results[pkl_file.name] = eval_result
            print(eval_result)

    # 汇总结果
    if all_results:
        summary_path = output_root / f"eval_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_path, "w") as f:
            f.write("UniAD Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            for pkl_name, result in all_results.items():
                f.write(f"\n{pkl_name}:\n")
                f.write("-" * 40 + "\n")
                f.write(str(result))
                f.write("\n")
        print(f"\n[UniAD Eval] Summary saved to {summary_path}")

    return {
        "task": "uniad_eval",
        "num_pkls": len(pkl_files),
        "output_dir": str(output_root),
        "results": all_results,
    }