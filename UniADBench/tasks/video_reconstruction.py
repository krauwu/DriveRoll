from __future__ import annotations

from pathlib import Path
from typing import Dict


def run_video_reconstruction(cfg: Dict, dataset=None) -> Dict:
    task_cfg = cfg["task"].get("video_reconstruction", {})
    output_dir = Path(task_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "video_reconstruction 只保留了统一入口。"
        "等你确定重建模型的输入/输出协议后，把具体逻辑写进这个函数即可。"
    )
