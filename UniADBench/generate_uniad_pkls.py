from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import argparse
from pprint import pprint

import yaml
import debugpy

from task_api import run_task


DEFAULT_CFG = (
    "<path-to-local-resource>"
)



def load_cfg(cfg_path: str) -> Dict[str, Any]:
    path = Path(cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "-c", type=str, default=DEFAULT_CFG, help="Path to config yaml")
    parser.add_argument("-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)

    if args.debug:
        debugpy.listen(("127.0.0.1", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

    result = run_task(cfg)
    pprint(result)


if __name__ == "__main__":
    main()
