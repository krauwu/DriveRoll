import os
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN_PY = ROOT / "sim_tools" / "nuplan_sim_DB.py"
CONFIG_PATH = ROOT / "sim_tools" / "config_paths.yaml"

DB_DIR = Path(os.environ.get("NUPLAN_DB_DIR", "/path/to/nuplan/nuplan-v1.1/splits/mini"))
DB_STEMS = [
    "2021.05.12.23.36.44_veh-35_02035_02387",
]

GPU_IDS = os.environ.get("NUPLAN_GPU_IDS", "0").split(",")
OUT_ROOT = Path(os.environ.get("STAGE3_OUTPUT_ROOT", "./dbsimgen"))

procs = []
stopping = False


def terminate_all(signum=None, frame=None):
    global stopping
    if stopping:
        return
    stopping = True
    print("\n[LAUNCH] Caught signal, terminating all children...")

    for stem, proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    try:
        for stem, proc in procs:
            if proc.poll() is None:
                proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    for stem, proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    print("[LAUNCH] All children terminated.")


signal.signal(signal.SIGINT, terminate_all)
signal.signal(signal.SIGTERM, terminate_all)

for index, stem in enumerate(DB_STEMS):
    db_path = DB_DIR / f"{stem}.db"
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    env = os.environ.copy()
    env["RUN_TAG"] = f"db{index}"
    env["NUPLAN_DB_FILES"] = str(db_path)
    env["STAGE3_OUTPUT_DIR"] = str(OUT_ROOT / stem)
    env["CUDA_VISIBLE_DEVICES"] = GPU_IDS[index] if index < len(GPU_IDS) else GPU_IDS[-1]
    env["NUPLAN_SIM_CONFIG_PATHS"] = str(CONFIG_PATH)

    proc = subprocess.Popen(
        ["python", str(MAIN_PY)],
        env=env,
        cwd=str(ROOT),
        start_new_session=True,
    )
    procs.append((stem, proc))
    print("[LAUNCH] started:", stem, "pid=", proc.pid)

ret = 0
try:
    for stem, proc in procs:
        code = proc.wait()
        print("[OK]" if code == 0 else "[FAIL]", stem, "exit=", code)
        if code != 0:
            ret = code
except KeyboardInterrupt:
    terminate_all()
    ret = 130
finally:
    terminate_all()

raise SystemExit(ret)
