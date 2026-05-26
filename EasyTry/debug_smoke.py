from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image

from tools.roll.data_agent import NuscRollingDataAgent
from tools.roll.rolling_gen_agent import RollingGenAgent
import debugpy

debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach on port 5678...")
debugpy.wait_for_client()
print("attached on port 5678...")

CFG_PATH = str(Path(__file__).resolve().parent / "rolling_demo.json")
OUTPUT_ROOT = str(Path(__file__).resolve().parent / "outputs" / "debug_smoke")
OUTPUT_VIDEO = f"{OUTPUT_ROOT}/online_roll.mp4"
GRID_DIR = f"{OUTPUT_ROOT}/grids"


def pil_to_rgb_array(image: Image.Image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb_video(frames, output_path, fps):
    if len(frames) == 0:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.pix_fmt = "yuv420p"
    stream.width = w
    stream.height = h
    stream.options = {"crf": "20", "preset": "slow"}

    for frame_rgb in frames:
        frame_av = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(frame_av):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def save_grid_image(image: Image.Image, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def print_segment_page(choices, page, page_size):
    total = len(choices)
    total_pages = (total + page_size - 1) // page_size
    start = page * page_size
    end = min(start + page_size, total)

    print("")
    print(f"segments page {page + 1}/{total_pages} | showing [{start}, {end})")
    for i in range(start, end):
        display_name, seg_id = choices[i]
        print(f"[{seg_id}] {display_name}")
    print("Enter a segment id to select it; use n/p to change pages; use q to quit")


def choose_segment_interactively(data_agent: NuscRollingDataAgent):
    choices = data_agent.get_segment_choices()
    if len(choices) == 0:
        raise RuntimeError("No selectable segment was found in the dataset")

    page_size = 20
    page = 0
    total_pages = (len(choices) + page_size - 1) // page_size

    while True:
        print_segment_page(choices, page, page_size)
        raw = input("Select segment: ").strip().lower()

        if raw == "q":
            raise SystemExit(0)

        if raw == "n":
            if page + 1 < total_pages:
                page += 1
            continue

        if raw == "p":
            if page > 0:
                page -= 1
            continue

        if raw.isdigit():
            seg_id = int(raw)
            if 0 <= seg_id < len(choices):
                return seg_id
            print(f"Invalid segment id: {seg_id}")
            continue

        print("Invalid input, please try again")


def read_user_command(step_idx: int):
    print("")
    print(f"step={step_idx} | commands: w=forward / a=left / d=right / s=slow / q")
    raw = input("Enter motion command: ").strip().lower()

    if raw in {"q", "quit", "exit"}:
        return None

    if raw == "":
        return "w"

    return raw

def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(GRID_DIR).mkdir(parents=True, exist_ok=True)

    data_agent = NuscRollingDataAgent(CFG_PATH)
    gen_agent = RollingGenAgent(CFG_PATH)

    segment_id = choose_segment_interactively(data_agent)

    gen_agent.reset()
    data_agent.gen_state = None

    init_image, init_status = data_agent.select_segment(segment_id)
    print("")
    print(init_status)

    save_grid_image(init_image, f"{GRID_DIR}/step_0000_init.png")

    frames = [pil_to_rgb_array(init_image)]
    step_idx = 0

    while True:
        if not data_agent.can_continue_generation():
            print("End of segment, stop generation")
            break

        command = read_user_command(step_idx)
        if command is None:
            print("Stopped by user")
            break

        try:
            infos_all, step_meta = data_agent.build_infos_all(command)
        except StopIteration:
            print("Insufficient remaining frames, stop generation")
            break

        cond_pack, hist_cam_list = data_agent.build_cond_from_infos(infos_all)

        next_views, new_state = gen_agent.generate_next_views(
            cond_pack=cond_pack,
            hist_cam_list=hist_cam_list,
            gen_state=data_agent.gen_state,
        )

        if next_views is None:
            print("generate_next_views returned None, stop generation")
            break

        data_agent.gen_state = new_state

        out_image, status = data_agent.commit_generated_frame(
            next_views=next_views,
            command=command,
            commit_ego_transform=step_meta["commit_ego_transform"],
        )

        print("")
        print(status)

        step_idx += 1
        save_grid_image(out_image, f"{GRID_DIR}/step_{step_idx:04d}_{command}.png")
        frames.append(pil_to_rgb_array(out_image))

        if not data_agent.can_continue_generation():
            print("End of segment, stop generation")
            break

    save_rgb_video(
        frames=frames,
        output_path=OUTPUT_VIDEO,
        fps=max(1, int(data_agent.fps)),
    )

    print("")
    print(f"Video saved: {OUTPUT_VIDEO}")
    print(f"Grid images saved: {GRID_DIR}")


if __name__ == "__main__":
    main()