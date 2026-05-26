import pickle
import numpy as np

# Paths
custom_pkl = "<path-to-local-resource>"
official_pkl = "<path-to-local-resource>"


def load_pkl(path):
    """Load pkl file"""
    print("\n==============================")
    print("Loading:", path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    print("Total frames:", len(infos))
    return infos


def check_future_traj(infos, name):
    """Check if each frame contains fut_traj"""
    print("\n==============================")
    print("Checking future traj:", name)
    missing = 0
    exist = 0
    for frame in infos:
        if "fut_traj" not in frame:
            missing += 1
        else:
            exist += 1
    print("Frames with fut_traj:", exist)
    print("Frames missing fut_traj:", missing)


def print_example_future(infos, name):
    """Print a sample containing fut_traj"""
    print("\n==============================")
    print("Example future traj from:", name)
    for frame in infos:
        if "fut_traj" in frame:
            fut_traj = frame["fut_traj"]
            mask = frame["fut_traj_valid_mask"]
            print("\nFound frame with fut_traj")
            print("fut_traj shape:", fut_traj.shape)
            print("mask shape:", mask.shape)

            print("\nFirst agent future traj:")
            print(fut_traj[0])

            print("\nFirst agent mask:")
            print(mask[0])

            break
    else:
        print("No fut_traj found in this dataset")


def inspect_frame_keys(infos, name):
    """View keys contained in each frame"""
    print("\n==============================")
    print("Frame keys in:", name)
    frame = infos[0]
    for k in frame.keys():
        print(k)


if __name__ == "__main__":
    # ---------- Official 2Hz ----------
    official_infos = load_pkl(official_pkl)
    inspect_frame_keys(official_infos, "official 2hz")
    check_future_traj(official_infos, "official 2hz")
    print_example_future(official_infos, "official 2hz")

    # ---------- Sliced data ----------
    custom_infos = load_pkl(custom_pkl)
    inspect_frame_keys(custom_infos, "custom sliced")
    check_future_traj(custom_infos, "custom sliced")
    print_example_future(custom_infos, "custom sliced")