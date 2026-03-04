#!/usr/bin/env python3
# Debug script to verify H1 axis/sign conventions against HOYO front-view assumptions.

import argparse
import os
import sys

# Ensure local extensions are discoverable (same as train.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
NAVILA_ROOT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
ISAACLAB_SOURCE = os.path.join(NAVILA_ROOT, "IsaacLab", "source")
if os.path.isdir(ISAACLAB_SOURCE) and ISAACLAB_SOURCE not in sys.path:
    sys.path.append(ISAACLAB_SOURCE)
ISAACLAB_TASKS_PKG = os.path.join(ISAACLAB_SOURCE, "isaaclab_tasks")
if os.path.isdir(ISAACLAB_TASKS_PKG) and ISAACLAB_TASKS_PKG not in sys.path:
    sys.path.append(ISAACLAB_TASKS_PKG)
LOCAL_EXT = os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.leggedloco")
if os.path.isdir(LOCAL_EXT) and LOCAL_EXT not in sys.path:
    sys.path.append(LOCAL_EXT)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Debug H1 axis signs for HOYO projection.")
parser.add_argument("--task", type=str, default="h1_vision_heading_fixed", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--steps", type=int, default=1, help="Number of sim steps before printing.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--hoyo_head_ratio", type=float, default=None, help="HOYO head-shoulder ratio (hs/hf) for offset estimation.")

# append RSL-RL cli arguments (needed for env creation parity)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
import omni.isaac.leggedloco.config  # noqa: F401  # registers gym tasks
from isaaclab_tasks.utils import parse_env_cfg


def _find_body_index(body_names, candidates):
    for cand in candidates:
        for idx, name in enumerate(body_names):
            if cand in name:
                return idx, name
    return None, None


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Reset once to initialize buffers
    env.reset()

    robot = env.unwrapped.scene["robot"]
    style_term = env.unwrapped.command_manager._terms.get("style_command")
    style_module = style_term.style_module if style_term is not None else None
    body_names = list(robot.data.body_names)

    left_idx, left_name = _find_body_index(
        body_names,
        ["left_shoulder_pitch_link", "left_shoulder", "left_shoulder_roll_link"],
    )
    right_idx, right_name = _find_body_index(
        body_names,
        ["right_shoulder_pitch_link", "right_shoulder", "right_shoulder_roll_link"],
    )
    l_ankle_idx, l_ankle_name = _find_body_index(body_names, ["left_ankle_link", "left_ankle"])
    r_ankle_idx, r_ankle_name = _find_body_index(body_names, ["right_ankle_link", "right_ankle"])
    torso_idx, torso_name = _find_body_index(body_names, ["torso_link", "torso"])

    print("[Debug] Body name samples:")
    print("  left_shoulder:", left_name)
    print("  right_shoulder:", right_name)

    if left_idx is None or right_idx is None:
        print("[Debug] Could not find shoulder links. Available names:")
        for name in body_names:
            if "shoulder" in name:
                print(" ", name)
        return

    body_pos_w = robot.data.body_pos_w  # (B, N, 3)
    l_pos = body_pos_w[0, left_idx].detach().cpu().numpy()
    r_pos = body_pos_w[0, right_idx].detach().cpu().numpy()

    print("[Debug] Shoulder positions (world):")
    print(f"  left  {left_name}:  x={l_pos[0]: .4f}, y={l_pos[1]: .4f}, z={l_pos[2]: .4f}")
    print(f"  right {right_name}: x={r_pos[0]: .4f}, y={r_pos[1]: .4f}, z={r_pos[2]: .4f}")

    if l_pos[1] > r_pos[1]:
        print("[Result] H1 world Y is left-positive (L shoulder Y > R shoulder Y).")
    elif l_pos[1] < r_pos[1]:
        print("[Result] H1 world Y is right-positive (L shoulder Y < R shoulder Y).")
    else:
        print("[Result] Shoulder Y is equal; check another frame or link.")

    # Local frame check (rotation-invariant)
    root_quat_w = robot.data.root_quat_w  # (B, 4) in (w, x, y, z)
    delta_world = body_pos_w[0, left_idx] - body_pos_w[0, right_idx]
    delta_local = math_utils.quat_rotate_inverse(root_quat_w[0:1], delta_world.unsqueeze(0))[0]
    if delta_local[1] > 0:
        print("[Result] H1 local Y is left-positive (L shoulder Y > R shoulder Y in local frame).")
    elif delta_local[1] < 0:
        print("[Result] H1 local Y is right-positive (L shoulder Y < R shoulder Y in local frame).")
    else:
        print("[Result] Shoulder Y is equal in local frame; check another frame or link.")

    # Ankle sanity check (optional but useful for scale correctness)
    if l_ankle_idx is not None and r_ankle_idx is not None:
        l_ank = body_pos_w[0, l_ankle_idx].detach().cpu().numpy()
        r_ank = body_pos_w[0, r_ankle_idx].detach().cpu().numpy()
        print("[Debug] Ankle positions (world):")
        print(f"  left  {l_ankle_name}:  x={l_ank[0]: .4f}, y={l_ank[1]: .4f}, z={l_ank[2]: .4f}")
        print(f"  right {r_ankle_name}: x={r_ank[0]: .4f}, y={r_ank[1]: .4f}, z={r_ank[2]: .4f}")

        if torso_idx is not None and args_cli.hoyo_head_ratio is not None:
            torso_pos = body_pos_w[0, torso_idx].detach().cpu().numpy()
            shoulder_mid = 0.5 * (l_pos + r_pos)
            feet_mid = 0.5 * (l_ank + r_ank)
            ratio = float(args_cli.hoyo_head_ratio)
            if abs(1.0 - ratio) < 1e-6:
                print("[Debug] hoyo_head_ratio too close to 1.0; cannot estimate head offset.")
            else:
                # Solve for head Z using (H - S) / (H - F) = ratio
                S = float(shoulder_mid[2])
                F = float(feet_mid[2])
                H = (S - ratio * F) / (1.0 - ratio)
                offset = H - float(torso_pos[2])
                print("[Debug] Head offset estimate from HOYO ratio:")
                print(f"  torso_z={torso_pos[2]: .4f}, shoulder_z={S: .4f}, feet_z={F: .4f}")
                print(f"  ratio(hs/hf)={ratio:.4f} => head_z={H: .4f}, offset_z={offset: .4f}")

    # Fill style buffer and check projection stats
    if style_module is not None:
        root_quat_w = robot.data.root_quat_w
        body_quat_w = robot.data.body_quat_w
        for _ in range(style_module.buffer_len):
            style_module.update_buffer(body_pos_w, root_quat_w, body_names, body_quat_w)

        buf_2d = style_module.get_processed_buffer_2d()  # (B, T, 14, 2) with stats
        head = buf_2d[0, -1, 0].detach().cpu().numpy()
        r_ank_2d = buf_2d[0, -1, 10].detach().cpu().numpy()
        l_ank_2d = buf_2d[0, -1, 13].detach().cpu().numpy()
        print("[Debug] HOYO 2D projection sample (x,y) after preprocessing:")
        print(f"  head:    x={head[0]: .4f}, y={head[1]: .4f}")
        print(f"  l_ankle: x={l_ank_2d[0]: .4f}, y={l_ank_2d[1]: .4f}")
        print(f"  r_ankle: x={r_ank_2d[0]: .4f}, y={r_ank_2d[1]: .4f}")

        # Head-feet 2D distance for scale sanity (should be ~1 before standardization)
        feet_mid = 0.5 * (l_ank_2d + r_ank_2d)
        dist = ((head - feet_mid) ** 2).sum() ** 0.5
        print(f"[Debug] Head-feet 2D distance (post-std): {dist: .4f} (not expected ~1.0)")

        # Also check distance before standardization (mean/std)
        saved_mean, saved_std = style_module.mean, style_module.std
        style_module.mean, style_module.std = None, None
        buf_2d_raw = style_module.get_processed_buffer_2d()
        style_module.mean, style_module.std = saved_mean, saved_std

        head_raw = buf_2d_raw[0, -1, 0].detach().cpu().numpy()
        r_ank_raw = buf_2d_raw[0, -1, 10].detach().cpu().numpy()
        l_ank_raw = buf_2d_raw[0, -1, 13].detach().cpu().numpy()
        feet_mid_raw = 0.5 * (l_ank_raw + r_ank_raw)
        dist_raw = ((head_raw - feet_mid_raw) ** 2).sum() ** 0.5
        print(f"[Debug] Head-feet 2D distance (pre-std):  {dist_raw: .4f} (target ~1.0)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
