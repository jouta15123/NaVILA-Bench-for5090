#!/usr/bin/env python3
"""
Debug script to visualize H1 robot's 2D projection (HOYO-compatible) as a stick figure animation.
Generates a GIF demonstrating the 'always frontal' view logic.
"""

import argparse
import os
import sys

# Strong headless configuration
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# === Path Setup (same as train.py) ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)  # legged-loco

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

NAVILA_ROOT = os.path.dirname(REPO_ROOT)  # NaVILA-Bench
if NAVILA_ROOT not in sys.path:
    sys.path.append(NAVILA_ROOT)

SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if os.path.isdir(SCRIPTS_DIR) and SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Assume IsaacLab is sibling of NaVILA-Bench
HOME_ROOT = os.path.dirname(NAVILA_ROOT)
ISAACLAB_SOURCE = os.path.join(HOME_ROOT, "IsaacLab", "source")

if os.path.isdir(ISAACLAB_SOURCE) and ISAACLAB_SOURCE not in sys.path:
    sys.path.append(ISAACLAB_SOURCE)

ISAACLAB_PKG = os.path.join(ISAACLAB_SOURCE, "isaaclab")
if os.path.isdir(ISAACLAB_PKG) and ISAACLAB_PKG not in sys.path:
    sys.path.append(ISAACLAB_PKG)
ISAACLAB_TASKS_PKG = os.path.join(ISAACLAB_SOURCE, "isaaclab_tasks")
if os.path.isdir(ISAACLAB_TASKS_PKG) and ISAACLAB_TASKS_PKG not in sys.path:
    sys.path.append(ISAACLAB_TASKS_PKG)
ISAACLAB_RL_PKG = os.path.join(ISAACLAB_SOURCE, "isaaclab_rl")
if os.path.isdir(ISAACLAB_RL_PKG) and ISAACLAB_RL_PKG not in sys.path:
    sys.path.append(ISAACLAB_RL_PKG)

# Site packages
py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
ISAACLAB_SITE_PACKAGES = os.path.join(
    os.path.dirname(ISAACLAB_SOURCE),
    "env_isaaclab",
    "lib",
    py_version,
    "site-packages",
)
if os.path.isdir(ISAACLAB_SITE_PACKAGES) and ISAACLAB_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, ISAACLAB_SITE_PACKAGES)

# Local plugins in legged-loco/isaaclab_exts
LOCAL_EXT_PATH_GROUPS = [
    (
        "omni.isaac.leggedloco",
        [
             os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.leggedloco"),
        ]
    ),
    (
        "omni.isaac.vlnce",
        [
            os.path.join(NAVILA_ROOT, "omni.isaac.vlnce"),
            os.path.join(NAVILA_ROOT, "isaaclab_exts", "omni.isaac.vlnce"),
        ],
    ),
    (
        "omni.isaac.matterport",
        [
            os.path.join(NAVILA_ROOT, "omni.isaac.matterport"),
            os.path.join(NAVILA_ROOT, "isaaclab_exts", "omni.isaac.matterport"),
        ],
    ),
]

for _, candidate_paths in LOCAL_EXT_PATH_GROUPS:
    for _ext_path in candidate_paths:
        if os.path.isdir(_ext_path) and _ext_path not in sys.path:
            sys.path.append(_ext_path)
# === End Path Setup ===

from isaaclab.app import AppLauncher

# local imports
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize H1 2D projection.")
parser.add_argument("--task", type=str, default="h1_base_rough", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--frames", type=int, default=500, help="Number of frames to record.")
parser.add_argument("--save_path", type=str, default="h1_stick_figure.gif", help="Output GIF path.")
# Missing args required by parse_rsl_rl_cfg
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Use CNN")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks
import omni.isaac.leggedloco.config
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import load_yaml
import yaml
from isaaclab.utils import update_class_from_dict
from rsl_rl.runners import OnPolicyRunner
import isaaclab_rl
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

def load_yaml_with_slices(path: str) -> dict:
    """
    Load YAML while supporting python/object/apply:builtins.slice tags that
    appear in IsaacLab env.yaml dumps.
    """
    class _Loader(yaml.FullLoader):
        pass

    def _construct_slice(loader, node):
        values = loader.construct_sequence(node)
        return slice(*values)

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice",
        _construct_slice,
    )

    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)


def apply_observations_override(env_cfg, obs_dict: dict) -> None:
    update_class_from_dict(env_cfg, {"observations": obs_dict})
    # If a group omits "style", explicitly disable it (base config may include it).
    try:
        if isinstance(obs_dict, dict) and "critic" in obs_dict:
            if isinstance(obs_dict["critic"], dict) and "style" not in obs_dict["critic"]:
                if hasattr(env_cfg.observations, "critic") and hasattr(env_cfg.observations.critic, "style"):
                    env_cfg.observations.critic.style = None
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            if isinstance(obs_dict["policy"], dict) and "style" not in obs_dict["policy"]:
                if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "style"):
                    env_cfg.observations.policy.style = None
    except Exception:
        pass


def enforce_num_envs(env_cfg, num_envs: int) -> None:
    try:
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = num_envs
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
            if hasattr(env_cfg.scene.terrain, "num_envs"):
                env_cfg.scene.terrain.num_envs = num_envs
    except Exception:
        pass


def infer_obs_dims_from_checkpoint(ckpt_path: str) -> tuple:
    """Infer actor and critic input dimensions from checkpoint."""
    try:
        data = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None, None
    sd = data.get("model_state_dict", data) if isinstance(data, dict) else data
    try:
        actor_in = int(sd["actor.0.weight"].shape[1])
        critic_in = int(sd["critic.0.weight"].shape[1])
        return actor_in, critic_in
    except Exception:
        return None, None


def maybe_disable_critic_style(env_cfg, actor_in, critic_in, style_dim: int = 512) -> None:
    """Disable critic style observation if checkpoint shows mismatch."""
    if actor_in is None or critic_in is None:
        return
    if actor_in - critic_in == style_dim and critic_in < actor_in:
        try:
            if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "critic"):
                if hasattr(env_cfg.observations.critic, "style"):
                    env_cfg.observations.critic.style = None
                    print("[INFO] Disabled critic style observation to match checkpoint dims.")
        except Exception:
            pass


def _get_command_tensor(command_term):
    for attr in ("_command", "command", "commands"):
        if hasattr(command_term, attr):
            tensor = getattr(command_term, attr)
            if torch.is_tensor(tensor):
                return tensor
    return None

def _set_base_velocity_for_envs(command_term, env_ids, cmd: dict) -> bool:
    cmd_tensor = _get_command_tensor(command_term)
    if cmd_tensor is None:
        return False
    lin_x = float(cmd.get("lin_vel_x", 0.0))
    lin_y = float(cmd.get("lin_vel_y", 0.0))
    ang_z = float(cmd.get("ang_vel_z", 0.0))
    heading = float(cmd.get("heading", 0.0))
    for env_id in env_ids:
        idx = int(env_id)
        if cmd_tensor.shape[1] > 0:
            cmd_tensor[idx, 0] = lin_x
        if cmd_tensor.shape[1] > 1:
            cmd_tensor[idx, 1] = lin_y
        if cmd_tensor.shape[1] > 2:
            cmd_tensor[idx, 2] = ang_z
        if cmd_tensor.shape[1] > 3:
            cmd_tensor[idx, 3] = heading
    return True

# HOYO Skeleton Connectivity (Indices)
# 0: Head, 1: Neck, 2: R-Shoulder, 3: R-Elbow, 4: R-Hand
# 5: L-Shoulder, 6: L-Elbow, 7: L-Hand
# 8: R-Hip, 9: R-Knee, 10: R-Ankle
# 11: L-Hip, 12: L-Knee, 13: L-Ankle

SKELETON_EDGES = [
    (0, 1),   # Head - Neck
    (1, 2),   # Neck - R-Shoulder
    (2, 3),   # R-Shoulder - R-Elbow
    (3, 4),   # R-Elbow - R-Hand
    (1, 5),   # Neck - L-Shoulder
    (5, 6),   # L-Shoulder - L-Elbow
    (6, 7),   # L-Elbow - L-Hand
    (1, 8),   # Neck - R-Hip (Approx, usually Spine->Hip)
    (8, 9),   # R-Hip - R-Knee
    (9, 10),  # R-Knee - R-Ankle
    (1, 11),  # Neck - L-Hip
    (11, 12), # L-Hip - L-Knee
    (12, 13), # L-Knee - L-Ankle
]

# Better connectivity for visual appeal (Torso/Spine logic might differ, but this is simple stickman)
# Note: HOYO uses:
# Head, Neck
# R_S, R_E, R_H
# L_S, L_E, L_H
# R_Hip, R_K, R_A
# L_Hip, L_K, L_A
# Usually hips are connected to a root or spine. Neck (1) is close to shoulders.
# We will connect Hips to Neck for simplicity in this visualization if no Spine joint exists.

import math
import imageio


def main():
    # Hardcoded base run for debugging
    # base_run_dir = os.path.join(REPO_ROOT, "logs", "rsl_rl", "h1_base_rough", "2025-12-06_10-01-18")
    base_run_dir = os.path.join(REPO_ROOT, "logs", "rsl_rl", "h1_vision_rough", "2026-01-05_03-51-55_trial_h1_vision_20260104_hoyo_m2t_repro")
    
    # Switch default task if not provided (though args are parsed already, we override if it matches default)
    if args_cli.task == "h1_base_rough": # or check if user didn't provide it? simpler to just use args_cli.task if user provided it
        pass # User control
    else:
        # If user passed h1_vision (default), we might want to switch to h1_base_rough for this policy
        # BUT user might want to test h1_vision environment structure.
        # Let's assume user knows what they are doing.
        # But for "base model" compatibility, likely best to use the task it was trained on.
        # We will warn if mismatch.
        pass

    # Parse config
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Load params from run
    # Load params from run
    log_env_cfg_path = os.path.join(base_run_dir, "params", "env.yaml")
    if os.path.exists(log_env_cfg_path):
        print(f"Loading env config from {log_env_cfg_path}")
        try:
            try:
                log_env_cfg_dict = load_yaml(log_env_cfg_path)
            except Exception as exc:
                print(f"[WARN] Failed to load env.yaml with default loader: {exc}")
                log_env_cfg_dict = load_yaml_with_slices(log_env_cfg_path)
            try:
                update_class_from_dict(env_cfg, log_env_cfg_dict)
                print(f"[INFO] Using env config from: {log_env_cfg_path}")
            except ValueError as exc:
                print(f"[WARN] Full env.yaml update failed: {exc}")
                # Fallback: only update observations to match policy input size.
                obs_only = {}
                if isinstance(log_env_cfg_dict, dict) and "observations" in log_env_cfg_dict:
                    obs_only["observations"] = log_env_cfg_dict["observations"]
                if obs_only:
                    apply_observations_override(env_cfg, obs_only["observations"])
                    print("[INFO] Applied observations-only override from env.yaml")
                else:
                    print("[WARN] env.yaml did not contain observations; skipping override")
        except Exception as exc:
            print(f"[WARN] Failed to apply env.yaml overrides: {exc}")
        
    enforce_num_envs(env_cfg, args_cli.num_envs)

    # Update agent config from the loaded run
    log_agent_cfg_dict = None
    log_agent_cfg_file_path = os.path.join(base_run_dir, "params", "agent.yaml")
    if os.path.exists(log_agent_cfg_file_path):
         log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
         update_class_from_dict(agent_cfg, log_agent_cfg_dict)
    
    # Re-apply CLI checkpoint override if provided (not applicable here as we force load)
    # Re-enforce num_envs from CLI arg (because env.yaml might have overwritten it)
    if args_cli.num_envs is not None:
         env_cfg.scene.num_envs = args_cli.num_envs

    # Use history_length from agent config when CLI does not override it.
    history_length = args_cli.history_length
    if history_length == 0 and log_agent_cfg_dict and "policy" in log_agent_cfg_dict:
        history_length = log_agent_cfg_dict["policy"].get("history_length", 0)
        if history_length:
            print(f"[INFO] Using history_length={history_length} from agent config.")

    # Infer obs dims from checkpoint and adjust config if needed
    resume_path = os.path.join(base_run_dir, "model_1800.pt")
    actor_in, critic_in = infer_obs_dims_from_checkpoint(resume_path)
    print(f"[INFO] Checkpoint obs dims: actor_in={actor_in}, critic_in={critic_in}")
    maybe_disable_critic_style(env_cfg, actor_in, critic_in)
    
    # Disable terrain curriculum for debug visualization to avoid 'size' error
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        print("[INFO] Disabling terrain_levels curriculum for debug script.")
        env_cfg.curriculum.terrain_levels = None

    # Replace reset_base event with simpler uniform reset to avoid flat patch error
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
        from isaaclab.managers import EventTermCfg as EventTerm
        import isaaclab.envs.mdp as base_mdp
        print("[INFO] Replacing reset_base with reset_root_state_uniform for debug script.")
        env_cfg.events.reset_base = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )

    # Enable rendering
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    # Wrap environment for RSL-RL (history-aware if needed)
    if history_length > 0:
        print(f"[INFO] Using RslRlVecEnvHistoryWrapper with history_length={history_length}")
        env = RslRlVecEnvHistoryWrapper(env, history_length=history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # Load Policy
    print(f"Loading policy from {base_run_dir}")
    # We need to construct runner to load
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Load model_9.pt (latest in that dir)
    resume_path = os.path.join(base_run_dir, "model_1800.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    env.reset()
    
    robot = env.unwrapped.scene["robot"]
    # Provide access to command manager
    command_manager = env.unwrapped.command_manager
    base_vel_term = command_manager._terms.get("base_velocity")
    
    # Try to get style module if available (h1_vision has it, h1_base_rough might NOT)
    # If h1_base_rough, we might fail to get style_module if not added.
    # The user asked to debug "coordinate transformation for style reward".
    # This implies we NEED style_module.
    # If h1_base_rough doesn't have style_module, this script fails.
    # So we MUST use h1_vision task (or similar) but force-load base policy.
    # We will assume args_cli.task is "h1_vision" (default).
    
    if "style_command" in command_manager._terms:
        style_module = command_manager._terms["style_command"].style_module
    else:
        print("Warning: 'style_command' not found. Visualization of stick figure might fail or be empty.")
        style_module = None

    history_2d = [] 
    video_frames = []
    
    init_root_pos = robot.data.root_pos_w[0].clone()
    init_root_pos[2] += 0.1 

    warmup_steps = 50
    total_steps = warmup_steps + args_cli.frames

    print(f"Collecting {args_cli.frames} frames (after {warmup_steps} warmup)...")
    
    for i in range(total_steps):
        # --- Command logic ---
        # 0-150: Forward
        # 150-350: Forward + Turn Left
        # 350-500: Forward + Turn Right
        
        rel_step = i - warmup_steps
        cmd = {"lin_vel_x": 0.5, "lin_vel_y": 0.0, "ang_vel_z": 0.0, "heading": 0.0}
        
        if rel_step > 150 and rel_step <= 300:
             cmd["ang_vel_z"] = 0.5 # Turn Left
        elif rel_step > 300:
             cmd["ang_vel_z"] = -0.5 # Turn Right
        
        if base_vel_term:
             _set_base_velocity_for_envs(base_vel_term, range(env.unwrapped.num_envs), cmd)
        
        # Inference
        with torch.no_grad():
             obs, _ = env.get_observations()
             actions = policy(obs)
             step_out = env.step(actions)
             if len(step_out) == 5:
                 obs, rew, terminated, truncated, info = step_out
                 done_mask = terminated | truncated
             else:
                 obs, rew, done_mask, info = step_out
                 terminated = done_mask
                 truncated = torch.zeros_like(done_mask)

        yaw = float(torch.atan2(
            2 * (
                robot.data.root_quat_w[0, 0] * robot.data.root_quat_w[0, 3]
                + robot.data.root_quat_w[0, 1] * robot.data.root_quat_w[0, 2]
            ),
            1 - 2 * (robot.data.root_quat_w[0, 2] ** 2 + robot.data.root_quat_w[0, 3] ** 2),
        ).item())
        if i % 50 == 0:
            print("yaw(rad)=", yaw)
        
        # --- Capture Video Frame ---
        if i >= warmup_steps:
             frame = env.unwrapped.render()
             if frame is not None:
                  video_frames.append(frame)
             
             # Update Camera to look at robot
             # logic from play.py
             robot_pos_w = robot.data.root_pos_w[0].cpu().numpy()
             # Eye: offset by (3,3,3), Target: robot pos
             cam_eye = (robot_pos_w[0] + 3.0, robot_pos_w[1] + 3.0, robot_pos_w[2] + 1.5)
             cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
             env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

        if style_module:
             # 2) Explicitly update buffer to ensure it works even if reward is not called
             # Warning: If env.step() already updates buffer (via StyleReward), this will cause double update (2x speed).
             # For debugging physics/mapping, this is acceptable, but be aware.
             body_pos_w = robot.data.body_pos_w
             body_quat_w = robot.data.body_quat_w
             root_quat_w = robot.data.root_quat_w
             body_names = robot.data.body_names 
             
             style_module.update_buffer(body_pos_w, root_quat_w, body_names, body_quat_w)
             
             # Capture 2D map
             if i >= warmup_steps:
                 full_buffer_2d = style_module.get_hoyo_compatible_keymap(standardize=False, normalize_height=False)
                 latest_frame_2d = full_buffer_2d[0, -1].detach().cpu().numpy() # (14, 2)
                 history_2d.append(latest_frame_2d)

        # Handle resets
        # If any env is done, we reset everything for simplicity in this single-stream debug
        if done_mask.any():
             env.reset() # This resets all envs
             # Reset style buffer for ALL envs (not just done_ids) to avoid scale corruption
             if style_module:
                 style_module.reset_buffer()  # No args = reset all envs
             continue

        if i < warmup_steps:
             continue
        
        if (i - warmup_steps) % 50 == 0:
            print(f"Collected frame {i - warmup_steps}/{args_cli.frames}")
        
    
    print(f"Generating animation: {args_cli.save_path}")
    
    # Save MP4
    mp4_path = args_cli.save_path.replace(".gif", ".mp4")
    print(f"Saving MP4 video to {mp4_path}...")
    try:
        mp4_out = os.path.abspath(mp4_path)
        writer = imageio.get_writer(mp4_out, fps=30)
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"Saved Video: {mp4_out}")
    except Exception as e:
        print(f"Error saving MP4: {e}")
        import traceback
        traceback.print_exc()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Raw meter scale: Robot is ~1.7m tall.
    # X: Left/Right (-1 to 1)
    # Y: Up/Down (-0.5 to 2.0)
    ax.set_xlim(-1.0, 1.0) 
    ax.set_ylim(-0.5, 2.0)
    
    # Plot y = -y to flip "Down is Positive" (HOYO) to "Up is Positive" (Plot)
    
    scat = ax.scatter([], [], c='r', s=20)
    lines = [ax.plot([], [], 'b-')[0] for _ in SKELETON_EDGES]
    text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def update(frame_idx):
        data = history_2d[frame_idx] # (14, 2)
        x = data[:, 0]
        y = -data[:, 1] # Flip Y so Up is Positive for visualization
        
        scat.set_offsets(np.c_[x, y])
        
        for line, (i, j) in zip(lines, SKELETON_EDGES):
            line.set_data([x[i], x[j]], [y[i], y[j]])
            
        text.set_text(f"Frame: {frame_idx}")
        return [scat, text] + lines

    ani = animation.FuncAnimation(fig, update, frames=len(history_2d), blit=True)
    try:
        print("Saving GIF... (this might take a moment)")
        out_path = os.path.abspath(args_cli.save_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ani.save(out_path, writer=PillowWriter(fps=30))
        print("Saved:", out_path)
    except Exception as e:
        print(f"Error saving animation: {e}")
        import traceback
        traceback.print_exc()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()