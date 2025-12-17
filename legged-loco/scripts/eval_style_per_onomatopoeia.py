# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
オノマトペごとのスタイル評価スクリプト
各オノマトペを固定して推論し、動きの特徴を比較する
"""

import argparse
import os
import sys

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
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate style per onomatopoeia")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="h1_vision")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--use_cnn", action="store_true", default=None)
parser.add_argument("--arm_fixed", action="store_true", default=False)
parser.add_argument("--use_rnn", action="store_true", default=False)
parser.add_argument("--history_length", default=0, type=int)
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps to evaluate per onomatopoeia")
parser.add_argument("--output_dir", type=str, default="eval_results/style_per_onomatopoeia")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import torch
import numpy as np
import json
import imageio
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import load_yaml
from isaaclab.utils import update_class_from_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from omni.isaac.leggedloco.config import *
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper
from omni.isaac.leggedloco.leggedloco.mdp.style_module import INSTRUCTION_ONOMATOPEIA

# List of onomatopoeia to evaluate
ONOMATOPOEIA_LIST = INSTRUCTION_ONOMATOPEIA


def quat2eulers(w, x, y, z):
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def evaluate_onomatopoeia(env, policy, onomatopoeia: str, num_steps: int, device, 
                          record_video: bool = False, video_path: str = None):
    """Evaluate a single onomatopoeia and collect metrics."""
    
    # Force set the style command for all envs
    style_gen = env.unwrapped.command_manager._terms["style_command"]
    
    # Manually set the onomatopoeia for all environments
    for env_id in range(env.unwrapped.num_envs):
        style_gen.current_texts[env_id] = onomatopoeia
        z_onm, centroid = style_gen.style_module.encode_instruction(onomatopoeia)
        z_onm = z_onm.squeeze(0)
        centroid = centroid.squeeze(0)
        style_gen.style_latents[env_id] = z_onm
        style_gen.centroids[env_id] = centroid
        style_gen._command[env_id, :512] = z_onm
        style_gen._command[env_id, 512:] = centroid
    
    # Reset environment and rebuild observation with history buffer (if enabled)
    # Note: some wrappers return the raw policy obs on reset; calling
    # get_observations() aligns shapes with history-augmented policies.
    env.reset()
    obs, _ = env.get_observations()
    
    # Video recording setup
    frames = []
    if record_video:
        base_env = env.unwrapped
        # Set initial camera
        robot_pos_w = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        cam_eye = (robot_pos_w[0] + 3.0, robot_pos_w[1] + 3.0, robot_pos_w[2] + 2.0)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        base_env.sim.set_camera_view(eye=cam_eye, target=cam_target)
    
    # Metrics storage
    metrics = {
        "velocities_x": [],
        "velocities_y": [],
        "velocities_z": [],
        "angular_vel_z": [],
        "roll": [],
        "pitch": [],
        "episode_lengths": [],
        "text_sims": [],
        "centroid_sims": [],
    }
    
    episode_step_counts = torch.zeros(env.unwrapped.num_envs, device=device)
    
    for step in range(num_steps):
        # Use no_grad instead of inference_mode to avoid creating inference tensors
        # that later reject in-place updates during env.reset().
        with torch.no_grad():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            
            # Get robot state
            robot = env.unwrapped.scene["robot"]
            base_lin_vel = robot.data.root_lin_vel_w  # (N, 3)
            base_ang_vel = robot.data.root_ang_vel_w  # (N, 3)
            root_quat = robot.data.root_quat_w  # (N, 4)
            
            # Extract metrics
            metrics["velocities_x"].append(base_lin_vel[:, 0].mean().item())
            metrics["velocities_y"].append(base_lin_vel[:, 1].mean().item())
            metrics["velocities_z"].append(base_lin_vel[:, 2].mean().item())
            metrics["angular_vel_z"].append(base_ang_vel[:, 2].mean().item())
            
            # Get roll/pitch from quaternion
            # quat: (w, x, y, z) or (x, y, z, w) - need to check
            # Assuming (w, x, y, z) format
            w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
            roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = torch.asin(torch.clamp(2*(w*y - z*x), -1, 1))
            
            metrics["roll"].append(roll.abs().mean().item())
            metrics["pitch"].append(pitch.abs().mean().item())
            
            # Get style similarity from extras
            if hasattr(env.unwrapped, "extras"):
                extras = env.unwrapped.extras
                if "metrics/style_text_sim" in extras:
                    metrics["text_sims"].append(extras["metrics/style_text_sim"].item())
                if "metrics/style_centroid_sim" in extras:
                    metrics["centroid_sims"].append(extras["metrics/style_centroid_sim"].item())
            
            # Track episode lengths
            episode_step_counts += 1
            if dones.any():
                for i in range(env.unwrapped.num_envs):
                    if dones[i]:
                        metrics["episode_lengths"].append(episode_step_counts[i].item())
                        episode_step_counts[i] = 0
            
            # Record video frame
            if record_video:
                base_env = env.unwrapped
                frame = base_env.render()
                if frame is not None:
                    frames.append(frame)
                
                # Update camera to follow robot
                robot_pos_w = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                robot_quat_w = base_env.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
                cam_eye = (robot_pos_w[0] + 3.0, robot_pos_w[1] + 3.0, robot_pos_w[2] + 2.0)
                cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
                base_env.sim.set_camera_view(eye=cam_eye, target=cam_target)
    
    # Save video
    if record_video and frames and video_path:
        print(f"    Saving video to: {video_path}")
        writer = imageio.get_writer(video_path, fps=50)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    
    # Compute summary statistics
    summary = {
        "onomatopoeia": onomatopoeia,
        "mean_velocity_x": np.mean(metrics["velocities_x"]),
        "std_velocity_x": np.std(metrics["velocities_x"]),
        "mean_velocity_y": np.mean(metrics["velocities_y"]),
        "mean_velocity_z": np.mean(metrics["velocities_z"]),
        "mean_angular_vel_z": np.mean(metrics["angular_vel_z"]),
        "mean_roll": np.mean(metrics["roll"]),
        "mean_pitch": np.mean(metrics["pitch"]),
        "mean_episode_length": np.mean(metrics["episode_lengths"]) if metrics["episode_lengths"] else num_steps,
        "mean_text_sim": np.mean(metrics["text_sims"]) if metrics["text_sims"] else 0.0,
        "mean_centroid_sim": np.mean(metrics["centroid_sims"]) if metrics["centroid_sims"] else 0.0,
    }
    
    return summary, metrics


def main():
    """Evaluate style per onomatopoeia."""
    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # Specify directory for logging experiments (relative to REPO_ROOT, not cwd)
    log_root_path = os.path.join(REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")
    
    # Update agent config from the loaded run
    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    assert os.path.exists(log_agent_cfg_file_path), f"Agent config file not found: {log_agent_cfg_file_path}"
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)
    
    # Get history_length from agent config if not specified in CLI
    history_length = args_cli.history_length
    if history_length == 0 and "policy" in log_agent_cfg_dict:
        history_length = log_agent_cfg_dict["policy"].get("history_length", 0)
        print(f"[INFO] Using history_length={history_length} from agent config")
    
    # Create environment - always use rgb_array for video recording
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if history_length > 0:
        print(f"[INFO] Using RslRlVecEnvHistoryWrapper with history_length={history_length}")
        env = RslRlVecEnvHistoryWrapper(env, history_length=history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    
    # Load model
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # Get inference policy
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    # Output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Video directory
    if args_cli.video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate each onomatopoeia
    all_results = []
    interrupted = False

    print("\n" + "="*60)
    print("STYLE EVALUATION PER ONOMATOPOEIA")
    print("="*60)
    
    try:
        for onomatopoeia in ONOMATOPOEIA_LIST:
            print(f"\n--- Evaluating: {onomatopoeia} ---")
            
            # Video path for this onomatopoeia
            video_path = None
            if args_cli.video:
                # Use safe filename (replace problematic characters)
                safe_name = onomatopoeia.replace("/", "_").replace("\\", "_")
                video_path = str(video_dir / f"{timestamp}_{safe_name}.mp4")
            
            summary, raw_metrics = evaluate_onomatopoeia(
                env, policy, onomatopoeia, 
                num_steps=args_cli.eval_steps if not args_cli.video else args_cli.video_length,
                device=env.unwrapped.device,
                record_video=args_cli.video,
                video_path=video_path
            )
            
            all_results.append(summary)
            
            print(f"  Mean Vel X: {summary['mean_velocity_x']:.3f} ± {summary['std_velocity_x']:.3f}")
            print(f"  Mean Angular Vel Z: {summary['mean_angular_vel_z']:.3f}")
            print(f"  Mean Roll: {summary['mean_roll']:.4f}")
            print(f"  Mean Pitch: {summary['mean_pitch']:.4f}")
            print(f"  Mean Text Sim: {summary['mean_text_sim']:.4f}")
            print(f"  Mean Centroid Sim: {summary['mean_centroid_sim']:.4f}")
            print(f"  Mean Episode Length: {summary['mean_episode_length']:.1f}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Ctrl+C detected. Stopping evaluation early...")
    
    # Save results (if any were collected)
    if all_results:
        results_file = output_dir / "style_evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Results saved to: {results_file}")
    else:
        print("\n[INFO] No results to save (evaluation interrupted before any episode completed).")
    
    if args_cli.video and all_results:
        print(f"[INFO] Videos saved to: {video_dir}")
    
    # Print summary table if we have data
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(f"{'Onomatopoeia':<15} {'Vel X':>8} {'AngVel Z':>10} {'Roll':>8} {'Pitch':>8} {'TextSim':>10} {'CentrSim':>10}")
        print("-"*80)
        for r in all_results:
            print(f"{r['onomatopoeia']:<15} {r['mean_velocity_x']:>8.3f} {r['mean_angular_vel_z']:>10.3f} "
                  f"{r['mean_roll']:>8.4f} {r['mean_pitch']:>8.4f} {r['mean_text_sim']:>10.4f} {r['mean_centroid_sim']:>10.4f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
