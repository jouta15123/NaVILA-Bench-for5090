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
parser.add_argument("--log_root", type=str, default=None, help="Override log root directory containing rsl_rl runs.")
parser.add_argument("--use_log_env", action="store_true", default=True, help="Use env.yaml from the run if available.")
parser.add_argument("--use_base_policy", action="store_true", default=False, help="Load base policy for residual actor.")
parser.add_argument("--compute_hoyo_error", action="store_true", default=False)
parser.add_argument("--hoyo_root", type=str, default=None)
parser.add_argument("--hoyo_samples_per_label", type=int, default=5)
parser.add_argument("--hoyo_eval_interval", type=int, default=10)
parser.add_argument("--hoyo_metric", type=str, choices=["dtw", "l2"], default="dtw")
parser.add_argument("--hoyo_dtw_band", type=int, default=10)
parser.add_argument("--hoyo_seed", type=int, default=42)

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
import random
import yaml
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

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
import omni.isaac.leggedloco.leggedloco.mdp as mdp
from hoyo_v1_1.models.common import HoyoInstructionDataset, apply_normalization_from_stats

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


def resolve_log_root_path(exp_name: str, load_run: str, override: str | None) -> str:
    candidates = []
    if override:
        candidates.append(override)
    candidates.append(os.path.join(REPO_ROOT, "logs", "rsl_rl"))
    candidates.append(os.path.join(NAVILA_ROOT, "logs", "rsl_rl"))
    for root in candidates:
        if not root:
            continue
        run_dir = os.path.join(root, exp_name, load_run)
        if os.path.isdir(run_dir):
            return root
    return candidates[0] if candidates else os.path.join(REPO_ROOT, "logs", "rsl_rl")


def resolve_checkpoint_path(path: str) -> str:
    if not path:
        return path
    if os.path.exists(path):
        return path
    # Docker path translation
    if path.startswith("/home/jouta/NaVILA-Bench"):
        alt = path.replace("/home/jouta/NaVILA-Bench", "/workspace/NaVILA-Bench")
        if os.path.exists(alt):
            return alt
    return path


def infer_obs_dims_from_checkpoint(ckpt_path: str) -> tuple[int | None, int | None]:
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


def maybe_disable_critic_style(env_cfg, actor_in: int | None, critic_in: int | None, style_dim: int = 512) -> None:
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


def resolve_norm_stats_path(hoyo_root: Path, run_name: str) -> Path:
    candidates = [
        hoyo_root / "joint_training_results" / "normalization_stats.json",
        hoyo_root / "joint_training_results" / run_name / "normalization_stats.json",
        hoyo_root / "data" / "normalization_stats.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Normalization stats not found under {hoyo_root}")


def build_hoyo_reference_cache(
    hoyo_root: Path,
    labels: list[str],
    target_len: int,
    samples_per_label: int,
    stats_path: Path,
    seed: int,
) -> dict:
    dataset = HoyoInstructionDataset(hoyo_root, labels, target_len=target_len, is_train=False, use_aug=False)
    apply_normalization_from_stats(dataset, stats_path)
    rng = random.Random(seed)
    cache = {}
    for lab in labels:
        cache[lab] = []
        if samples_per_label <= 0:
            continue
        for _ in range(samples_per_label):
            # dataset.get_sample applies cropping + centering + normalization
            cache[lab].append(dataset.get_sample(lab))
    return cache


def _frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: (14, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band: int | None = None) -> float:
    # seq_a, seq_b: (T, 14, 2)
    T = seq_a.shape[0]
    U = seq_b.shape[0]
    dp = np.full((T + 1, U + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        if band is None:
            j_start, j_end = 1, U
        else:
            j_start = max(1, i - band)
            j_end = min(U, i + band)
        for j in range(j_start, j_end + 1):
            cost = _frame_distance(seq_a[i - 1], seq_b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    # Normalize by path length to keep scale comparable across runs
    return float(dp[T, U] / max(1, T + U))


def l2_sequence_error(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    T = min(seq_a.shape[0], seq_b.shape[0])
    diff = seq_a[:T] - seq_b[:T]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def compute_hoyo_error(
    seq: np.ndarray,
    refs: list[np.ndarray],
    metric: str,
    band: int,
) -> float:
    if not refs:
        return float("nan")
    if metric == "dtw":
        return min(dtw_distance(seq, ref, band=band) for ref in refs)
    return min(l2_sequence_error(seq, ref) for ref in refs)


def _set_style_for_envs(style_gen, env_ids, onomatopoeia: str) -> None:
    z_onm, centroid = style_gen.style_module.encode_instruction(onomatopoeia)
    z_onm = z_onm.squeeze(0)
    centroid = centroid.squeeze(0)
    for env_id in env_ids:
        env_id = int(env_id)
        style_gen.current_texts[env_id] = onomatopoeia
        style_gen.style_latents[env_id] = z_onm
        style_gen.centroids[env_id] = centroid
        style_gen._command[env_id, :512] = z_onm
        style_gen._command[env_id, 512:] = centroid


def _detect_falls(env, dones: torch.Tensor, extras: dict | None) -> torch.Tensor:
    done_mask = dones.bool()
    if not done_mask.any():
        return done_mask
    term_mgr = getattr(env.unwrapped, "termination_manager", None)
    if term_mgr is not None and hasattr(term_mgr, "find_terms") and hasattr(term_mgr, "get_term"):
        try:
            term_names = term_mgr.find_terms("base_contact")
            if term_names:
                base_contact = term_mgr.get_term(term_names[0]).bool()
                return done_mask & base_contact
        except Exception:
            pass
    time_outs = None
    if isinstance(extras, dict):
        time_outs = extras.get("time_outs", None)
    if time_outs is None and term_mgr is not None:
        time_outs = getattr(term_mgr, "time_outs", None)
    if time_outs is not None:
        return done_mask & (~time_outs.bool())
    return done_mask


def evaluate_onomatopoeia(env, policy, onomatopoeia: str, num_steps: int, device, 
                          record_video: bool = False, video_path: str = None,
                          teacher_cache: dict | None = None,
                          hoyo_metric: str = "dtw",
                          hoyo_eval_interval: int = 10,
                          hoyo_dtw_band: int = 10):
    """Evaluate a single onomatopoeia and collect metrics."""
    
    # Force set the style command for all envs
    style_gen = env.unwrapped.command_manager._terms["style_command"]
    
    # Reset environment and rebuild observation with history buffer (if enabled)
    # Note: some wrappers return the raw policy obs on reset; calling
    # get_observations() aligns shapes with history-augmented policies.
    env.reset()
    
    # Manually set the onomatopoeia for all environments AFTER reset
    # (reset triggers command resampling, so set style after it).
    _set_style_for_envs(style_gen, range(env.unwrapped.num_envs), onomatopoeia)
    
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
        "hoyo_errors": [],
    }
    
    episode_step_counts = torch.zeros(env.unwrapped.num_envs, device=device)
    fall_count = 0
    episode_count = 0
    
    for step in tqdm(range(num_steps), desc=f"Evaluating {onomatopoeia}"):
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
            
            # Track episode lengths and falls
            episode_step_counts += 1
            if dones.any():
                for i in range(env.unwrapped.num_envs):
                    if dones[i]:
                        metrics["episode_lengths"].append(episode_step_counts[i].item())
                        episode_step_counts[i] = 0
                done_ids = torch.where(dones)[0].tolist()
                fall_mask = _detect_falls(env, dones, infos if isinstance(infos, dict) else None)
                fall_count += int(fall_mask.sum().item())
                episode_count += int(dones.sum().item())
                # Enforce fixed style after reset
                _set_style_for_envs(style_gen, done_ids, onomatopoeia)

            # HOYO error (optional, interval-based)
            if teacher_cache is not None and (step % max(1, hoyo_eval_interval) == 0):
                try:
                    # Use HOYO-compatible preprocessing (matching HOYO dataset format)
                    buf_2d = style_gen.style_module.get_buffer_for_hoyo_comparison()
                    warm_mask = style_gen.style_module.warmup_counter >= style_gen.style_module.warmup_frames
                    buf_np = buf_2d.detach().cpu().numpy()
                    warm_np = warm_mask.detach().cpu().numpy()
                    refs = teacher_cache.get(onomatopoeia, [])
                    for env_id in range(env.unwrapped.num_envs):
                        if not warm_np[env_id]:
                            continue
                        err = compute_hoyo_error(
                            buf_np[env_id],
                            refs,
                            metric=hoyo_metric,
                            band=hoyo_dtw_band,
                        )
                        if not np.isnan(err):
                            metrics["hoyo_errors"].append(err)
                except Exception:
                    pass
            
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
        "fall_rate": (fall_count / episode_count) if episode_count > 0 else 0.0,
        "fall_count": fall_count,
        "episode_count": episode_count,
        "mean_hoyo_error": np.mean(metrics["hoyo_errors"]) if metrics["hoyo_errors"] else None,
        "std_hoyo_error": np.std(metrics["hoyo_errors"]) if metrics["hoyo_errors"] else None,
    }
    
    return summary, metrics


def save_results_to_csv(all_results, output_path, timestamp):
    """Save evaluation results to CSV format.

    Args:
        all_results: List of summary dicts from evaluate_onomatopoeia()
        output_path: Path to output directory
        timestamp: Timestamp string for filename
    """
    try:
        import pandas as pd

        # Create DataFrame from results
        df = pd.DataFrame(all_results)

        # Reorder columns for better readability
        column_order = [
            "onomatopoeia",
            "mean_velocity_x",
            "std_velocity_x",
            "mean_velocity_y",
            "mean_velocity_z",
            "mean_angular_vel_z",
            "mean_roll",
            "mean_pitch",
            "mean_episode_length",
            "mean_text_sim",
            "mean_centroid_sim",
            "fall_rate",
            "fall_count",
            "episode_count",
            "mean_hoyo_error",
            "std_hoyo_error",
        ]

        # Keep only existing columns
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]

        # Save to CSV with timestamp
        csv_filename = f"{timestamp}_metrics.csv"
        csv_path = output_path / csv_filename
        df.to_csv(csv_path, index=False, encoding="utf-8")

        print(f"[INFO] Metrics saved to CSV: {csv_path}")
        return csv_path

    except Exception as exc:
        print(f"[WARN] Failed to save CSV: {exc}")
        return None


def main():
    """Evaluate style per onomatopoeia."""
    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # Specify directory for logging experiments (relative to REPO_ROOT, not cwd)
    log_root_base = resolve_log_root_path(agent_cfg.experiment_name, args_cli.load_run, args_cli.log_root)
    log_root_path = os.path.join(log_root_base, agent_cfg.experiment_name)
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    # Update env config from the loaded run (to match observation sizes)
    log_env_cfg_file_path = os.path.join(log_dir, "params", "env.yaml")
    if args_cli.use_log_env and os.path.exists(log_env_cfg_file_path):
        try:
            try:
                log_env_cfg_dict = load_yaml(log_env_cfg_file_path)
            except Exception as exc:
                print(f"[WARN] Failed to load env.yaml with default loader: {exc}")
                log_env_cfg_dict = load_yaml_with_slices(log_env_cfg_file_path)
            try:
                update_class_from_dict(env_cfg, log_env_cfg_dict)
                print(f"[INFO] Using env config from: {log_env_cfg_file_path}")
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
        # Ensure eval uses CLI num_envs even if env.yaml overrides it.
        enforce_num_envs(env_cfg, args_cli.num_envs)
    
    # Update agent config from the loaded run
    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    if not os.path.exists(log_agent_cfg_file_path):
        raise FileNotFoundError(
            f"Agent config file not found: {log_agent_cfg_file_path}\n"
            f"Hint: pass --log_root to point at the directory containing rsl_rl/{agent_cfg.experiment_name}/..."
        )
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)

    # Force override num_envs back to CLI arg (because env.yaml might have overwritten it with 4096 etc.)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        print(f"[INFO] Forced num_envs to {env_cfg.scene.num_envs} (from CLI) overriding env.yaml")

    # Re-apply CLI checkpoint override if provided
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    # Fix base policy checkpoint path if needed (Docker path translation)
    base_ckpt = getattr(agent_cfg.policy, "base_policy_checkpoint", None)
    if base_ckpt:
        agent_cfg.policy.base_policy_checkpoint = resolve_checkpoint_path(base_ckpt)
    if not args_cli.use_base_policy:
        agent_cfg.policy.base_policy_checkpoint = None
        print("[INFO] Base policy loading disabled for evaluation.")

    # Resolve checkpoint path early to align observation sizes
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    actor_in, critic_in = infer_obs_dims_from_checkpoint(resume_path)
    style_dim = getattr(agent_cfg.policy, "style_dim", 512)
    maybe_disable_critic_style(env_cfg, actor_in, critic_in, style_dim=style_dim)
    
    # Get history_length from agent config if not specified in CLI
    history_length = args_cli.history_length
    if history_length == 0 and "policy" in log_agent_cfg_dict:
        history_length = log_agent_cfg_dict["policy"].get("history_length", 0)
        print(f"[INFO] Using history_length={history_length} from agent config")
    
    # Disable terrain curriculum for evaluation
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
        print("[INFO] Disabled terrain_levels curriculum for evaluation.")

    # Override reset_base to use uniform reset instead of terrain-based reset
    # This avoids "valid flat patches" errors during evaluation where terrain might result in empty patches
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
        env_cfg.events.reset_base.func = mdp.reset_root_state_uniform
        env_cfg.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        }
        print("[INFO] Overrode reset_base to use reset_root_state_uniform for evaluation.")
    
    # Create environment - always use rgb_array for video recording
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if history_length > 0:
        print(f"[INFO] Using RslRlVecEnvHistoryWrapper with history_length={history_length}")
        env = RslRlVecEnvHistoryWrapper(env, history_length=history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    
    # Load model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # Get inference policy
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Optional HOYO reference cache
    teacher_cache = None
    if args_cli.compute_hoyo_error:
        try:
            hoyo_root = Path(args_cli.hoyo_root) if args_cli.hoyo_root else (Path(NAVILA_ROOT) / "hoyo_v1_1")
            run_name = env.unwrapped.command_manager._terms["style_command"].style_module.run_name
            stats_path = resolve_norm_stats_path(hoyo_root, run_name)
            teacher_cache = build_hoyo_reference_cache(
                hoyo_root=hoyo_root,
                labels=ONOMATOPOEIA_LIST,
                target_len=60,
                samples_per_label=args_cli.hoyo_samples_per_label,
                stats_path=stats_path,
                seed=args_cli.hoyo_seed,
            )
            print(f"[INFO] HOYO references loaded from: {hoyo_root}")
            print(f"[INFO] HOYO stats path: {stats_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load HOYO references: {exc}")
            teacher_cache = None
    
    # Output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Video directory
    if args_cli.video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
    
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
                video_path=video_path,
                teacher_cache=teacher_cache,
                hoyo_metric=args_cli.hoyo_metric,
                hoyo_eval_interval=args_cli.hoyo_eval_interval,
                hoyo_dtw_band=args_cli.hoyo_dtw_band,
            )
            
            all_results.append(summary)
            
            print(f"  Mean Vel X: {summary['mean_velocity_x']:.3f} ± {summary['std_velocity_x']:.3f}")
            print(f"  Mean Angular Vel Z: {summary['mean_angular_vel_z']:.3f}")
            print(f"  Mean Roll: {summary['mean_roll']:.4f}")
            print(f"  Mean Pitch: {summary['mean_pitch']:.4f}")
            print(f"  Mean Text Sim: {summary['mean_text_sim']:.4f}")
            print(f"  Mean Centroid Sim: {summary['mean_centroid_sim']:.4f}")
            print(f"  Mean Episode Length: {summary['mean_episode_length']:.1f}")
            print(f"  Fall Rate: {summary['fall_rate']:.3f} ({summary['fall_count']}/{summary['episode_count']})")
            if summary["mean_hoyo_error"] is not None:
                print(f"  Mean HOYO Error: {summary['mean_hoyo_error']:.4f} (std={summary['std_hoyo_error']:.4f})")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Ctrl+C detected. Stopping evaluation early...")
    
    # Save results (if any were collected)
    if all_results:
        results_file = output_dir / "style_evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Results saved to: {results_file}")

        # Save CSV
        try:
            save_results_to_csv(all_results, output_dir, timestamp)
        except Exception as exc:
            print(f"[WARN] Failed to save CSV: {exc}")
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
