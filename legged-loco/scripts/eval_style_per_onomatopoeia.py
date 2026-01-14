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

HOYO_UNIFIED_FPS = 50

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
parser.add_argument("--no_base_policy", action="store_true", default=False, help="Disable base policy loading (loaded by default for residual actor).")
parser.add_argument(
    "--compute_hoyo_error",
    action="store_true",
    default=False,
    help="(Legacy) Enable HOYO error computation. HOYO error is enabled by default unless --no_hoyo_error is set.",
)
parser.add_argument(
    "--no_hoyo_error",
    action="store_true",
    default=False,
    help="Disable HOYO error computation (enabled by default).",
)
parser.add_argument("--hoyo_root", type=str, default=None)
parser.add_argument("--hoyo_samples_per_label", type=int, default=5)
parser.add_argument("--hoyo_eval_interval", type=int, default=10)
parser.add_argument("--hoyo_metric", type=str, choices=["dtw", "l2"], default="dtw")
parser.add_argument("--hoyo_dtw_band", type=int, default=10)
parser.add_argument("--hoyo_seed", type=int, default=42)
parser.add_argument(
    "--no_hoyo_yaw_correction",
    action="store_true",
    default=False,
    help="Disable yaw correction when computing HOYO error (enabled by default).",
)
parser.add_argument(
    "--style_list",
    type=str,
    default=None,
    help="Comma-separated style list to evaluate (overrides default onomatopoeia list).",
)
parser.add_argument(
    "--base_velocity_mode",
    type=str,
    choices=["env", "fixed", "style_table"],
    default="fixed",
    help="How to set base_velocity command during evaluation.",
)
parser.add_argument("--base_lin_vel_x", type=float, default=0.5, help="Fixed command lin_vel_x.")
parser.add_argument("--base_lin_vel_y", type=float, default=0.0, help="Fixed command lin_vel_y.")
parser.add_argument("--base_ang_vel_z", type=float, default=0.0, help="Fixed command ang_vel_z.")
parser.add_argument("--base_heading", type=float, default=0.0, help="Fixed command heading.")
parser.add_argument(
    "--style_speed_table",
    type=str,
    default=None,
    help="JSON mapping for style -> speed. Value can be float (lin_vel_x) or dict.",
)
parser.add_argument("--debug_quat", action="store_true", default=False, help="Log one quaternion to verify order.")
parser.add_argument("--debug_dones", action="store_true", default=False, help="Log done/reset signals once.")
parser.add_argument(
    "--debug_hoyo_range",
    action="store_true",
    default=False,
    help="Log H1/HOYO axis ranges for GIF comparison (processed input to motion encoder).",
)

# HOYO mapping visualization
parser.add_argument(
    "--save_hoyo_gif",
    action="store_true",
    default=False,
    help="Save HOYO 2D keypoint mapping as animated GIF for each onomatopoeia.",
)
parser.add_argument(
    "--save_hoyo_comparison_gif",
    action="store_true",
    default=False,
    help="Save side-by-side comparison GIF (H1 vs HOYO reference).",
)
parser.add_argument(
    "--hoyo_gif_fps",
    type=int,
    default=HOYO_UNIFIED_FPS,
    help="FPS for HOYO GIF animation.",
)
parser.add_argument(
    "--hoyo_gif_stride",
    type=int,
    default=2,
    help="Frame stride for HOYO GIF (skip frames for smaller file).",
)

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
import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from omni.isaac.leggedloco.config import *
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper
from omni.isaac.leggedloco.leggedloco.mdp.style_module import INSTRUCTION_ONOMATOPEIA
import omni.isaac.leggedloco.leggedloco.mdp as mdp

# HOYO visualization (optional)
try:
    from hoyo_viz_utils import save_hoyo_gif, save_hoyo_comparison_gif
    HOYO_VIZ_AVAILABLE = True
except ImportError:
    HOYO_VIZ_AVAILABLE = False
from hoyo_v1_1.models.common import HoyoInstructionDataset, apply_normalization_from_stats

# List of onomatopoeia to evaluate
ONOMATOPOEIA_LIST = INSTRUCTION_ONOMATOPEIA


def _get_command_tensor(command_term):
    for attr in ("_command", "command", "commands"):
        if hasattr(command_term, attr):
            tensor = getattr(command_term, attr)
            if torch.is_tensor(tensor):
                return tensor
    return None


def _as_env_ids(env_ids, num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    if torch.is_tensor(env_ids):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.tensor(list(env_ids), device=device, dtype=torch.long)


def _set_base_velocity_for_envs(command_term, env_ids, cmd: dict) -> bool:
    cmd_tensor = _get_command_tensor(command_term)
    if cmd_tensor is None:
        return False
    env_ids_tensor = _as_env_ids(env_ids, cmd_tensor.shape[0], cmd_tensor.device)
    if env_ids_tensor.numel() == 0:
        return True
    lin_x = float(cmd.get("lin_vel_x", 0.0))
    lin_y = float(cmd.get("lin_vel_y", 0.0))
    ang_z = float(cmd.get("ang_vel_z", 0.0))
    heading = float(cmd.get("heading", 0.0))
    if cmd_tensor.shape[1] > 0:
        cmd_tensor[env_ids_tensor, 0] = lin_x
    if cmd_tensor.shape[1] > 1:
        cmd_tensor[env_ids_tensor, 1] = lin_y
    if cmd_tensor.shape[1] > 2:
        cmd_tensor[env_ids_tensor, 2] = ang_z
    if cmd_tensor.shape[1] > 3:
        cmd_tensor[env_ids_tensor, 3] = heading
    return True


def _normalize_style_speed_entry(value, default_cmd: dict) -> dict:
    if isinstance(value, (int, float)):
        return {
            "lin_vel_x": float(value),
            "lin_vel_y": default_cmd["lin_vel_y"],
            "ang_vel_z": default_cmd["ang_vel_z"],
            "heading": default_cmd["heading"],
        }
    if isinstance(value, (list, tuple)) and len(value) > 0:
        cmd = dict(default_cmd)
        cmd["lin_vel_x"] = float(value[0])
        if len(value) > 1:
            cmd["lin_vel_y"] = float(value[1])
        if len(value) > 2:
            cmd["ang_vel_z"] = float(value[2])
        if len(value) > 3:
            cmd["heading"] = float(value[3])
        return cmd
    if isinstance(value, dict):
        cmd = dict(default_cmd)
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
            if key in value and value[key] is not None:
                cmd[key] = float(value[key])
        return cmd
    return dict(default_cmd)


def load_style_speed_table(path: str | None, default_cmd: dict) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load style_speed_table: {exc}")
        return {}
    table = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            table[key] = _normalize_style_speed_entry(value, default_cmd)
    return table


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


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_seed_env(env, seed: int | None) -> None:
    if seed is None:
        return
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        if hasattr(target, "reset"):
            try:
                target.reset(seed=seed)
                return
            except TypeError:
                pass
        if hasattr(target, "seed"):
            try:
                target.seed(seed)
                return
            except Exception:
                pass


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
        hoyo_root / "joint_training_results" / run_name / "normalization_stats.json",
        hoyo_root / "joint_training_results" / "normalization_stats.json",
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
    centering: str = "first_frame_com",
) -> dict:
    dataset = HoyoInstructionDataset(
        hoyo_root,
        labels,
        target_len=target_len,
        is_train=False,
        use_aug=False,
        centering=centering,
    )
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


def _axis_range_stats(arr: np.ndarray) -> dict:
    """Return min/max/mean for x/y axes of (T, 14, 2) array."""
    x = arr[..., 0]
    y = arr[..., 1]
    return {
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_mean": float(np.mean(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
    }


def _set_style_for_envs(style_gen, env_ids, onomatopoeia: str, style_cache: dict | None = None) -> None:
    if style_cache is not None and onomatopoeia in style_cache:
        z_onm, teacher_motion = style_cache[onomatopoeia]
    else:
        z_onm, teacher_motion = style_gen.style_module.encode_instruction(onomatopoeia)
        z_onm = z_onm.squeeze(0)
        teacher_motion = teacher_motion.squeeze(0)
        if style_cache is not None:
            style_cache[onomatopoeia] = (z_onm, teacher_motion)
    env_ids_tensor = _as_env_ids(env_ids, style_gen.num_envs, style_gen._command.device)
    if env_ids_tensor.numel() == 0:
        return
    for env_id in env_ids_tensor.tolist():
        style_gen.current_texts[int(env_id)] = onomatopoeia
    style_gen.style_latents[env_ids_tensor] = z_onm
    style_gen.teacher_motion_latents[env_ids_tensor] = teacher_motion
    style_gen._command[env_ids_tensor, :512] = z_onm
    style_gen._command[env_ids_tensor, 512:] = teacher_motion


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
                          record_video: bool = False, video_path: str | None = None,
                          teacher_cache: dict | None = None,
                          hoyo_metric: str = "dtw",
                          hoyo_eval_interval: int = 10,
                          hoyo_dtw_band: int = 10,
                          hoyo_apply_yaw_correction: bool = True,
                          base_velocity_mode: str = "env",
                          base_cmd_default: dict | None = None,
                          style_speed_table: dict | None = None,
                          style_cache: dict | None = None,
                          debug_quat: bool = False,
                          debug_dones: bool = False,
                          collect_hoyo_2d: bool = False):
    """Evaluate a single onomatopoeia and collect metrics.
    
    Returns:
        tuple: (metrics_dict, hoyo_2d_history) where hoyo_2d_history is a list of (14, 2) arrays
               if collect_hoyo_2d=True, else None.
    """
    
    # Force set the style command for all envs
    style_gen = env.unwrapped.command_manager._terms["style_command"]
    base_cmd_term = env.unwrapped.command_manager._terms.get("base_velocity", None)
    base_cmd_default = base_cmd_default or {
        "lin_vel_x": 0.5,
        "lin_vel_y": 0.0,
        "ang_vel_z": 0.0,
        "heading": 0.0,
    }
    style_speed_table = style_speed_table or {}
    
    # Reset environment and rebuild observation with history buffer (if enabled)
    # Note: some wrappers return the raw policy obs on reset; calling
    # get_observations() aligns shapes with history-augmented policies.
    env.reset()
    
    # Manually set the onomatopoeia for all environments AFTER reset
    # (reset triggers command resampling, so set style after it).
    _set_style_for_envs(style_gen, range(env.unwrapped.num_envs), onomatopoeia, style_cache=style_cache)

    # Optionally override base_velocity command (fixed or style-dependent)
    if base_cmd_term is not None and base_velocity_mode != "env":
        if base_velocity_mode == "style_table":
            base_cmd = style_speed_table.get(onomatopoeia, base_cmd_default)
        else:
            base_cmd = base_cmd_default
        _set_base_velocity_for_envs(base_cmd_term, range(env.unwrapped.num_envs), base_cmd)
    
    obs, _ = env.get_observations()
    
    # Video recording setup
    writer = None
    if record_video and video_path:
        base_env = env.unwrapped
        # Set initial camera
        robot_pos_w = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        cam_eye = (robot_pos_w[0] + 3.0, robot_pos_w[1] + 3.0, robot_pos_w[2] + 2.0)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        base_env.sim.set_camera_view(eye=cam_eye, target=cam_target)
        writer = imageio.get_writer(video_path, fps=HOYO_UNIFIED_FPS)
    
    # Metrics storage
    metrics = {
        "velocities_x": [],
        "velocities_y": [],
        "velocities_z": [],
        "angular_vel_z": [],
        "cmd_lin_vel_x": [],
        "cmd_lin_vel_y": [],
        "cmd_ang_vel_z": [],
        "cmd_heading": [],
        "roll": [],
        "pitch": [],
        "episode_lengths": [],
        "text_sims": [],
        "teacher_motion_sims": [],
        "hoyo_errors": [],
    }
    # Per-env accumulators (lightweight)
    num_envs = env.unwrapped.num_envs
    vel_x_sum = torch.zeros(num_envs, device=device)
    vel_x_count = torch.zeros(num_envs, device=device)
    
    episode_step_counts = torch.zeros(env.unwrapped.num_envs, device=device)
    fall_count = 0
    episode_count = 0
    
    # HOYO 2D history for visualization (only env 0)
    hoyo_2d_history = [] if collect_hoyo_2d else None
    style_module = style_gen.style_module if hasattr(style_gen, "style_module") else None
    
    quat_logged = False
    done_logged = False
    for step in tqdm(range(num_steps), desc=f"Evaluating {onomatopoeia}"):
        # Use no_grad instead of inference_mode to avoid creating inference tensors
        # that later reject in-place updates during env.reset().
        with torch.no_grad():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            
            # Get robot state
            robot = env.unwrapped.scene["robot"]
            # Use body-frame velocities for consistency with training rewards.
            base_lin_vel = getattr(robot.data, "root_lin_vel_b", robot.data.root_lin_vel_w)  # (N, 3)
            base_ang_vel = getattr(robot.data, "root_ang_vel_b", robot.data.root_ang_vel_w)  # (N, 3)
            root_quat = robot.data.root_quat_w  # (N, 4)

            # Log commanded base_velocity (if available)
            cmd = env.unwrapped.command_manager.get_command("base_velocity")
            if cmd is not None and cmd.numel() > 0:
                metrics["cmd_lin_vel_x"].append(cmd[:, 0].mean().item())
                if cmd.shape[1] > 1:
                    metrics["cmd_lin_vel_y"].append(cmd[:, 1].mean().item())
                if cmd.shape[1] > 2:
                    metrics["cmd_ang_vel_z"].append(cmd[:, 2].mean().item())
                if cmd.shape[1] > 3:
                    metrics["cmd_heading"].append(cmd[:, 3].mean().item())
            
            # Extract metrics
            metrics["velocities_x"].append(base_lin_vel[:, 0].mean().item())
            metrics["velocities_y"].append(base_lin_vel[:, 1].mean().item())
            metrics["velocities_z"].append(base_lin_vel[:, 2].mean().item())
            metrics["angular_vel_z"].append(base_ang_vel[:, 2].mean().item())
            
            # Get roll/pitch via IsaacLab math_utils (expects wxyz)
            rpy = math_utils.euler_xyz_from_quat(root_quat)
            if isinstance(rpy, tuple):
                roll, pitch = rpy[0], rpy[1]
            else:
                roll = rpy[:, 0]
                pitch = rpy[:, 1]
            # Wrap to [-pi, pi] before taking abs to avoid 2π wrap artifacts.
            roll = math_utils.wrap_to_pi(roll)
            pitch = math_utils.wrap_to_pi(pitch)
            
            metrics["roll"].append(roll.abs().mean().item())
            metrics["pitch"].append(pitch.abs().mean().item())
            # Per-env mean velocity (for dispersion diagnostics)
            vel_x_sum += base_lin_vel[:, 0]
            vel_x_count += 1.0

            if debug_quat and not quat_logged:
                q0 = root_quat[0].detach().cpu().numpy()
                rpy_wxyz = quat2eulers(q0[0], q0[1], q0[2], q0[3])
                rpy_xyzw = quat2eulers(q0[3], q0[0], q0[1], q0[2])
                print(f"[DEBUG] root_quat_w[0]={q0} (wxyz? w={q0[0]:.4f}, w_last={q0[3]:.4f})")
                print(f"[DEBUG] rpy(wxyz)={rpy_wxyz} | rpy(xyzw)={rpy_xyzw}")
                quat_logged = True
            
            # Get style similarity from extras
            if hasattr(env.unwrapped, "extras"):
                extras = env.unwrapped.extras
                if "metrics/style_text_sim" in extras:
                    metrics["text_sims"].append(extras["metrics/style_text_sim"].item())
                if "metrics/style_teacher_motion_sim" in extras:
                    metrics["teacher_motion_sims"].append(extras["metrics/style_teacher_motion_sim"].item())
                elif "metrics/style_centroid_sim" in extras:
                    metrics["teacher_motion_sims"].append(extras["metrics/style_centroid_sim"].item())
            
            # Collect HOYO 2D keypoints for visualization (env 0 only)
            if collect_hoyo_2d and style_module is not None:
                try:
                    # Get the 2D projection (last frame from buffer)
                    # get_hoyo_compatible_keymap returns (B, T, 14, 2)
                    full_buffer_2d = style_module.get_hoyo_compatible_keymap(
                        standardize=True, normalize_height=True
                    )
                    latest_frame_2d = full_buffer_2d[:, -1]  # (B, 14, 2)
                    hoyo_2d_history.append(latest_frame_2d[0].detach().cpu().numpy())
                except Exception:
                    pass  # Skip if buffer not ready
            
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
                _set_style_for_envs(style_gen, done_ids, onomatopoeia, style_cache=style_cache)
                if base_cmd_term is not None and base_velocity_mode != "env":
                    if base_velocity_mode == "style_table":
                        base_cmd = style_speed_table.get(onomatopoeia, base_cmd_default)
                    else:
                        base_cmd = base_cmd_default
                    _set_base_velocity_for_envs(base_cmd_term, done_ids, base_cmd)

                if debug_dones and not done_logged:
                    ep_len_buf = getattr(env.unwrapped, "episode_length_buf", None)
                    reset_buf = getattr(env.unwrapped, "reset_buf", None)
                    msg = f"[DEBUG] dones env_ids={done_ids}"
                    if ep_len_buf is not None:
                        msg += f" | episode_length_buf={ep_len_buf[done_ids].detach().cpu().tolist()}"
                    if reset_buf is not None:
                        msg += f" | reset_buf={reset_buf[done_ids].detach().cpu().tolist()}"
                    print(msg)
                    done_logged = True

            # Enforce base_velocity every step to avoid resampling drift
            if base_cmd_term is not None and base_velocity_mode != "env":
                if base_velocity_mode == "style_table":
                    base_cmd = style_speed_table.get(onomatopoeia, base_cmd_default)
                else:
                    base_cmd = base_cmd_default
                _set_base_velocity_for_envs(base_cmd_term, range(env.unwrapped.num_envs), base_cmd)
                obs, _ = env.get_observations()

            # HOYO error (optional, interval-based)
            if teacher_cache is not None and (step % max(1, hoyo_eval_interval) == 0):
                try:
                    # Use HOYO-compatible preprocessing (matching HOYO dataset format)
                    buf_2d = style_gen.style_module.get_buffer_for_hoyo_comparison(
                        apply_yaw_correction=hoyo_apply_yaw_correction,
                        centering="first_frame_com",
                    )
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
            if record_video and writer is not None:
                base_env = env.unwrapped
                frame = base_env.render()
                if frame is not None:
                    writer.append_data(frame)
                
                # Update camera to follow robot
                robot_pos_w = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                robot_quat_w = base_env.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
                cam_eye = (robot_pos_w[0] + 3.0, robot_pos_w[1] + 3.0, robot_pos_w[2] + 2.0)
                cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
                base_env.sim.set_camera_view(eye=cam_eye, target=cam_target)
    
    # Save video
    if writer is not None:
        print(f"    Saving video to: {video_path}")
        writer.close()
    
    # Compute summary statistics
    summary = {
        "onomatopoeia": onomatopoeia,
        "mean_velocity_x": np.mean(metrics["velocities_x"]),
        "std_velocity_x": np.std(metrics["velocities_x"]),
        "mean_velocity_y": np.mean(metrics["velocities_y"]),
        "mean_velocity_z": np.mean(metrics["velocities_z"]),
        "mean_angular_vel_z": np.mean(metrics["angular_vel_z"]),
        "mean_cmd_lin_vel_x": np.mean(metrics["cmd_lin_vel_x"]) if metrics["cmd_lin_vel_x"] else None,
        "mean_cmd_lin_vel_y": np.mean(metrics["cmd_lin_vel_y"]) if metrics["cmd_lin_vel_y"] else None,
        "mean_cmd_ang_vel_z": np.mean(metrics["cmd_ang_vel_z"]) if metrics["cmd_ang_vel_z"] else None,
        "mean_cmd_heading": np.mean(metrics["cmd_heading"]) if metrics["cmd_heading"] else None,
        "mean_roll": np.mean(metrics["roll"]),
        "mean_pitch": np.mean(metrics["pitch"]),
        "mean_episode_length": np.mean(metrics["episode_lengths"]) if metrics["episode_lengths"] else num_steps,
        "mean_text_sim": np.mean(metrics["text_sims"]) if metrics["text_sims"] else 0.0,
        "mean_teacher_motion_sim": np.mean(metrics["teacher_motion_sims"]) if metrics["teacher_motion_sims"] else 0.0,
        "fall_rate": (fall_count / episode_count) if episode_count > 0 else 0.0,
        "fall_count": fall_count,
        "episode_count": episode_count,
        "mean_hoyo_error": np.mean(metrics["hoyo_errors"]) if metrics["hoyo_errors"] else None,
        "std_hoyo_error": np.std(metrics["hoyo_errors"]) if metrics["hoyo_errors"] else None,
    }
    if vel_x_count.sum().item() > 0:
        vel_x_means = (vel_x_sum / vel_x_count.clamp(min=1.0)).detach().cpu().numpy()
        summary["env_velocity_x_mean"] = float(np.mean(vel_x_means))
        summary["env_velocity_x_std"] = float(np.std(vel_x_means))
        summary["env_velocity_x_median"] = float(np.median(vel_x_means))
        summary["env_velocity_x_iqr"] = float(np.percentile(vel_x_means, 75) - np.percentile(vel_x_means, 25))
    
    return summary, metrics, hoyo_2d_history


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
            "mean_cmd_lin_vel_x",
            "mean_cmd_lin_vel_y",
            "mean_cmd_ang_vel_z",
            "mean_cmd_heading",
            "mean_roll",
            "mean_pitch",
            "mean_episode_length",
            "mean_text_sim",
            "mean_teacher_motion_sim",
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
    if args_cli.no_base_policy:
        agent_cfg.policy.base_policy_checkpoint = None
        print("[INFO] Base policy loading disabled for evaluation.")
    else:
        print("[INFO] Base policy will be loaded for residual actor.")

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
        # For heading_fixed tasks, use yaw=0 to match training (robot starts facing forward)
        # For other tasks, use random yaw
        if "heading_fixed" in args_cli.task:
            yaw_range = (0.0, 0.0)
            print("[INFO] Using yaw=0 for heading_fixed task (matching training).")
        else:
            yaw_range = (-3.14, 3.14)
        # Match training reset distribution (no initial velocity/rotation noise).
        env_cfg.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": yaw_range},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        print("[INFO] Overrode reset_base to use reset_root_state_uniform for evaluation.")

    # Prepare base_velocity command settings before environment creation.
    base_cmd_default = {
        "lin_vel_x": args_cli.base_lin_vel_x,
        "lin_vel_y": args_cli.base_lin_vel_y,
        "ang_vel_z": args_cli.base_ang_vel_z,
        "heading": args_cli.base_heading,
    }
    effective_base_velocity_mode = args_cli.base_velocity_mode
    if args_cli.base_velocity_mode == "fixed":
        cmd_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
        if cmd_cfg is not None and hasattr(cmd_cfg, "ranges"):
            cmd_cfg.ranges.lin_vel_x = (args_cli.base_lin_vel_x, args_cli.base_lin_vel_x)
            cmd_cfg.ranges.lin_vel_y = (args_cli.base_lin_vel_y, args_cli.base_lin_vel_y)
            cmd_cfg.ranges.ang_vel_z = (args_cli.base_ang_vel_z, args_cli.base_ang_vel_z)
            if hasattr(cmd_cfg.ranges, "heading"):
                cmd_cfg.ranges.heading = (args_cli.base_heading, args_cli.base_heading)
            # Avoid sampling "standing" commands when fixed velocity is requested.
            if hasattr(cmd_cfg, "rel_standing_envs"):
                cmd_cfg.rel_standing_envs = 0.0
            # Prevent resampling from changing the fixed command.
            cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)
            effective_base_velocity_mode = "env"
            print("[INFO] base_velocity_mode=fixed -> using command generator with constant ranges.")
        else:
            print("[WARN] base_velocity_mode=fixed requested but env_cfg.commands.base_velocity not found.")

    # Create environment - always use rgb_array for video recording
    seed_everything(args_cli.seed)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if history_length > 0:
        print(f"[INFO] Using RslRlVecEnvHistoryWrapper with history_length={history_length}")
        env = RslRlVecEnvHistoryWrapper(env, history_length=history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    maybe_seed_env(env, args_cli.seed)
    
    # Load model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Optimizer state isn't needed for evaluation and may mismatch when base_policy
    # parameters are frozen/absent in the checkpoint.
    ppo_runner.load(resume_path, load_optimizer=False)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # Verify base policy is loaded for ResidualActorCritic
    actor_critic = ppo_runner.alg.actor_critic
    if hasattr(actor_critic, "base_policy"):
        if actor_critic.base_policy is None:
            if args_cli.no_base_policy:
                print("[WARN] Running without base policy (--no_base_policy specified).")
                print("[WARN] ResidualActorCritic without base policy will likely fail to walk!")
            else:
                raise RuntimeError(
                    "ResidualActorCritic requires base_policy but it is None!\n"
                    "This means the base policy checkpoint failed to load or was disabled.\n"
                    "Check the base_policy_checkpoint path in agent config.\n"
                    "If you intentionally want to run without base policy, use --no_base_policy."
                )
        else:
            print("[INFO] Base policy is loaded and ready for evaluation.")
    
    # Get inference policy
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Optional HOYO reference cache (enabled by default)
    teacher_cache = None
    styles_to_eval = ONOMATOPOEIA_LIST
    if args_cli.style_list is not None:
        styles_to_eval = [s.strip() for s in args_cli.style_list.split(",") if s.strip()]
        if not styles_to_eval:
            raise ValueError("style_list is empty after parsing. Provide at least one style.")
        # Validate against known list (warn but allow)
        unknown = [s for s in styles_to_eval if s not in ONOMATOPOEIA_LIST]
        if unknown:
            print(f"[WARN] Unknown styles requested: {unknown}. They will still be evaluated.")
    compute_hoyo_error = True
    if args_cli.no_hoyo_error:
        compute_hoyo_error = False
    elif args_cli.compute_hoyo_error:
        compute_hoyo_error = True
    hoyo_apply_yaw_correction = not args_cli.no_hoyo_yaw_correction
    if compute_hoyo_error:
        try:
            hoyo_root = Path(args_cli.hoyo_root) if args_cli.hoyo_root else (Path(NAVILA_ROOT) / "hoyo_v1_1")
            run_name = env.unwrapped.command_manager._terms["style_command"].style_module.run_name
            stats_path = resolve_norm_stats_path(hoyo_root, run_name)
            style_term = env.unwrapped.command_manager._terms.get("style_command", None)
            buffer_len = 100
            if style_term is not None and hasattr(style_term, "style_module"):
                buffer_len = int(getattr(style_term.style_module, "buffer_len", buffer_len))
            teacher_cache = build_hoyo_reference_cache(
                hoyo_root=hoyo_root,
                labels=styles_to_eval,
                target_len=buffer_len,
                samples_per_label=args_cli.hoyo_samples_per_label,
                stats_path=stats_path,
                seed=args_cli.hoyo_seed,
                centering="first_frame_com",
            )
            print(f"[INFO] HOYO references loaded from: {hoyo_root}")
            print(f"[INFO] HOYO stats path: {stats_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load HOYO references: {exc}")
            teacher_cache = None

    # Precompute style latents to avoid repeated text encoding during resets.
    style_cache = {}
    style_term = env.unwrapped.command_manager._terms.get("style_command", None)
    if style_term is not None:
        for style in styles_to_eval:
            z_onm, teacher_motion = style_term.style_module.encode_instruction(style)
            style_cache[style] = (z_onm.squeeze(0), teacher_motion.squeeze(0))
    
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
    style_speed_table = load_style_speed_table(args_cli.style_speed_table, base_cmd_default)

    print("\n" + "="*60)
    print("STYLE EVALUATION PER ONOMATOPOEIA")
    print("="*60)
    print(f"[INFO] Evaluating {len(styles_to_eval)} styles: {styles_to_eval}")
    if compute_hoyo_error:
        print("[INFO] HOYO error computation: enabled")
        print(f"[INFO] HOYO yaw correction: {'enabled' if hoyo_apply_yaw_correction else 'disabled'}")
    else:
        print("[INFO] HOYO error computation: disabled")
    if effective_base_velocity_mode != "env":
        print(f"[INFO] base_velocity_mode={effective_base_velocity_mode}")
        print(f"[INFO] base_velocity_default={base_cmd_default}")
        if effective_base_velocity_mode == "style_table":
            print(f"[INFO] style_speed_table entries={len(style_speed_table)}")
    
    try:
        for onomatopoeia in styles_to_eval:
            print(f"\n--- Evaluating: {onomatopoeia} ---")
            
            # Video path for this onomatopoeia
            video_path = None
            if args_cli.video:
                # Use safe filename (replace problematic characters)
                safe_name = onomatopoeia.replace("/", "_").replace("\\", "_")
                video_path = str(video_dir / f"{timestamp}_{safe_name}.mp4")
            
            summary, raw_metrics, hoyo_2d_history = evaluate_onomatopoeia(
                env, policy, onomatopoeia, 
                num_steps=args_cli.eval_steps if not args_cli.video else args_cli.video_length,
                device=env.unwrapped.device,
                record_video=args_cli.video,
                video_path=video_path,
                teacher_cache=teacher_cache,
                hoyo_metric=args_cli.hoyo_metric,
                hoyo_eval_interval=args_cli.hoyo_eval_interval,
                hoyo_dtw_band=args_cli.hoyo_dtw_band,
                hoyo_apply_yaw_correction=hoyo_apply_yaw_correction,
                base_velocity_mode=effective_base_velocity_mode,
                base_cmd_default=base_cmd_default,
                style_speed_table=style_speed_table,
                style_cache=style_cache,
                debug_quat=args_cli.debug_quat,
                debug_dones=args_cli.debug_dones,
                collect_hoyo_2d=args_cli.save_hoyo_gif or args_cli.save_hoyo_comparison_gif,
            )
            
            # Save HOYO GIF if requested
            if args_cli.save_hoyo_gif and hoyo_2d_history and HOYO_VIZ_AVAILABLE:
                safe_name = onomatopoeia.replace("/", "_").replace("\\", "_")
                hoyo_gif_dir = output_dir / "hoyo_gifs"
                hoyo_gif_dir.mkdir(exist_ok=True)
                hoyo_gif_path = str(hoyo_gif_dir / f"{timestamp}_{safe_name}_hoyo.gif")
                save_hoyo_gif(
                    hoyo_2d_history,
                    hoyo_gif_path,
                    fps=args_cli.hoyo_gif_fps,
                    stride=args_cli.hoyo_gif_stride,
                    title=f"H1 → HOYO: {onomatopoeia}",
                )
                if args_cli.debug_hoyo_range:
                    h1_arr = np.stack(hoyo_2d_history, axis=0)
                    stats = _axis_range_stats(h1_arr)
                    print(
                        "[DEBUG] H1 axis range (processed): "
                        f"x[{stats['x_min']:.3f}, {stats['x_max']:.3f}] "
                        f"y[{stats['y_min']:.3f}, {stats['y_max']:.3f}] "
                        f"mean(x,y)=({stats['x_mean']:.3f}, {stats['y_mean']:.3f})"
                    )
            
            # Save comparison GIF (H1 vs HOYO reference)
            if args_cli.save_hoyo_comparison_gif and hoyo_2d_history and HOYO_VIZ_AVAILABLE:
                if teacher_cache is not None and onomatopoeia in teacher_cache:
                    hoyo_refs = teacher_cache[onomatopoeia]
                    if hoyo_refs:
                        # Use the first reference sample for comparison
                        hoyo_ref_sample = hoyo_refs[0]  # (T, 14, 2)
                        if args_cli.debug_hoyo_range:
                            stats = _axis_range_stats(hoyo_ref_sample)
                            print(
                                "[DEBUG] HOYO axis range (processed): "
                                f"x[{stats['x_min']:.3f}, {stats['x_max']:.3f}] "
                                f"y[{stats['y_min']:.3f}, {stats['y_max']:.3f}] "
                                f"mean(x,y)=({stats['x_mean']:.3f}, {stats['y_mean']:.3f})"
                            )
                        safe_name = onomatopoeia.replace("/", "_").replace("\\", "_")
                        comparison_gif_dir = output_dir / "hoyo_comparison_gifs"
                        comparison_gif_dir.mkdir(exist_ok=True)
                        comparison_gif_path = str(comparison_gif_dir / f"{timestamp}_{safe_name}_comparison.gif")
                        save_hoyo_comparison_gif(
                            hoyo_2d_history,
                            hoyo_ref_sample,
                            comparison_gif_path,
                            fps=args_cli.hoyo_gif_fps,
                            stride=args_cli.hoyo_gif_stride,
                            title=f"{onomatopoeia}: H1 vs HOYO Reference",
                            loop_hoyo=True,
                        )
                else:
                    print(f"  [WARN] No HOYO reference for {onomatopoeia}. Skipping comparison GIF.")
            
            all_results.append(summary)
            
            print(f"  Mean Vel X: {summary['mean_velocity_x']:.3f} ± {summary['std_velocity_x']:.3f}")
            if summary.get("mean_cmd_lin_vel_x") is not None:
                print(f"  Mean Cmd Vel X: {summary['mean_cmd_lin_vel_x']:.3f}")
            if summary.get("env_velocity_x_median") is not None:
                print(
                    "  Env Vel X (median/IQR): "
                    f"{summary['env_velocity_x_median']:.3f} / {summary['env_velocity_x_iqr']:.3f}"
                )
            print(f"  Mean Angular Vel Z: {summary['mean_angular_vel_z']:.3f}")
            print(f"  Mean Roll: {summary['mean_roll']:.4f}")
            print(f"  Mean Pitch: {summary['mean_pitch']:.4f}")
            print(f"  Mean Text Sim: {summary['mean_text_sim']:.4f}")
            print(f"  Mean Teacher Motion Sim: {summary['mean_teacher_motion_sim']:.4f}")
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
        print(f"{'Onomatopoeia':<15} {'Vel X':>8} {'Cmd X':>8} {'AngVel Z':>10} {'Roll':>8} {'Pitch':>8} {'TextSim':>10} {'CentrSim':>10}")
        print("-"*80)
        for r in all_results:
            cmd_x = r.get("mean_cmd_lin_vel_x")
            cmd_x_str = f"{cmd_x:>8.3f}" if cmd_x is not None else f"{'n/a':>8}"
            print(f"{r['onomatopoeia']:<15} {r['mean_velocity_x']:>8.3f} {cmd_x_str} {r['mean_angular_vel_z']:>10.3f} "
                  f"{r['mean_roll']:>8.4f} {r['mean_pitch']:>8.4f} {r['mean_text_sim']:>10.4f} {r['mean_teacher_motion_sim']:>10.4f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
