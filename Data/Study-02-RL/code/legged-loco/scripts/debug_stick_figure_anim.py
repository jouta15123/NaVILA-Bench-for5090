#!/usr/bin/env python3
"""
Debug script to visualize H1 robot's 2D projection (HOYO-compatible) as a stick figure animation.
Generates a GIF demonstrating the 'always frontal' view logic.
"""

import argparse
import json
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
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="Override robot USD path (e.g., /workspace/NaVILA-Bench/h1_hand_head.usd for explicit markers).",
)
parser.add_argument("--frames", type=int, default=100, help="Number of frames to record (default: 100 = 2sec @ 50Hz).")
parser.add_argument("--save_path", type=str, default="h1_stick_figure_offset.gif", help="Output GIF path.")
parser.add_argument("--gif_fps", type=int, default=50, help="GIF/MP4 playback FPS (default: 50 to match Isaac Sim control rate).")
parser.add_argument("--disable_gravity", action="store_true", help="Disable gravity to keep robot upright.")
parser.add_argument("--use_markers", action="store_true", help="Use marker prims (head/neck/hand) if present.")
parser.add_argument(
    "--no_marker_override",
    action="store_true",
    help="Disable marker-based buffer override (still allows offset dumping).",
)
parser.add_argument("--no_policy", action="store_true", help="Skip loading policy and use zero actions.")
parser.add_argument(
    "--disable_sensors",
    action="store_true",
    help="Disable contact/height sensors and dependent terms (useful if custom USD breaks sensor paths).",
)
parser.add_argument(
    "--no_env_render",
    action="store_true",
    help="Skip env.render() frames (stick-figure GIF only). Use if headless without --enable_cameras.",
)
parser.add_argument(
    "--log_root",
    type=str,
    default=None,
    help="Override log root directory containing rsl_rl runs (e.g., /workspace/NaVILA-Bench/legged-loco/logs/rsl_rl).",
)
parser.add_argument(
    "--use_log_env",
    action="store_true",
    default=True,
    help="Apply env.yaml from the run to match observation sizes.",
)
parser.add_argument(
    "--use_base_policy",
    action="store_true",
    default=False,
    help="Load base policy for residual actor (default: off, matches eval_style_per_onomatopoeia).",
)
# Stick-figure preprocessing controls
parser.add_argument(
    "--coord_mode",
    type=str,
    default="hoyo_front",
    choices=["hoyo_front", "legacy_xz_yaw"],
    help="Coordinate preprocessing mode for keymap generation.",
)
parser.add_argument(
    "--standardize",
    action="store_true",
    help="Apply mean/std normalization to keymap (default: off for visualization).",
)
parser.add_argument(
    "--no_normalize_height",
    action="store_true",
    help="Disable height normalization (default: on).",
)
parser.add_argument(
    "--head_ratio",
    type=float,
    default=None,
    help="Override head/shoulder ratio used for head estimation (HOYO_HEAD_SHOULDER_RATIO).",
)
parser.add_argument(
    "--neck_ratio",
    type=float,
    default=None,
    help="Override neck/shoulder ratio used for neck estimation (HOYO_NECK_SHOULDER_RATIO).",
)
parser.add_argument(
    "--dump_offsets",
    action="store_true",
    help="Print parent-local offsets from marker prims.",
)
parser.add_argument(
    "--dump_offsets_path",
    type=str,
    default=None,
    help="Write offsets JSON to path (e.g., configs/h1_to_hoyo_offsets.json).",
)
parser.add_argument(
    "--dump_offsets_only",
    action="store_true",
    help="Exit after dumping offsets (no rollout/animation).",
)
parser.add_argument(
    "--dump_offsets_body",
    action="store_true",
    help="Compute offsets in physics body frame (uses body_pos_w/body_quat_w).",
)
parser.add_argument(
    "--save_per_env",
    action="store_true",
    help="Save per-env GIFs with _env{idx} suffix.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="Directory to save per-env GIFs/features (default: directory of save_path).",
)
parser.add_argument(
    "--max_envs_to_save",
    type=int,
    default=None,
    help="Limit number of envs to save (useful for large num_envs).",
)
parser.add_argument(
    "--feature_mode",
    type=str,
    default="keymap",
    choices=["none", "keymap", "latent", "both"],
    help="Features to collect (keymap=2D joints, latent=MotionCLIP).",
)
parser.add_argument(
    "--feature_stride",
    type=int,
    default=1,
    help="Stride (frames) for feature collection.",
)
parser.add_argument(
    "--save_features",
    action="store_true",
    help="Save collected features to NPZ files.",
)
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
import isaaclab.utils.math as math_utils
import isaaclab_tasks
import omni.isaac.leggedloco.config
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
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


def _resolve_marker_prims(stage, marker_names: set[str], env_hint: str = "/World/envs/env_0") -> dict:
    """Find marker prims by exact name, prefer env_0 if present."""
    from pxr import Usd

    found = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        name = prim.GetName()
        if name in marker_names:
            path = str(prim.GetPath())
            if env_hint in path:
                found[name] = prim
            elif name not in found:
                found[name] = prim
    return found


def _pick_marker_prims(stage) -> dict:
    """Pick best marker prims for HOYO endpoints if available."""
    marker_candidates = {
        "head": ["head", "head_marker", "head_tip"],
        "neck": ["neck", "neck_marker", "neck_tip"],
        "r_hand": ["right_hand_marker", "right_hand_tip", "right_hand"],
        "l_hand": ["left_hand_marker", "left_hand_tip", "left_hand"],
    }
    all_names = {name for names in marker_candidates.values() for name in names}
    prims = _resolve_marker_prims(stage, all_names)
    picked = {}
    for role, names in marker_candidates.items():
        for name in names:
            if name in prims:
                picked[role] = prims[name]
                break
    return picked


def _apply_marker_overrides(style_module, marker_prims, xform_cache) -> None:
    """Override latest motion buffer frame with marker prim positions."""
    if style_module is None or not marker_prims:
        return
    # HOYO indices: head=0, neck=1, r_hand=4, l_hand=7
    index_map = {0: "head", 1: "neck", 4: "r_hand", 7: "l_hand"}
    xform_cache.Clear()
    for idx, role in index_map.items():
        prim = marker_prims.get(role)
        if prim is None:
            continue
        mat = xform_cache.GetLocalToWorldTransform(prim)
        pos = mat.ExtractTranslation()
        style_module.motion_buffer[:, -1, idx, :] = torch.tensor(
            [pos[0], pos[1], pos[2]], device=style_module.motion_buffer.device
        )


def _find_parent_by_name(prim, name_candidates):
    """Walk up the prim hierarchy to find a parent whose name matches candidates."""
    current = prim.GetParent()
    while current and current.IsValid():
        name = current.GetName()
        for cand in name_candidates:
            if cand in name:
                return current
        current = current.GetParent()
    return prim.GetParent()


def _extract_local_offset(parent_prim, child_prim, xform_cache):
    """Compute child translation in parent local frame."""
    xform_cache.Clear()
    parent_w = xform_cache.GetLocalToWorldTransform(parent_prim)
    child_w = xform_cache.GetLocalToWorldTransform(child_prim)
    local = parent_w.GetInverse() * child_w
    t = local.ExtractTranslation()
    return [float(t[0]), float(t[1]), float(t[2])]


def _compute_offsets_from_markers(marker_prims, xform_cache):
    """Compute HOYO offsets from marker prims and their intended parents."""
    offsets = {}
    parents = {}
    # head/neck -> torso_link
    if "head" in marker_prims:
        head_parent = _find_parent_by_name(marker_prims["head"], ["torso_link", "torso"])
        offsets["head"] = _extract_local_offset(head_parent, marker_prims["head"], xform_cache)
        parents["head"] = str(head_parent.GetPath())
    if "neck" in marker_prims:
        neck_parent = _find_parent_by_name(marker_prims["neck"], ["torso_link", "torso"])
        offsets["neck"] = _extract_local_offset(neck_parent, marker_prims["neck"], xform_cache)
        parents["neck"] = str(neck_parent.GetPath())
    # hands -> elbow links
    if "r_hand" in marker_prims:
        r_parent = _find_parent_by_name(marker_prims["r_hand"], ["right_elbow_link", "right_elbow"])
        offsets["r_hand"] = _extract_local_offset(r_parent, marker_prims["r_hand"], xform_cache)
        parents["r_hand"] = str(r_parent.GetPath())
    if "l_hand" in marker_prims:
        l_parent = _find_parent_by_name(marker_prims["l_hand"], ["left_elbow_link", "left_elbow"])
        offsets["l_hand"] = _extract_local_offset(l_parent, marker_prims["l_hand"], xform_cache)
        parents["l_hand"] = str(l_parent.GetPath())
    return offsets, parents


def _find_body_index_by_name(body_names, target_name):
    for idx, name in enumerate(body_names):
        if target_name in name:
            return idx
    return None


def _compute_offsets_from_markers_body(marker_prims, xform_cache, robot):
    """Compute offsets using physics body frame (body_pos_w/body_quat_w)."""
    offsets = {}
    parents = {}
    body_pos_w = robot.data.body_pos_w
    body_quat_w = robot.data.body_quat_w
    body_names = list(robot.data.body_names)
    device = body_pos_w.device

    def _offset_for(role, parent_candidates):
        prim = marker_prims.get(role)
        if prim is None:
            return
        parent = _find_parent_by_name(prim, parent_candidates)
        parent_name = parent.GetName()
        body_idx = _find_body_index_by_name(body_names, parent_name)
        if body_idx is None:
            return
        xform_cache.Clear()
        mat = xform_cache.GetLocalToWorldTransform(prim)
        pos = mat.ExtractTranslation()
        marker_pos = torch.tensor([pos[0], pos[1], pos[2]], device=device, dtype=body_pos_w.dtype)
        delta = marker_pos - body_pos_w[:, body_idx]
        offset_local = math_utils.quat_rotate_inverse(body_quat_w[:, body_idx], delta)
        offsets[role] = [float(v) for v in offset_local[0].tolist()]
        parents[role] = str(parent.GetPath())

    _offset_for("head", ["torso_link", "torso"])
    _offset_for("neck", ["torso_link", "torso"])
    _offset_for("r_hand", ["right_elbow_link", "right_elbow"])
    _offset_for("l_hand", ["left_elbow_link", "left_elbow"])
    return offsets, parents


def _write_offsets_json(path, offsets):
    data = {}
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(offsets)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def _strip_terms_with_entity(cfg_section, entity_name: str) -> None:
    """Disable terms in a manager cfg section that reference a given SceneEntityCfg name."""
    try:
        from isaaclab.managers import SceneEntityCfg
    except Exception:
        SceneEntityCfg = None
    if cfg_section is None:
        return
    for term_name, term_cfg in cfg_section.__dict__.items():
        if term_cfg is None:
            continue
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict):
            continue
        for _, value in params.items():
            if SceneEntityCfg is not None and isinstance(value, SceneEntityCfg):
                if value.name == entity_name:
                    setattr(cfg_section, term_name, None)
                    break

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
    # Resolve run directory (align with eval_style_per_onomatopoeia defaults)
    default_run = "2026-01-05_03-51-55_trial_h1_vision_20260104_hoyo_m2t_repro"
    
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

    # Normalize task name (gym registry uses h1_vision)
    if args_cli.task == "h1_vision_rough":
        print("[WARN] Task 'h1_vision_rough' not registered. Using 'h1_vision' instead.")
        args_cli.task = "h1_vision"

    # Parse config
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    load_run = args_cli.load_run or default_run
    log_root_base = args_cli.log_root or os.path.join(REPO_ROOT, "logs", "rsl_rl")
    log_root_path = os.path.join(log_root_base, agent_cfg.experiment_name)
    log_dir = os.path.join(log_root_path, load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    # Load params from run
    # Load params from run
    log_env_cfg_path = os.path.join(log_dir, "params", "env.yaml")
    if args_cli.use_log_env and os.path.exists(log_env_cfg_path):
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
    if args_cli.usd_path:
        try:
            env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
            if args_cli.disable_gravity:
                env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
            print(f"[INFO] Using custom USD: {args_cli.usd_path}")
        except Exception as exc:
            print(f"[WARN] Failed to apply USD override: {exc}")

    # Update agent config from the loaded run
    log_agent_cfg_dict = None
    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
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

    # Respect checkpoint override from CLI (if any)
    if args_cli.checkpoint is not None:
         agent_cfg.load_checkpoint = args_cli.checkpoint

    # Disable base policy unless explicitly requested (matches eval script)
    if not args_cli.use_base_policy:
         if hasattr(agent_cfg, "policy") and hasattr(agent_cfg.policy, "base_policy_checkpoint"):
             agent_cfg.policy.base_policy_checkpoint = None
             print("[INFO] Base policy loading disabled.")

    # Infer obs dims from checkpoint and adjust config if needed
    resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)
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

    # Disable sensors/terms when using custom USD (they require specific prim paths that may differ)
    if args_cli.disable_sensors:
        print("[INFO] Disabling contact/height sensors and dependent terms")
        if hasattr(env_cfg, "scene"):
            if hasattr(env_cfg.scene, "contact_forces"):
                env_cfg.scene.contact_forces = None
            if hasattr(env_cfg.scene, "height_scan"):
                env_cfg.scene.height_scan = None
            if hasattr(env_cfg.scene, "height_scanner"):
                env_cfg.scene.height_scanner = None
        # Also disable observations that depend on these sensors
        if hasattr(env_cfg, "observations"):
            if hasattr(env_cfg.observations, "policy"):
                for key in ("height_scan", "height_scanner"):
                    if hasattr(env_cfg.observations.policy, key):
                        setattr(env_cfg.observations.policy, key, None)
            if hasattr(env_cfg.observations, "critic"):
                for key in ("height_scan", "height_scanner"):
                    if hasattr(env_cfg.observations.critic, key):
                        setattr(env_cfg.observations.critic, key, None)
        # Disable rewards/terminations that reference contact_forces
        if hasattr(env_cfg, "rewards"):
            _strip_terms_with_entity(env_cfg.rewards, "contact_forces")
            if hasattr(env_cfg.rewards, "feet_stumble"):
                env_cfg.rewards.feet_stumble = None
        if hasattr(env_cfg, "terminations"):
            _strip_terms_with_entity(env_cfg.terminations, "contact_forces")
            if hasattr(env_cfg.terminations, "base_contact"):
                env_cfg.terminations.base_contact = None

    # Enable rendering
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    marker_prims = {}
    xform_cache = None
    use_markers = bool(args_cli.use_markers or args_cli.usd_path)
    if use_markers:
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            marker_prims = _pick_marker_prims(stage)
            if marker_prims:
                print("[INFO] Found marker prims:")
                for role, prim in marker_prims.items():
                    print(f"  {role}: {prim.GetPath()}")
            else:
                print("[INFO] No marker prims found (head/neck/hand).")
            xform_cache = UsdGeom.XformCache()
        except Exception as exc:
            print(f"[WARN] Failed to resolve marker prims: {exc}")
    marker_override_enabled = bool(marker_prims and xform_cache and not args_cli.no_marker_override)
    
    # Wrap environment for RSL-RL (history-aware if needed)
    if history_length > 0:
        print(f"[INFO] Using RslRlVecEnvHistoryWrapper with history_length={history_length}")
        env = RslRlVecEnvHistoryWrapper(env, history_length=history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # Load Policy (optional)
    policy = None
    if not args_cli.no_policy:
        print(f"Loading policy from {log_dir}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # Resolve checkpoint path from run folder
        resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)
        try:
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
        except Exception as exc:
            print(f"[WARN] Failed to load policy: {exc}")
            print("[WARN] Falling back to zero actions. Use --no_policy to silence this.")

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
    if style_module is not None:
        # Force desired preprocessing mode for debugging visualization
        if getattr(style_module, "coord_mode", None) != args_cli.coord_mode:
            print(f"[INFO] Overriding coord_mode: {getattr(style_module, 'coord_mode', None)} -> {args_cli.coord_mode}")
        style_module.coord_mode = args_cli.coord_mode
        if args_cli.head_ratio is not None:
            style_module.head_shoulder_ratio = float(args_cli.head_ratio)
        if args_cli.neck_ratio is not None:
            style_module.neck_shoulder_ratio = float(args_cli.neck_ratio)
        print(
            f"[INFO] Keymap preprocess: coord_mode={style_module.coord_mode}, "
            f"standardize={args_cli.standardize}, "
            f"normalize_height={not args_cli.no_normalize_height}"
        )
        if args_cli.head_ratio is not None or args_cli.neck_ratio is not None:
            print(
                f"[INFO] Head/Neck ratios: head={style_module.head_shoulder_ratio}, "
                f"neck={style_module.neck_shoulder_ratio}"
            )

    # Dump offsets from marker prims if requested.
    if marker_prims and xform_cache and (args_cli.dump_offsets or args_cli.dump_offsets_path):
        if args_cli.dump_offsets_body:
            offsets, parents = _compute_offsets_from_markers_body(marker_prims, xform_cache, robot)
        else:
            offsets, parents = _compute_offsets_from_markers(marker_prims, xform_cache)
        if offsets:
            print("[INFO] Marker parent paths:")
            for key, path in parents.items():
                print(f"  {key}: {path}")
            print("[INFO] Computed offsets (parent-local):")
            for key, vec in offsets.items():
                print(f"  {key}: {vec}")
            if args_cli.dump_offsets_path:
                _write_offsets_json(args_cli.dump_offsets_path, offsets)
                print(f"[INFO] Wrote offsets to: {args_cli.dump_offsets_path}")
        else:
            print("[WARN] No offsets computed (missing marker prims).")
        if args_cli.dump_offsets_only:
            env.close()
            simulation_app.close()
            return

    num_envs = env.unwrapped.num_envs
    history_2d = [[] for _ in range(num_envs)]
    collect_keymap = args_cli.feature_mode in ("keymap", "both")
    collect_latent = args_cli.feature_mode in ("latent", "both")
    feature_stride = max(1, int(args_cli.feature_stride))
    history_latent = [[] for _ in range(num_envs)] if collect_latent else None
    video_frames = []
    env_render_enabled = not args_cli.no_env_render
    
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
        
        # Inference or zero actions
        with torch.no_grad():
             obs, _ = env.get_observations()
             if policy is None:
                 act_dim = env.action_space.shape[0]
                 actions = torch.zeros((env.unwrapped.num_envs, act_dim), device=env.unwrapped.device)
             else:
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
        if i >= warmup_steps and env_render_enabled:
             try:
                  frame = env.unwrapped.render()
                  if frame is not None:
                       video_frames.append(frame)
             except RuntimeError as exc:
                  print(f"[WARN] env.render() disabled: {exc}")
                  print("[WARN] Re-run with --enable_cameras or pass --no_env_render to skip.")
                  env_render_enabled = False
             
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
             if marker_override_enabled:
                 _apply_marker_overrides(style_module, marker_prims, xform_cache)
             
             # Capture 2D map
             if i >= warmup_steps:
                 full_buffer_2d = style_module.get_hoyo_compatible_keymap(
                     standardize=args_cli.standardize,
                     normalize_height=not args_cli.no_normalize_height,
                 )
                 latest_frames_2d = full_buffer_2d[:, -1].detach().cpu().numpy()  # (B, 14, 2)
                 for env_id in range(num_envs):
                     history_2d[env_id].append(latest_frames_2d[env_id])
                 if collect_latent and ((i - warmup_steps) % feature_stride == 0):
                     z_m = style_module.encode_buffer()  # (B, 512)
                     z_m_np = z_m.detach().cpu().numpy()
                     for env_id in range(num_envs):
                         history_latent[env_id].append(z_m_np[env_id])
                 
                 # Debug: print coordinates every 100 frames
                 if (i - warmup_steps) % 100 == 0:
                     env0 = latest_frames_2d[0]
                     print(f"[Frame {i - warmup_steps}] 2D coords (env0, x=lateral, y=down):")
                     print(f"  Head(0): {env0[0]}")
                     print(f"  Neck(1): {env0[1]}")
                     print(f"  R-Shoulder(2): {env0[2]}")
                     print(f"  L-Shoulder(5): {env0[5]}")
                     print(f"  R-Hip(8): {env0[8]}")
                     print(f"  R-Ankle(10): {env0[10]}")

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
    
    # Save MP4 (only if we captured env frames)
    if video_frames:
        mp4_path = args_cli.save_path.replace(".gif", ".mp4")
        print(f"Saving MP4 video to {mp4_path}...")
        try:
            mp4_out = os.path.abspath(mp4_path)
            writer = imageio.get_writer(mp4_out, fps=args_cli.gif_fps)
            for frame in video_frames:
                writer.append_data(frame)
            writer.close()
            print(f"Saved Video: {mp4_out}")
        except Exception as e:
            print(f"Error saving MP4: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[INFO] No env frames captured. Skipping MP4 export.")

    def _save_gif(history_env, out_path):
        if len(history_env) == 0:
            print(f"[WARN] No frames collected for {out_path}. Skipping GIF.")
            return
        fig, ax = plt.subplots(figsize=(6, 6))

        # Auto-scale axes from collected keypoints so the full body is visible.
        all_data = np.stack(history_env, axis=0)  # (T, 14, 2)
        x_all = all_data[..., 0]
        y_all = -all_data[..., 1]  # flip for display (up positive)
        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        # Add padding to avoid clipping at edges
        pad_x = max(0.05, 0.1 * (x_max - x_min))
        pad_y = max(0.05, 0.1 * (y_max - y_min))
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        
        scat = ax.scatter([], [], c="r", s=20)
        lines = [ax.plot([], [], "b-")[0] for _ in SKELETON_EDGES]
        text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def update(frame_idx):
            data = history_env[frame_idx]  # (14, 2)
            x = data[:, 0]
            y = -data[:, 1]  # Flip Y so Up is Positive for visualization
            scat.set_offsets(np.c_[x, y])
            for line, (i, j) in zip(lines, SKELETON_EDGES):
                line.set_data([x[i], x[j]], [y[i], y[j]])
            text.set_text(f"Frame: {frame_idx}")
            return [scat, text] + lines

        ani = animation.FuncAnimation(fig, update, frames=len(history_env), blit=True)
        try:
            print(f"Saving GIF: {out_path}")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            ani.save(out_path, writer=PillowWriter(fps=args_cli.gif_fps))
        except Exception as e:
            print(f"Error saving animation: {e}")
            import traceback
            traceback.print_exc()

    base_out = os.path.abspath(args_cli.save_path)
    base_dir = args_cli.save_dir or os.path.dirname(base_out) or "."
    base_root, base_ext = os.path.splitext(os.path.basename(base_out))
    if not base_ext:
        base_ext = ".gif"
    env_ids = list(range(num_envs))
    if args_cli.max_envs_to_save is not None:
        env_ids = env_ids[: max(0, int(args_cli.max_envs_to_save))]

    if args_cli.save_per_env:
        for env_id in env_ids:
            out_path = os.path.join(base_dir, f"{base_root}_env{env_id}{base_ext}")
            _save_gif(history_2d[env_id], out_path)
    else:
        out_path = base_out if args_cli.save_dir is None else os.path.join(base_dir, f"{base_root}{base_ext}")
        _save_gif(history_2d[0], out_path)

    if args_cli.save_features and args_cli.feature_mode != "none":
        for env_id in env_ids if args_cli.save_per_env else [0]:
            feat_path = os.path.join(base_dir, f"{base_root}_env{env_id}_features.npz")
            payload = {}
            if collect_keymap:
                if len(history_2d[env_id]) > 0:
                    payload["keymap_2d"] = np.stack(history_2d[env_id], axis=0)
                else:
                    payload["keymap_2d"] = np.empty((0, 14, 2), dtype=np.float32)
            if collect_latent:
                if history_latent and len(history_latent[env_id]) > 0:
                    payload["latent"] = np.stack(history_latent[env_id], axis=0)
                else:
                    payload["latent"] = np.empty((0, 512), dtype=np.float32)
                payload["latent_stride"] = np.array([feature_stride], dtype=np.int32)
            payload["coord_mode"] = np.array([style_module.coord_mode if style_module else "unknown"])
            np.savez(feat_path, **payload)
            print(f"[INFO] Saved features: {feat_path}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
