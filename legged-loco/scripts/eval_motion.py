# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
シンプルな動作評価スクリプト

eval_style_per_onomatopoeia.py のリファクタリング版。
必要な機能:
- velocity x (x方向速度)
- com x (x方向重心位置)
- cos類似度 (motion embedding間)
- 教師との関節誤差

Usage:
    python scripts/eval_motion.py --load_run RUN_NAME --checkpoint MODEL_PATH
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# === Path Setup (Isaac Sim requires specific import order) ===
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

# --- Argument Parser (must be before AppLauncher) ---
parser = argparse.ArgumentParser(description="Evaluate motion style metrics")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="h1_vision")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps per style")
parser.add_argument("--output_dir", type=str, default="eval_results/motion")
parser.add_argument("--log_root", type=str, default=None, help="Log root directory")
parser.add_argument("--use_log_env", action="store_true", default=True)
parser.add_argument("--no_base_policy", action="store_true", default=False)
parser.add_argument("--style_list", type=str, default=None, help="Comma-separated styles to evaluate")
parser.add_argument(
    "--terrain",
    type=str,
    default="flat",
    choices=["rough", "flat", "plane"],
    help="Terrain type for evaluation (default: flat). Use flat/plane for a flat ground plane.",
)
# Policy arguments (required by cli_args.parse_rsl_rl_cfg)
parser.add_argument("--use_cnn", action="store_true", default=None)
parser.add_argument("--use_rnn", action="store_true", default=False)
parser.add_argument("--history_length", type=int, default=0)
# Velocity command
parser.add_argument("--lin_vel_x", type=float, default=0.5)
parser.add_argument("--lin_vel_y", type=float, default=0.0)
parser.add_argument("--ang_vel_z", type=float, default=0.0)
# Video
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
# HOYO reference
parser.add_argument("--hoyo_root", type=str, default=None)
parser.add_argument("--hoyo_seed", type=int, default=42)
parser.add_argument(
    "--enable_joint_error_baseline",
    action="store_true",
    default=False,
    help="Compute one global structural baseline for joint error using static prototypes.",
)
parser.add_argument(
    "--baseline_style_label",
    type=str,
    default="通常",
    help="HOYO label used to build frame0 static prototype for joint-error baseline.",
)
parser.add_argument(
    "--baseline_h1_warmup_steps",
    type=int,
    default=150,
    help="Warmup steps with zero velocity before collecting H1 static prototype.",
)
parser.add_argument(
    "--baseline_h1_collect_steps",
    type=int,
    default=100,
    help="Collection steps for H1 static prototype (after warmup).",
)
parser.add_argument(
    "--baseline_h1_collect_stride",
    type=int,
    default=2,
    help="Stride for collecting H1 frames during static prototype capture.",
)
parser.add_argument(
    "--baseline_h1_source",
    type=str,
    default="policy_zero_velocity",
    choices=["policy_zero_velocity", "usd_stand"],
    help=(
        "Source of H1 static prototype for baseline: "
        "'policy_zero_velocity' (existing behavior) or 'usd_stand' (hold default joint stand pose)."
    ),
)
parser.add_argument(
    "--save_joint_error_baseline_gif",
    action="store_true",
    default=False,
    help="Save side-by-side GIF for baseline prototypes (H1 static vs HOYO static).",
)
parser.add_argument(
    "--baseline_gif_fps",
    type=int,
    default=8,
    help="FPS for baseline comparison GIF.",
)
parser.add_argument(
    "--baseline_gif_frames",
    type=int,
    default=60,
    help="Max frames to save for baseline comparison GIF.",
)
# Time-shift analysis
parser.add_argument("--time_shift_analyze", action="store_true", default=False,
                    help="Analyze cos similarity with time-shifted H1 buffer")
parser.add_argument("--time_shift_stride", type=int, default=10,
                    help="Stride for time-shift analysis (in frames)")
parser.add_argument("--time_shift_max", type=int, default=100,
                    help="Maximum time shift (in frames)")
# Reward breakdown
parser.add_argument(
    "--log_reward_terms",
    action="store_true",
    default=False,
    help="Log per-term reward contributions (weighted, per-step) via env.reward_manager.",
)
parser.add_argument(
    "--reward_terms_topk",
    type=int,
    default=8,
    help="How many reward terms to print (by |mean|) when --log_reward_terms is enabled.",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=None,
    help="Discount factor for rate^adv (override agent config).",
)
parser.add_argument(
    "--log_adv_stats",
    action="store_true",
    default=False,
    help="Log min/mean/max of values_pre/values_next/delta_t/r_style/r_base for one episode.",
)
parser.add_argument(
    "--log_hoyo_stats",
    action="store_true",
    default=False,
    help="Log HOYO embedding similarity and keypoint stats for debugging.",
)
parser.add_argument(
    "--debug_progress",
    action="store_true",
    default=False,
    help="Print progress markers to stdout for debugging early exits.",
)

# RSL-RL and AppLauncher args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
DEBUG_PROGRESS = args_cli.debug_progress or os.environ.get("EVAL_MOTION_DEBUG", "").strip() == "1"

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === After AppLauncher: Safe to import torch, gymnasium, etc. ===

import gymnasium as gym
import imageio
import json
import numpy as np
import random
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass  # Japanese labels may not display correctly

# DTW for phase-aligned joint error
try:
    from tslearn.metrics import dtw_path
    DTW_AVAILABLE = True
except ImportError:
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        DTW_AVAILABLE = True
    except ImportError:
        DTW_AVAILABLE = False

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

# HOYO dataset
from hoyo_v1_1.models.common import HoyoInstructionDataset, apply_normalization_from_stats

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
NORMAL_STYLE_LABEL_CANDIDATES = ("通常", "normal", "Normal")
_CLEANUP_ENV = None
DATASET_SPEED_ORDER = [
    "すたすた",
    "せかせか",
    "通常",
    "てくてく",
    "どっしどっし",
    "ぶらぶら",
    "のしのし",
    "よたよた",
    "のろのろ",
    "とぼとぼ",
    "よろよろ",
]
_DATASET_SPEED_ORDER_INDEX = {label: idx for idx, label in enumerate(DATASET_SPEED_ORDER)}


def order_labels_by_dataset_speed(labels: list[str]) -> list[str]:
    """Sort labels by dataset speed order; keep unknown labels in their original order at the end."""
    unique_labels = list(dict.fromkeys(labels))
    original_index = {label: idx for idx, label in enumerate(unique_labels)}
    return sorted(
        unique_labels,
        key=lambda label: (
            _DATASET_SPEED_ORDER_INDEX.get(label, len(DATASET_SPEED_ORDER)),
            original_index[label],
        ),
    )


def order_results_by_dataset_speed(results: list["EvaluationResult"]) -> list["EvaluationResult"]:
    """Sort results by dataset speed order using onomatopoeia labels."""
    ordered_labels = order_labels_by_dataset_speed([r.onomatopoeia for r in results])
    rank = {label: idx for idx, label in enumerate(ordered_labels)}
    return sorted(results, key=lambda r: rank.get(r.onomatopoeia, len(ordered_labels)))


def _debug(msg: str) -> None:
    if not DEBUG_PROGRESS:
        return
    try:
        print(f"[eval_motion debug] {msg}", flush=True)
    except Exception:
        pass


_SCENE_ENTITY_ID_FIELDS = {
    "joint_ids",
    "fixed_tendon_ids",
    "body_ids",
    "object_collection_ids",
}


def _normalize_scene_entity_ids(obj, _visited=None) -> int:
    """Replace None -> slice(None) for SceneEntityCfg id fields inside config trees."""
    if _visited is None:
        _visited = set()
    if obj is None:
        return 0
    if isinstance(obj, (str, int, float, bool, bytes, slice)):
        return 0
    obj_id = id(obj)
    if obj_id in _visited:
        return 0
    _visited.add(obj_id)

    changed = 0
    if isinstance(obj, dict):
        for v in obj.values():
            changed += _normalize_scene_entity_ids(v, _visited)
        return changed
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            changed += _normalize_scene_entity_ids(v, _visited)
        return changed
    # Avoid heavy traversal into tensors/arrays
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return 0
    except Exception:
        pass
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return 0
    except Exception:
        pass

    if not hasattr(obj, "__dict__"):
        return 0

    for key, val in vars(obj).items():
        if key in _SCENE_ENTITY_ID_FIELDS and val is None:
            setattr(obj, key, slice(None))
            changed += 1
        changed += _normalize_scene_entity_ids(val, _visited)
    return changed


def _cleanup() -> None:
    """Ensure Isaac Sim and env are closed even on exceptions."""
    global _CLEANUP_ENV
    if _CLEANUP_ENV is not None:
        try:
            _CLEANUP_ENV.close()
        except Exception:
            pass
        _CLEANUP_ENV = None
    try:
        simulation_app.close()
    except Exception:
        pass


def _load_yaml_with_fallback(path: str) -> tuple[dict, str]:
    """Load YAML with a fallback that supports python/object tags (e.g. slice)."""
    try:
        data = load_yaml(path)
        if not isinstance(data, dict):
            raise TypeError(f"Top-level YAML is not dict: {type(data)}")
        return data, "isaaclab"
    except Exception as primary_error:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.unsafe_load(f)
            if not isinstance(data, dict):
                raise TypeError(f"Top-level YAML is not dict: {type(data)}")
            logger.warning(
                "load_yaml failed for %s; fallback to yaml.unsafe_load (trusted local run config): %s",
                path,
                primary_error,
            )
            return data, "pyyaml_unsafe"
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load YAML '{path}'. primary={primary_error}; fallback={fallback_error}"
            ) from fallback_error


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_envs: int = 16
    eval_steps: int = 500
    seed: int | None = None
    output_dir: str = "eval_results/motion"
    log_reward_terms: bool = False
    reward_terms_topk: int = 8
    gamma: float | None = None
    log_adv_stats: bool = False
    log_hoyo_stats: bool = False
    enable_joint_error_baseline: bool = False
    baseline_style_label: str = "通常"
    baseline_h1_warmup_steps: int = 150
    baseline_h1_collect_steps: int = 100
    baseline_h1_collect_stride: int = 2
    baseline_h1_source: str = "policy_zero_velocity"
    save_joint_error_baseline_gif: bool = False
    baseline_gif_fps: int = 8
    baseline_gif_frames: int = 60
    # Velocity command
    lin_vel_x: float = 0.5
    lin_vel_y: float = 0.0
    ang_vel_z: float = 0.0
    # Video
    record_video: bool = False
    video_fps: int = 50
    video_length: int = 500
    # Time-shift analysis
    time_shift_analyze: bool = False
    time_shift_stride: int = 10
    time_shift_max: int = 100

    @classmethod
    def from_args(cls, args) -> "EvalConfig":
        return cls(
            num_envs=args.num_envs,
            eval_steps=args.eval_steps,
            seed=args.seed,
            output_dir=args.output_dir,
            log_reward_terms=getattr(args, "log_reward_terms", False),
            reward_terms_topk=getattr(args, "reward_terms_topk", 8),
            gamma=getattr(args, "gamma", None),
            log_adv_stats=getattr(args, "log_adv_stats", False),
            log_hoyo_stats=getattr(args, "log_hoyo_stats", False),
            enable_joint_error_baseline=getattr(args, "enable_joint_error_baseline", False),
            baseline_style_label=getattr(args, "baseline_style_label", "通常"),
            baseline_h1_warmup_steps=getattr(args, "baseline_h1_warmup_steps", 150),
            baseline_h1_collect_steps=getattr(args, "baseline_h1_collect_steps", 100),
            baseline_h1_collect_stride=getattr(args, "baseline_h1_collect_stride", 2),
            baseline_h1_source=getattr(args, "baseline_h1_source", "policy_zero_velocity"),
            save_joint_error_baseline_gif=getattr(args, "save_joint_error_baseline_gif", False),
            baseline_gif_fps=getattr(args, "baseline_gif_fps", 8),
            baseline_gif_frames=getattr(args, "baseline_gif_frames", 60),
            lin_vel_x=args.lin_vel_x,
            lin_vel_y=args.lin_vel_y,
            ang_vel_z=args.ang_vel_z,
            record_video=args.video,
            video_length=args.video_length,
            time_shift_analyze=getattr(args, 'time_shift_analyze', False),
            time_shift_stride=getattr(args, 'time_shift_stride', 10),
            time_shift_max=getattr(args, 'time_shift_max', 100),
        )


@dataclass
class RobotState:
    """Robot state at a single timestep."""
    lin_vel: torch.Tensor  # (N, 3) body-frame linear velocity
    ang_vel: torch.Tensor  # (N, 3) body-frame angular velocity
    root_pos: torch.Tensor  # (N, 3) world position
    root_quat: torch.Tensor  # (N, 4) quaternion (wxyz)


@dataclass
class MotionMetrics:
    """Collected metrics for a single evaluation run."""
    velocity_x: list[float] = field(default_factory=list)
    velocity_y: list[float] = field(default_factory=list)
    velocity_z: list[float] = field(default_factory=list)
    com_x: list[float] = field(default_factory=list)  # 重心 x位置
    # cos類似度: H1 motion embedding vs セントロイド (クラス平均latent)
    cos_centroid: list[float] = field(default_factory=list)
    # cos類似度: H1 motion embedding vs ランダムサンプル (latent_snapshotから)
    cos_random_sample: list[float] = field(default_factory=list)
    # 相対スタイル指標: cos(H1, teacher_style) - cos(H1, teacher_normal)
    style_score: list[float] = field(default_factory=list)
    # cos類似度: H1 motion embedding vs 教師埋め込み（スタイル別）
    teacher_similarity: dict[str, list[float]] = field(default_factory=dict)
    # cos類似度: H1 motion embedding vs HOYO reference (styleごと)
    hoyo_similarity: dict[str, list[float]] = field(default_factory=dict)
    # 中心化 cos類似度: H1 motion embedding vs HOYO reference (styleごと)
    hoyo_similarity_centered: dict[str, list[float]] = field(default_factory=dict)
    # 時間シフト解析結果: {shift_amount: [cos_values]}
    cos_time_shift: dict[int, list[float]] = field(default_factory=dict)
    joint_error: list[float] = field(default_factory=list)
    joint_error_dtw: list[float] = field(default_factory=list)  # Soft-DTWで位相アライメント後
    roll: list[float] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    fall_count: int = 0
    episode_count: int = 0


@dataclass
class EvaluationResult:
    """Summary result for one onomatopoeia."""
    onomatopoeia: str
    # Velocity
    mean_velocity_x: float
    std_velocity_x: float
    # CoM (center of mass)
    mean_com_x: float
    std_com_x: float
    # Similarity (H1 vs centroid - クラス平均latent)
    mean_cos_centroid: float | None
    std_cos_centroid: float | None
    # Similarity (H1 vs random sample - latent_snapshotからランダム選択)
    mean_cos_random_sample: float | None
    std_cos_random_sample: float | None
    # Relative style score (teacher style - teacher normal)
    mean_style_score: float | None
    std_style_score: float | None
    # Joint error
    mean_joint_error: float | None
    std_joint_error: float | None
    # Joint error with DTW (phase-aligned)
    mean_joint_error_dtw: float | None
    std_joint_error_dtw: float | None
    # Episode stats
    mean_episode_length: float
    fall_rate: float
    episode_count: int
    mean_joint_error_delta: float | None = None
    mean_joint_error_ratio: float | None = None
    mean_joint_error_dtw_delta: float | None = None
    mean_joint_error_dtw_ratio: float | None = None
    cos_centroid_count: int | None = None
    cos_random_sample_count: int | None = None
    style_score_count: int | None = None
    # Similarity (H1 vs teacher embedding) per label
    teacher_similarity_mean: dict[str, float] | None = None
    teacher_similarity_std: dict[str, float] | None = None
    # 時間シフト解析結果: {shift: mean_cos}
    time_shift_results: dict[int, float] | None = None
    # Similarity (H1 vs HOYO reference) per label
    hoyo_similarity_mean: dict[str, float] | None = None
    hoyo_similarity_std: dict[str, float] | None = None
    # Similarity (H1 vs HOYO reference, centered) per label
    hoyo_similarity_centered_mean: dict[str, float] | None = None
    hoyo_similarity_centered_std: dict[str, float] | None = None
    # Reward breakdown (weighted, per-step)
    mean_reward_total: float | None = None
    std_reward_total: float | None = None
    reward_terms_mean: dict[str, float] | None = None
    reward_terms_std: dict[str, float] | None = None
    reward_step_dt: float | None = None
    mean_action_sq: float | None = None
    # Contribution rates (kiyoritu.md)
    rate_mag: dict[str, float] | None = None  # 報酬内訳寄与率(%)
    rate_adv: dict[str, float] | None = None  # Advantage寄与率(%)
    share_e: dict[str, float] | None = None   # 関節エネルギー割合(%)

    def to_dict(self) -> dict:
        d = {
            "onomatopoeia": self.onomatopoeia,
            "mean_velocity_x": float(self.mean_velocity_x),
            "std_velocity_x": float(self.std_velocity_x),
            "mean_com_x": float(self.mean_com_x),
            "std_com_x": float(self.std_com_x),
            "mean_cos_centroid": self.mean_cos_centroid,
            "std_cos_centroid": self.std_cos_centroid,
            "mean_cos_random_sample": self.mean_cos_random_sample,
            "std_cos_random_sample": self.std_cos_random_sample,
            "mean_style_score": self.mean_style_score,
            "std_style_score": self.std_style_score,
            "mean_joint_error": self.mean_joint_error,
            "std_joint_error": self.std_joint_error,
            "mean_joint_error_dtw": self.mean_joint_error_dtw,
            "std_joint_error_dtw": self.std_joint_error_dtw,
            "mean_joint_error_delta": self.mean_joint_error_delta,
            "mean_joint_error_ratio": self.mean_joint_error_ratio,
            "mean_joint_error_dtw_delta": self.mean_joint_error_dtw_delta,
            "mean_joint_error_dtw_ratio": self.mean_joint_error_dtw_ratio,
            "mean_episode_length": float(self.mean_episode_length),
            "fall_rate": float(self.fall_rate),
            "episode_count": self.episode_count,
        }
        if self.cos_centroid_count is not None:
            d["cos_centroid_count"] = int(self.cos_centroid_count)
        if self.cos_random_sample_count is not None:
            d["cos_random_sample_count"] = int(self.cos_random_sample_count)
        if self.style_score_count is not None:
            d["style_score_count"] = int(self.style_score_count)
        if self.teacher_similarity_mean is not None:
            d["teacher_similarity_mean"] = {
                k: float(v) for k, v in self.teacher_similarity_mean.items()
            }
        if self.teacher_similarity_std is not None:
            d["teacher_similarity_std"] = {
                k: float(v) for k, v in self.teacher_similarity_std.items()
            }
        if self.hoyo_similarity_mean is not None:
            d["hoyo_similarity_mean"] = {k: float(v) for k, v in self.hoyo_similarity_mean.items()}
        if self.hoyo_similarity_std is not None:
            d["hoyo_similarity_std"] = {k: float(v) for k, v in self.hoyo_similarity_std.items()}
        if self.hoyo_similarity_centered_mean is not None:
            d["hoyo_similarity_centered_mean"] = {
                k: float(v) for k, v in self.hoyo_similarity_centered_mean.items()
            }
        if self.hoyo_similarity_centered_std is not None:
            d["hoyo_similarity_centered_std"] = {
                k: float(v) for k, v in self.hoyo_similarity_centered_std.items()
            }
        if self.time_shift_results is not None:
            d["time_shift_results"] = self.time_shift_results
        if self.mean_reward_total is not None:
            d["mean_reward_total"] = float(self.mean_reward_total)
        if self.std_reward_total is not None:
            d["std_reward_total"] = float(self.std_reward_total)
        if self.reward_terms_mean is not None:
            d["reward_terms_mean"] = {k: float(v) for k, v in self.reward_terms_mean.items()}
        if self.reward_terms_std is not None:
            d["reward_terms_std"] = {k: float(v) for k, v in self.reward_terms_std.items()}
        if self.reward_step_dt is not None:
            d["reward_step_dt"] = float(self.reward_step_dt)
        if self.mean_action_sq is not None:
            d["mean_action_sq"] = float(self.mean_action_sq)
        # Contribution rates
        if self.rate_mag is not None:
            d["rate_mag"] = {k: float(v) for k, v in self.rate_mag.items()}
        if self.rate_adv is not None:
            d["rate_adv"] = {k: float(v) for k, v in self.rate_adv.items()}
        if self.share_e is not None:
            d["share_e"] = {k: float(v) for k, v in self.share_e.items()}
        return d


def _summarize_reward_terms(
    reward_terms_mean: dict[str, float] | None,
    reward_terms_std: dict[str, float] | None,
    topk: int,
) -> str:
    if not reward_terms_mean:
        return "(no reward term stats)"
    items = [(k, float(v)) for k, v in reward_terms_mean.items()]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    lines = []
    for name, mean in items[: max(0, int(topk))]:
        std = None
        if reward_terms_std and name in reward_terms_std:
            std = float(reward_terms_std[name])
        if std is None:
            lines.append(f"      - {name}: {mean:+.6f}")
        else:
            lines.append(f"      - {name}: {mean:+.6f} ± {std:.6f}")
    return "\n".join(lines)


def _summarize_contribution_rate(
    rate_dict: dict[str, float] | None,
    topk: int,
    title: str,
) -> str:
    """寄与率(%)を表示用にフォーマット."""
    if not rate_dict:
        return f"    {title}: (no data)"
    items = [(k, float(v)) for k, v in rate_dict.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    lines = [f"    {title}:"]
    for name, pct in items[: max(0, int(topk))]:
        lines.append(f"      - {name}: {pct:.2f}%")
    return "\n".join(lines)


# =============================================================================
# Metric Calculators (明確なエラーハンドリング)
# =============================================================================

class CosineSimilarityError(Exception):
    """cos類似度計算時のエラー."""
    pass


class JointErrorCalculationError(Exception):
    """関節誤差計算時のエラー."""
    pass


def compute_cosine_similarity(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> float:
    """
    2つのmotion embedding間のcos類似度を計算.

    cos類似度の計算フロー:
    1. H1 motion → get_buffer_for_hoyo_comparison() (前処理) → encode_hoyo_sample() → z_h1 (512次元)
    2. HOYO reference → 同様の前処理済み → encode_hoyo_sample() → z_hoyo (512次元)
    3. cos_sim = dot(z_h1, z_hoyo) (両方正規化済みなのでdot product = cos類似度)

    Args:
        embedding_a: 埋め込みベクトルA (D,) or (1, D) - 正規化済み
        embedding_b: 埋め込みベクトルB (D,) or (1, D) - 正規化済み

    Returns:
        cos類似度 (-1.0 ~ 1.0)

    Raises:
        CosineSimilarityError: 計算に失敗した場合
    """
    if embedding_a is None or embedding_b is None:
        raise CosineSimilarityError("Embedding is None")

    try:
        a_flat = embedding_a.view(-1)
        b_flat = embedding_b.view(-1)

        if a_flat.shape != b_flat.shape:
            raise CosineSimilarityError(
                f"Shape mismatch: a={a_flat.shape}, b={b_flat.shape}"
            )

        # 正規化（既に正規化されているはずだが安全のため）
        a_norm = F.normalize(a_flat.unsqueeze(0), dim=-1)
        b_norm = F.normalize(b_flat.unsqueeze(0), dim=-1)
        # dot product = cos similarity (for normalized vectors)
        sim = (a_norm * b_norm).sum().item()
        return float(sim)
    except Exception as e:
        raise CosineSimilarityError(f"Failed to compute similarity: {e}") from e


def build_centered_hoyo_embeddings(
    embedding_cache: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """HOYO参照埋め込みを中心化+正規化したキャッシュを作成."""
    if not embedding_cache:
        return {}, None

    valid = {k: v for k, v in embedding_cache.items() if isinstance(v, torch.Tensor)}
    if len(valid) < 2:
        return {}, None

    try:
        stacked = torch.stack([emb.view(-1) for emb in valid.values()], dim=0)
    except Exception as e:
        logger.debug(f"Failed to stack HOYO embeddings for centering: {e}")
        return {}, None

    center = stacked.mean(dim=0)
    centered = {
        label: F.normalize(emb.view(-1) - center, dim=0)
        for label, emb in valid.items()
    }
    return centered, center


def compute_l2_joint_error(
    current_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    関節位置のL2誤差を計算.

    Args:
        current_keypoints: 現在のキーポイント (T, J, 2) or (J, 2)
        reference_keypoints: 参照キーポイント (T, J, 2) or (J, 2)
        weights: 関節ごとの重み (J,), optional

    Returns:
        重み付きL2誤差

    Raises:
        JointErrorCalculationError: 計算に失敗した場合
    """
    if current_keypoints is None or reference_keypoints is None:
        raise JointErrorCalculationError("Keypoints is None")

    try:
        # Shape alignment
        if current_keypoints.ndim == 2:
            current_keypoints = current_keypoints[np.newaxis, ...]
        if reference_keypoints.ndim == 2:
            reference_keypoints = reference_keypoints[np.newaxis, ...]

        T = min(current_keypoints.shape[0], reference_keypoints.shape[0])
        curr = current_keypoints[:T]
        ref = reference_keypoints[:T]

        # Per-joint L2 distance
        per_joint = np.mean(np.linalg.norm(curr - ref, axis=-1), axis=0)  # (J,)

        if weights is not None:
            error = float(np.sum(per_joint * weights))
        else:
            error = float(np.mean(per_joint))

        return error
    except Exception as e:
        raise JointErrorCalculationError(f"Failed to compute joint error: {e}") from e


def compute_l2_joint_error_with_dtw(
    current_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
) -> float:
    """
    DTWで位相アライメントした上での関節L2誤差を計算.

    DTWで最適なワーピングパスを見つけ、そのパスに沿った平均L2誤差を返す。
    これにより、compute_l2_joint_error と同じ単位で比較可能。

    位相が完全に一致 → L2 Err ≈ DTW Err
    位相がずれている → L2 Err > DTW Err（DTWが位相を補正）

    Args:
        current_keypoints: 現在のキーポイント (T1, J, 2) or (J, 2)
        reference_keypoints: 参照キーポイント (T2, J, 2) or (J, 2)

    Returns:
        位相アライメント後の平均L2誤差（L2 Errと同じ単位）

    Raises:
        JointErrorCalculationError: 計算に失敗した場合
    """
    if not DTW_AVAILABLE:
        raise JointErrorCalculationError(
            "DTW not available. Install: pip install tslearn or pip install fastdtw"
        )

    if current_keypoints is None or reference_keypoints is None:
        raise JointErrorCalculationError("Keypoints is None")

    try:
        # Shape alignment: ensure 3D
        if current_keypoints.ndim == 2:
            current_keypoints = current_keypoints[np.newaxis, ...]
        if reference_keypoints.ndim == 2:
            reference_keypoints = reference_keypoints[np.newaxis, ...]

        # Flatten joints: (T, J, 2) → (T, J*2)
        T1, J, D = current_keypoints.shape
        T2 = reference_keypoints.shape[0]
        curr_flat = current_keypoints.reshape(T1, -1)  # (T1, J*D)
        ref_flat = reference_keypoints.reshape(T2, -1)  # (T2, J*D)

        # DTWでワーピングパスを取得
        try:
            # tslearn
            path, _ = dtw_path(curr_flat, ref_flat)
        except NameError:
            # fastdtw
            _, path = fastdtw(curr_flat, ref_flat, dist=euclidean)

        # パスに沿った各関節のL2誤差を計算
        # path: [(i0, j0), (i1, j1), ...] のリスト
        aligned_errors = []
        for i, j in path:
            # (J, 2) の形に戻して関節ごとのL2距離を計算
            curr_frame = current_keypoints[i]  # (J, 2)
            ref_frame = reference_keypoints[j]  # (J, 2)
            per_joint_dist = np.linalg.norm(curr_frame - ref_frame, axis=-1)  # (J,)
            aligned_errors.append(np.mean(per_joint_dist))

        # 平均（L2 Err と同じ単位）
        return float(np.mean(aligned_errors))

    except Exception as e:
        raise JointErrorCalculationError(f"Failed to compute DTW joint error: {e}") from e


# =============================================================================
# Environment Utilities
# =============================================================================

class EvalEnvWrapper:
    """評価用の環境ラッパー."""

    def __init__(self, env, style_cache: dict | None = None, long_buffer_size: int = 300):
        self.env = env
        self._style_cache = style_cache or {}
        self._style_gen = env.unwrapped.command_manager._terms.get("style_command")
        self._base_cmd_term = env.unwrapped.command_manager._terms.get("base_velocity")
        
        # 長いヒストリーバッファ (時間シフト解析用)
        # shape: (num_envs, long_buffer_size, 14, 2)
        self._long_buffer_size = long_buffer_size
        self._long_buffer: torch.Tensor | None = None
        self._long_buffer_ptr = 0  # 現在の書き込み位置
        self._long_buffer_filled = 0  # 蓄積されたフレーム数
        self._longbuf_warned = False  # サイレント失敗対策フラグ

    def is_known_style(self, onomatopoeia: str) -> bool:
        """Return True if the style label exists in training labels."""
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return False
        label_to_id = getattr(self._style_gen.style_module, "label_to_id", None)
        if not isinstance(label_to_id, dict):
            return False
        return onomatopoeia in label_to_id

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    def reset(self) -> tuple[torch.Tensor, dict]:
        """環境をリセット."""
        self.env.reset()
        # ロングバッファをリセット
        self._long_buffer = None
        self._long_buffer_ptr = 0
        self._long_buffer_filled = 0
        return self.env.get_observations()

    def step(self, actions: torch.Tensor) -> tuple:
        """1ステップ実行."""
        result = self.env.step(actions)
        if isinstance(result, (tuple, list)) and len(result) == 5:
            obs, rew, terminated, truncated, info = result
            dones = terminated | truncated
            result = (obs, rew, dones, info)
        
        # ロングバッファへの蓄積（時間シフト解析用）
        self._accumulate_to_long_buffer()
        
        return result

    def _accumulate_to_long_buffer(self) -> None:
        """現在の2Dキーポイントをロングバッファに蓄積."""
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return
        style_module = self._style_gen.style_module
        
        try:
            # 現在のフレームの2Dキーポイントを取得
            buf_2d = style_module._prepare_centered_2d(
                apply_yaw_correction=True,
                coord_mode=getattr(style_module, 'coord_mode', 'hoyo_front'),
            )
            # buf_2d: (N, T, 14, 2) - 最後のフレームだけ使う
            current_frame = buf_2d[:, -1:, :, :]  # (N, 1, 14, 2)
            
            # バッファ初期化
            if self._long_buffer is None:
                N = current_frame.shape[0]
                self._long_buffer = torch.zeros(
                    (N, self._long_buffer_size, 14, 2),
                    device=current_frame.device,
                    dtype=current_frame.dtype
                )
            
            # 循環バッファに追加
            self._long_buffer[:, self._long_buffer_ptr, :, :] = current_frame[:, 0, :, :]
            self._long_buffer_ptr = (self._long_buffer_ptr + 1) % self._long_buffer_size
            self._long_buffer_filled = min(self._long_buffer_filled + 1, self._long_buffer_size)
        except Exception as e:
            # サイレント失敗対策: 最初の1回だけ警告
            if not self._longbuf_warned:
                logger.warning(f"Long buffer accumulation disabled: {e}")
                self._longbuf_warned = True
            return

    def set_style(self, onomatopoeia: str, env_ids: list[int] | None = None) -> None:
        """
        指定した環境にオノマトペスタイルを設定.

        Args:
            onomatopoeia: 設定するオノマトペ
            env_ids: 対象のenv ID (None = 全環境)

        Raises:
            ValueError: style_generatorが見つからない場合
        """
        if self._style_gen is None:
            raise ValueError("style_command term not found in command_manager")

        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # キャッシュから取得または計算
        if onomatopoeia in self._style_cache:
            z_onm, teacher_motion = self._style_cache[onomatopoeia]
        else:
            z_onm, teacher_motion = self._style_gen.style_module.encode_instruction(onomatopoeia)
            z_onm = z_onm.squeeze(0)
            teacher_motion = teacher_motion.squeeze(0)
            self._style_cache[onomatopoeia] = (z_onm, teacher_motion)

        # 各環境に設定
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        for env_id in env_ids:
            self._style_gen.current_texts[env_id] = onomatopoeia
        self._style_gen.style_latents[env_ids_tensor] = z_onm
        self._style_gen.teacher_motion_latents[env_ids_tensor] = teacher_motion
        self._style_gen._command[env_ids_tensor, :512] = z_onm
        self._style_gen._command[env_ids_tensor, 512:] = teacher_motion

    def set_velocity_command(self, lin_vel_x: float, lin_vel_y: float, ang_vel_z: float) -> None:
        """速度コマンドを設定."""
        if self._base_cmd_term is None:
            logger.warning("base_velocity command not found")
            return

        cmd_tensor = self._base_cmd_term.command
        for i in range(self.num_envs):
            cmd_tensor[i, 0] = lin_vel_x
            if cmd_tensor.shape[1] > 1:
                cmd_tensor[i, 1] = lin_vel_y
            if cmd_tensor.shape[1] > 2:
                cmd_tensor[i, 2] = ang_vel_z

    def get_robot_state(self) -> RobotState:
        """現在のロボット状態を取得."""
        robot = self.env.unwrapped.scene["robot"]
        return RobotState(
            lin_vel=getattr(robot.data, "root_lin_vel_b", robot.data.root_lin_vel_w),
            ang_vel=getattr(robot.data, "root_ang_vel_b", robot.data.root_ang_vel_w),
            root_pos=robot.data.root_pos_w,
            root_quat=robot.data.root_quat_w,
        )

    def get_teacher_embedding(self) -> torch.Tensor | None:
        """現在の教師motion embedding を取得."""
        if self._style_gen is None:
            return None
        return self._style_gen.teacher_motion_latents

    def get_teacher_embedding_for_style(self, onomatopoeia: str) -> torch.Tensor | None:
        """指定スタイルの教師motion embedding を取得."""
        if onomatopoeia in self._style_cache:
            return self._style_cache[onomatopoeia][1]
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return None
        z_onm, teacher_motion = self._style_gen.style_module.encode_instruction(onomatopoeia)
        z_onm = z_onm.squeeze(0)
        teacher_motion = teacher_motion.squeeze(0)
        self._style_cache[onomatopoeia] = (z_onm, teacher_motion)
        return teacher_motion

    def encode_hoyo_reference(self, hoyo_sample: np.ndarray | torch.Tensor) -> torch.Tensor | None:
        """HOYO参照キーポイントをMotionCLIPでエンコード."""
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return None
        try:
            style_module = self._style_gen.style_module
            z = style_module.encode_hoyo_sample(hoyo_sample)
            return z.to(self.device)
        except Exception as e:
            logger.debug(f"Failed to encode HOYO reference: {e}")
            return None

    def get_current_motion_embedding(self) -> torch.Tensor | None:
        """現在のH1 motion embeddingを取得."""
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return None
        style_module = self._style_gen.style_module

        # Warmup check
        if hasattr(style_module, "warmup_counter") and hasattr(style_module, "warmup_frames"):
            if (style_module.warmup_counter < style_module.warmup_frames).all():
                return None

        try:
            buf_2d = style_module.get_buffer_for_hoyo_comparison(
                apply_yaw_correction=True,
                centering="first_frame_com",
                coord_mode=getattr(style_module, "coord_mode", "hoyo_front"),
            )
            return style_module.encode_hoyo_sample(buf_2d)
        except AttributeError:
            return style_module.encode_buffer() if hasattr(style_module, "encode_buffer") else None

    def get_2d_keypoints(self) -> np.ndarray | None:
        """HOYO形式の2Dキーポイントを取得 (N, T, 14, 2)."""
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return None
        style_module = self._style_gen.style_module
        try:
            buf_2d = style_module.get_buffer_for_hoyo_comparison(
                apply_yaw_correction=True,
                centering="first_frame_com",
                coord_mode=getattr(style_module, "coord_mode", "hoyo_front"),
            )
            return buf_2d.detach().cpu().numpy()
        except Exception:
            return None

    def get_extras(self) -> dict:
        """環境のextras情報を取得."""
        return getattr(self.env.unwrapped, "extras", {})

    def get_centroid_embedding(self, onomatopoeia: str) -> torch.Tensor | None:
        """
        指定オノマトペのセントロイド（クラス平均）embeddingを取得.
        
        latent_snapshot_final.npz から事前計算済みのセントロイドを取得。
        訓練時と同じ前処理で計算されているため、前処理の不一致問題がない。

        Args:
            onomatopoeia: オノマトペ文字列

        Returns:
            セントロイド embedding (512,) or None
        """
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return None
        style_module = self._style_gen.style_module
        
        label_to_id = getattr(style_module, "label_to_id", {})
        class_centroids = getattr(style_module, "class_centroids", {})
        
        if onomatopoeia not in label_to_id:
            return None
        
        lab_idx = label_to_id[onomatopoeia]
        if lab_idx not in class_centroids:
            return None
        
        return class_centroids[lab_idx]

    def get_random_sample_embedding(self, onomatopoeia: str, rng: random.Random | None = None) -> torch.Tensor | None:
        """
        指定オノマトペのランダムサンプルembeddingを取得.
        
        latent_snapshot_final.npz から事前計算済みのlatentをランダムに1つ選択。
        訓練時と同じ前処理で計算されているため、前処理の不一致問題がない。

        Args:
            onomatopoeia: オノマトペ文字列
            rng: 乱数生成器（再現性のため）

        Returns:
            ランダムサンプル embedding (512,) or None
        """
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return None
        style_module = self._style_gen.style_module
        
        label_to_id = getattr(style_module, "label_to_id", {})
        class_latents = getattr(style_module, "class_latents", {})
        
        if onomatopoeia not in label_to_id:
            return None
        
        lab_idx = label_to_id[onomatopoeia]
        if lab_idx not in class_latents:
            return None
        
        z_lab = class_latents[lab_idx]
        if z_lab.shape[0] == 0:
            return None
        
        # ランダムに1つ選択
        if rng is None:
            rng = random.Random()
        idx = rng.randint(0, z_lab.shape[0] - 1)
        return z_lab[idx]

    def encode_sliding_window(self, window_size: int, shift: int) -> torch.Tensor | None:
        """
        ロングバッファからスライディングウィンドウでエンコード.
        
        ロングバッファに蓄積された2Dキーポイントから、指定したshift位置から始まる
        window_sizeフレームの窓を切り出してMotionCLIPでエンコードする。
        
        Args:
            window_size: 窓サイズ（フレーム数、通常100）
            shift: シフト量（0で最新のwindow_sizeフレーム）
        
        Returns:
            Motion embedding (N, 512) or None
        """
        if self._long_buffer is None or self._long_buffer_filled < window_size + shift:
            return None
        if self._style_gen is None or not hasattr(self._style_gen, "style_module"):
            return None
        
        style_module = self._style_gen.style_module
        
        # 循環バッファから連続したwindow_sizeフレームを切り出す
        # _long_buffer_ptr は次の書き込み位置なので、最新フレームは ptr-1
        # shift=0: [ptr-window_size : ptr] の範囲
        # shift=10: [ptr-window_size-10 : ptr-10] の範囲
        N = self._long_buffer.shape[0]
        start = (self._long_buffer_ptr - window_size - shift) % self._long_buffer_size
        
        # 循環バッファから窓を取り出す
        indices = [(start + i) % self._long_buffer_size for i in range(window_size)]
        window = self._long_buffer[:, indices, :, :]  # (N, window_size, 14, 2)
        
        try:
            z = style_module.encode_hoyo_sample(window)
            return F.normalize(z, dim=-1)
        except Exception as e:
            logger.debug(f"Sliding window encode error: {e}")
            return None


def detect_falls(env, dones: torch.Tensor, extras: dict | None) -> torch.Tensor:
    """落下を検出."""
    done_mask = dones.bool()
    if not done_mask.any():
        return done_mask

    term_mgr = getattr(env.unwrapped, "termination_manager", None)
    if term_mgr is not None and hasattr(term_mgr, "find_terms"):
        try:
            term_names = term_mgr.find_terms("base_contact")
            if term_names:
                base_contact = term_mgr.get_term(term_names[0]).bool()
                return done_mask & base_contact
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Could not detect base_contact: {e}")

    # Fallback: timeout以外をfallとみなす
    time_outs = None
    if isinstance(extras, dict):
        time_outs = extras.get("time_outs")
    if time_outs is not None:
        return done_mask & (~time_outs.bool())
    return done_mask


# =============================================================================
# Video Recorder
# =============================================================================

class VideoRecorder:
    """ビデオ録画クラス."""

    def __init__(self, path: str, fps: int = 50):
        self.path = path
        self.fps = fps
        self._writer = None

    def start(self) -> None:
        """録画開始."""
        self._writer = imageio.get_writer(self.path, fps=self.fps)
        logger.info(f"Video recording started: {self.path}")

    def add_frame(self, frame: np.ndarray) -> None:
        """フレームを追加."""
        if self._writer is not None and frame is not None:
            self._writer.append_data(frame)

    def stop(self) -> None:
        """録画停止."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info(f"Video saved: {self.path}")


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_single_style(
    env_wrapper: EvalEnvWrapper,
    policy,
    onomatopoeia: str,
    config: EvalConfig,
    reference_keypoints: np.ndarray | None = None,
    teacher_embeddings: dict[str, torch.Tensor] | None = None,
    hoyo_embeddings: dict[str, torch.Tensor] | None = None,
    hoyo_embeddings_centered: dict[str, torch.Tensor] | None = None,
    hoyo_center: torch.Tensor | None = None,
    video_path: str | None = None,
    actor_critic=None,  # rate^adv 用 (V(s) 計算)
) -> EvaluationResult:
    """
    1つのオノマトペスタイルを評価.

    Args:
        env_wrapper: 環境ラッパー
        policy: 推論ポリシー
        onomatopoeia: 評価するオノマトペ
        config: 評価設定
        reference_keypoints: 参照動作のキーポイント (T, 14, 2) - 前処理済み (optional)
        teacher_embeddings: 教師埋め込み（スタイル別）
        hoyo_embeddings: HOYO参照埋め込み (raw)
        hoyo_embeddings_centered: HOYO参照埋め込み (中心化済み)
        hoyo_center: HOYO参照埋め込みの中心ベクトル
        video_path: ビデオ保存パス (optional)

    Returns:
        EvaluationResult: 評価結果

    cos類似度の計算（訓練時と同じ前処理）:
        1. H1 motion → encode_buffer() と同等の処理 → z_h1
        2. セントロイド: latent_snapshot_final.npz の class_centroids → z_centroid  
        3. ランダムサンプル: latent_snapshot_final.npz の class_latents からランダム選択 → z_random
        - cos_centroid = dot(z_h1, z_centroid)
        - cos_random_sample = dot(z_h1, z_random)
    """
    metrics = MotionMetrics()
    episode_step_counts = torch.zeros(env_wrapper.num_envs, device=env_wrapper.device)

    def _pick_value_tensor(out):
        if isinstance(out, (tuple, list)):
            candidates = list(out)
        else:
            candidates = [out]
        best = None
        for x in candidates:
            if not isinstance(x, torch.Tensor) or x.ndim < 1:
                continue
            if x.shape[0] != env_wrapper.num_envs:
                continue
            if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
                return x
            if best is None:
                best = x
        return best

    def _init_adv_stats():
        return {"min": None, "max": None, "sum": 0.0, "count": 0}

    def _update_adv_stats(stats: dict, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        if not isinstance(tensor, torch.Tensor):
            return
        values = tensor.detach().float().view(-1)
        if values.numel() == 0:
            return
        cur_min = float(values.min().item())
        cur_max = float(values.max().item())
        entry = stats.setdefault(name, _init_adv_stats())
        entry["min"] = cur_min if entry["min"] is None else min(entry["min"], cur_min)
        entry["max"] = cur_max if entry["max"] is None else max(entry["max"], cur_max)
        entry["sum"] += float(values.sum().item())
        entry["count"] += int(values.numel())

    def _summarize_adv_stats(stats: dict) -> str:
        lines = []
        for key in ("values_pre", "values_next", "delta_t", "r_style", "r_base"):
            entry = stats.get(key)
            if not entry or entry["count"] == 0:
                lines.append(f"  - {key}: (no samples)")
                continue
            mean = entry["sum"] / entry["count"]
            lines.append(
                f"  - {key}: min={entry['min']:.6f}, mean={mean:.6f}, max={entry['max']:.6f} (n={entry['count']})"
            )
        return "\n".join(lines)

    def _get_critic_obs_dim() -> int | None:
        if actor_critic is None:
            return None
        critic = getattr(actor_critic, "critic", None)
        if critic is None:
            return None
        for module in critic.modules():
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
        return None

    def _match_critic_obs(tensor: torch.Tensor | None, target_dim: int | None) -> torch.Tensor | None:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim < 2:
            return None
        if target_dim is None or tensor.shape[1] == target_dim:
            return tensor
        return None

    def _extract_critic_obs(obs, info_like, target_dim: int | None) -> torch.Tensor | None:
        candidates = []
        if isinstance(info_like, dict):
            obs_dict = info_like.get("observations")
            if isinstance(obs_dict, dict):
                for key in ("critic", "critic_obs", "critic_observations"):
                    if key in obs_dict:
                        candidates.append(obs_dict[key])
            for key in ("critic", "critic_obs", "critic_observations"):
                if key in info_like:
                    candidates.append(info_like[key])
        if isinstance(obs, dict):
            for key in ("critic", "critic_obs", "critic_observations"):
                if key in obs:
                    candidates.append(obs[key])
            obs_dict = obs.get("observations")
            if isinstance(obs_dict, dict):
                for key in ("critic", "critic_obs", "critic_observations"):
                    if key in obs_dict:
                        candidates.append(obs_dict[key])
        for candidate in candidates:
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.to(device=env_wrapper.device)
                matched = _match_critic_obs(candidate, target_dim)
                if matched is not None:
                    return matched
        if isinstance(obs, torch.Tensor):
            return _match_critic_obs(obs, target_dim)
        return None

    # Reward breakdown accumulators (optional)
    reward_term_names: list[str] | None = None
    reward_terms_sum: torch.Tensor | None = None
    reward_terms_sumsq: torch.Tensor | None = None
    reward_terms_active_mask: torch.Tensor | None = None
    reward_terms_count: int = 0
    reward_total_sum: torch.Tensor | None = None
    reward_total_sumsq: torch.Tensor | None = None
    reward_total_count: int = 0
    reward_step_dt: float | None = None
    # rate^mag: 絶対値の合計
    reward_terms_abs_sum: torch.Tensor | None = None
    # share^E: 関節エネルギー (action^2)
    joint_names: list[str] | None = None
    action_sq_sum: torch.Tensor | None = None  # (num_joints,)
    action_count: int = 0
    # rate^adv: online TD-residual samples (episode endを待たない)
    adv_r_samples: list[torch.Tensor] = []  # list of (N, K)
    adv_a_samples: list[torch.Tensor] = []  # list of (N,)
    # debug stats for rate^adv (min/mean/max over one episode)
    adv_debug_enabled = bool(config.log_adv_stats and config.log_reward_terms)
    adv_debug_active = adv_debug_enabled
    adv_debug_stats: dict[str, dict] = {}
    adv_debug_missing_terms_warned = False
    adv_debug_missing_critic_warned = False
    style_term_index: int | None = None
    critic_obs_dim = _get_critic_obs_dim()

    # 事前にセントロイドとランダムサンプルのembeddingを取得
    centroid_emb = env_wrapper.get_centroid_embedding(onomatopoeia)
    random_sample_emb = env_wrapper.get_random_sample_embedding(onomatopoeia, rng=random.Random(42))
    teacher_style_emb = env_wrapper.get_teacher_embedding_for_style(onomatopoeia)
    normal_teacher_emb = None
    normal_label = None
    for candidate in NORMAL_STYLE_LABEL_CANDIDATES:
        normal_teacher_emb = env_wrapper.get_teacher_embedding_for_style(candidate)
        if normal_teacher_emb is not None:
            normal_label = candidate
            break
    
    # デバイス統一 (GPU/CPU不一致対策)
    device = env_wrapper.device
    if centroid_emb is not None:
        centroid_emb = centroid_emb.to(device)
        print(f"    [INFO] Centroid embedding loaded for {onomatopoeia}")
    if random_sample_emb is not None:
        random_sample_emb = random_sample_emb.to(device)
        print(f"    [INFO] Random sample embedding loaded for {onomatopoeia}")
    if teacher_style_emb is not None:
        teacher_style_emb = teacher_style_emb.to(device)
    if normal_teacher_emb is not None:
        normal_teacher_emb = normal_teacher_emb.to(device)
    if teacher_style_emb is None or normal_teacher_emb is None:
        logger.warning(
            "Style score disabled: teacher embedding missing "
            f"(style={onomatopoeia}, normal={normal_label or 'N/A'})."
        )

    if teacher_embeddings:
        teacher_embeddings = {
            k: v.to(device) for k, v in teacher_embeddings.items() if v is not None
        }
    if hoyo_embeddings:
        hoyo_embeddings = {k: v.to(device) for k, v in hoyo_embeddings.items() if v is not None}
    if hoyo_embeddings_centered:
        hoyo_embeddings_centered = {
            k: v.to(device) for k, v in hoyo_embeddings_centered.items() if v is not None
        }
    if hoyo_center is not None:
        hoyo_center = hoyo_center.to(device)

    # Video recorder
    recorder = None
    if video_path and config.record_video:
        recorder = VideoRecorder(video_path, fps=config.video_fps)
        recorder.start()

    # Reset and set style
    obs, reset_extras = env_wrapper.reset()
    critic_obs = _extract_critic_obs(obs, reset_extras, critic_obs_dim)
    env_wrapper.set_style(onomatopoeia)
    env_wrapper.set_velocity_command(config.lin_vel_x, config.lin_vel_y, config.ang_vel_z)

    # 初期位置を記録 (CoM相対位置計算用)
    initial_state = env_wrapper.get_robot_state()
    initial_pos_x = initial_state.root_pos[:, 0].clone()

    num_steps = config.eval_steps

    for step in tqdm(range(num_steps), desc=f"Evaluating {onomatopoeia}", leave=False):
        with torch.no_grad():
            actions = policy(obs)
            # rate^adv: V(s_t) を事前に計算（obs は step 前の状態）
            values_pre = None
            if config.log_reward_terms and actor_critic is not None:
                try:
                    critic_input = critic_obs if critic_obs is not None else obs
                    if hasattr(actor_critic, "critic"):
                        values_pre = actor_critic.critic(critic_input)
                    elif hasattr(actor_critic, "evaluate"):
                        values_pre = _pick_value_tensor(actor_critic.evaluate(critic_input))
                except Exception as e:
                    logger.debug(f"rate^adv V(s) pre-step calculation skipped: {e}")
                if adv_debug_active and values_pre is None and not adv_debug_missing_critic_warned:
                    logger.warning("rate^adv debug: critic_obs not available or shape mismatch; values_pre is None.")
                    adv_debug_missing_critic_warned = True
            if adv_debug_active:
                _update_adv_stats(adv_debug_stats, "values_pre", values_pre)
            obs, rewards, dones, infos = env_wrapper.step(actions)
            critic_obs_next = _extract_critic_obs(obs, infos, critic_obs_dim)

            # Get robot state
            state = env_wrapper.get_robot_state()

            # Reward breakdown (weighted, per-step contributions)
            if config.log_reward_terms:
                base_env = env_wrapper.env.unwrapped
                reward_mgr = getattr(base_env, "reward_manager", None)
                per_step_terms = None
                if reward_mgr is not None and hasattr(reward_mgr, "active_terms") and hasattr(reward_mgr, "_step_reward"):
                    if reward_step_dt is None:
                        reward_step_dt = float(getattr(base_env, "step_dt", 1.0))
                    term_names = list(getattr(reward_mgr, "active_terms", []))
                    step_terms = reward_mgr._step_reward
                    if (
                        reward_term_names is None
                        and isinstance(step_terms, torch.Tensor)
                        and step_terms.ndim == 2
                        and step_terms.shape[1] == len(term_names)
                    ):
                        reward_term_names = term_names
                        reward_terms_sum = torch.zeros(step_terms.shape[1], device=step_terms.device, dtype=torch.float)
                        reward_terms_sumsq = torch.zeros(step_terms.shape[1], device=step_terms.device, dtype=torch.float)
                        # Mask out terms with zero weight (RewardManager does not update _step_reward for them).
                        try:
                            weights = []
                            for name in reward_term_names:
                                try:
                                    weights.append(float(reward_mgr.get_term_cfg(name).weight))
                                except Exception:
                                    weights.append(1.0)
                            w = torch.tensor(weights, device=step_terms.device, dtype=torch.float)
                            reward_terms_active_mask = (w != 0.0).to(dtype=torch.float)
                        except Exception:
                            reward_terms_active_mask = None
                        if style_term_index is None:
                            for i, name in enumerate(reward_term_names):
                                if "style" in name:
                                    style_term_index = i
                                    break
                    if reward_term_names is not None and reward_terms_sum is not None and reward_terms_sumsq is not None:
                        # Convert to per-step contributions (RewardManager stores raw*weight in _step_reward)
                        per_step_terms = step_terms * reward_step_dt
                        if reward_terms_active_mask is not None and reward_terms_active_mask.numel() == per_step_terms.shape[1]:
                            per_step_terms = per_step_terms * reward_terms_active_mask
                        reward_terms_sum += per_step_terms.sum(dim=0)
                        reward_terms_sumsq += (per_step_terms * per_step_terms).sum(dim=0)
                        reward_terms_count += int(per_step_terms.shape[0])
                        # rate^mag: 絶対値の合計
                        if reward_terms_abs_sum is None:
                            reward_terms_abs_sum = torch.zeros(per_step_terms.shape[1], device=per_step_terms.device, dtype=torch.float)
                        reward_terms_abs_sum += per_step_terms.abs().sum(dim=0)

                        # --- rate^adv: online TD residual (episode end不要) ---
                        if (
                            actor_critic is not None
                            and values_pre is not None
                            and isinstance(per_step_terms, torch.Tensor)
                        ):
                            gamma = config.gamma if config.gamma is not None else 0.99
                            values_next = None
                            try:
                                critic_input_next = critic_obs_next if critic_obs_next is not None else obs
                                if hasattr(actor_critic, "critic"):
                                    values_next = actor_critic.critic(critic_input_next)
                            except Exception as e:
                                logger.debug(f"rate^adv values_next skipped: {e}")
                            if values_next is not None:
                                if adv_debug_active:
                                    _update_adv_stats(adv_debug_stats, "values_next", values_next)
                                v_t = values_pre.squeeze(-1)
                                v_next = values_next.squeeze(-1)
                                if v_t.shape[0] == per_step_terms.shape[0] and v_next.shape[0] == per_step_terms.shape[0]:
                                    r_style = None
                                    if style_term_index is not None and style_term_index < per_step_terms.shape[1]:
                                        r_style = per_step_terms[:, style_term_index]
                                        _update_adv_stats(adv_debug_stats, "r_style", r_style)
                                    r_base = per_step_terms.sum(dim=1)
                                    if r_style is not None:
                                        r_base = r_base - r_style
                                    _update_adv_stats(adv_debug_stats, "r_base", r_base)
                                    r_total = per_step_terms.sum(dim=1)  # (N,)
                                    done_mask = torch.as_tensor(dones, device=per_step_terms.device, dtype=torch.float32)
                                    if isinstance(infos, dict) and "time_outs" in infos:
                                        time_outs = torch.as_tensor(infos["time_outs"], device=per_step_terms.device).bool()
                                        done_mask = done_mask * (~time_outs).to(dtype=torch.float32)
                                    td_residual = r_total + gamma * (1.0 - done_mask) * v_next - v_t  # (N,)
                                    _update_adv_stats(adv_debug_stats, "delta_t", td_residual)
                                    adv_r_samples.append(per_step_terms.detach())
                                    adv_a_samples.append(td_residual.detach())
                            if values_next is None and adv_debug_active and not adv_debug_missing_critic_warned:
                                logger.warning("rate^adv debug: critic_obs_next unavailable or shape mismatch; values_next is None.")
                                adv_debug_missing_critic_warned = True
                elif adv_debug_active and not adv_debug_missing_terms_warned:
                    logger.warning("rate^adv debug: reward_manager or step_reward is unavailable; per_step_terms empty.")
                    adv_debug_missing_terms_warned = True

                # Total reward stats (as returned by env.step)
                if isinstance(rewards, torch.Tensor) and rewards.numel() > 0:
                    if reward_total_sum is None:
                        reward_total_sum = torch.zeros((), device=rewards.device, dtype=torch.float)
                        reward_total_sumsq = torch.zeros((), device=rewards.device, dtype=torch.float)
                    reward_total_sum += rewards.sum()
                    reward_total_sumsq += (rewards * rewards).sum()
                    reward_total_count += int(rewards.numel())

            # share^E: 関節エネルギー (action^2)
            if config.log_reward_terms and actions is not None:
                if action_sq_sum is None:
                    action_sq_sum = torch.zeros(actions.shape[-1], device=actions.device, dtype=torch.float)
                    # 関節名を取得（可能であれば）
                    try:
                        robot = env_wrapper.env.unwrapped.scene.get("robot")
                        if robot is not None and hasattr(robot, "joint_names"):
                            joint_names = list(robot.joint_names)
                        else:
                            joint_names = [f"joint_{i}" for i in range(actions.shape[-1])]
                    except Exception:
                        joint_names = [f"joint_{i}" for i in range(actions.shape[-1])]
                action_sq_sum += (actions ** 2).sum(dim=0)
                action_count += actions.shape[0]

            # --- Collect metrics ---
            # Velocity
            metrics.velocity_x.append(state.lin_vel[:, 0].mean().item())
            metrics.velocity_y.append(state.lin_vel[:, 1].mean().item())
            metrics.velocity_z.append(state.lin_vel[:, 2].mean().item())

            # CoM X (初期位置からの相対移動距離)
            com_x_relative = (state.root_pos[:, 0] - initial_pos_x).mean().item()
            metrics.com_x.append(com_x_relative)

            # Roll/Pitch
            rpy = math_utils.euler_xyz_from_quat(state.root_quat)
            if isinstance(rpy, tuple):
                roll, pitch = rpy[0], rpy[1]
            else:
                roll, pitch = rpy[:, 0], rpy[:, 1]
            roll = math_utils.wrap_to_pi(roll)
            pitch = math_utils.wrap_to_pi(pitch)
            metrics.roll.append(roll.abs().mean().item())
            metrics.pitch.append(pitch.abs().mean().item())

            # Cosine similarity (10ステップごとに計算)
            if step % 10 == 0:
                try:
                    # H1の現在のmotion embedding (encode_buffer と同等)
                    motion_emb = env_wrapper.get_current_motion_embedding()
                    if motion_emb is not None:
                        # 1. H1 vs Centroid (クラス平均latent)
                        if centroid_emb is not None:
                            sim = F.cosine_similarity(
                                motion_emb, centroid_emb.view(1, -1), dim=-1
                            )
                            metrics.cos_centroid.extend(sim.detach().cpu().tolist())

                        # 2. H1 vs Random Sample (latent_snapshotからランダム選択)
                        if random_sample_emb is not None:
                            sim = F.cosine_similarity(
                                motion_emb, random_sample_emb.view(1, -1), dim=-1
                            )
                            metrics.cos_random_sample.extend(sim.detach().cpu().tolist())

                        # 2.5 相対スタイル指標 (教師スタイル - 教師通常)
                        if teacher_style_emb is not None and normal_teacher_emb is not None:
                            sim_style = F.cosine_similarity(
                                motion_emb, teacher_style_emb.view(1, -1), dim=-1
                            )
                            sim_normal = F.cosine_similarity(
                                motion_emb, normal_teacher_emb.view(1, -1), dim=-1
                            )
                            metrics.style_score.extend((sim_style - sim_normal).detach().cpu().tolist())

                        # 2.55 H1 vs Teacher embeddings (スタイル別)
                        if teacher_embeddings:
                            for label, teacher_emb in teacher_embeddings.items():
                                sim = F.cosine_similarity(
                                    motion_emb, teacher_emb.view(1, -1), dim=-1
                                )
                                if label not in metrics.teacher_similarity:
                                    metrics.teacher_similarity[label] = []
                                metrics.teacher_similarity[label].append(float(sim.mean().item()))

                        # 2.6 H1 vs HOYO reference (スタイル別)
                        if hoyo_embeddings:
                            for label, hoyo_emb in hoyo_embeddings.items():
                                sim = F.cosine_similarity(
                                    motion_emb, hoyo_emb.view(1, -1), dim=-1
                                )
                                if label not in metrics.hoyo_similarity:
                                    metrics.hoyo_similarity[label] = []
                                metrics.hoyo_similarity[label].append(float(sim.mean().item()))

                        # 2.7 H1 vs HOYO reference (中心化・スタイル別)
                        if hoyo_embeddings_centered and hoyo_center is not None:
                            motion_centered = F.normalize(motion_emb - hoyo_center, dim=-1)
                            for label, hoyo_emb in hoyo_embeddings_centered.items():
                                sim = (motion_centered * hoyo_emb.view(1, -1)).sum(dim=-1)
                                if label not in metrics.hoyo_similarity_centered:
                                    metrics.hoyo_similarity_centered[label] = []
                                metrics.hoyo_similarity_centered[label].append(float(sim.mean().item()))
                        
                        # 3. 時間シフト解析 (オプション) - スライディングウィンドウ方式
                        if config.time_shift_analyze and random_sample_emb is not None:
                            # ロングバッファからスライディングウィンドウで解析
                            # window_size=100 (style_moduleと同じ)
                            window_size = 100
                            stride = config.time_shift_stride
                            max_shift = config.time_shift_max
                            
                            for shift in range(0, max_shift + 1, stride):
                                # エピソード跨ぎ対策: 十分なステップ数が経過したenvのみ対象
                                valid_env_mask = episode_step_counts >= (window_size + shift)
                                valid_ids = torch.where(valid_env_mask)[0].tolist()
                                
                                if len(valid_ids) == 0:
                                    continue
                                
                                z_shifted = env_wrapper.encode_sliding_window(window_size, shift)
                                if z_shifted is not None:
                                    sim = F.cosine_similarity(
                                        z_shifted, random_sample_emb.view(1, -1), dim=-1
                                    )
                                    sim_vals = sim[valid_ids].detach().cpu().tolist()
                                    if sim_vals:
                                        if shift not in metrics.cos_time_shift:
                                            metrics.cos_time_shift[shift] = []
                                        metrics.cos_time_shift[shift].extend(sim_vals)
                except CosineSimilarityError as e:
                    logger.debug(f"Skipping cos similarity: {e}")

            # Joint error (with reference)
            if reference_keypoints is not None and step % 20 == 0:
                try:
                    current_kp = env_wrapper.get_2d_keypoints()
                    if current_kp is not None:
                        # 1. 従来のL2誤差（位相アライメントなし）
                        error = compute_l2_joint_error(current_kp[0], reference_keypoints)
                        metrics.joint_error.append(error)

                        # 2. DTW誤差（位相アライメントあり、L2 Errと同じ単位）
                        if DTW_AVAILABLE:
                            try:
                                error_dtw = compute_l2_joint_error_with_dtw(
                                    current_kp[0], reference_keypoints
                                )
                                metrics.joint_error_dtw.append(error_dtw)
                            except JointErrorCalculationError as e:
                                logger.debug(f"Skipping DTW joint error: {e}")
                except JointErrorCalculationError as e:
                    logger.debug(f"Skipping joint error: {e}")

            # Episode tracking
            episode_step_counts += 1
            if dones.any():
                for i in range(env_wrapper.num_envs):
                    if dones[i]:
                        metrics.episode_lengths.append(int(episode_step_counts[i].item()))
                        episode_step_counts[i] = 0
                        metrics.episode_count += 1

                fall_mask = detect_falls(env_wrapper.env, dones, infos if isinstance(infos, dict) else None)
                metrics.fall_count += int(fall_mask.sum().item())

                # Reset style after episode ends
                done_ids = torch.where(dones)[0].tolist()
                env_wrapper.set_style(onomatopoeia, env_ids=done_ids)

                if adv_debug_active:
                    logger.info("rate^adv debug stats (first episode):\n%s", _summarize_adv_stats(adv_debug_stats))
                    adv_debug_active = False

            if critic_obs_next is not None:
                critic_obs = critic_obs_next

            # Video frame
            if recorder is not None:
                base_env = env_wrapper.env.unwrapped
                frame = base_env.render()
                recorder.add_frame(frame)

                # Update camera
                robot_pos = state.root_pos[0].detach().cpu().numpy()
                cam_eye = (robot_pos[0] + 3.0, robot_pos[1] + 3.0, robot_pos[2] + 2.0)
                cam_target = (robot_pos[0], robot_pos[1], robot_pos[2])
                base_env.sim.set_camera_view(eye=cam_eye, target=cam_target)

                # Stop recording after video_length frames (keep evaluation running)
                if config.video_length is not None and config.video_length > 0:
                    if step + 1 >= config.video_length:
                        recorder.stop()
                        recorder = None

    if recorder is not None:
        recorder.stop()

    if adv_debug_active and adv_debug_stats:
        logger.info("rate^adv debug stats (no episode end, summary over eval):\n%s", _summarize_adv_stats(adv_debug_stats))

    # Reward breakdown summary
    reward_terms_mean = None
    reward_terms_std = None
    mean_reward_total = None
    std_reward_total = None
    if config.log_reward_terms and reward_term_names and reward_terms_sum is not None and reward_terms_sumsq is not None and reward_terms_count > 0:
        mean_t = reward_terms_sum / float(reward_terms_count)
        var_t = reward_terms_sumsq / float(reward_terms_count) - mean_t * mean_t
        std_t = torch.sqrt(torch.clamp(var_t, min=0.0))
        reward_terms_mean = {name: float(mean_t[i].item()) for i, name in enumerate(reward_term_names)}
        reward_terms_std = {name: float(std_t[i].item()) for i, name in enumerate(reward_term_names)}

    if config.log_reward_terms and reward_total_sum is not None and reward_total_sumsq is not None and reward_total_count > 0:
        mean_total = reward_total_sum / float(reward_total_count)
        var_total = reward_total_sumsq / float(reward_total_count) - mean_total * mean_total
        mean_reward_total = float(mean_total.item())
        std_reward_total = float(torch.sqrt(torch.clamp(var_total, min=0.0)).item())

    hoyo_similarity_mean = None
    hoyo_similarity_std = None
    if metrics.hoyo_similarity:
        hoyo_similarity_mean = {
            label: float(np.mean(vals)) for label, vals in metrics.hoyo_similarity.items() if vals
        }
        hoyo_similarity_std = {
            label: float(np.std(vals)) for label, vals in metrics.hoyo_similarity.items() if vals
        }

    teacher_similarity_mean = None
    teacher_similarity_std = None
    if metrics.teacher_similarity:
        teacher_similarity_mean = {
            label: float(np.mean(vals)) for label, vals in metrics.teacher_similarity.items() if vals
        }
        teacher_similarity_std = {
            label: float(np.std(vals)) for label, vals in metrics.teacher_similarity.items() if vals
        }

    hoyo_similarity_centered_mean = None
    hoyo_similarity_centered_std = None
    if metrics.hoyo_similarity_centered:
        hoyo_similarity_centered_mean = {
            label: float(np.mean(vals)) for label, vals in metrics.hoyo_similarity_centered.items() if vals
        }
        hoyo_similarity_centered_std = {
            label: float(np.std(vals)) for label, vals in metrics.hoyo_similarity_centered.items() if vals
        }

    # rate^mag: 報酬内訳寄与率(%)
    rate_mag = None
    if config.log_reward_terms and reward_term_names and reward_terms_abs_sum is not None:
        total_abs = reward_terms_abs_sum.sum().item()
        if total_abs > 1e-9:
            rate_mag = {
                name: float(reward_terms_abs_sum[i].item() / total_abs * 100)
                for i, name in enumerate(reward_term_names)
            }

    # share^E: 関節エネルギー割合(%) + 全体平均
    share_e = None
    mean_action_sq = None
    total_energy = None
    if config.log_reward_terms and action_sq_sum is not None and action_count > 0:
        total_energy = action_sq_sum.sum().item()
        mean_action_sq = float(total_energy / float(action_count))
    if joint_names and total_energy is not None and total_energy > 1e-9:
        share_e = {
            name: float(action_sq_sum[i].item() / total_energy * 100)
            for i, name in enumerate(joint_names)
        }

    # rate^adv: Advantage寄与率(%) - 共分散ベース
    rate_adv = None
    if config.log_reward_terms and reward_term_names and len(adv_r_samples) > 0 and len(adv_a_samples) > 0:
        try:
            # 全エピソードのサンプルを結合
            all_rewards = torch.cat(adv_r_samples, dim=0)  # (N_total, K)
            all_advantages = torch.cat(adv_a_samples, dim=0)  # (N_total,)
            K = all_rewards.shape[1]
            # 共分散を計算: Cov(r_k, A) = E[(r_k - E[r_k])(A - E[A])]
            r_mean = all_rewards.mean(dim=0)  # (K,)
            a_mean = all_advantages.mean()
            r_centered = all_rewards - r_mean  # (N, K)
            a_centered = all_advantages - a_mean  # (N,)
            cov = (r_centered * a_centered.unsqueeze(1)).mean(dim=0)  # (K,)
            cov_abs = cov.abs()
            total_cov = cov_abs.sum().item()
            if total_cov > 1e-9:
                rate_adv = {
                    name: float(cov_abs[i].item() / total_cov * 100)
                    for i, name in enumerate(reward_term_names)
                }
        except Exception as e:
            logger.debug(f"rate^adv calculation failed: {e}")

    # Compute summary
    return EvaluationResult(
        onomatopoeia=onomatopoeia,
        mean_velocity_x=float(np.mean(metrics.velocity_x)),
        std_velocity_x=float(np.std(metrics.velocity_x)),
        mean_com_x=float(np.mean(metrics.com_x)),
        std_com_x=float(np.std(metrics.com_x)),
        mean_cos_centroid=float(np.mean(metrics.cos_centroid)) if metrics.cos_centroid else None,
        std_cos_centroid=float(np.std(metrics.cos_centroid)) if metrics.cos_centroid else None,
        cos_centroid_count=len(metrics.cos_centroid),
        mean_cos_random_sample=float(np.mean(metrics.cos_random_sample)) if metrics.cos_random_sample else None,
        std_cos_random_sample=float(np.std(metrics.cos_random_sample)) if metrics.cos_random_sample else None,
        cos_random_sample_count=len(metrics.cos_random_sample),
        mean_style_score=float(np.mean(metrics.style_score)) if metrics.style_score else None,
        std_style_score=float(np.std(metrics.style_score)) if metrics.style_score else None,
        style_score_count=len(metrics.style_score),
        teacher_similarity_mean=teacher_similarity_mean,
        teacher_similarity_std=teacher_similarity_std,
        hoyo_similarity_mean=hoyo_similarity_mean,
        hoyo_similarity_std=hoyo_similarity_std,
        hoyo_similarity_centered_mean=hoyo_similarity_centered_mean,
        hoyo_similarity_centered_std=hoyo_similarity_centered_std,
        mean_joint_error=float(np.mean(metrics.joint_error)) if metrics.joint_error else None,
        std_joint_error=float(np.std(metrics.joint_error)) if metrics.joint_error else None,
        mean_joint_error_dtw=float(np.mean(metrics.joint_error_dtw)) if metrics.joint_error_dtw else None,
        std_joint_error_dtw=float(np.std(metrics.joint_error_dtw)) if metrics.joint_error_dtw else None,
        mean_episode_length=float(np.mean(metrics.episode_lengths)) if metrics.episode_lengths else float(num_steps),
        fall_rate=metrics.fall_count / max(1, metrics.episode_count),
        episode_count=metrics.episode_count,
        time_shift_results={shift: float(np.mean(vals)) for shift, vals in metrics.cos_time_shift.items()} if metrics.cos_time_shift else None,
        mean_reward_total=mean_reward_total,
        std_reward_total=std_reward_total,
        reward_terms_mean=reward_terms_mean,
        reward_terms_std=reward_terms_std,
        reward_step_dt=reward_step_dt,
        mean_action_sq=mean_action_sq,
        rate_mag=rate_mag,
        rate_adv=rate_adv,
        share_e=share_e,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def seed_everything(seed: int | None) -> None:
    """乱数シードを設定."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hoyo_reference(
    hoyo_root: Path,
    label: str,
    stats_path: Path | None = None,
    seed: int = 42,
) -> np.ndarray | None:
    """HOYO参照動作を読み込み."""
    try:
        # HoyoInstructionDatasetは target_labels が必須
        dataset = HoyoInstructionDataset(
            root=hoyo_root,
            target_labels=[label],
            target_len=100,
            is_train=False,  # センタークロップを使用
        )
        
        # 正規化統計を適用（あれば）
        resolved_stats_path = _resolve_hoyo_stats_path(hoyo_root, stats_path)
        if resolved_stats_path is not None:
            apply_normalization_from_stats(dataset, resolved_stats_path)
            logger.info(f"Applied normalization stats from: {resolved_stats_path}")
        
        if len(dataset) == 0:
            logger.debug(f"No samples found for label: {label}")
            return None
        
        # ランダムにサンプルを選択
        rng = random.Random(seed)
        idx = rng.randint(0, len(dataset) - 1)
        keypoints, _ = dataset[idx]
        return np.array(keypoints)
    except Exception as e:
        logger.warning(f"Failed to load HOYO reference for {label}: {e}")
        return None


def _resolve_hoyo_stats_path(hoyo_root: Path, stats_path: Path | None = None) -> Path | None:
    """Resolve normalization_stats.json path with run-specific fallback order."""
    if stats_path is not None and stats_path.exists():
        return stats_path

    style_run_name = os.environ.get("STYLE_RUN_NAME")
    candidates: list[Path] = []
    if style_run_name:
        candidates.append(
            hoyo_root / "joint_training_results" / style_run_name / "normalization_stats.json"
        )
    candidates.extend(
        [
            hoyo_root / "joint_training_results" / "normalization_stats.json",
            hoyo_root / "data" / "normalization_stats.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def compute_hoyo_static_prototype_frame0_median(
    hoyo_root: Path,
    label: str,
    target_len: int = 100,
    stats_path: Path | None = None,
) -> np.ndarray | None:
    """Build a static HOYO sequence from representative near-start static pose.

    NOTE:
    - Per-joint averaging/median can create anatomically broken poses.
    - We therefore choose one *actual* frame as representative.
    - For each sample, we choose an anchor pose from early frames with lowest local motion
      (instead of strict frame0), because some clips already start moving at frame0.
    """
    try:
        dataset = HoyoInstructionDataset(
            root=hoyo_root,
            target_labels=[label],
            target_len=target_len,
            is_train=False,
        )
        resolved_stats_path = _resolve_hoyo_stats_path(hoyo_root, stats_path)
        if resolved_stats_path is not None:
            apply_normalization_from_stats(dataset, resolved_stats_path)

        if len(dataset) == 0:
            return None

        def _center_frame0_like_dataset(frame0_xy: np.ndarray) -> np.ndarray:
            # Align with dataset centering used for HOYO comparison, but using true sequence-start frame.
            mode = getattr(dataset, "centering", "first_frame_com")
            frame = frame0_xy.astype(np.float32, copy=True)
            if mode == "pelvis":
                pelvis = 0.5 * (frame[8] + frame[11])
                frame = frame - pelvis
            elif mode == "pelvis_mean":
                pelvis = 0.5 * (frame[8] + frame[11])
                frame = frame - pelvis
            elif mode == "first_frame_pelvis":
                pelvis = 0.5 * (frame[8] + frame[11])
                frame = frame - pelvis
            else:  # "first_frame_com"
                com = frame.mean(axis=0)
                frame = frame - com

            if hasattr(dataset, "mean") and dataset.mean is not None and hasattr(dataset, "std") and dataset.std is not None:
                frame = (frame - dataset.mean) / dataset.std
            return frame.astype(np.float32)

        candidates: list[tuple[np.ndarray, float, int, int]] = []
        raw_samples = dataset.samples_by_label.get(label, [])
        for sample_idx, raw_seq in enumerate(raw_samples):
            arr = np.asarray(raw_seq)
            if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1:] != (14, 2):
                continue
            # Pick near-start low-motion anchor (robust against clips that start mid-motion).
            search_frames = min(arr.shape[0], 20)
            if search_frames <= 1:
                anchor_idx = 0
            else:
                prefix = arr[:search_frames]  # (S, 14, 2)
                step_disp = np.linalg.norm(prefix[1:] - prefix[:-1], axis=-1).mean(axis=-1)  # (S-1,)
                frame_score = np.zeros(search_frames, dtype=np.float32)
                frame_score[0] = step_disp[0]
                frame_score[-1] = step_disp[-1]
                if search_frames > 2:
                    frame_score[1:-1] = 0.5 * (step_disp[:-1] + step_disp[1:])
                anchor_idx = int(np.argmin(frame_score))
            anchor_score = float(frame_score[anchor_idx]) if search_frames > 1 else 0.0
            centered_anchor = _center_frame0_like_dataset(arr[anchor_idx])
            candidates.append((centered_anchor, anchor_score, sample_idx, anchor_idx))

        if not candidates:
            return None

        if candidates:
            selected_anchor_indices = [c[3] for c in candidates]
            logger.info(
                "HOYO static anchor indices for '%s': mean=%.2f min=%d max=%d n=%d",
                label,
                float(np.mean(selected_anchor_indices)),
                int(np.min(selected_anchor_indices)),
                int(np.max(selected_anchor_indices)),
                len(candidates),
            )

        # Select the most static candidate directly.
        # This prioritizes "clearly standing" posture over representativeness.
        best_pose, best_score, best_sample_idx, best_anchor_idx = min(candidates, key=lambda c: c[1])
        logger.info(
            "HOYO static representative for '%s': sample_idx=%d anchor_frame=%d score=%.6f",
            label,
            best_sample_idx,
            best_anchor_idx,
            best_score,
        )
        representative_pose = best_pose.astype(np.float32)

        return np.repeat(representative_pose[np.newaxis, ...], target_len, axis=0)
    except Exception as e:
        logger.warning(f"Failed to build HOYO static prototype for {label}: {e}")
        return None


def compute_h1_static_prototype(
    env_wrapper: EvalEnvWrapper,
    policy,
    style_label: str,
    warmup_steps: int,
    collect_steps: int,
    collect_stride: int,
    source: str = "policy_zero_velocity",
    target_len: int = 100,
) -> np.ndarray | None:
    """Build a static H1 sequence and return median-pose repeated sequence."""
    warmup_steps = max(0, int(warmup_steps))
    collect_steps = max(1, int(collect_steps))
    collect_stride = max(1, int(collect_stride))
    total_steps = warmup_steps + collect_steps

    def _infer_action_dim() -> int:
        base_env = env_wrapper.env.unwrapped
        action_mgr = getattr(base_env, "action_manager", None)
        if action_mgr is not None and hasattr(action_mgr, "action"):
            action_tensor = getattr(action_mgr, "action")
            if isinstance(action_tensor, torch.Tensor) and action_tensor.ndim == 2:
                return int(action_tensor.shape[1])
        for space_obj in (getattr(env_wrapper.env, "action_space", None), getattr(base_env, "action_space", None)):
            if space_obj is not None and hasattr(space_obj, "shape") and space_obj.shape:
                return int(space_obj.shape[0])
        raise RuntimeError("Failed to infer action dimension for baseline rollout.")

    def _try_apply_usd_stand_pose(reset_root: bool = False) -> bool:
        base_env = env_wrapper.env.unwrapped
        scene = getattr(base_env, "scene", None)
        if scene is None:
            return False

        def _resolve_robot_from_scene():
            # Prefer direct access used in this repo, then fall back to map-like getters.
            try:
                robot_asset = scene["robot"]
                if robot_asset is not None:
                    return robot_asset
            except Exception:
                pass
            try:
                robot_asset = scene.get("robot")
                if robot_asset is not None:
                    return robot_asset
            except Exception:
                pass
            try:
                articulations = getattr(scene, "articulations", None)
                if articulations is not None:
                    robot_asset = articulations.get("robot")
                    if robot_asset is not None:
                        return robot_asset
            except Exception:
                pass
            return None

        def _call_with_fallbacks(method, variants: list[tuple[tuple, dict]]) -> bool:
            for args, kwargs in variants:
                try:
                    method(*args, **kwargs)
                    return True
                except Exception:
                    continue
            return False

        robot = _resolve_robot_from_scene()
        if robot is None or not hasattr(robot, "data"):
            return False

        num_instances = int(getattr(robot, "num_instances", env_wrapper.num_envs))
        env_ids = torch.arange(num_instances, device=env_wrapper.device, dtype=torch.long)
        applied = False

        if reset_root and hasattr(robot.data, "default_root_state"):
            try:
                root_state = robot.data.default_root_state.clone()
                env_origins = getattr(scene, "env_origins", None)
                if isinstance(env_origins, torch.Tensor) and env_origins.shape[0] == root_state.shape[0]:
                    root_state[:, :3] += env_origins
                if hasattr(robot, "write_root_pose_to_sim"):
                    applied |= _call_with_fallbacks(
                        robot.write_root_pose_to_sim,
                        variants=[
                            ((root_state[:, :7],), {"env_ids": env_ids}),
                            ((root_state[:, :7], env_ids), {}),
                            ((root_state[:, :7],), {}),
                        ],
                    )
                if hasattr(robot, "write_root_velocity_to_sim"):
                    applied |= _call_with_fallbacks(
                        robot.write_root_velocity_to_sim,
                        variants=[
                            ((root_state[:, 7:],), {"env_ids": env_ids}),
                            ((root_state[:, 7:], env_ids), {}),
                            ((root_state[:, 7:],), {}),
                        ],
                    )
            except Exception:
                pass

        if hasattr(robot.data, "default_joint_pos") and hasattr(robot.data, "default_joint_vel"):
            try:
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                if hasattr(robot, "write_joint_state_to_sim"):
                    applied |= _call_with_fallbacks(
                        robot.write_joint_state_to_sim,
                        variants=[
                            ((default_joint_pos, default_joint_vel), {"env_ids": env_ids}),
                            ((default_joint_pos, default_joint_vel, env_ids), {}),
                            ((default_joint_pos, default_joint_vel, None, env_ids), {}),
                            ((default_joint_pos, default_joint_vel), {}),
                        ],
                    )
                if hasattr(robot, "set_joint_position_target"):
                    applied |= _call_with_fallbacks(
                        robot.set_joint_position_target,
                        variants=[
                            ((default_joint_pos,), {"env_ids": env_ids}),
                            ((default_joint_pos, env_ids), {}),
                            ((default_joint_pos,), {}),
                        ],
                    )
            except Exception:
                pass

        try:
            if applied and hasattr(scene, "write_data_to_sim"):
                scene.write_data_to_sim()
        except Exception:
            pass
        return applied

    try:
        obs, _ = env_wrapper.reset()
        env_wrapper.set_velocity_command(0.0, 0.0, 0.0)
        source = str(source).strip().lower()

        frames: list[np.ndarray] = []

        if source == "policy_zero_velocity":
            try:
                env_wrapper.set_style(style_label)
            except Exception as e:
                logger.warning(f"Failed to set baseline style '{style_label}', continuing: {e}")

            for step in range(total_steps):
                with torch.no_grad():
                    actions = policy(obs)
                    obs, _, dones, _ = env_wrapper.step(actions)

                if dones.any():
                    done_ids = torch.where(dones)[0].tolist()
                    if done_ids:
                        try:
                            env_wrapper.set_style(style_label, env_ids=done_ids)
                        except Exception:
                            pass
                        env_wrapper.set_velocity_command(0.0, 0.0, 0.0)

                if step < warmup_steps:
                    continue
                if ((step - warmup_steps) % collect_stride) != 0:
                    continue

                current_kp = env_wrapper.get_2d_keypoints()
                if current_kp is None:
                    continue
                arr = np.asarray(current_kp)
                if arr.ndim != 4 or arr.shape[1] < 1:
                    continue
                frames.append(arr[0, -1].astype(np.float32))
        elif source == "usd_stand":
            action_dim = _infer_action_dim()
            zero_actions = torch.zeros(
                (env_wrapper.num_envs, action_dim),
                device=env_wrapper.device,
                dtype=torch.float32,
            )
            applied = _try_apply_usd_stand_pose(reset_root=True)
            if not applied:
                logger.warning("USD stand baseline: failed to apply default stand pose; continuing with zero-actions only.")

            for step in range(total_steps):
                _try_apply_usd_stand_pose(reset_root=False)
                with torch.no_grad():
                    obs, _, dones, _ = env_wrapper.step(zero_actions)

                if dones.any():
                    env_wrapper.set_velocity_command(0.0, 0.0, 0.0)
                    _try_apply_usd_stand_pose(reset_root=True)

                if step < warmup_steps:
                    continue
                if ((step - warmup_steps) % collect_stride) != 0:
                    continue

                current_kp = env_wrapper.get_2d_keypoints()
                if current_kp is None:
                    continue
                arr = np.asarray(current_kp)
                if arr.ndim != 4 or arr.shape[1] < 1:
                    continue
                frames.append(arr[0, -1].astype(np.float32))
        else:
            raise ValueError(f"Unsupported H1 baseline source: {source}")

        if not frames:
            return None

        median_pose = np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)
        return np.repeat(median_pose[np.newaxis, ...], target_len, axis=0)
    except Exception as e:
        logger.warning(f"Failed to build H1 static prototype: {e}")
        return None


def apply_joint_error_baseline(
    result: EvaluationResult,
    baseline_l2: float | None,
    baseline_dtw: float | None,
) -> None:
    """Attach baseline-normalized joint-error metrics to result."""
    if baseline_l2 is not None and result.mean_joint_error is not None:
        result.mean_joint_error_delta = float(result.mean_joint_error - baseline_l2)
        if abs(baseline_l2) > 1e-8:
            result.mean_joint_error_ratio = float(result.mean_joint_error / baseline_l2)
        else:
            result.mean_joint_error_ratio = None

    if baseline_dtw is not None and result.mean_joint_error_dtw is not None:
        result.mean_joint_error_dtw_delta = float(result.mean_joint_error_dtw - baseline_dtw)
        if abs(baseline_dtw) > 1e-8:
            result.mean_joint_error_dtw_ratio = float(result.mean_joint_error_dtw / baseline_dtw)
        else:
            result.mean_joint_error_dtw_ratio = None


HOYO_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
]


def _as_keypoint_sequence(arr: np.ndarray) -> np.ndarray | None:
    data = np.asarray(arr)
    if data.ndim == 2 and data.shape == (14, 2):
        return data[np.newaxis, ...].astype(np.float32)
    if data.ndim == 3 and data.shape[1:] == (14, 2):
        return data.astype(np.float32)
    return None


def _draw_pose_frame(
    ax,
    pose: np.ndarray,
    title: str,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> None:
    x = pose[:, 0]
    y = -pose[:, 1]  # display-friendly (up-positive)

    for i, j in HOYO_SKELETON_EDGES:
        ax.plot([x[i], x[j]], [y[i], y[j]], color="#1f77b4", linewidth=2.0)
    ax.scatter(x, y, c="#d62728", s=20, zorder=5)
    ax.set_title(title)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def save_baseline_comparison_gif(
    h1_static_seq: np.ndarray,
    hoyo_static_seq: np.ndarray,
    output_path: Path,
    fps: int = 8,
    max_frames: int = 60,
    baseline_l2: float | None = None,
    baseline_dtw: float | None = None,
) -> Path | None:
    """Save side-by-side baseline prototype comparison as GIF."""
    h1_seq = _as_keypoint_sequence(h1_static_seq)
    hoyo_seq = _as_keypoint_sequence(hoyo_static_seq)
    if h1_seq is None or hoyo_seq is None:
        return None

    num_frames = min(h1_seq.shape[0], hoyo_seq.shape[0], max(1, int(max_frames)))
    if num_frames <= 0:
        return None

    h1_used = h1_seq[:num_frames]
    hoyo_used = hoyo_seq[:num_frames]

    x_all = np.concatenate([h1_used[..., 0].reshape(-1), hoyo_used[..., 0].reshape(-1)])
    y_all = np.concatenate([(-h1_used[..., 1]).reshape(-1), (-hoyo_used[..., 1]).reshape(-1)])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad_x = max(0.1, 0.12 * max(1e-6, x_max - x_min))
    pad_y = max(0.1, 0.12 * max(1e-6, y_max - y_min))
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=max(1, int(fps))) as writer:
        for t in range(num_frames):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
            _draw_pose_frame(axes[0], h1_used[t], "H1 Static Prototype", x_lim, y_lim)
            _draw_pose_frame(axes[1], hoyo_used[t], "HOYO Static Prototype", x_lim, y_lim)

            if baseline_l2 is not None:
                if baseline_dtw is not None:
                    fig.suptitle(f"Baseline L2={baseline_l2:.4f}, DTW={baseline_dtw:.4f}, frame={t+1}/{num_frames}")
                else:
                    fig.suptitle(f"Baseline L2={baseline_l2:.4f}, frame={t+1}/{num_frames}")
            else:
                fig.suptitle(f"Baseline Prototype Comparison, frame={t+1}/{num_frames}")

            fig.tight_layout()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
            writer.append_data(frame)
            plt.close(fig)

    return output_path


def _log_hoyo_stats(
    reference_cache: dict[str, np.ndarray | None],
    embedding_cache: dict[str, torch.Tensor],
) -> None:
    """Log HOYO reference and embedding stats to diagnose similarity collapse."""
    def _hoyo_log(msg: str, *args) -> None:
        if logger.isEnabledFor(logging.INFO):
            logger.info(msg, *args)
            return
        try:
            print(msg % args if args else msg)
        except Exception:
            print(msg)

    if not embedding_cache:
        _hoyo_log("HOYO stats: no embeddings to report.")
        return

    labels = [label for label in reference_cache.keys() if label in embedding_cache]
    if not labels:
        _hoyo_log("HOYO stats: no overlapping labels between references and embeddings.")
        return

    _hoyo_log("HOYO stats: reference keypoint summary (mean/std/min/max)")
    ref_flat = []
    ref_labels = []
    for label in labels:
        reference = reference_cache.get(label)
        if reference is None:
            continue
        arr = np.asarray(reference)
        if arr.size == 0:
            continue
        _hoyo_log(
            "  - %s: shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
            label,
            arr.shape,
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
        )
        # Joint-wise std average (more sensitive than global std)
        try:
            if arr.ndim == 3 and arr.shape[-1] == 2:
                joint_std = arr.reshape(arr.shape[0], arr.shape[1], -1).std(axis=0)  # (J, 2)
                joint_std_mean = float(np.mean(joint_std))
                _hoyo_log("    - joint_std_mean=%.6f", joint_std_mean)
        except Exception:
            pass
        ref_flat.append(arr.reshape(-1).astype(np.float32))
        ref_labels.append(label)

    if len(ref_flat) >= 2:
        ref_mat = np.stack(ref_flat, axis=0)
        ref_norm = np.linalg.norm(ref_mat, axis=1) + 1e-8
        ref_unit = ref_mat / ref_norm[:, None]
        cos_mat = ref_unit @ ref_unit.T
        # Pairwise L2 in raw space
        diff = ref_mat[:, None, :] - ref_mat[None, :, :]
        l2_mat = np.sqrt(np.sum(diff * diff, axis=-1))
        pair_cos = []
        pair_l2 = []
        for i in range(len(ref_labels)):
            for j in range(i + 1, len(ref_labels)):
                pair_cos.append((float(cos_mat[i, j]), ref_labels[i], ref_labels[j]))
                pair_l2.append((float(l2_mat[i, j]), ref_labels[i], ref_labels[j]))
        if pair_cos:
            vals = [v[0] for v in pair_cos]
            _hoyo_log(
                "HOYO stats: reference pairwise cos (off-diag) min=%.6f mean=%.6f max=%.6f (n=%d)",
                float(np.min(vals)),
                float(np.mean(vals)),
                float(np.max(vals)),
                len(vals),
            )
            pair_cos.sort(key=lambda x: x[0])
            lowest = ", ".join([f"{a}-{b}:{v:.4f}" for v, a, b in pair_cos[: min(3, len(pair_cos))]])
            highest = ", ".join([f"{a}-{b}:{v:.4f}" for v, a, b in pair_cos[-min(3, len(pair_cos)) :]])
            _hoyo_log("HOYO stats: reference lowest cos pairs: %s", lowest)
            _hoyo_log("HOYO stats: reference highest cos pairs: %s", highest)
        if pair_l2:
            vals = [v[0] for v in pair_l2]
            _hoyo_log(
                "HOYO stats: reference pairwise L2 (off-diag) min=%.6f mean=%.6f max=%.6f (n=%d)",
                float(np.min(vals)),
                float(np.mean(vals)),
                float(np.max(vals)),
                len(vals),
            )
            pair_l2.sort(key=lambda x: x[0])
            lowest = ", ".join([f"{a}-{b}:{v:.4f}" for v, a, b in pair_l2[: min(3, len(pair_l2))]])
            highest = ", ".join([f"{a}-{b}:{v:.4f}" for v, a, b in pair_l2[-min(3, len(pair_l2)) :]])
            _hoyo_log("HOYO stats: reference lowest L2 pairs: %s", lowest)
            _hoyo_log("HOYO stats: reference highest L2 pairs: %s", highest)

    embeddings = []
    emb_labels = []
    emb_norms = []
    for label in labels:
        emb = embedding_cache.get(label)
        if not isinstance(emb, torch.Tensor):
            continue
        emb_flat = emb.detach().view(-1).float()
        if emb_flat.numel() == 0:
            continue
        emb_labels.append(label)
        emb_norms.append(float(emb_flat.norm().item()))
        embeddings.append(F.normalize(emb_flat, dim=0))

    if not embeddings:
        _hoyo_log("HOYO stats: no valid embeddings after filtering.")
        return

    _hoyo_log("HOYO stats: embedding norms (per label)")
    for label, norm in zip(emb_labels, emb_norms):
        _hoyo_log("  - %s: norm=%.6f", label, norm)

    if len(embeddings) < 2:
        return

    mat = torch.stack(embeddings, dim=0)
    cos_mat = (mat @ mat.t()).detach().cpu().numpy()

    pair_vals = []
    for i in range(len(emb_labels)):
        for j in range(i + 1, len(emb_labels)):
            pair_vals.append((float(cos_mat[i, j]), emb_labels[i], emb_labels[j]))

    if pair_vals:
        vals = [v[0] for v in pair_vals]
        mean_val = float(np.mean(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        _hoyo_log(
            "HOYO stats: embedding pairwise cos (off-diag) min=%.6f mean=%.6f max=%.6f (n=%d)",
            min_val,
            mean_val,
            max_val,
            len(vals),
        )
        pair_vals.sort(key=lambda x: x[0])
        lowest = ", ".join([f"{a}-{b}:{v:.4f}" for v, a, b in pair_vals[: min(3, len(pair_vals))]])
        highest = ", ".join([f"{a}-{b}:{v:.4f}" for v, a, b in pair_vals[-min(3, len(pair_vals)) :]])
        _hoyo_log("HOYO stats: lowest cos pairs: %s", lowest)
        _hoyo_log("HOYO stats: highest cos pairs: %s", highest)


def save_confusion_heatmap(
    results: list[EvaluationResult],
    hoyo_labels: list[str],
    output_dir: Path,
    timestamp: str,
    use_centered: bool = True,
) -> Path | None:
    """
    H1スタイル vs HOYO教師motionの類似度行列をヒートマップとして保存.

    Args:
        results: 評価結果リスト（各要素がH1で実行したスタイル）
        hoyo_labels: HOYOリファレンスのラベル一覧
        output_dir: 出力ディレクトリ
        timestamp: タイムスタンプ文字列
        use_centered: 中心化cos類似度を使用するか

    Returns:
        保存先パス、または失敗時None
    """
    if not results or not hoyo_labels:
        logger.warning("No results or HOYO labels for heatmap")
        return None

    sorted_results = order_results_by_dataset_speed(results)
    sorted_hoyo_labels = order_labels_by_dataset_speed(hoyo_labels)

    # 行列データを構築
    h1_labels = [r.onomatopoeia for r in sorted_results]
    matrix = np.zeros((len(h1_labels), len(sorted_hoyo_labels)))

    for i, r in enumerate(sorted_results):
        sim_dict = r.hoyo_similarity_centered_mean if use_centered else r.hoyo_similarity_mean
        if sim_dict is None:
            continue
        for j, label in enumerate(sorted_hoyo_labels):
            if label in sim_dict:
                matrix[i, j] = sim_dict[label]

    # ヒートマップ作成
    fig, ax = plt.subplots(figsize=(max(8, len(sorted_hoyo_labels) * 0.8), max(6, len(h1_labels) * 0.6)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # 軸ラベル
    ax.set_xticks(np.arange(len(sorted_hoyo_labels)))
    ax.set_yticks(np.arange(len(h1_labels)))
    ax.set_xticklabels(sorted_hoyo_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(h1_labels, fontsize=10)

    # セル内に数値を表示
    for i in range(len(h1_labels)):
        for j in range(len(sorted_hoyo_labels)):
            val = matrix[i, j]
            # 対角成分（同じスタイル）をハイライト
            is_diagonal = (h1_labels[i] == sorted_hoyo_labels[j])
            color = 'white' if abs(val) > 0.5 else 'black'
            weight = 'bold' if is_diagonal else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=9, fontweight=weight)

    # タイトルとラベル
    sim_type = "Centered Cos" if use_centered else "Raw Cos"
    ax.set_title(f'H1 vs HOYO Similarity Matrix ({sim_type})\n'
                 f'Row: H1 executed style, Col: HOYO reference', fontsize=12)
    ax.set_xlabel('HOYO Reference Style', fontsize=11)
    ax.set_ylabel('H1 Executed Style', fontsize=11)

    # カラーバー
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity', fontsize=10)

    plt.tight_layout()

    # 保存
    suffix = "centered" if use_centered else "raw"
    output_path = output_dir / f"confusion_heatmap_{suffix}_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Heatmap saved to: {output_path}")
    return output_path


def save_margin_heatmap(
    results: list[EvaluationResult],
    hoyo_labels: list[str],
    output_dir: Path,
    timestamp: str,
    use_centered: bool = True,
) -> Path | None:
    """
    H1スタイル vs HOYO教師motionの類似度マージン行列をヒートマップとして保存.

    M[i, j] = S[i, i] - S[i, j]
    SはH1 vs HOYOの類似度行列（raw or centered）.
    """
    if not results or not hoyo_labels:
        logger.warning("No results or HOYO labels for margin heatmap")
        return None

    sorted_results = order_results_by_dataset_speed(results)
    sorted_hoyo_labels = order_labels_by_dataset_speed(hoyo_labels)
    h1_labels = [r.onomatopoeia for r in sorted_results]
    label_to_col = {label: j for j, label in enumerate(sorted_hoyo_labels)}

    # 類似度行列 S を構築（欠損は NaN）
    sim_mat = np.full((len(h1_labels), len(sorted_hoyo_labels)), np.nan, dtype=np.float32)
    for i, r in enumerate(sorted_results):
        sim_dict = r.hoyo_similarity_centered_mean if use_centered else r.hoyo_similarity_mean
        if not sim_dict:
            continue
        for label, val in sim_dict.items():
            j = label_to_col.get(label)
            if j is not None:
                sim_mat[i, j] = float(val)

    # マージン行列 M[i, j] = S[i, i] - S[i, j]
    margin = np.full_like(sim_mat, np.nan)
    for i, label in enumerate(h1_labels):
        diag_j = label_to_col.get(label)
        if diag_j is None:
            continue
        diag_val = sim_mat[i, diag_j]
        if not np.isfinite(diag_val):
            continue
        row = sim_mat[i, :]
        valid = np.isfinite(row)
        margin[i, valid] = diag_val - row[valid]

    if not np.isfinite(margin).any():
        logger.warning("Margin heatmap skipped: no valid entries")
        return None

    max_abs = float(np.nanmax(np.abs(margin)))
    if max_abs < 1e-6:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_hoyo_labels) * 0.8), max(6, len(h1_labels) * 0.6)))
    masked = np.ma.masked_invalid(margin)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(np.arange(len(sorted_hoyo_labels)))
    ax.set_yticks(np.arange(len(h1_labels)))
    ax.set_xticklabels(sorted_hoyo_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(h1_labels, fontsize=10)

    for i in range(len(h1_labels)):
        for j in range(len(sorted_hoyo_labels)):
            val = margin[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "N/A", ha='center', va='center', color='gray', fontsize=8)
                continue
            color = 'white' if abs(val) > 0.5 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=9, fontweight=weight)

    sim_type = "Centered Cos" if use_centered else "Raw Cos"
    ax.set_title(
        f'H1 vs HOYO Margin Matrix ({sim_type})\n'
        f'M[i,j] = S[i,i] - S[i,j] (Row: H1 style, Col: HOYO style)',
        fontsize=12,
    )
    ax.set_xlabel('HOYO Reference Style', fontsize=11)
    ax.set_ylabel('H1 Executed Style', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Margin (diag - col)', fontsize=10)

    plt.tight_layout()

    suffix = "centered" if use_centered else "raw"
    output_path = output_dir / f"margin_heatmap_{suffix}_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Margin heatmap saved to: {output_path}")
    return output_path


def save_teacher_margin_heatmap(
    results: list[EvaluationResult],
    teacher_labels: list[str],
    output_dir: Path,
    timestamp: str,
) -> Path | None:
    """
    H1スタイル vs 教師埋め込みの類似度マージン行列をヒートマップとして保存.

    M[i, j] = S[i, i] - S[i, j]
    SはH1 vs teacherの類似度行列.
    """
    if not results or not teacher_labels:
        logger.warning("No results or teacher labels for margin heatmap")
        return None

    sorted_results = order_results_by_dataset_speed(results)
    sorted_teacher_labels = order_labels_by_dataset_speed(teacher_labels)
    h1_labels = [r.onomatopoeia for r in sorted_results]
    label_to_col = {label: j for j, label in enumerate(sorted_teacher_labels)}

    sim_mat = np.full((len(h1_labels), len(sorted_teacher_labels)), np.nan, dtype=np.float32)
    for i, r in enumerate(sorted_results):
        sim_dict = r.teacher_similarity_mean
        if not sim_dict:
            continue
        for label, val in sim_dict.items():
            j = label_to_col.get(label)
            if j is not None:
                sim_mat[i, j] = float(val)

    margin = np.full_like(sim_mat, np.nan)
    for i, label in enumerate(h1_labels):
        diag_j = label_to_col.get(label)
        if diag_j is None:
            continue
        diag_val = sim_mat[i, diag_j]
        if not np.isfinite(diag_val):
            continue
        row = sim_mat[i, :]
        valid = np.isfinite(row)
        margin[i, valid] = diag_val - row[valid]

    if not np.isfinite(margin).any():
        logger.warning("Teacher margin heatmap skipped: no valid entries")
        return None

    max_abs = float(np.nanmax(np.abs(margin)))
    if max_abs < 1e-6:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_teacher_labels) * 0.8), max(6, len(h1_labels) * 0.6)))
    masked = np.ma.masked_invalid(margin)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(np.arange(len(sorted_teacher_labels)))
    ax.set_yticks(np.arange(len(h1_labels)))
    ax.set_xticklabels(sorted_teacher_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(h1_labels, fontsize=10)

    for i in range(len(h1_labels)):
        for j in range(len(sorted_teacher_labels)):
            val = margin[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "N/A", ha='center', va='center', color='gray', fontsize=8)
                continue
            color = 'white' if abs(val) > 0.5 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=9, fontweight=weight)

    ax.set_title(
        'H1 vs Teacher Margin Matrix\\nM[i,j] = S[i,i] - S[i,j] (Row: H1 style, Col: Teacher style)',
        fontsize=12,
    )
    ax.set_xlabel('Teacher Style', fontsize=11)
    ax.set_ylabel('H1 Executed Style', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Margin (diag - col)', fontsize=10)

    plt.tight_layout()

    output_path = output_dir / f"teacher_margin_heatmap_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Teacher margin heatmap saved to: {output_path}")
    return output_path


def save_hoyo_pairwise_heatmap(
    hoyo_embedding_cache: dict[str, torch.Tensor],
    output_dir: Path,
    timestamp: str,
) -> Path | None:
    """
    HOYO embedding同士のcos類似度行列をヒートマップとして保存（診断用）.

    これが全て0.99に近い場合、エンコーダのfeature collapse問題を示唆。
    """
    if not hoyo_embedding_cache or len(hoyo_embedding_cache) < 2:
        return None

    labels = order_labels_by_dataset_speed(list(hoyo_embedding_cache.keys()))
    n = len(labels)

    # cos類似度行列を計算
    embeddings = []
    for label in labels:
        emb = hoyo_embedding_cache[label]
        emb_flat = F.normalize(emb.view(-1).float(), dim=0)
        embeddings.append(emb_flat)

    mat = torch.stack(embeddings, dim=0)
    cos_mat = (mat @ mat.t()).detach().cpu().numpy()

    # ヒートマップ作成
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))

    im = ax.imshow(cos_mat, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = cos_mat[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=9, fontweight=weight)

    # 対角以外の平均を計算（診断用）
    off_diag = cos_mat[~np.eye(n, dtype=bool)]
    off_diag_mean = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0

    ax.set_title(f'HOYO Pairwise Similarity (Diagnostic)\n'
                 f'Off-diagonal mean: {off_diag_mean:.3f} (should be < 0.9 for good separation)',
                 fontsize=12)
    ax.set_xlabel('HOYO Style', fontsize=11)
    ax.set_ylabel('HOYO Style', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity', fontsize=10)

    plt.tight_layout()

    output_path = output_dir / f"hoyo_pairwise_heatmap_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"HOYO pairwise heatmap saved to: {output_path}")

    # 警告: 対角以外が高すぎる場合
    if off_diag_mean > 0.9:
        logger.warning(
            f"HOYO embeddings are too similar (off-diag mean={off_diag_mean:.3f}). "
            "This suggests feature collapse in the encoder. "
            "Consider checking the MotionCLIP encoder or using centered cosine similarity."
        )

    return output_path


def save_results(
    results: list[EvaluationResult],
    output_dir: Path,
    timestamp: str,
    joint_error_baseline: dict | None = None,
) -> None:
    """結果をJSON形式で保存."""
    output_path = output_dir / f"eval_motion_{timestamp}.json"
    data = {
        "timestamp": timestamp,
        "results": [r.to_dict() for r in results],
    }
    if joint_error_baseline is not None:
        data["joint_error_baseline"] = joint_error_baseline
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    _debug("main start")
    config = EvalConfig.from_args(args_cli)
    global _CLEANUP_ENV

    # Parse env config
    _debug(f"args: task={args_cli.task} num_envs={args_cli.num_envs} load_run={args_cli.load_run} checkpoint={args_cli.checkpoint}")
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Ensure terrain env_spacing type matches log env.yaml (avoid None->float type mismatch)
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain_cfg is not None and getattr(terrain_cfg, "env_spacing", None) is None:
        terrain_cfg.env_spacing = getattr(env_cfg.scene, "env_spacing", None)

    # Resolve log path (search multiple candidates)
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

    log_root = resolve_log_root_path(agent_cfg.experiment_name, args_cli.load_run, args_cli.log_root)
    log_dir = os.path.join(log_root, agent_cfg.experiment_name, args_cli.load_run)
    _debug(f"log_root={log_root} exists={os.path.isdir(log_root)}")
    _debug(f"log_dir={log_dir} exists={os.path.isdir(log_dir)}")
    logger.info(f"Loading run from: {log_dir}")

    # Load env config from run
    log_env_cfg_path = os.path.join(log_dir, "params", "env.yaml")
    if args_cli.use_log_env and os.path.exists(log_env_cfg_path):
        try:
            log_env_cfg_dict, loader_name = _load_yaml_with_fallback(log_env_cfg_path)
            update_class_from_dict(env_cfg, log_env_cfg_dict)
            logger.info(f"Loaded env config from: {log_env_cfg_path} (loader={loader_name})")
        except Exception as e:
            logger.warning(f"Failed to load env config: {e}")
    _debug(f"log_env_cfg_path={log_env_cfg_path} exists={os.path.exists(log_env_cfg_path)}")
    fixed_ids = _normalize_scene_entity_ids(env_cfg)
    if fixed_ids:
        logger.info(f"Normalized {fixed_ids} SceneEntityCfg id fields (None -> slice(None)).")

    # Force CLI num_envs
    env_cfg.scene.num_envs = args_cli.num_envs

    # Load agent config
    agent_cfg_path = os.path.join(log_dir, "params", "agent.yaml")
    _debug(f"agent_cfg_path={agent_cfg_path} exists={os.path.exists(agent_cfg_path)}")
    if not os.path.exists(agent_cfg_path):
        raise FileNotFoundError(f"Agent config not found: {agent_cfg_path}")
    agent_cfg_dict, agent_loader_name = _load_yaml_with_fallback(agent_cfg_path)
    _debug(f"agent_cfg_loader={agent_loader_name}")
    update_class_from_dict(agent_cfg, agent_cfg_dict)

    # Resolve gamma for rate^adv (CLI override > agent config > default)
    if config.gamma is None:
        gamma = None
        algo_cfg = getattr(agent_cfg, "algorithm", None)
        if algo_cfg is not None and hasattr(algo_cfg, "gamma"):
            gamma = float(algo_cfg.gamma)
        if gamma is None and isinstance(agent_cfg_dict, dict):
            algo_dict = agent_cfg_dict.get("algorithm", {})
            if isinstance(algo_dict, dict) and "gamma" in algo_dict:
                gamma = algo_dict.get("gamma")
        if gamma is None:
            gamma = 0.99
        config.gamma = float(gamma)

    # Checkpoint
    if args_cli.checkpoint:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    # Base policy
    if args_cli.no_base_policy:
        agent_cfg.policy.base_policy_checkpoint = None
        logger.info("Base policy disabled")

    # History length
    history_length = args_cli.history_length
    if history_length == 0 and "policy" in agent_cfg_dict:
        history_length = agent_cfg_dict["policy"].get("history_length", 0)

    # Disable terrain curriculum
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    # Terrain override (flat/plane)
    if args_cli.terrain in ("flat", "plane"):
        terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain_cfg is None:
            logger.warning("Terrain override requested but env_cfg.scene.terrain is not available.")
        else:
            terrain_cfg.terrain_type = "plane"
            terrain_cfg.terrain_generator = None
            if getattr(terrain_cfg, "env_spacing", None) is None:
                terrain_cfg.env_spacing = getattr(env_cfg.scene, "env_spacing", 2.5)
            logger.info("Using plane terrain for evaluation.")

    # Reset function override
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
        env_cfg.events.reset_base.func = mdp.reset_root_state_uniform
        env_cfg.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

    # Fixed velocity command
    cmd_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if cmd_cfg is not None and hasattr(cmd_cfg, "ranges"):
        cmd_cfg.ranges.lin_vel_x = (config.lin_vel_x, config.lin_vel_x)
        cmd_cfg.ranges.lin_vel_y = (config.lin_vel_y, config.lin_vel_y)
        cmd_cfg.ranges.ang_vel_z = (config.ang_vel_z, config.ang_vel_z)
        if hasattr(cmd_cfg, "rel_standing_envs"):
            cmd_cfg.rel_standing_envs = 0.0
        cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)

    # Create environment
    _debug("creating env")
    seed_everything(config.seed)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    _CLEANUP_ENV = env
    _debug("env created")

    # Load policy
    resume_path = get_checkpoint_path(
        os.path.join(log_root, agent_cfg.experiment_name),
        args_cli.load_run,
        agent_cfg.load_checkpoint,
    )
    _debug(f"resume_path={resume_path} exists={os.path.exists(resume_path)}")
    _debug("creating OnPolicyRunner")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _debug("loading checkpoint")
    ppo_runner.load(resume_path, load_optimizer=False)
    logger.info(f"Loaded checkpoint: {resume_path}")
    _debug("checkpoint loaded")

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # rate^adv 用に actor_critic を取得
    actor_critic = getattr(ppo_runner.alg, "actor_critic", None)

    # Prepare style cache
    style_cache = {}
    style_term = env.unwrapped.command_manager._terms.get("style_command")
    teacher_embedding_cache: dict[str, torch.Tensor] | None = None
    if style_term is not None:
        styles_to_eval = INSTRUCTION_ONOMATOPEIA
        if args_cli.style_list:
            styles_to_eval = [s.strip() for s in args_cli.style_list.split(",") if s.strip()]

        for style in styles_to_eval:
            z_onm, teacher_motion = style_term.style_module.encode_instruction(style)
            style_cache[style] = (z_onm.squeeze(0), teacher_motion.squeeze(0))
        teacher_embedding_cache = {style: teacher for style, (_, teacher) in style_cache.items()}
    else:
        styles_to_eval = ["default"]
    _debug(f"style_term={'yes' if style_term is not None else 'no'} style_list={args_cli.style_list}")
    _debug(f"styles_to_eval({len(styles_to_eval)}): {styles_to_eval}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Video directory
    video_dir = None
    if config.record_video:
        video_dir = output_dir / "videos" / timestamp
        video_dir.mkdir(parents=True, exist_ok=True)

    # HOYO reference
    hoyo_root = Path(args_cli.hoyo_root) if args_cli.hoyo_root else Path(NAVILA_ROOT) / "hoyo_v1_1"

    # Wrap environment
    # ロングバッファサイズ = window_size(100) + max_shift(200) + 余裕
    long_buffer_size = 100 + config.time_shift_max + 50 if config.time_shift_analyze else 300
    env_wrapper = EvalEnvWrapper(env, style_cache=style_cache, long_buffer_size=long_buffer_size)
    _debug("env_wrapper ready")

    # HOYO references + embeddings cache (for similarity matrix)
    hoyo_reference_cache: dict[str, np.ndarray | None] = {}
    hoyo_embedding_cache: dict[str, torch.Tensor] = {}
    for style in styles_to_eval:
        hoyo_reference_cache[style] = load_hoyo_reference(hoyo_root, style, seed=args_cli.hoyo_seed)

    for style, reference in hoyo_reference_cache.items():
        if reference is None:
            continue
        emb = env_wrapper.encode_hoyo_reference(reference)
        if emb is not None:
            hoyo_embedding_cache[style] = emb
        else:
            logger.warning(f"Failed to encode HOYO reference embedding for: {style}")

    hoyo_embedding_centered_cache: dict[str, torch.Tensor] | None = None
    hoyo_center: torch.Tensor | None = None
    if hoyo_embedding_cache:
        hoyo_embedding_centered_cache, hoyo_center = build_centered_hoyo_embeddings(hoyo_embedding_cache)

    if config.log_hoyo_stats:
        _log_hoyo_stats(hoyo_reference_cache, hoyo_embedding_cache)

    joint_error_baseline: dict = {
        "enabled": bool(config.enable_joint_error_baseline),
        "style_label": str(config.baseline_style_label),
        "baseline_l2": None,
        "baseline_dtw": None,
        "comparison_gif": None,
        "status": "disabled",
        "config": {
            "h1_source": str(config.baseline_h1_source),
            "h1_warmup_steps": int(config.baseline_h1_warmup_steps),
            "h1_collect_steps": int(config.baseline_h1_collect_steps),
            "h1_collect_stride": int(config.baseline_h1_collect_stride),
            "save_gif": bool(config.save_joint_error_baseline_gif),
            "gif_fps": int(config.baseline_gif_fps),
            "gif_frames": int(config.baseline_gif_frames),
            "target_len": 100,
        },
    }
    if config.enable_joint_error_baseline:
        joint_error_baseline["status"] = "running"
        baseline_label = str(config.baseline_style_label)
        logger.info("Computing joint-error baseline with style label: %s", baseline_label)

        hoyo_static_seq = compute_hoyo_static_prototype_frame0_median(
            hoyo_root=hoyo_root,
            label=baseline_label,
            target_len=100,
        )
        if hoyo_static_seq is None:
            joint_error_baseline["status"] = "failed_hoyo_prototype"
            logger.warning("Joint-error baseline disabled: failed to build HOYO static prototype.")
        else:
            h1_static_seq = compute_h1_static_prototype(
                env_wrapper=env_wrapper,
                policy=policy,
                style_label=baseline_label,
                warmup_steps=config.baseline_h1_warmup_steps,
                collect_steps=config.baseline_h1_collect_steps,
                collect_stride=config.baseline_h1_collect_stride,
                source=config.baseline_h1_source,
                target_len=100,
            )
            if h1_static_seq is None:
                joint_error_baseline["status"] = "failed_h1_prototype"
                logger.warning("Joint-error baseline disabled: failed to build H1 static prototype.")
            else:
                try:
                    baseline_l2 = compute_l2_joint_error(h1_static_seq, hoyo_static_seq)
                    joint_error_baseline["baseline_l2"] = float(baseline_l2)
                except Exception as e:
                    baseline_l2 = None
                    joint_error_baseline["status"] = "failed_l2"
                    logger.warning("Joint-error baseline L2 computation failed: %s", e)

                baseline_dtw = None
                if baseline_l2 is not None:
                    if DTW_AVAILABLE:
                        try:
                            baseline_dtw = compute_l2_joint_error_with_dtw(h1_static_seq, hoyo_static_seq)
                            joint_error_baseline["baseline_dtw"] = float(baseline_dtw)
                            joint_error_baseline["status"] = "ok"
                        except Exception as e:
                            joint_error_baseline["status"] = "ok_l2_only_dtw_failed"
                            logger.warning("Joint-error baseline DTW computation failed: %s", e)
                    else:
                        joint_error_baseline["status"] = "ok_l2_only_dtw_unavailable"

                if config.save_joint_error_baseline_gif:
                    try:
                        baseline_gif_path = output_dir / f"baseline_compare_{timestamp}.gif"
                        saved_gif = save_baseline_comparison_gif(
                            h1_static_seq=h1_static_seq,
                            hoyo_static_seq=hoyo_static_seq,
                            output_path=baseline_gif_path,
                            fps=config.baseline_gif_fps,
                            max_frames=config.baseline_gif_frames,
                            baseline_l2=joint_error_baseline.get("baseline_l2"),
                            baseline_dtw=joint_error_baseline.get("baseline_dtw"),
                        )
                        if saved_gif is not None:
                            joint_error_baseline["comparison_gif"] = str(saved_gif)
                            logger.info("Saved baseline comparison GIF: %s", saved_gif)
                    except Exception as e:
                        logger.warning("Failed to save baseline comparison GIF: %s", e)

        print("")
        print("=" * 80)
        print("JOINT ERROR BASELINE")
        print("=" * 80)
        print(f"enabled: {joint_error_baseline['enabled']}")
        print(f"style_label: {joint_error_baseline['style_label']}")
        print(f"h1_source: {joint_error_baseline['config']['h1_source']}")
        print(f"status: {joint_error_baseline['status']}")
        b_l2 = joint_error_baseline.get("baseline_l2")
        b_dtw = joint_error_baseline.get("baseline_dtw")
        b_gif = joint_error_baseline.get("comparison_gif")
        print(f"baseline_l2: {b_l2:.6f}" if b_l2 is not None else "baseline_l2: N/A")
        print(f"baseline_dtw: {b_dtw:.6f}" if b_dtw is not None else "baseline_dtw: N/A")
        print(f"comparison_gif: {b_gif}" if b_gif else "comparison_gif: N/A")
        print("=" * 80)

    # Evaluate each style
    results = []
    logger.info("=" * 60)
    logger.info("MOTION EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Evaluating {len(styles_to_eval)} styles")
    _debug("begin evaluation loop")

    for onomatopoeia in styles_to_eval:
        logger.info(f"\n--- Evaluating: {onomatopoeia} ---")
        _debug(f"evaluating {onomatopoeia}")

        # Load reference
        reference = hoyo_reference_cache.get(onomatopoeia)

        # Video path
        video_path = None
        if video_dir:
            safe_name = onomatopoeia.replace("/", "_").replace("\\", "_")
            known_suffix = "__known" if env_wrapper.is_known_style(onomatopoeia) else "__unknown"
            video_path = str(video_dir / f"{safe_name}{known_suffix}.mp4")

        result = evaluate_single_style(
            env_wrapper=env_wrapper,
            policy=policy,
            onomatopoeia=onomatopoeia,
            config=config,
            reference_keypoints=reference,
            teacher_embeddings=teacher_embedding_cache if teacher_embedding_cache else None,
            hoyo_embeddings=hoyo_embedding_cache if hoyo_embedding_cache else None,
            hoyo_embeddings_centered=hoyo_embedding_centered_cache if hoyo_embedding_centered_cache else None,
            hoyo_center=hoyo_center,
            video_path=video_path,
            actor_critic=actor_critic,
        )
        if joint_error_baseline.get("baseline_l2") is not None:
            apply_joint_error_baseline(
                result=result,
                baseline_l2=joint_error_baseline.get("baseline_l2"),
                baseline_dtw=joint_error_baseline.get("baseline_dtw"),
            )
        results.append(result)

        # Print summary (use print for immediate output)
        print(f"\n  [RESULT] {onomatopoeia}")
        print(f"    mean_velocity_x: {result.mean_velocity_x:.4f} ± {result.std_velocity_x:.4f}")
        if result.mean_cos_centroid is not None:
            print(f"    cos_centroid (H1 vs Centroid): {result.mean_cos_centroid:.4f}")
        if result.cos_centroid_count is not None:
            print(f"    cos_centroid_samples: {result.cos_centroid_count}")
        if result.mean_cos_random_sample is not None:
            print(f"    cos_random_sample (H1 vs Random): {result.mean_cos_random_sample:.4f}")
        if result.cos_random_sample_count is not None:
            print(f"    cos_random_sample_samples: {result.cos_random_sample_count}")
        if result.mean_style_score is not None:
            print(f"    style_score (teacher - normal): {result.mean_style_score:.4f} ± {result.std_style_score:.4f}")
        if result.style_score_count is not None:
            print(f"    style_score_samples: {result.style_score_count}")
        if result.mean_joint_error is not None:
            print(f"    mean_joint_error (L2): {result.mean_joint_error:.4f}")
            if result.mean_joint_error_delta is not None or result.mean_joint_error_ratio is not None:
                delta_str = (
                    f"{result.mean_joint_error_delta:+.4f}"
                    if result.mean_joint_error_delta is not None
                    else "N/A"
                )
                ratio_str = (
                    f"{result.mean_joint_error_ratio:.3f}"
                    if result.mean_joint_error_ratio is not None
                    else "N/A"
                )
                print(f"    mean_joint_error (raw/delta/ratio): {result.mean_joint_error:.4f} / {delta_str} / {ratio_str}")
        if result.mean_joint_error_dtw is not None:
            print(f"    mean_joint_error_dtw (Soft-DTW): {result.mean_joint_error_dtw:.4f}")
            if result.mean_joint_error_dtw_delta is not None or result.mean_joint_error_dtw_ratio is not None:
                delta_str = (
                    f"{result.mean_joint_error_dtw_delta:+.4f}"
                    if result.mean_joint_error_dtw_delta is not None
                    else "N/A"
                )
                ratio_str = (
                    f"{result.mean_joint_error_dtw_ratio:.3f}"
                    if result.mean_joint_error_dtw_ratio is not None
                    else "N/A"
                )
                print(
                    "    mean_joint_error_dtw (raw/delta/ratio): "
                    f"{result.mean_joint_error_dtw:.4f} / {delta_str} / {ratio_str}"
                )
        print(f"    fall_rate: {result.fall_rate:.2%}")
        if config.log_reward_terms:
            if result.mean_reward_total is not None:
                print(f"    mean_reward_total: {result.mean_reward_total:+.6f} ± {result.std_reward_total:.6f}")
            if result.mean_action_sq is not None:
                print(f"    mean_action_sq: {result.mean_action_sq:.6f}")
            print("    reward_terms (top-|mean|):")
            print(_summarize_reward_terms(result.reward_terms_mean, result.reward_terms_std, config.reward_terms_topk))
            # 寄与率表示
            if result.rate_mag:
                print(_summarize_contribution_rate(result.rate_mag, config.reward_terms_topk, "rate^mag (報酬寄与率)"))
            if result.rate_adv:
                print(_summarize_contribution_rate(result.rate_adv, config.reward_terms_topk, "rate^adv (Advantage寄与率)"))
            if result.share_e:
                print(_summarize_contribution_rate(result.share_e, config.reward_terms_topk, "share^E (関節エネルギー割合)"))
        sys.stdout.flush()

    # Save results
    _debug(f"saving results to {output_dir}")
    save_results(results, output_dir, timestamp, joint_error_baseline=joint_error_baseline)

    # Print final summary table
    print("")
    print("=" * 108)
    print("EVALUATION SUMMARY")
    print(f"Command: lin_vel_x={config.lin_vel_x}, lin_vel_y={config.lin_vel_y}, ang_vel_z={config.ang_vel_z}")
    print("=" * 108)
    print(f"{'Onomatopoeia':<12} {'Cmd X':>6} {'Vel X':>8} {'Style':>8} {'Cos(C)':>8} {'Cos(R)':>8} {'L2 Err':>8} {'DTW Err':>8} {'Fall%':>8}")
    print("-" * 108)
    for r in results:
        cos_c = f"{r.mean_cos_centroid:.3f}" if r.mean_cos_centroid is not None else "N/A"
        cos_r = f"{r.mean_cos_random_sample:.3f}" if r.mean_cos_random_sample is not None else "N/A"
        style = f"{r.mean_style_score:.3f}" if r.mean_style_score is not None else "N/A"
        j_err = f"{r.mean_joint_error:.4f}" if r.mean_joint_error is not None else "N/A"
        j_dtw = f"{r.mean_joint_error_dtw:.4f}" if r.mean_joint_error_dtw is not None else "N/A"
        print(
            f"{r.onomatopoeia:<12} {config.lin_vel_x:>6.2f} {r.mean_velocity_x:>8.3f} "
            f"{style:>8} {cos_c:>8} {cos_r:>8} {j_err:>8} {j_dtw:>8} {r.fall_rate:>7.1%}"
        )
    print("=" * 108)
    if joint_error_baseline.get("baseline_l2") is not None:
        print("JOINT ERROR VS BASELINE")
        print(
            f"{'Onomatopoeia':<12} {'L2(raw)':>10} {'L2(delta)':>11} {'L2(ratio)':>10} "
            f"{'DTW(raw)':>10} {'DTW(delta)':>12} {'DTW(ratio)':>11}"
        )
        print("-" * 84)
        for r in results:
            l2_raw = f"{r.mean_joint_error:.4f}" if r.mean_joint_error is not None else "N/A"
            l2_delta = f"{r.mean_joint_error_delta:+.4f}" if r.mean_joint_error_delta is not None else "N/A"
            l2_ratio = f"{r.mean_joint_error_ratio:.3f}" if r.mean_joint_error_ratio is not None else "N/A"
            dtw_raw = f"{r.mean_joint_error_dtw:.4f}" if r.mean_joint_error_dtw is not None else "N/A"
            dtw_delta = (
                f"{r.mean_joint_error_dtw_delta:+.4f}"
                if r.mean_joint_error_dtw_delta is not None
                else "N/A"
            )
            dtw_ratio = (
                f"{r.mean_joint_error_dtw_ratio:.3f}"
                if r.mean_joint_error_dtw_ratio is not None
                else "N/A"
            )
            print(
                f"{r.onomatopoeia:<12} {l2_raw:>10} {l2_delta:>11} {l2_ratio:>10} "
                f"{dtw_raw:>10} {dtw_delta:>12} {dtw_ratio:>11}"
            )
        print("=" * 108)
    print("Style = cos(teacher_style) - cos(teacher_normal), Cos(C/R) = embedding similarity, L2/DTW Err = joint error")
    print(f"Results saved to: {output_dir / f'eval_motion_{timestamp}.json'}")

    # Print HOYO similarity matrix if available
    any_hoyo_centered = any(r.hoyo_similarity_centered_mean for r in results)
    any_hoyo_raw = any(r.hoyo_similarity_mean for r in results)
    hoyo_labels = [label for label in styles_to_eval if label in hoyo_embedding_cache]
    any_teacher = any(r.teacher_similarity_mean for r in results)
    teacher_labels = list(teacher_embedding_cache.keys()) if teacher_embedding_cache else []
    if (any_hoyo_centered or any_hoyo_raw) and hoyo_labels:
        sorted_results = order_results_by_dataset_speed(results)
        sorted_hoyo_labels = order_labels_by_dataset_speed(hoyo_labels)
        print("")
        print("=" * 80)
        if any_hoyo_centered:
            print("HOYO SIMILARITY MATRIX (CENTERED COS, H1 vs HOYO)")
            value_label = "Values = mean centered cos similarity (H1 vs HOYO, centered by HOYO mean)"
        else:
            print("HOYO SIMILARITY MATRIX (RAW COS, H1 vs HOYO)")
            value_label = "Values = mean cos similarity (H1 motion embedding vs HOYO reference embedding)"
        header = "H1\\HOYO".ljust(12) + " " + " ".join([f"{label:>8}" for label in sorted_hoyo_labels])
        print(header)
        print("-" * len(header))
        for r in sorted_results:
            row_vals = []
            for label in sorted_hoyo_labels:
                val = None
                if any_hoyo_centered:
                    if r.hoyo_similarity_centered_mean and label in r.hoyo_similarity_centered_mean:
                        val = r.hoyo_similarity_centered_mean[label]
                else:
                    if r.hoyo_similarity_mean and label in r.hoyo_similarity_mean:
                        val = r.hoyo_similarity_mean[label]
                row_vals.append(f"{val:.3f}" if val is not None else "N/A")
            row = f"{r.onomatopoeia:<12} " + " ".join([f"{v:>8}" for v in row_vals])
            print(row)
        print("=" * 80)
        print(value_label)

        # Save heatmaps
        if any_hoyo_centered:
            save_confusion_heatmap(sorted_results, sorted_hoyo_labels, output_dir, timestamp, use_centered=True)
            save_margin_heatmap(sorted_results, sorted_hoyo_labels, output_dir, timestamp, use_centered=True)
        if any_hoyo_raw:
            save_confusion_heatmap(sorted_results, sorted_hoyo_labels, output_dir, timestamp, use_centered=False)
            save_margin_heatmap(sorted_results, sorted_hoyo_labels, output_dir, timestamp, use_centered=False)

        # Save HOYO pairwise similarity heatmap (diagnostic)
        save_hoyo_pairwise_heatmap(hoyo_embedding_cache, output_dir, timestamp)

    # Save teacher margin heatmap if available
    if any_teacher and teacher_labels:
        sorted_results = order_results_by_dataset_speed(results)
        sorted_teacher_labels = order_labels_by_dataset_speed(teacher_labels)
        print("")
        print("=" * 80)
        print("TEACHER MARGIN MATRIX (H1 vs Teacher)")
        print("Values = cos(H1_i, teacher_i) - cos(H1_i, teacher_j)")
        header = "H1\\Teacher".ljust(12) + " " + " ".join([f"{label:>8}" for label in sorted_teacher_labels])
        print(header)
        print("-" * len(header))
        for r in sorted_results:
            sim_dict = r.teacher_similarity_mean or {}
            diag = sim_dict.get(r.onomatopoeia)
            row_vals = []
            for label in sorted_teacher_labels:
                sim = sim_dict.get(label)
                if diag is None or sim is None:
                    row_vals.append("N/A")
                else:
                    row_vals.append(f"{(diag - sim):.3f}")
            row = f"{r.onomatopoeia:<12} " + " ".join([f"{v:>8}" for v in row_vals])
            print(row)
        print("=" * 80)
        save_teacher_margin_heatmap(sorted_results, sorted_teacher_labels, output_dir, timestamp)

    # Print time-shift analysis if available
    any_time_shift = any(r.time_shift_results for r in results)
    if any_time_shift:
        print("")
        print("=" * 80)
        print("TIME-SHIFT ANALYSIS (Cos(R) at different shifts)")
        print("=" * 80)
        # Get all shift values from first result with time_shift_results
        for r in results:
            if r.time_shift_results:
                shifts = sorted(r.time_shift_results.keys())
                header = "Onomatopoeia  " + " ".join([f"{s:>6}" for s in shifts])
                print(header)
                print("-" * len(header))
                break
        for r in results:
            if r.time_shift_results:
                vals = " ".join([f"{r.time_shift_results.get(s, 0):.3f}" for s in shifts])
                print(f"{r.onomatopoeia:<12}  {vals}")
        print("=" * 80)
    
    sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            logger.exception("eval_motion failed")
        except Exception:
            pass
        raise
    finally:
        _cleanup()
