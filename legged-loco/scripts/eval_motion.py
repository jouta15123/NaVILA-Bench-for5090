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

# RSL-RL and AppLauncher args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

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
    # Velocity command
    lin_vel_x: float = 0.5
    lin_vel_y: float = 0.0
    ang_vel_z: float = 0.0
    # Video
    record_video: bool = False
    video_fps: int = 50
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
            lin_vel_x=args.lin_vel_x,
            lin_vel_y=args.lin_vel_y,
            ang_vel_z=args.ang_vel_z,
            record_video=args.video,
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
    # 時間シフト解析結果: {shift: mean_cos}
    time_shift_results: dict[int, float] | None = None
    # Reward breakdown (weighted, per-step)
    mean_reward_total: float | None = None
    std_reward_total: float | None = None
    reward_terms_mean: dict[str, float] | None = None
    reward_terms_std: dict[str, float] | None = None
    reward_step_dt: float | None = None

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
            "mean_joint_error": self.mean_joint_error,
            "std_joint_error": self.std_joint_error,
            "mean_joint_error_dtw": self.mean_joint_error_dtw,
            "std_joint_error_dtw": self.std_joint_error_dtw,
            "mean_episode_length": float(self.mean_episode_length),
            "fall_rate": float(self.fall_rate),
            "episode_count": self.episode_count,
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
    video_path: str | None = None,
) -> EvaluationResult:
    """
    1つのオノマトペスタイルを評価.

    Args:
        env_wrapper: 環境ラッパー
        policy: 推論ポリシー
        onomatopoeia: 評価するオノマトペ
        config: 評価設定
        reference_keypoints: 参照動作のキーポイント (T, 14, 2) - 前処理済み (optional)
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

    # 事前にセントロイドとランダムサンプルのembeddingを取得
    centroid_emb = env_wrapper.get_centroid_embedding(onomatopoeia)
    random_sample_emb = env_wrapper.get_random_sample_embedding(onomatopoeia, rng=random.Random(42))
    
    # デバイス統一 (GPU/CPU不一致対策)
    device = env_wrapper.device
    if centroid_emb is not None:
        centroid_emb = centroid_emb.to(device)
        print(f"    [INFO] Centroid embedding loaded for {onomatopoeia}")
    if random_sample_emb is not None:
        random_sample_emb = random_sample_emb.to(device)
        print(f"    [INFO] Random sample embedding loaded for {onomatopoeia}")

    # Video recorder
    recorder = None
    if video_path and config.record_video:
        recorder = VideoRecorder(video_path, fps=config.video_fps)
        recorder.start()

    # Reset and set style
    obs, _ = env_wrapper.reset()
    env_wrapper.set_style(onomatopoeia)
    env_wrapper.set_velocity_command(config.lin_vel_x, config.lin_vel_y, config.ang_vel_z)

    # 初期位置を記録 (CoM相対位置計算用)
    initial_state = env_wrapper.get_robot_state()
    initial_pos_x = initial_state.root_pos[:, 0].clone()

    num_steps = config.eval_steps

    for step in tqdm(range(num_steps), desc=f"Evaluating {onomatopoeia}", leave=False):
        with torch.no_grad():
            actions = policy(obs)
            obs, rewards, dones, infos = env_wrapper.step(actions)

            # Get robot state
            state = env_wrapper.get_robot_state()

            # Reward breakdown (weighted, per-step contributions)
            if config.log_reward_terms:
                base_env = env_wrapper.env.unwrapped
                reward_mgr = getattr(base_env, "reward_manager", None)
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
                    if reward_term_names is not None and reward_terms_sum is not None and reward_terms_sumsq is not None:
                        # Convert to per-step contributions (RewardManager stores raw*weight in _step_reward)
                        per_step_terms = step_terms * reward_step_dt
                        if reward_terms_active_mask is not None and reward_terms_active_mask.numel() == per_step_terms.shape[1]:
                            per_step_terms = per_step_terms * reward_terms_active_mask
                        reward_terms_sum += per_step_terms.sum(dim=0)
                        reward_terms_sumsq += (per_step_terms * per_step_terms).sum(dim=0)
                        reward_terms_count += int(per_step_terms.shape[0])

                # Total reward stats (as returned by env.step)
                if isinstance(rewards, torch.Tensor) and rewards.numel() > 0:
                    if reward_total_sum is None:
                        reward_total_sum = torch.zeros((), device=rewards.device, dtype=torch.float)
                        reward_total_sumsq = torch.zeros((), device=rewards.device, dtype=torch.float)
                    reward_total_sum += rewards.sum()
                    reward_total_sumsq += (rewards * rewards).sum()
                    reward_total_count += int(rewards.numel())

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
                            for i in range(min(env_wrapper.num_envs, motion_emb.shape[0])):
                                sim = compute_cosine_similarity(motion_emb[i], centroid_emb)
                                metrics.cos_centroid.append(sim)

                        # 2. H1 vs Random Sample (latent_snapshotからランダム選択)
                        if random_sample_emb is not None:
                            for i in range(min(env_wrapper.num_envs, motion_emb.shape[0])):
                                sim = compute_cosine_similarity(motion_emb[i], random_sample_emb)
                                metrics.cos_random_sample.append(sim)
                        
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
                                    for i in valid_ids:
                                        if i < z_shifted.shape[0]:
                                            sim = compute_cosine_similarity(z_shifted[i], random_sample_emb)
                                            if shift not in metrics.cos_time_shift:
                                                metrics.cos_time_shift[shift] = []
                                            metrics.cos_time_shift[shift].append(sim)
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

    if recorder is not None:
        recorder.stop()

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

    # Compute summary
    return EvaluationResult(
        onomatopoeia=onomatopoeia,
        mean_velocity_x=float(np.mean(metrics.velocity_x)),
        std_velocity_x=float(np.std(metrics.velocity_x)),
        mean_com_x=float(np.mean(metrics.com_x)),
        std_com_x=float(np.std(metrics.com_x)),
        mean_cos_centroid=float(np.mean(metrics.cos_centroid)) if metrics.cos_centroid else None,
        std_cos_centroid=float(np.std(metrics.cos_centroid)) if metrics.cos_centroid else None,
        mean_cos_random_sample=float(np.mean(metrics.cos_random_sample)) if metrics.cos_random_sample else None,
        std_cos_random_sample=float(np.std(metrics.cos_random_sample)) if metrics.cos_random_sample else None,
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
        if stats_path and stats_path.exists():
            apply_normalization_from_stats(dataset, stats_path)
        else:
            # デフォルトのパスを試す
            default_stats = hoyo_root / "data" / "normalization_stats.json"
            if default_stats.exists():
                apply_normalization_from_stats(dataset, default_stats)
        
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


def save_results(results: list[EvaluationResult], output_dir: Path, timestamp: str) -> None:
    """結果をJSON形式で保存."""
    output_path = output_dir / f"eval_motion_{timestamp}.json"
    data = {
        "timestamp": timestamp,
        "results": [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    config = EvalConfig.from_args(args_cli)

    # Parse env config
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

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
    logger.info(f"Loading run from: {log_dir}")

    # Load env config from run
    log_env_cfg_path = os.path.join(log_dir, "params", "env.yaml")
    if args_cli.use_log_env and os.path.exists(log_env_cfg_path):
        try:
            log_env_cfg_dict = load_yaml(log_env_cfg_path)
            update_class_from_dict(env_cfg, log_env_cfg_dict)
            logger.info(f"Loaded env config from: {log_env_cfg_path}")
        except Exception as e:
            logger.warning(f"Failed to load env config: {e}")

    # Force CLI num_envs
    env_cfg.scene.num_envs = args_cli.num_envs

    # Load agent config
    agent_cfg_path = os.path.join(log_dir, "params", "agent.yaml")
    if not os.path.exists(agent_cfg_path):
        raise FileNotFoundError(f"Agent config not found: {agent_cfg_path}")
    agent_cfg_dict = load_yaml(agent_cfg_path)
    update_class_from_dict(agent_cfg, agent_cfg_dict)

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
    seed_everything(config.seed)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # Load policy
    resume_path = get_checkpoint_path(
        os.path.join(log_root, agent_cfg.experiment_name),
        args_cli.load_run,
        agent_cfg.load_checkpoint,
    )
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)
    logger.info(f"Loaded checkpoint: {resume_path}")

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Prepare style cache
    style_cache = {}
    style_term = env.unwrapped.command_manager._terms.get("style_command")
    if style_term is not None:
        styles_to_eval = INSTRUCTION_ONOMATOPEIA
        if args_cli.style_list:
            styles_to_eval = [s.strip() for s in args_cli.style_list.split(",") if s.strip()]

        for style in styles_to_eval:
            z_onm, teacher_motion = style_term.style_module.encode_instruction(style)
            style_cache[style] = (z_onm.squeeze(0), teacher_motion.squeeze(0))
    else:
        styles_to_eval = ["default"]

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

    # Evaluate each style
    results = []
    logger.info("=" * 60)
    logger.info("MOTION EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Evaluating {len(styles_to_eval)} styles")

    for onomatopoeia in styles_to_eval:
        logger.info(f"\n--- Evaluating: {onomatopoeia} ---")

        # Load reference
        reference = load_hoyo_reference(hoyo_root, onomatopoeia, seed=args_cli.hoyo_seed)

        # Video path
        video_path = None
        if video_dir:
            safe_name = onomatopoeia.replace("/", "_").replace("\\", "_")
            video_path = str(video_dir / f"{safe_name}.mp4")

        result = evaluate_single_style(
            env_wrapper=env_wrapper,
            policy=policy,
            onomatopoeia=onomatopoeia,
            config=config,
            reference_keypoints=reference,
            video_path=video_path,
        )
        results.append(result)

        # Print summary (use print for immediate output)
        print(f"\n  [RESULT] {onomatopoeia}")
        print(f"    mean_velocity_x: {result.mean_velocity_x:.4f} ± {result.std_velocity_x:.4f}")
        if result.mean_cos_centroid is not None:
            print(f"    cos_centroid (H1 vs Centroid): {result.mean_cos_centroid:.4f}")
        if result.mean_cos_random_sample is not None:
            print(f"    cos_random_sample (H1 vs Random): {result.mean_cos_random_sample:.4f}")
        if result.mean_joint_error is not None:
            print(f"    mean_joint_error (L2): {result.mean_joint_error:.4f}")
        if result.mean_joint_error_dtw is not None:
            print(f"    mean_joint_error_dtw (Soft-DTW): {result.mean_joint_error_dtw:.4f}")
        print(f"    fall_rate: {result.fall_rate:.2%}")
        if config.log_reward_terms:
            if result.mean_reward_total is not None:
                print(f"    mean_reward_total: {result.mean_reward_total:+.6f} ± {result.std_reward_total:.6f}")
            print("    reward_terms (top-|mean|):")
            print(_summarize_reward_terms(result.reward_terms_mean, result.reward_terms_std, config.reward_terms_topk))
        sys.stdout.flush()

    # Save results
    save_results(results, output_dir, timestamp)

    # Print final summary table
    print("")
    print("=" * 100)
    print("EVALUATION SUMMARY")
    print(f"Command: lin_vel_x={config.lin_vel_x}, lin_vel_y={config.lin_vel_y}, ang_vel_z={config.ang_vel_z}")
    print("=" * 100)
    print(f"{'Onomatopoeia':<12} {'Cmd X':>6} {'Vel X':>8} {'Cos(C)':>8} {'Cos(R)':>8} {'L2 Err':>8} {'DTW Err':>8} {'Fall%':>8}")
    print("-" * 100)
    for r in results:
        cos_c = f"{r.mean_cos_centroid:.3f}" if r.mean_cos_centroid is not None else "N/A"
        cos_r = f"{r.mean_cos_random_sample:.3f}" if r.mean_cos_random_sample is not None else "N/A"
        j_err = f"{r.mean_joint_error:.4f}" if r.mean_joint_error is not None else "N/A"
        j_dtw = f"{r.mean_joint_error_dtw:.4f}" if r.mean_joint_error_dtw is not None else "N/A"
        print(
            f"{r.onomatopoeia:<12} {config.lin_vel_x:>6.2f} {r.mean_velocity_x:>8.3f} "
            f"{cos_c:>8} {cos_r:>8} {j_err:>8} {j_dtw:>8} {r.fall_rate:>7.1%}"
        )
    print("=" * 100)
    print("Cos(C/R) = embedding similarity, L2/DTW Err = joint error (same unit, DTW=phase-aligned)")
    print(f"Results saved to: {output_dir / f'eval_motion_{timestamp}.json'}")
    
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

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
