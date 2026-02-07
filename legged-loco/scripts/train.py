# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

import atexit
import csv
import faulthandler
import json
import os
import re
import shlex
import subprocess
import sys

# ensure repository modules are discoverable when launched via Isaac Sim kit python
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT) # legged-loco

# Add legged-loco to path
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

NAVILA_ROOT = os.path.dirname(REPO_ROOT) # NaVILA-Bench
# Add NaVILA-Bench to path (for hoyo imports etc)
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
            # break # Don't break? Or break per group?
            
# Import AppLauncher first (needed before other isaaclab imports)
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--style_weight", type=float, default=None, help="Override style reward weight.")
parser.add_argument(
    "--no_style_reward",
    action="store_true",
    default=False,
    help="Disable style reward by forcing rewards.style_tracking.weight=0.0 (ablation).",
)
parser.add_argument("--style_beta_text", type=float, default=None, help="Override style reward beta_text.")
parser.add_argument(
    "--style_beta_teacher_motion",
    type=float,
    default=None,
    help="Override style reward beta_teacher_motion (teacher motion latent similarity).",
)
parser.add_argument(
    "--style_beta_centroid",
    type=float,
    default=None,
    help="Deprecated. Use --style_beta_teacher_motion instead.",
)
parser.add_argument("--style_ramp_steps", type=int, default=None, help="Override style reward ramp steps.")
parser.add_argument(
    "--style_centroid_mode",
    type=str,
    default=None,
    choices=["centroid", "random"],
    help="Centroid selection mode for style reward. Overrides STYLE_CENTROID_MODE env var.",
)
parser.add_argument(
    "--terrain",
    type=str,
    default="flat",
    choices=["flat", "rough", "keep"],
    help="Override terrain type for training. flat=plane, rough=use config generator, keep=no override.",
)
parser.add_argument(
    "--style_list",
    type=str,
    default=None,
    help="Comma-separated style list to sample from (overrides INSTRUCTION_ONOMATOPEIA).",
)
parser.add_argument(
    "--run_note",
    type=str,
    default=None,
    help="Short change summary used in run name and RUN.md.",
)
parser.add_argument(
    "--run_purpose",
    type=str,
    default=None,
    help="One-line experiment purpose written to RUN.md.",
)
parser.add_argument(
    "--run_changes",
    type=str,
    default=None,
    help="Change summary (reward/obs/command/arch) written to RUN.md.",
)
parser.add_argument(
    "--log_contrib",
    action="store_true",
    default=False,
    help="Log contribution rates (rate^mag/rate^adv/share^E) during training.",
)
parser.add_argument(
    "--contrib_interval",
    type=int,
    default=1,
    help="Iteration interval for contribution logging.",
)
parser.add_argument(
    "--contrib_topk",
    type=int,
    default=8,
    help="Top-k terms/joints to log for contribution rates.",
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
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# from omni.isaac.viplanner.config import H1RoughEnvCfg, H1BaseRoughEnvCfg, H12DoFRoughEnvCfg, H1VisionRoughEnvCfg, G1VisionRoughEnvCfg
# from omni.isaac.viplanner.config import H1RoughEnvCfg_PLAY, H1BaseRoughEnvCfg_PLAY, H12DoFRoughEnvCfg_PLAY, H1VisionRoughEnvCfg_PLAY, G1VisionRoughEnvCfg_PLAY
from omni.isaac.leggedloco.config import *
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

EXPERIMENTS_CSV_COLUMNS = [
    "run_name",
    "run_name_base",
    "date",
    "task",
    "seed",
    "note",
    "experiment_name",
    "reward_weights",
    "best_metric",
    "fall_rate",
    "mean_vel_x",
    "checkpoint_best_path",
    "checkpoint_last_path",
    "commit_hash",
    "log_dir",
    "video_dir",
]


def _format_float_token(value: float) -> str:
    sign = "m" if value < 0 else ""
    s = str(abs(value)).replace(".", "p")
    return f"{sign}{s}"


def _sanitize_token(text: str) -> str:
    cleaned = text.strip().replace(".", "p")
    cleaned = cleaned.replace(" ", "")
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", cleaned)
    return cleaned or "default"


class _TeeStream:
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        try:
            self._primary.write(data)
        except Exception:
            pass
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self._primary.flush()
        except Exception:
            pass
        try:
            self._secondary.flush()
        except Exception:
            pass

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def isatty(self):
        if hasattr(self._primary, "isatty"):
            try:
                return bool(self._primary.isatty())
            except Exception:
                return False
        return False

    def fileno(self):
        if hasattr(self._primary, "fileno"):
            return self._primary.fileno()
        raise OSError("fileno")

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _enable_run_logging(log_dir: str) -> str | None:
    """Tee stdout/stderr into <log_dir>/train.log and enable faulthandler.

    Returns the log file path on success, otherwise None.
    """
    log_path = os.path.join(log_dir, "train.log")
    try:
        log_file = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
    except Exception as exc:
        print(f"[WARN] Failed to open train log at {log_path}: {exc}")
        return None

    stdout_original = sys.stdout
    stderr_original = sys.stderr
    sys.stdout = _TeeStream(stdout_original, log_file)
    sys.stderr = _TeeStream(stderr_original, log_file)

    try:
        faulthandler.enable(file=log_file, all_threads=True)
    except Exception as exc:
        print(f"[WARN] Failed to enable faulthandler: {exc}")

    def _cleanup():
        try:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = stdout_original
            sys.stderr = stderr_original
        finally:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass

    atexit.register(_cleanup)
    print(f"[INFO] Stdout/stderr are being tee'd to: {log_path}")
    return log_path


def _collect_reward_weights(env_cfg) -> dict:
    weights = {}
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return weights
    for name in dir(rewards):
        if name.startswith("_"):
            continue
        term = getattr(rewards, name)
        if hasattr(term, "weight"):
            try:
                weights[name] = float(term.weight)
            except Exception:
                continue
    return weights


def _format_reward_summary(weights: dict) -> str:
    if not weights:
        return ""
    parts = [f"{k}={v}" for k, v in sorted(weights.items())]
    return ";".join(parts)


def _build_run_note(args_cli, env_cfg) -> str:
    if args_cli.run_note:
        return args_cli.run_note
    if args_cli.run_name:
        return args_cli.run_name
    parts = []
    if args_cli.history_length:
        parts.append(f"obsHist{args_cli.history_length}")
    if args_cli.use_rnn:
        parts.append("useRnn")
    if args_cli.use_cnn:
        parts.append("useCnn")
    if args_cli.arm_fixed:
        parts.append("armFixed")
    if args_cli.terrain and args_cli.terrain != "flat":
        parts.append(f"terrain{args_cli.terrain}")
    if args_cli.no_style_reward:
        parts.append("noStyle")
    if args_cli.style_weight is not None:
        parts.append(f"rStyle{_format_float_token(args_cli.style_weight)}")
    if args_cli.style_beta_text is not None:
        parts.append(f"bText{_format_float_token(args_cli.style_beta_text)}")
    beta_teacher = args_cli.style_beta_teacher_motion
    if beta_teacher is None and args_cli.style_beta_centroid is not None:
        beta_teacher = args_cli.style_beta_centroid
    if beta_teacher is not None:
        parts.append(f"bTeach{_format_float_token(beta_teacher)}")
    if args_cli.style_ramp_steps is not None:
        parts.append(f"ramp{args_cli.style_ramp_steps}")
    if args_cli.style_centroid_mode is not None:
        parts.append(f"centroid{args_cli.style_centroid_mode}")
    if args_cli.style_list is not None:
        styles = [s.strip() for s in args_cli.style_list.split(",") if s.strip()]
        parts.append(f"styles{len(styles)}")
    if not parts:
        reward_weights = _collect_reward_weights(env_cfg)
        if "style_tracking" in reward_weights:
            parts.append(f"rStyle{_format_float_token(reward_weights['style_tracking'])}")
    return "_".join(parts) if parts else "default"


def _build_run_name(args_cli, agent_cfg, env_cfg) -> tuple[str, str]:
    date_tag = datetime.now().strftime("%Y-%m-%d")
    task = _sanitize_token(args_cli.task or "task")
    note = _sanitize_token(_build_run_note(args_cli, env_cfg))
    seed_val = agent_cfg.seed if agent_cfg.seed is not None else args_cli.seed
    seed = str(seed_val) if seed_val is not None else "na"
    run_name_base = f"{date_tag}_{task}_{note}_seed{seed}"
    return run_name_base, note


def _get_git_commit_hash(root: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    except Exception:
        return "unknown"


def _stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _update_experiments_csv(path: str, row: dict, key_field: str = "run_name") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    fieldnames = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    if not fieldnames:
        fieldnames = list(EXPERIMENTS_CSV_COLUMNS)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    updated = False
    for existing in rows:
        if existing.get(key_field) == row.get(key_field):
            existing.update({k: _stringify_value(v) for k, v in row.items()})
            updated = True
            break
    if not updated:
        rows.append({k: _stringify_value(v) for k, v in row.items()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing in rows:
            writer.writerow(existing)


def _find_checkpoint_paths(log_dir: str) -> tuple[str | None, str | None]:
    if not os.path.isdir(log_dir):
        return None, None
    candidates = [f for f in os.listdir(log_dir) if f.endswith(".pt")]
    best = None
    last = None
    for fname in candidates:
        lower = fname.lower()
        if "best" in lower:
            best = fname
        if "last" in lower:
            last = fname
    if last is None:
        step_re = re.compile(r"model_(\d+)\.pt")
        best_step = -1
        for fname in candidates:
            match = step_re.match(fname)
            if match:
                step = int(match.group(1))
                if step > best_step:
                    best_step = step
                    last = fname
    best_path = os.path.join(log_dir, best) if best else None
    last_path = os.path.join(log_dir, last) if last else None
    return best_path, last_path


def _write_run_md(path: str, metadata: dict) -> None:
    reward_weights = metadata.get("reward_weights", {})
    reward_lines = [f"- {k}: {v}" for k, v in sorted(reward_weights.items())] if reward_weights else ["- (none)"]
    lines = [
        "# RUN METADATA",
        "",
        "## Experiment",
        f"- Run name: {metadata.get('run_name')}",
        f"- Base run name: {metadata.get('run_name_base')}",
        f"- Task: {metadata.get('task')}",
        f"- Date: {metadata.get('date')}",
        f"- Seed: {metadata.get('seed')}",
        f"- Note: {metadata.get('note')}",
        f"- Purpose: {metadata.get('purpose') or 'N/A'}",
        f"- Changes: {metadata.get('changes') or 'N/A'}",
        "",
        "## Hyperparameters",
        f"- num_envs: {metadata.get('num_envs')}",
        f"- max_iterations: {metadata.get('max_iterations')}",
        f"- learning_rate: {metadata.get('learning_rate')}",
        "",
        "## Reward weights",
        *reward_lines,
        "",
        "## Checkpoints",
        f"- best: {metadata.get('checkpoint_best_path') or 'N/A'}",
        f"- last: {metadata.get('checkpoint_last_path') or 'N/A'}",
        "",
        "## Reproducibility",
        f"- git_commit: {metadata.get('commit_hash')}",
        f"- command: `{metadata.get('command')}`",
        "",
        "## Artifacts",
        f"- log_dir: {metadata.get('log_dir')}",
        f"- stdout_log: {metadata.get('stdout_log_path') or 'N/A'}",
        f"- video_dir: {metadata.get('video_dir') or 'N/A'}",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """Train with RSL-RL agent."""
    # Apply centroid mode before env construction (StyleModule reads env var at init).
    if args_cli.style_centroid_mode is not None:
        os.environ["STYLE_CENTROID_MODE"] = args_cli.style_centroid_mode

    # parse configuration
    # env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
    #     args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    # )
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.no_style_reward and args_cli.style_weight is not None:
        raise ValueError("Cannot set both --no_style_reward and --style_weight. Choose one.")

    # Optional style overrides for experiments
    if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "style_tracking"):
        if args_cli.no_style_reward:
            env_cfg.rewards.style_tracking.weight = 0.0
            print("[INFO] Style reward disabled: rewards.style_tracking.weight=0.0")
        elif args_cli.style_weight is not None:
            env_cfg.rewards.style_tracking.weight = args_cli.style_weight
        if args_cli.style_beta_text is not None:
            env_cfg.rewards.style_tracking.params["beta_text"] = args_cli.style_beta_text
        beta_teacher_motion = args_cli.style_beta_teacher_motion
        if beta_teacher_motion is None and args_cli.style_beta_centroid is not None:
            beta_teacher_motion = args_cli.style_beta_centroid
        if beta_teacher_motion is not None:
            env_cfg.rewards.style_tracking.params["beta_teacher_motion"] = beta_teacher_motion
        if args_cli.style_ramp_steps is not None:
            env_cfg.rewards.style_tracking.params["ramp_steps"] = args_cli.style_ramp_steps
    elif args_cli.no_style_reward:
        print("[WARN] --no_style_reward was requested but rewards.style_tracking is not available in this task.")
    if args_cli.style_list is not None and hasattr(env_cfg, "commands"):
        styles = [s.strip() for s in args_cli.style_list.split(",") if s.strip()]
        if len(styles) == 0:
            raise ValueError("style_list is empty after parsing. Provide at least one style.")
        if hasattr(env_cfg.commands, "style_command"):
            env_cfg.commands.style_command.styles = styles

    # Terrain override (default: flat) for style-focused training
    if args_cli.terrain != "keep":
        if not hasattr(env_cfg, "scene") or not hasattr(env_cfg.scene, "terrain"):
            print(f"[WARN] Terrain override requested but env_cfg has no scene.terrain (task={args_cli.task}).")
        else:
            if args_cli.terrain == "flat":
                current_type = getattr(env_cfg.scene.terrain, "terrain_type", None)
                if current_type not in (None, "generator", "plane"):
                    print(
                        "[WARN] Terrain override 'flat' skipped for non-generator terrain type: "
                        f"{current_type}. Use --terrain=keep to suppress this warning."
                    )
                else:
                    env_cfg.scene.terrain.terrain_type = "plane"
                    env_cfg.scene.terrain.terrain_generator = None
                    env_cfg.scene.terrain.max_init_terrain_level = None
                    if hasattr(env_cfg, "curriculum") and getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
                        env_cfg.curriculum.terrain_levels = None
                    # Switch reset event to uniform when flat patches are not available.
                    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
                        try:
                            import omni.isaac.leggedloco.leggedloco.mdp as mdp

                            env_cfg.events.reset_base.func = mdp.reset_root_state_uniform
                        except Exception as exc:
                            print(f"[WARN] Failed to override reset_base for flat terrain: {exc}")
                    print("[INFO] Terrain override: flat (plane).")
            elif args_cli.terrain == "rough":
                # Keep the default generator-based terrain from the config.
                print("[INFO] Terrain override: rough (use config generator).")
            else:
                raise ValueError(f"Unsupported terrain option: {args_cli.terrain}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # max iterations for training (override before RUN.md generation)
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    run_name_base, run_note = _build_run_name(args_cli, agent_cfg, env_cfg)
    run_name_effective = run_name_base
    log_dir = os.path.join(log_root_path, run_name_effective)
    if os.path.exists(log_dir):
        suffix = 1
        while os.path.exists(f"{log_dir}_dup{suffix}"):
            suffix += 1
        run_name_effective = f"{run_name_base}_dup{suffix}"
        log_dir = os.path.join(log_root_path, run_name_effective)
    agent_cfg.run_name = run_name_effective
    os.makedirs(log_dir, exist_ok=True)

    stdout_log_path = _enable_run_logging(log_dir)
    print(f"[INFO] Run name: {run_name_effective}")
    print(f"[INFO] Log dir: {log_dir}")

    reward_weights = _collect_reward_weights(env_cfg)
    reward_summary = _format_reward_summary(reward_weights)
    git_hash = _get_git_commit_hash(REPO_ROOT)
    command = "python " + " ".join(shlex.quote(arg) for arg in sys.argv)
    num_envs = getattr(getattr(env_cfg, "scene", None), "num_envs", None) or args_cli.num_envs
    learning_rate = None
    for candidate in (
        ("alg", "learning_rate"),
        ("alg", "lr"),
        ("algorithm", "learning_rate"),
        ("algorithm", "lr"),
    ):
        parent = getattr(agent_cfg, candidate[0], None)
        if parent is not None and hasattr(parent, candidate[1]):
            learning_rate = getattr(parent, candidate[1])
            break

    run_metadata = {
        "run_name": run_name_effective,
        "run_name_base": run_name_base,
        "experiment_name": agent_cfg.experiment_name,
        "task": args_cli.task,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "seed": agent_cfg.seed,
        "note": run_note,
        "purpose": args_cli.run_purpose,
        "changes": args_cli.run_changes,
        "num_envs": num_envs,
        "max_iterations": agent_cfg.max_iterations,
        "learning_rate": learning_rate,
        "reward_weights": reward_weights,
        "checkpoint_best_path": None,
        "checkpoint_last_path": None,
        "commit_hash": git_hash,
        "command": command,
        "log_dir": log_dir,
        "stdout_log_path": stdout_log_path,
        "video_dir": os.path.join(log_dir, "videos") if args_cli.video else None,
    }

    run_md_path = os.path.join(log_dir, "RUN.md")
    _write_run_md(run_md_path, run_metadata)
    experiments_csv_path = os.path.join(os.path.dirname(log_root_path), "experiments.csv")
    _update_experiments_csv(
        experiments_csv_path,
        {
            "run_name": run_name_effective,
            "run_name_base": run_name_base,
            "date": run_metadata["date"],
            "task": args_cli.task,
            "seed": run_metadata["seed"],
            "note": run_note,
            "experiment_name": agent_cfg.experiment_name,
            "reward_weights": reward_summary,
            "best_metric": "",
            "fall_rate": "",
            "mean_vel_x": "",
            "checkpoint_best_path": "",
            "checkpoint_last_path": "",
            "commit_hash": git_hash,
            "log_dir": log_dir,
            "video_dir": run_metadata["video_dir"],
        },
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    train_cfg = agent_cfg.to_dict()
    train_cfg["contrib"] = {
        "enabled": bool(args_cli.log_contrib),
        "interval": int(args_cli.contrib_interval),
        "topk": int(args_cli.contrib_topk),
    }
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # ---- trainable params summary (minimal) ----
    ac = runner.alg.actor_critic
    total = sum(p.numel() for p in ac.parameters())
    trainable = sum(p.numel() for p in ac.parameters() if p.requires_grad)
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({ratio:.1f}%)")

    base = getattr(ac, "base_policy", None)
    if base is not None:
        base_total = sum(p.numel() for p in base.parameters())
        base_trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
        print(f"[INFO] Base policy trainable: {base_trainable:,} / {base_total:,}")
    # -------------------------------------------

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    checkpoint_best, checkpoint_last = _find_checkpoint_paths(log_dir)
    run_metadata["checkpoint_best_path"] = checkpoint_best
    run_metadata["checkpoint_last_path"] = checkpoint_last
    _write_run_md(run_md_path, run_metadata)
    _update_experiments_csv(
        experiments_csv_path,
        {
            "run_name": run_name_effective,
            "checkpoint_best_path": checkpoint_best or "",
            "checkpoint_last_path": checkpoint_last or "",
        },
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
