#!/usr/bin/env python3
"""Record HOYO keypoints (world coords) from H1 in IsaacLab and save to NPZ."""

import argparse
import os
import sys
from pathlib import Path

# Ensure local extensions are discoverable (same pattern as other scripts)
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

parser = argparse.ArgumentParser(description="Record HOYO keypoints (world coords) from H1.")
parser.add_argument("--task", type=str, default="h1_vision_heading_fixed", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--env_id", type=int, default=0, help="Env index to record.")
parser.add_argument("--steps", type=int, default=300, help="Number of steps to record.")
parser.add_argument(
    "--out",
    type=str,
    default="eval_results/keymapping_world/latest_keymapping_world.npz",
    help="Output npz path.",
)
parser.add_argument("--base_velocity_mode", type=str, choices=["env", "fixed"], default="fixed")
parser.add_argument("--lin_vel_x", type=float, default=0.5)
parser.add_argument("--lin_vel_y", type=float, default=0.0)
parser.add_argument("--ang_vel_z", type=float, default=0.0)
parser.add_argument("--heading", type=float, default=0.0)
parser.add_argument("--use_log_env", action="store_true", default=False, help="Load env.yaml from run.")
parser.add_argument("--log_root", type=str, default=None, help="Override log root for runs.")
parser.add_argument("--no_policy", action="store_true", default=False, help="Disable policy (zero actions).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_cnn", action="store_true", default=None, help="Match CNN policy if needed.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Enable RNN policy if needed.")
parser.add_argument("--history_length", default=0, type=int, help="History length for RNN policies.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml

from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import load_yaml
from isaaclab.utils import update_class_from_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.leggedloco.config import *  # noqa: F401,F403

HOYO_JOINT_NAMES = [
    "head",
    "neck",
    "r_shoulder",
    "r_elbow",
    "r_hand",
    "l_shoulder",
    "l_elbow",
    "l_hand",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
]


def load_yaml_with_slices(path: str) -> dict:
    """Load YAML while supporting python/object/apply:builtins.slice tags."""
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
    # If a group omits "style", explicitly disable it.
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


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    resume_path = None
    log_dir = None
    use_policy = not args_cli.no_policy

    if use_policy:
        if args_cli.load_run is None and args_cli.checkpoint is None:
            print("[WARN] --load_run/--checkpoint not provided; disabling policy.")
            use_policy = False
        else:
            log_root_base = resolve_log_root_path(agent_cfg.experiment_name, args_cli.load_run, args_cli.log_root)
            log_dir = os.path.join(log_root_base, agent_cfg.experiment_name, args_cli.load_run)
            if args_cli.use_log_env and log_dir and os.path.exists(os.path.join(log_dir, "params", "env.yaml")):
                try:
                    try:
                        log_env_cfg_dict = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
                    except Exception as exc:
                        print(f"[WARN] Failed to load env.yaml with default loader: {exc}")
                        log_env_cfg_dict = load_yaml_with_slices(os.path.join(log_dir, "params", "env.yaml"))
                    update_class_from_dict(env_cfg, log_env_cfg_dict)
                except ValueError as exc:
                    print(f"[WARN] Full env.yaml update failed: {exc}")
                    if isinstance(log_env_cfg_dict, dict) and "observations" in log_env_cfg_dict:
                        apply_observations_override(env_cfg, log_env_cfg_dict["observations"])
                        print("[INFO] Applied observations-only override from env.yaml")
                except Exception as exc:
                    print(f"[WARN] Failed to apply env.yaml overrides: {exc}")
                enforce_num_envs(env_cfg, args_cli.num_envs or env_cfg.scene.num_envs)

            if log_dir and os.path.exists(os.path.join(log_dir, "params", "agent.yaml")):
                try:
                    log_agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
                    update_class_from_dict(agent_cfg, log_agent_cfg_dict)
                except Exception as exc:
                    print(f"[WARN] Failed to apply agent.yaml overrides: {exc}")

            # resolve checkpoint
            if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
                resume_path = args_cli.checkpoint
            else:
                resume_path = get_checkpoint_path(
                    os.path.join(log_root_base, agent_cfg.experiment_name),
                    args_cli.load_run,
                    agent_cfg.load_checkpoint,
                )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    policy = None
    if use_policy and resume_path:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        print(f"[INFO] Loaded policy checkpoint: {resume_path}")
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    else:
        print("[WARN] Policy disabled; using zero actions (robot may not move).")

    # Initialize
    obs, _ = env.get_observations()

    base_cmd_term = env.unwrapped.command_manager._terms.get("base_velocity")
    if args_cli.base_velocity_mode == "fixed" and base_cmd_term is not None:
        cmd = {
            "lin_vel_x": args_cli.lin_vel_x,
            "lin_vel_y": args_cli.lin_vel_y,
            "ang_vel_z": args_cli.ang_vel_z,
            "heading": args_cli.heading,
        }
        _set_base_velocity_for_envs(base_cmd_term, range(env.unwrapped.num_envs), cmd)

    style_term = env.unwrapped.command_manager._terms.get("style_command")
    if style_term is None:
        raise RuntimeError("style_command not found in command_manager. Cannot access StyleModule.")
    style_module = style_term.style_module

    robot = env.unwrapped.scene["robot"]
    body_names = list(robot.data.body_names)

    keypoints = []
    root_positions = []

    for step in range(args_cli.steps):
        with torch.inference_mode():
            if policy is None:
                action_dim = env.action_space.shape[0]
                actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
            else:
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # enforce fixed base velocity each step if needed
            if args_cli.base_velocity_mode == "fixed" and base_cmd_term is not None:
                cmd = {
                    "lin_vel_x": args_cli.lin_vel_x,
                    "lin_vel_y": args_cli.lin_vel_y,
                    "ang_vel_z": args_cli.ang_vel_z,
                    "heading": args_cli.heading,
                }
                _set_base_velocity_for_envs(base_cmd_term, range(env.unwrapped.num_envs), cmd)

            body_pos_w = robot.data.body_pos_w
            body_quat_w = robot.data.body_quat_w
            root_pos_w = robot.data.root_pos_w
            # Use the same mapping as style reward
            hoyo_3d = style_module._get_hoyo_joints_from_h1(body_pos_w, body_names, body_quat_w)
            env_id = int(args_cli.env_id)
            keypoints.append(hoyo_3d[env_id].detach().cpu().numpy())
            root_positions.append(root_pos_w[env_id].detach().cpu().numpy())

    keypoints = np.stack(keypoints, axis=0)  # (T, 14, 3)
    root_positions = np.stack(root_positions, axis=0)

    # resolve mapping
    mapping_indices = None
    mapping_target_names = []
    mapping_body_names = []
    if hasattr(style_module, "body_indices") and style_module.body_indices is not None:
        mapping_indices = style_module.body_indices.detach().cpu().numpy()
        map_dict = getattr(style_module, "_resolved_map_dict", None) or (style_module.mapping_dict or {})
        for joint in HOYO_JOINT_NAMES:
            mapping_target_names.append(map_dict.get(joint, "torso_link"))
        for idx in mapping_indices:
            mapping_body_names.append(body_names[int(idx)])

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        keypoints_w=keypoints,
        root_pos_w=root_positions,
        body_names=np.array(body_names, dtype=object),
        hoyo_joint_names=np.array(HOYO_JOINT_NAMES, dtype=object),
        mapping_indices=mapping_indices,
        mapping_target_names=np.array(mapping_target_names, dtype=object),
        mapping_body_names=np.array(mapping_body_names, dtype=object),
        coord_mode=getattr(style_module, "coord_mode", "unknown"),
        task=args_cli.task,
    )
    print(f"[INFO] Saved keypoints: {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
