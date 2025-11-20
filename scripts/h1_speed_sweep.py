#!/usr/bin/env python3
# Copyright (c) 2022-2024
# SPDX-License-Identifier: BSD-3-Clause
"""
Sweep commanded base velocities for H1 and measure realized speeds.

- Loads the trained low-level policy (RSL-RL) for the given task.
- Issues commanded velocities [vx, 0.0, omega] through the RSL-RL history wrapper.
- Measures realized base velocities over a hold window per command.
"""

import argparse
import os
import sys
import math
import time
from typing import List, Tuple

import torch
import numpy as np

# ensure local extensions (omni.isaac.vlnce, etc.) are on the import path when launched via isaaclab.sh
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
ISAACLAB_SOURCE = os.path.join(os.path.dirname(REPO_ROOT), "IsaacLab", "source")
if ISAACLAB_SOURCE not in sys.path:
    sys.path.append(ISAACLAB_SOURCE)
ISAACLAB_PKG = os.path.join(ISAACLAB_SOURCE, "isaaclab")
if ISAACLAB_PKG not in sys.path:
    sys.path.append(ISAACLAB_PKG)
LOCAL_EXT_PATH_GROUPS = [
    (
        "omni.isaac.vlnce",
        [
            os.path.join(REPO_ROOT, "omni.isaac.vlnce"),
            os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.vlnce"),
            os.path.join(REPO_ROOT, "legged-loco", "isaaclab_exts", "omni.isaac.vlnce"),
            os.path.join(os.path.dirname(REPO_ROOT), "IsaacLab", "source", "omni.isaac.vlnce"),
        ],
    ),
    (
        "omni.isaac.matterport",
        [
            os.path.join(REPO_ROOT, "omni.isaac.matterport"),
            os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.matterport"),
            os.path.join(REPO_ROOT, "legged-loco", "isaaclab_exts", "omni.isaac.matterport"),
            os.path.join(os.path.dirname(REPO_ROOT), "IsaacLab", "source", "omni.isaac.matterport"),
        ],
    ),
    (
        "omni.isaac.leggedloco",
        [
            os.path.join(REPO_ROOT, "omni.isaac.leggedloco"),
            os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.leggedloco"),
            os.path.join(REPO_ROOT, "legged-loco", "isaaclab_exts", "omni.isaac.leggedloco"),
            os.path.join(os.path.dirname(REPO_ROOT), "IsaacLab", "source", "omni.isaac.leggedloco"),
        ],
    ),
]

for _, candidate_paths in LOCAL_EXT_PATH_GROUPS:
    for _ext_path in candidate_paths:
        if os.path.isdir(_ext_path) and _ext_path not in sys.path:
            sys.path.append(_ext_path)
            break

try:
    from omni.isaac.lab.app import AppLauncher
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def frange(vmin: float, vmax: float, step: float) -> List[float]:
    if step <= 0:
        return [vmin]
    n = int(math.floor((vmax - vmin) / step)) + 1
    return [round(vmin + i * step, 6) for i in range(max(n, 1))]


def parse_csv_floats(s: str | None) -> List[float]:
    if not s:
        return []
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except Exception:
            pass
    return vals


def measure_over_window(env, cmd: torch.Tensor, window_steps: int, simulation_app) -> Tuple[float, float]:
    lin_samples = []
    ang_samples = []
    for _ in range(window_steps):
        # advance physics with the same command while collecting samples
        _ = env.step(cmd)
        simulation_app.update()
        # measure base velocities in body frame
        try:
            lin_b = env.unwrapped.scene["robot"].data.root_lin_vel_b[0].detach().cpu().numpy()
            ang_b = env.unwrapped.scene["robot"].data.root_ang_vel_b[0].detach().cpu().numpy()
        except Exception:
            lin_b = np.zeros(3, dtype=np.float32)
            ang_b = np.zeros(3, dtype=np.float32)
        lin_samples.append(float(lin_b[0]))
        ang_samples.append(float(ang_b[2]))
    return float(np.mean(lin_samples)), float(np.mean(ang_samples))


# isaaclab argparse arguments
parser = argparse.ArgumentParser(description="H1 speed sweep (vx/omega) experiment.")
parser.add_argument("--task", type=str, default="h1_matterport_vision", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--visualize_path", action="store_true", default=False, help="Visualize the path in the simulator.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# Sweep definition (either explicit lists or ranges)
parser.add_argument("--vx_list", type=str, default="", help="Comma-separated list of vx (m/s).")
parser.add_argument("--omega_list", type=str, default="", help="Comma-separated list of omega (rad/s).")
parser.add_argument("--vx_min", type=float, default=0.0)
parser.add_argument("--vx_max", type=float, default=1.0)
parser.add_argument("--vx_step", type=float, default=0.1)
parser.add_argument("--omega_min", type=float, default=-1.0, help="Min angular velocity (rad/s). Default -1.0 to test both directions.")
parser.add_argument("--omega_max", type=float, default=1.0)
parser.add_argument("--omega_step", type=float, default=0.2)

# Timing
parser.add_argument("--hold_time", type=float, default=1.5, help="Seconds to hold each command.")
parser.add_argument("--stabilize_time", type=float, default=0.5, help="Seconds before measuring.")
parser.add_argument("--ramp_time", type=float, default=0.0, help="Seconds to ramp to target command.")

# Output
parser.add_argument("--csv_out", type=str, default="", help="Optional CSV path to save results.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during run.")
parser.add_argument("--high_level_obs_key", type=str, default=None, help="Override high-level obs key (e.g., 'camera_obs' or 'policy'). Default: auto-detect.")
parser.add_argument("--scene_id_override", type=str, default=None, help="Override Matterport scene id (e.g., 'QUCTc6BB5sX').")
parser.add_argument("--scene_usd", type=str, default=None, help="Direct USD path to use for Matterport terrain.")
parser.add_argument("--episode_idx", type=int, default=0, help="Episode index within the selected scene (for spawn pose).")
parser.add_argument("--preset", type=str, choices=["matterport", "open"], default="matterport", help="Preset to simplify setup. 'open' switches to h1_vision.")
parser.add_argument("--open_env_camera", action="store_true", default=False, help="Position camera behind the robot (non-Matterport tasks).")

# RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for vision-based tasks (required for camera sensors)
# This is needed when the task uses camera sensors (like h1_matterport_vision)
if not getattr(args_cli, 'enable_cameras', False):
    args_cli.enable_cameras = True

# launch omniverse app - MUST be done before importing other isaaclab modules
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner

try:
    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.lab.utils.io import load_yaml
    import omni.isaac.lab.utils.math as math_utils
    from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from omni.isaac.lab.utils import update_class_from_dict
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlVecEnvWrapper,
    )
    import omni.isaac.lab.sim as sim_utils
except ModuleNotFoundError:
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab.utils.io import load_yaml
    import isaaclab.utils.math as math_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.utils import update_class_from_dict
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlVecEnvWrapper,
    )
    import isaaclab.sim as sim_utils

from omni.isaac.vlnce.config import *  # noqa: F403
# Register non-Matterport H1 environments (flat/rough terrains)
try:
    from omni.isaac.leggedloco.config.h1 import *  # noqa: F403
    _LEGLOCO_OK = True
except Exception as _e:
    _LEGLOCO_OK = False
    print(f"[WARN] Failed to import omni.isaac.leggedloco.config.h1: {_e}")
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper, read_episodes

import gymnasium as gym


def main():

    # Build env cfg and agent cfg
    selected_task = args_cli.task
    if args_cli.preset == "open":
        # Use legged loco vision env for open terrain tests
        if "matterport" in (selected_task or "").lower():
            print(f"[INFO] Preset 'open' requested. Switching task from '{selected_task}' to 'h1_vision'.")
        selected_task = "h1_vision"
        # If legged-loco wasn't imported properly, warn early
        if not _LEGLOCO_OK:
            print("[WARN] 'h1_vision' preset requested but legged-loco envs failed to import. "
                  "Please ensure 'legged-loco/isaaclab_exts/omni.isaac.leggedloco' is on PYTHONPATH.")
    # Preflight: ensure task is registered in Gym
    try:
        import gymnasium as _gym_check
        _ = _gym_check.spec(selected_task)
    except Exception as _e:
        print(f"[WARN] Task '{selected_task}' not found in Gym registry: {_e}")
        if selected_task == "h1_vision":
            print("[HINT] Try running without '--preset=open' or verify legged-loco extension import.")
    env_cfg = parse_env_cfg(selected_task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(selected_task, args_cli, play=True)

    # Optional: override Matterport scene USD when using Matterport tasks
    try:
        if "matterport" in str(selected_task).lower():
            usd_path = None
            target_scene_id = None
            if args_cli.scene_usd:
                if os.path.isfile(args_cli.scene_usd):
                    usd_path = os.path.abspath(args_cli.scene_usd)
                    target_scene_id = os.path.splitext(os.path.basename(usd_path))[0]
                else:
                    print(f"[WARN] --scene_usd not found: {args_cli.scene_usd}")
            elif args_cli.scene_id_override:
                scene_id = args_cli.scene_id_override.strip()
                candidate = os.path.join(ASSETS_DIR, "matterport_usd", scene_id, f"{scene_id}.usd")
                if os.path.isfile(candidate):
                    usd_path = candidate
                    target_scene_id = scene_id
                else:
                    print(f"[WARN] USD for scene_id '{scene_id}' not found at: {candidate}")
            if usd_path:
                print(f"[INFO] Overriding Matterport USD: {usd_path}")
                try:
                    env_cfg.scene.terrain.obj_filepath = usd_path
                except Exception as e:
                    print(f"[WARN] Failed to set env_cfg.scene.terrain.obj_filepath: {e}")
            # Align spawn pose to dataset when possible
            if target_scene_id:
                try:
                    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
                    all_episodes = read_episodes(r2r_data_path)
                    matching = [ep for ep in all_episodes if os.path.splitext(os.path.basename(ep["scene_id"]))[0] == target_scene_id]
                    if not matching:
                        print(f"[WARN] No dataset episodes found for scene '{target_scene_id}'. Using default spawn.")
                    else:
                        ep_idx = max(0, min(int(args_cli.episode_idx), len(matching) - 1))
                        episode = matching[ep_idx]
                        start_pos = episode["start_position"]
                        start_rot = episode["start_rotation"]
                        goal_pos = episode["reference_path"][-1]
                        env_cfg.scene.robot.init_state.rot = start_rot
                        if "go2" in selected_task:
                            env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + 0.4)
                        elif ("h1" in selected_task) or ("g1" in selected_task):
                            env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + 1.0)
                        else:
                            env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + 0.5)
                        try:
                            env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos
                        except Exception:
                            pass
                        try:
                            env_cfg.scene.disk_1.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + 2.5)
                            env_cfg.scene.disk_2.init_state.pos = (goal_pos[0], goal_pos[1], goal_pos[2] + 2.5)
                        except Exception:
                            pass
                        print(f"[INFO] Spawn aligned to scene '{target_scene_id}' episode {ep_idx}.")
                except Exception as e:
                    print(f"[WARN] Failed to align spawn to dataset: {e}")
    except Exception:
        pass

    # specify directory for logging experiments
    log_root_path = os.path.join(os.path.dirname(__file__), "../logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    # update agent config with the one from the loaded run
    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    assert os.path.exists(log_agent_cfg_file_path), f"Agent config file not found: {log_agent_cfg_file_path}"
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)

    # specify directory for logging experiments
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(selected_task, cfg=env_cfg, render_mode=None)
    # wrap around environment for rsl-rl (match working eval scripts)
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Determine high-level observation key (non-Matterport tasks may not provide 'camera_obs')
    detected_key = None
    try:
        space = getattr(env, "observation_space", None)
        keys = list(getattr(space, "spaces", {}).keys()) if space is not None else []
        if args_cli.high_level_obs_key:
            detected_key = args_cli.high_level_obs_key
        elif "camera_obs" in keys:
            detected_key = "camera_obs"
        elif "policy" in keys:
            detected_key = "policy"
        elif keys:
            detected_key = keys[0]
        else:
            detected_key = "policy"
    except Exception:
        detected_key = args_cli.high_level_obs_key or "policy"
    print(f"[INFO] Using high_level_obs_key='{detected_key}'")

    # Wrap with VLNEnvWrapper to support update_command on both history/non-history envs
    # We don't need measures for speed sweep, so keep measure_names empty
    env = VLNEnvWrapper(
        env=env,
        low_level_policy=policy,
        task_name=selected_task,
        episode={},  # no dataset episode needed for sweep
        high_level_obs_key=detected_key,
        measure_names=[],
    )

    # Reset
    obs, infos = env.reset()
    # Optional: camera positioning for open environments (similar to demo_planner.py)
    try:
        if args_cli.open_env_camera and ("matterport" not in str(selected_task).lower()):
            robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu()
            robot_yaw_quat = math_utils.yaw_quat(robot_quat_w).unsqueeze(0)
            yaw = float(math_utils.euler_xyz_from_quat(robot_yaw_quat)[2].cpu().numpy())
            cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
            cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
            env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)
            print("[INFO] Open env camera positioned behind robot.")
    except Exception:
        pass
    step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    if step_dt <= 0:
        step_dt = 0.02

    # Prepare sweeps
    vx_vals = parse_csv_floats(args_cli.vx_list) or frange(args_cli.vx_min, args_cli.vx_max, args_cli.vx_step)
    om_vals = parse_csv_floats(args_cli.omega_list) or frange(args_cli.omega_min, args_cli.omega_max, args_cli.omega_step)

    print("[INFO] Learned command range (H1 config): vx in [0.0, 1.0], vy=0.0, omega in [-1.0, 1.0].")
    print(f"[INFO] Testing vx: {vx_vals}")
    print(f"[INFO] Testing omega: {om_vals}")

    results: List[Tuple[str, float, float, float, float]] = []

    def step_with_command(cmd: torch.Tensor, steps: int):
        # Send high-level velocity command; VLNEnvWrapper handles low-level policy internally
        for _ in range(steps):
            _, _, _, _ = env.step(cmd)
            simulation_app.update()

    # Linear sweep (omega=0)
    for vx in vx_vals:
        target = torch.tensor([vx, 0.0, 0.0], device=env.unwrapped.device)
        ramp_steps = int(max(0.0, args_cli.ramp_time) / step_dt)
        stab_steps = int(max(0.0, args_cli.stabilize_time) / step_dt)
        hold_steps = int(max(0.0, args_cli.hold_time) / step_dt)

        if ramp_steps > 0:
            for i in range(ramp_steps):
                alpha = float(i + 1) / float(ramp_steps)
                cmd = target * alpha
                step_with_command(cmd, 1)
        else:
            step_with_command(target, 1)

        if stab_steps > 0:
            step_with_command(target, stab_steps)

        lin_meas, ang_meas = measure_over_window(env, target, max(1, hold_steps), simulation_app)
        vx_error = abs(vx - lin_meas)
        results.append(("vx", vx, 0.0, lin_meas, ang_meas))
        print(f"[VX] cmd=({vx:.2f},0,0) -> meas vx={lin_meas:.3f} m/s (err={vx_error:.3f}), omega={ang_meas:.3f} rad/s")

    # Angular sweep (vx=0)
    for om in om_vals:
        target = torch.tensor([0.0, 0.0, om], device=env.unwrapped.device)
        ramp_steps = int(max(0.0, args_cli.ramp_time) / step_dt)
        stab_steps = int(max(0.0, args_cli.stabilize_time) / step_dt)
        hold_steps = int(max(0.0, args_cli.hold_time) / step_dt)

        if ramp_steps > 0:
            for i in range(ramp_steps):
                alpha = float(i + 1) / float(ramp_steps)
                cmd = target * alpha
                step_with_command(cmd, 1)
        else:
            step_with_command(target, 1)

        if stab_steps > 0:
            step_with_command(target, stab_steps)

        lin_meas, ang_meas = measure_over_window(env, target, max(1, hold_steps), simulation_app)
        om_error = abs(om - ang_meas)
        results.append(("om", 0.0, om, lin_meas, ang_meas))
        print(f"[OM] cmd=(0,0,{om:.2f}) -> meas vx={lin_meas:.3f} m/s, omega={ang_meas:.3f} rad/s (err={om_error:.3f})")

    # Save CSV if requested
    if args_cli.csv_out:
        try:
            os.makedirs(os.path.dirname(args_cli.csv_out), exist_ok=True)
        except Exception:
            pass
        with open(args_cli.csv_out, "w", encoding="utf-8") as f:
            f.write("mode,cmd_vx,cmd_omega,meas_vx,meas_omega,err_vx,err_omega\n")
            for mode, vx, om, lin_meas, ang_meas in results:
                # per-sample absolute errors
                err_vx = abs(vx - lin_meas) if mode == "vx" else abs(0.0 - lin_meas)
                err_om = abs(om - ang_meas) if mode == "om" else abs(0.0 - ang_meas)
                f.write(f"{mode},{vx:.6f},{om:.6f},{lin_meas:.6f},{ang_meas:.6f},{err_vx:.6f},{err_om:.6f}\n")
        print(f"[INFO] Wrote results to: {args_cli.csv_out}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: Velocity Tracking Performance")
    print("=" * 60)
    
    vx_results = [(vx, lin_meas) for mode, vx, om, lin_meas, ang_meas in results if mode == "vx"]
    om_results = [(om, ang_meas) for mode, vx, om, lin_meas, ang_meas in results if mode == "om"]
    
    if vx_results:
        vx_errors = [abs(cmd - meas) for cmd, meas in vx_results]
        vx_rmse = math.sqrt(sum(e * e for e in vx_errors) / len(vx_errors))
        vx_max_err = max(vx_errors)
        vx_mean_err = sum(vx_errors) / len(vx_errors)
        print(f"\nLinear Velocity (vx) Tracking:")
        print(f"  Tested range: [{min(vx for vx, _ in vx_results):.2f}, {max(vx for vx, _ in vx_results):.2f}] m/s")
        print(f"  RMSE: {vx_rmse:.4f} m/s")
        print(f"  Mean error: {vx_mean_err:.4f} m/s")
        print(f"  Max error: {vx_max_err:.4f} m/s")
        print(f"  Samples: {len(vx_results)}")
    
    if om_results:
        om_errors = [abs(cmd - meas) for cmd, meas in om_results]
        om_rmse = math.sqrt(sum(e * e for e in om_errors) / len(om_errors))
        om_max_err = max(om_errors)
        om_mean_err = sum(om_errors) / len(om_errors)
        print(f"\nAngular Velocity (omega) Tracking:")
        print(f"  Tested range: [{min(om for om, _ in om_results):.2f}, {max(om for om, _ in om_results):.2f}] rad/s")
        print(f"  RMSE: {om_rmse:.4f} rad/s")
        print(f"  Mean error: {om_mean_err:.4f} rad/s")
        print(f"  Max error: {om_max_err:.4f} rad/s")
        print(f"  Samples: {len(om_results)}")
    
    # Save summary CSV next to results
    if args_cli.csv_out:
        try:
            base, ext = os.path.splitext(args_cli.csv_out)
            summary_path = f"{base}_summary.csv"
            with open(summary_path, "w", encoding="utf-8") as fsum:
                fsum.write("mode,rmse,mean_error,max_error,samples,range_min,range_max\n")
                if vx_results:
                    fsum.write(
                        "vx,"
                        f"{vx_rmse:.6f},{vx_mean_err:.6f},{vx_max_err:.6f},{len(vx_results)},"
                        f"{min(vx for vx, _ in vx_results):.6f},{max(vx for vx, _ in vx_results):.6f}\n"
                    )
                if om_results:
                    fsum.write(
                        "omega,"
                        f"{om_rmse:.6f},{om_mean_err:.6f},{om_max_err:.6f},{len(om_results)},"
                        f"{min(om for om, _ in om_results):.6f},{max(om for om, _ in om_results):.6f}\n"
                    )
            print(f"[INFO] Wrote summary to: {summary_path}")
        except Exception as e:
            print(f"[WARN] Failed to write summary CSV: {e}")
    
    print("=" * 60 + "\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


