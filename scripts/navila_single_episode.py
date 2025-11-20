# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import gymnasium as gym
import os
import sys
import json
import math
import torch
import numpy as np
import imageio
from PIL import Image
import time
import base64
import io
import socket
import threading
from typing import Optional

# ensure repository modules are discoverable when launched via Isaac Sim kit python
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if os.path.isdir(SCRIPTS_DIR) and SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
ISAACLAB_SOURCE = os.path.join(os.path.dirname(REPO_ROOT), "IsaacLab", "source")
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
LOCAL_EXT_PATH_GROUPS = [
    (
        "omni.isaac.vlnce",
        [
            os.path.join(REPO_ROOT, "omni.isaac.vlnce"),
            os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.vlnce"),
            os.path.join(os.path.dirname(REPO_ROOT), "IsaacLab", "source", "omni.isaac.vlnce"),
        ],
    ),
    (
        "omni.isaac.matterport",
        [
            os.path.join(REPO_ROOT, "omni.isaac.matterport"),
            os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.matterport"),
            os.path.join(os.path.dirname(REPO_ROOT), "IsaacLab", "source", "omni.isaac.matterport"),
        ],
    ),
]

for _, candidate_paths in LOCAL_EXT_PATH_GROUPS:
    for _ext_path in candidate_paths:
        if os.path.isdir(_ext_path) and _ext_path not in sys.path:
            sys.path.append(_ext_path)
            break

# Import AppLauncher first (needed before other isaaclab imports)
try:
    from omni.isaac.lab.app import AppLauncher
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# isaaclab argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--visualize_path", action="store_true", default=False, help="Visualize the path in the simulator.")

# navila argparse arguments
parser.add_argument("--vlm_host", type=str, default="localhost")
parser.add_argument("--vlm_port", type=int, default=54321)
parser.add_argument(
    "--vlm_timeout",
    type=float,
    default=10.0,
    help="Socket timeout (seconds) for VLM server connect/send/recv.",
)
parser.add_argument(
    "--vlm_num_frames",
    type=int,
    default=8,
    help="Number of frames to send to the VLM (sampled/padded).",
)
parser.add_argument(
    "--max_action_duration",
    type=float,
    default=None,
    help="Cap each VLM command duration in seconds (None = no cap, matches original).",
)
parser.add_argument(
    "--linear_scale",
    type=float,
    default=1.0,
    help="Scale factor for linear velocity (vx).",
)
parser.add_argument(
    "--angular_scale",
    type=float,
    default=1.0,
    help="Scale factor for angular velocity (yaw rate).",
)
parser.add_argument(
    "--duration_scale",
    type=float,
    default=1.0,
    help="Multiply time_to_go by this factor before converting to steps (>=1.0 slows requery).",
)
parser.add_argument(
    "--stuck_replan_steps",
    type=int,
    default=0,
    help="If >0 and robot barely moves for this many steps, force early requery.",
)
parser.add_argument(
    "--post_action_wait",
    type=float,
    default=0.0,
    help="Seconds to hold zero velocity after finishing an action before next VLM query.",
)
parser.add_argument(
    "--auto_replan",
    action="store_true",
    default=False,
    help="After executing a segment, requery VLM only if progress is insufficient.",
)
parser.add_argument(
    "--replan_delta",
    type=float,
    default=0.25,
    help="Minimum distance-to-goal improvement (meters) required to skip replanning.",
)
parser.add_argument(
    "--requery_interval",
    type=float,
    default=1.5,
    help="Minimum seconds between VLM queries (throttle).",
)
parser.add_argument(
    "--max_vlm_queries",
    type=int,
    default=0,
    help="If >0, cap the total number of VLM queries to avoid infinite loops.",
)
parser.add_argument(
    "--min_new_frames_after_query",
    type=int,
    default=2,
    help="Require at least N newly captured camera frames since the last VLM query.",
)
parser.add_argument(
    "--min_stable_steps_for_query",
    type=int,
    default=3,
    help="Require at least N consecutive stable steps (low vel/ang vel) before querying VLM.",
)
parser.add_argument(
    "--max_body_speed_for_query",
    type=float,
    default=0.10,
    help="Linear speed (m/s) threshold to consider the robot stable enough to query VLM.",
)
parser.add_argument(
    "--max_body_ang_speed_for_query",
    type=float,
    default=0.6,
    help="Angular speed (rad/s) threshold to consider the robot stable enough to query VLM.",
)
parser.add_argument(
    "--stop_confirmations",
    type=int,
    default=1,
    help="Require N consecutive 'stop' decisions from VLM before accepting stop.",
)
parser.add_argument(
    "--no_video",
    action="store_true",
    default=False,
    help="Disable saving output video to disk.",
)
parser.add_argument(
    "--log_vlm",
    action="store_true",
    default=False,
    help="Append VLM interactions to logs/navila_events/events.jsonl.",
)
parser.add_argument(
    "--scene_id_override",
    type=str,
    default=None,
    help="Force selection of episodes from the specified Matterport scene (e.g., '17DRP5sb8fy').",
)
parser.add_argument(
    "--scene_usd",
    type=str,
    default=None,
    help="Direct USD path to use for Matterport terrain (overrides dataset scene file).",
)
parser.add_argument(
    "--instruction_mode",
    type=str,
    choices=["dataset", "text", "stdin", "socket"],
    default="socket",
    help="Source of the navigation instruction: dataset (default), direct text, stdin prompt, or socket.",
)
parser.add_argument(
    "--instruction_text",
    type=str,
    default=None,
    help="Instruction string when --instruction_mode=text is used.",
)
parser.add_argument(
    "--instruction_timeout",
    type=float,
    default=None,
    help="Timeout (seconds) when waiting for socket instructions. None waits indefinitely.",
)
parser.add_argument(
    "--instruction_host",
    type=str,
    default="127.0.0.1",
    help="Host/IP to bind the instruction socket server when --instruction_mode=socket.",
)
parser.add_argument(
    "--instruction_port",
    type=int,
    default=5557,
    help="Port for the instruction socket server when --instruction_mode=socket.",
)

parser.add_argument(
    "--goal_tolerance",
    type=float,
    default=0.5,
    help="Distance (m) to goal considered success (stops automatically).",
)

# r2r argparse arguments
parser.add_argument("--episode_idx", type=int, default=0)

# RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app - MUST be done before importing other isaaclab modules
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import isaaclab modules after AppLauncher has initialized Isaac Sim
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

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper
from omni.isaac.vlnce.utils.eval_utils import (
    get_vel_command,
    read_episodes,
    add_instruction_on_img,
    InstructionData,
)

# Reuse robust helpers for VLM I/O and parsing (interactive utils)
from navila_vla_utils import (
    sample_and_pad_images,
    encode_images_to_base64,
    parse_vlm_response,
    quantise_commands,
    commands_to_velocity_plan,
    commands_to_dicts,
    summarize_commands,
)


def quat2eulers(q0, q1, q2, q3):
    """
    Calculates the roll, pitch, and yaw angles from a quaternion.

    Args:
        q0: The scalar component of the quaternion.
        q1: The x-component of the quaternion.
        q2: The y-component of the quaternion.
        q3: The z-component of the quaternion.

    Returns:
        A tuple containing the roll, pitch, and yaw angles in radians.
    """

    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0**2 + q1**2 - q2**2 - q3**2)

    return roll, pitch, yaw


def define_markers() -> VisualizationMarkers:
    """Define path markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pathMarkers",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def reset_start_pos_rot(env_cfg, args_cli, episode):
    # Prefer direct USD override if provided
    if args_cli.scene_usd and os.path.isfile(args_cli.scene_usd):
        env_cfg.scene.terrain.obj_filepath = os.path.abspath(args_cli.scene_usd)
        scene_id = os.path.splitext(os.path.basename(env_cfg.scene.terrain.obj_filepath))[0]
    else:
        scene_id = os.path.splitext(os.path.basename(episode["scene_id"]))[0]
        env_cfg.scene.terrain.obj_filepath = os.path.join(ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")

    start_pos, start_rot, goal_pos = episode["start_position"], episode["start_rotation"], episode["reference_path"][-1]
    env_cfg.scene.robot.init_state.rot = start_rot

    if "go2" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + 0.4)
    elif "h1" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + 1.0)
    else:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + 0.5)

    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos

    env_cfg.scene.disk_1.init_state.pos = ([start_pos[0], start_pos[1], start_pos[2] + 2.5])
    env_cfg.scene.disk_2.init_state.pos = ([goal_pos[0], goal_pos[1], goal_pos[2] + 2.5])

    return env_cfg




def sample_images_and_send_to_vlm(image_list, vlm_host, vlm_port, query, num_frames: int = 8, timeout: float | None = None):
    if len(image_list) == 0:
        print("Did not receive any images.")
        return None
    # Use the shared sampler to mirror Habitat behavior (history + latest)
    sampled_images = sample_and_pad_images(image_list, num_frames=num_frames)
    encoded_images = encode_images_to_base64(sampled_images)
    history_frames = max(0, len(sampled_images) - 1)

    request_data = {
        "images": encoded_images,
        "query": query,
        "history_frames": history_frames,
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if timeout is not None:
                s.settimeout(timeout)
            s.connect((vlm_host, vlm_port))
            data_bytes = json.dumps(request_data).encode()
            s.sendall(len(data_bytes).to_bytes(8, "big"))
            s.sendall(data_bytes)

            size_data = s.recv(8)
            size = int.from_bytes(size_data, "big")

            response_data = b""
            while len(response_data) < size:
                packet = s.recv(4096)
                if not packet:
                    break
                response_data += packet

            response = json.loads(response_data.decode())
            return response
    except socket.timeout:
        print("[WARN] VLM request timed out.")
        return None
    except Exception as e:
        print(f"[ERROR] VLM request failed: {e}")
        return None


def _log_vlm_event(
    enabled: bool,
    instruction_text: str | None,
    vlm_text: str | None,
    vlm_vel_commands,
    planned_steps: int,
    step_idx: int,
) -> None:
    if not enabled:
        return
    try:
        log_dir = os.path.join(PROJECT_ROOT, "..", "logs", "navila_events")
        os.makedirs(log_dir, exist_ok=True)
        event = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "step": step_idx,
            "instruction": instruction_text or "",
            "vlm_text": vlm_text or "",
            "vlm_vel_commands": vlm_vel_commands,
            "env_steps_to_go": planned_steps,
        }
        with open(os.path.join(log_dir, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write VLM log: {e}")


def select_episode(all_episodes, episode_idx: int, scene_id_override: Optional[str]):
    if scene_id_override:
        target_scene = scene_id_override.strip()
        matching_indices = [
            idx
            for idx, ep in enumerate(all_episodes)
            if os.path.splitext(os.path.basename(ep["scene_id"]))[0] == target_scene
        ]
        if not matching_indices:
            raise ValueError(
                f"No episodes found in dataset for scene_id '{target_scene}'. "
                "Please verify the ID or update the dataset."
            )
        if episode_idx >= len(matching_indices):
            raise ValueError(
                f"--episode_idx={episode_idx} exceeds available episodes ({len(matching_indices)}) "
                f"for scene '{target_scene}'."
            )
        selected_idx = matching_indices[episode_idx]
        print(
            f"[INFO] Using episode {episode_idx} (dataset index {selected_idx}) "
            f"within scene '{target_scene}' ({len(matching_indices)} available)."
        )
        return all_episodes[selected_idx]

    if episode_idx >= len(all_episodes):
        raise ValueError(
            f"--episode_idx={episode_idx} exceeds total episodes ({len(all_episodes)})."
        )
    return all_episodes[episode_idx]


class InstructionSocketServer:
    """Background TCP server that keeps updating the latest instruction text."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._stop_event = threading.Event()
        self._server_socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._current_instruction: Optional[str] = None
        self._thread = threading.Thread(target=self._socket_loop, name="InstructionSocket", daemon=True)
        self._thread.start()
        print(f"[INFO] Instruction socket server listening on {host}:{port}")
        print(f"[INFO] Send instruction via: echo 'your instruction' | nc {host} {port}")

    def _socket_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            self._server_socket = server
            while not self._stop_event.is_set():
                try:
                    server.settimeout(1.0)
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    buffer = b""
                    while not self._stop_event.is_set():
                        try:
                            data = conn.recv(1024)
                        except ConnectionResetError:
                            break
                        if not data:
                            break
                        buffer += data
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            text = line.decode("utf-8", errors="ignore").strip()
                            if not text:
                                continue
                            with self._lock:
                                self._current_instruction = text
                            print(f"[INFO] Instruction received: {text}")
                    # continue accepting new connections

    def get_instruction(self) -> Optional[str]:
        with self._lock:
            return self._current_instruction

    def shutdown(self):
        self._stop_event.set()
        if self._server_socket:
            try:
                with socket.create_connection((self.host, self.port), timeout=0.2):
                    pass
            except Exception:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def resolve_instruction(args_cli, episode) -> Optional[InstructionData]:
    dataset_instruction = InstructionData(**episode["instruction"])
    mode = args_cli.instruction_mode

    if mode == "dataset":
        print("[INFO] Using dataset-provided instruction.")
        return dataset_instruction

    if mode == "text":
        if not args_cli.instruction_text:
            raise ValueError("--instruction_mode=text requires --instruction_text.")
        print("[INFO] Using instruction provided via --instruction_text.")
        return InstructionData(instruction_text=args_cli.instruction_text)

    if mode == "stdin":
        try:
            user_input = input("Enter navigation instruction: ").strip()
        except EOFError:
            user_input = ""
        if not user_input:
            print("[WARN] Empty stdin instruction. Falling back to dataset instruction.")
            return dataset_instruction
        print("[INFO] Using instruction entered via stdin.")
        return InstructionData(instruction_text=user_input)

    if mode == "socket":
        print("[INFO] Waiting for instruction from socket after scene load.")
        return None

    print(f"[WARN] Unknown instruction_mode '{mode}'. Falling back to dataset instruction.")
    return dataset_instruction


def main():
    """IsaacSim Evaluation using NaViLA and trained low-level policy."""

    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    all_episodes = read_episodes(r2r_data_path)
    # If scene_usd is provided but scene_id_override is not, derive scene id from USD to avoid mismatch
    effective_scene_id = args_cli.scene_id_override
    if (not effective_scene_id) and args_cli.scene_usd and os.path.isfile(args_cli.scene_usd):
        effective_scene_id = os.path.splitext(os.path.basename(args_cli.scene_usd))[0]
        print(f"[INFO] Using scene_id_override derived from --scene_usd: {effective_scene_id}")
    episode = select_episode(all_episodes, args_cli.episode_idx, effective_scene_id)

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg = reset_start_pos_rot(env_cfg, args_cli, episode)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli, play=True
    )

    log_root_path = os.path.join(os.path.dirname(__file__), "../logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    assert os.path.exists(log_agent_cfg_file_path), f"Agent config file not found: {log_agent_cfg_file_path}"
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)

    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ベンチマーク評価は不要（任意指示テスト用のため）
    # 最小限のmeasure（PathLengthのみ）で動作確認
    env = VLNEnvWrapper(
        env,
        policy,
        args_cli.task,
        episode,
        high_level_obs_key="camera_obs",
        measure_names=[],  # ベンチマーク評価不要
    )

    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
    cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
    cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    obs, infos = env.reset()

    steps_per_image = 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    steps_per_viz_image = 0.1 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)

    rgb_obs = infos["observations"]["camera_obs"]
    init_frame = rgb_obs[0, :, :, :3].cpu().numpy()
    image_observations = []
    image_observations.append(Image.fromarray(init_frame))

    instruction_server = InstructionSocketServer(args_cli.instruction_host, args_cli.instruction_port)

    initial_instruction = resolve_instruction(args_cli, episode)
    instruction_text = initial_instruction.instruction_text if initial_instruction else None
    
    init_overlay = instruction_text or "Waiting for instruction..."
    add_instruction_on_img(init_frame, init_overlay)
    vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    add_instruction_on_img(vis_frame, "")
    record_video = not args_cli.no_video
    rgb_obses = [np.concatenate([init_frame, vis_frame], axis=1)] if record_video else []

    step_duration = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    num_steps = 0
    same_pos_count = 0
    prev_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    max_episode_steps = 100 * 0.5 / step_duration

    # Goal position and progress tracking (2D distance)
    goal_pos = episode["reference_path"][-1]
    def _distance_to_goal_xy(robot_pos):
        try:
            return math.hypot(float(goal_pos[0]) - float(robot_pos[0]), float(goal_pos[1]) - float(robot_pos[1]))
        except Exception:
            return 0.0
    current_distance_to_goal = _distance_to_goal_xy(prev_pos)
    last_query_distance = current_distance_to_goal
    last_vlm_query_time = 0.0
    num_vlm_queries = 0

    vlm_vel_commands = [0.0, 0.0, 0.0]
    curr_frame_copy = init_frame.copy()
    stream_output = None
    # Track execution state for low-level actions derived from VLM outputs.
    action_steps_remaining = 0
    action_active = False
    pending_vlm_query = instruction_text is not None
    post_action_hold_steps = 0
    planned_action_steps = 0
    consecutive_stop_votes = 0
    stop_latched = False
    steps_since_last_query = 1_000_000
    new_frames_since_last_query = 0
    stable_steps_count = 0

    while simulation_app.is_running():
        simulation_app.update()
        hold_started_this_loop = False

        new_instruction_text = instruction_server.get_instruction()
        if new_instruction_text:
            new_instruction_text = new_instruction_text.strip()
            if new_instruction_text and new_instruction_text != instruction_text:
                instruction_text = new_instruction_text
                if len(image_observations) > 8:
                    image_observations = image_observations[-8:]
                # interrupt current action gracefully; wait for next query once stable
                if action_active:
                    action_active = False
                    action_steps_remaining = 0
                    vlm_vel_commands = [0.0, 0.0, 0.0]
                    post_action_hold_steps = max(post_action_hold_steps, 1)
                    hold_started_this_loop = True
                env.set_stop_called(False)
                stop_latched = False
                pending_vlm_query = instruction_text is not None and not action_active and post_action_hold_steps <= 0
                stream_output = None
                print(f"[INFO] Instruction updated via socket: {instruction_text}")

        with torch.inference_mode():
            cooldown_ok = (time.time() - last_vlm_query_time) >= float(args_cli.requery_interval)
            quota_ok = (int(args_cli.max_vlm_queries) <= 0) or (num_vlm_queries < int(args_cli.max_vlm_queries))
            min_requery_steps = max(1, int(float(args_cli.requery_interval) / max(step_duration, 1e-6)))
            steps_ok = steps_since_last_query >= min_requery_steps
            frames_ok = new_frames_since_last_query >= int(args_cli.min_new_frames_after_query)
            stable_ok = stable_steps_count >= int(args_cli.min_stable_steps_for_query)
            ready_for_vlm = (
                instruction_text
                and pending_vlm_query
                and not action_active
                and post_action_hold_steps <= 0
                and cooldown_ok
                and steps_ok
                and frames_ok
                and stable_ok
                and quota_ok
            )
            if ready_for_vlm:
                stream_output = sample_images_and_send_to_vlm(
                    image_observations,
                    args_cli.vlm_host,
                    args_cli.vlm_port,
                    instruction_text,
                    args_cli.vlm_num_frames,
                    args_cli.vlm_timeout,
                )
                if isinstance(stream_output, dict):
                    vlm_text = stream_output.get("response", "")
                else:
                    vlm_text = str(stream_output) if stream_output is not None else ""
                stream_output = vlm_text

                try:
                    last_vlm_query_time = time.time()
                    num_vlm_queries += 1
                    last_query_distance = current_distance_to_goal
                    steps_since_last_query = 0
                    new_frames_since_last_query = 0
                    stable_steps_count = 0
                    # Decode VLM text into mid-level commands and build a velocity plan
                    commands = parse_vlm_response(vlm_text)
                    commands = quantise_commands(commands)
                    velocity_plan = commands_to_velocity_plan(commands)

                    # Strict closed-loop: execute only the next mid-level action (first segment)
                    planned_action_steps = 0
                    if commands and getattr(commands[0], "kind", "") == "stop":
                        consecutive_stop_votes += 1
                        accept_stop = (
                            (consecutive_stop_votes >= int(args_cli.stop_confirmations))
                            or (float(args_cli.goal_tolerance) > 0.0 and current_distance_to_goal <= float(args_cli.goal_tolerance))
                        )
                        if accept_stop:
                            vlm_vel_commands = [0.0, 0.0, 0.0]
                            action_steps_remaining = 0
                            action_active = False
                            env.set_stop_called(False)
                            stop_latched = True
                            if args_cli.post_action_wait and args_cli.post_action_wait > 0.0:
                                post_action_hold_steps = int(max(0.0, float(args_cli.post_action_wait)) / step_duration)
                                hold_started_this_loop = True
                            print("[INFO] Stop accepted. Holding.")
                            pending_vlm_query = False
                            consecutive_stop_votes = 0
                        else:
                            # Not enough evidence to stop: keep zero and requery after cooldown
                            vlm_vel_commands = [0.0, 0.0, 0.0]
                            action_steps_remaining = 0
                            action_active = False
                            env.set_stop_called(False)
                            pending_vlm_query = True
                            print(f"[INFO] Stop deferred (votes={consecutive_stop_votes}).")
                    else:
                        if velocity_plan:
                            velocity_vec, segment_duration = velocity_plan[0]
                            time_to_go = float(segment_duration) * float(args_cli.duration_scale)
                            if args_cli.max_action_duration is not None:
                                time_to_go = min(time_to_go, float(args_cli.max_action_duration))
                            planned_action_steps = int(math.ceil(time_to_go / step_duration)) if step_duration > 0 else 0
                            scaled_vec = [
                                float(velocity_vec[0]) * float(args_cli.linear_scale),
                                0.0,
                                float(velocity_vec[2]) * float(args_cli.angular_scale),
                            ]
                            if planned_action_steps > 0 and any(abs(v) > 0 for v in scaled_vec):
                                vlm_vel_commands = scaled_vec
                                action_steps_remaining = planned_action_steps
                                action_active = True
                                env.set_stop_called(False)
                                pending_vlm_query = False
                                consecutive_stop_votes = 0
                            else:
                                vlm_vel_commands = [0.0, 0.0, 0.0]
                                action_steps_remaining = 0
                                action_active = False
                                env.set_stop_called(False)
                                pending_vlm_query = False
                                stop_latched = True
                        else:
                            vlm_vel_commands = [0.0, 0.0, 0.0]
                            action_steps_remaining = 0
                            action_active = False
                            env.set_stop_called(False)
                            pending_vlm_query = False
                            stop_latched = True

                    print(
                        f"VLM output: {stream_output}\n"
                        f"Vel Command: {vlm_vel_commands}, Planned Env Steps: {planned_action_steps}\n"
                        f"Plan: {summarize_commands(commands)}\n"
                    )
                    _log_vlm_event(
                        args_cli.log_vlm,
                        instruction_text,
                        vlm_text,
                        vlm_vel_commands,
                        planned_action_steps,
                        num_steps,
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to parse/execute VLM response: {e}")
                    vlm_vel_commands = [0.0, 0.0, 0.0]
                    action_steps_remaining = 0
                    action_active = False
                    pending_vlm_query = True
                    stream_output = None

        obs, _, done, infos = env.step(torch.tensor(vlm_vel_commands, device=obs.device))

        if done or num_steps > max_episode_steps:
            break

        cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_vel = np.linalg.norm(env.unwrapped.scene["robot"].data.root_vel_w[0].detach().cpu().numpy())
        try:
            robot_ang_vel = np.linalg.norm(env.unwrapped.scene["robot"].data.root_ang_vel_w[0].detach().cpu().numpy())
        except Exception:
            robot_ang_vel = 0.0
        if np.linalg.norm(cur_pos - prev_pos) < 0.01 and robot_vel < 0.01:
            same_pos_count += 1
        else:
            same_pos_count = 0
        prev_pos = cur_pos
        current_distance_to_goal = _distance_to_goal_xy(cur_pos)

        # Update stability counters
        if robot_vel <= float(args_cli.max_body_speed_for_query) and robot_ang_vel <= float(args_cli.max_body_ang_speed_for_query):
            stable_steps_count += 1
        else:
            stable_steps_count = 0

        # Stop automatically when close enough to goal: just hold and wait for new instruction
        if float(args_cli.goal_tolerance) > 0.0 and current_distance_to_goal <= float(args_cli.goal_tolerance):
            vlm_vel_commands = [0.0, 0.0, 0.0]
            action_steps_remaining = 0
            action_active = False
            stop_latched = True
            pending_vlm_query = False
            print(f"[INFO] Goal tolerance reached ({current_distance_to_goal:.2f} m). Holding.")

        # Early requery when stuck (optional)
        if args_cli.stuck_replan_steps > 0 and same_pos_count >= args_cli.stuck_replan_steps:
            print(f"Robot seems stuck for {same_pos_count} steps. Forcing early requery.")
            same_pos_count = 0
            action_steps_remaining = 0
            action_active = False
            vlm_vel_commands = [0.0, 0.0, 0.0]
            post_action_hold_steps = max(post_action_hold_steps, 1)
            hold_started_this_loop = True
            env.set_stop_called(False)
            pending_vlm_query = instruction_text is not None and post_action_hold_steps <= 0

        if action_active and action_steps_remaining > 0:
            action_steps_remaining -= 1
            if action_steps_remaining <= 0:
                action_steps_remaining = 0
                action_active = False
                vlm_vel_commands = [0.0, 0.0, 0.0]
                if args_cli.post_action_wait and args_cli.post_action_wait > 0.0:
                    post_action_hold_steps = int(max(0.0, float(args_cli.post_action_wait)) / step_duration)
                    hold_started_this_loop = True
        elif not action_active:
            vlm_vel_commands = [0.0, 0.0, 0.0]

        if post_action_hold_steps > 0 and not hold_started_this_loop:
            post_action_hold_steps -= 1
            if post_action_hold_steps < 0:
                post_action_hold_steps = 0

        # Re-arm query when robot is idle (but not after an accepted stop)
        if not action_active and post_action_hold_steps <= 0 and instruction_text and not stop_latched:
            pending_vlm_query = True


        if num_steps % steps_per_image == 0:
            curr_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            image_observations.append(Image.fromarray(curr_frame))
            if len(image_observations) > 32:
                image_observations = image_observations[-32:]
            curr_frame_copy = curr_frame.copy()
            overlay_text = instruction_text or "Waiting for instruction..."
            add_instruction_on_img(curr_frame_copy, overlay_text)
            new_frames_since_last_query += 1
            
        if num_steps % steps_per_viz_image == 0 and record_video:
            curr_vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
            if stream_output is not None:
                add_instruction_on_img(curr_vis_frame, stream_output)
            rgb_obses.append(np.concatenate([curr_frame_copy, curr_vis_frame], axis=1))

        num_steps += 1
        steps_since_last_query += 1

    # ベンチマーク評価結果の保存は不要（任意指示テスト用のため）
    if record_video:
        result_dir = f"eval_results/{args_cli.task}_loco_{args_cli.load_run}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        video_dir = os.path.join(result_dir, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # episode_idが無い場合でも動作するように
        episode_id_str = str(int(episode.get('episode_id', 0)) - 1) if 'episode_id' in episode else "custom"
        video_filename = f"output_{episode_id_str}_{timestamp}.mp4"

        HD_WIDTH = 1280
        HD_HEIGHT = 720

        writer = imageio.get_writer(
            os.path.join(video_dir, video_filename),
            fps=10,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        for frame in rgb_obses:
            frame = frame.astype(np.uint8)
            if frame.shape[1] != HD_WIDTH or frame.shape[0] != HD_HEIGHT:
                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(frame)
                pil_img = pil_img.resize((HD_WIDTH, HD_HEIGHT), PILImage.Resampling.LANCZOS)
                frame = np.array(pil_img)
            writer.append_data(frame)

        writer.close()

    instruction_server.shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
