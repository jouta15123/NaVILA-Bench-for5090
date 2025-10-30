# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive evaluation script that queries a VLM server with custom instructions."""

import argparse
import json
import math
import os
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Sequence

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image

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
ISAACLAB_SITE_PACKAGES = os.path.join(
    os.path.dirname(ISAACLAB_SOURCE), "env_isaaclab", "lib", "python3.11", "site-packages"
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

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# isaaclab argparse arguments
parser = argparse.ArgumentParser(description="Interactive NaViLA control with custom language commands.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="h1_matterport_vision", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")

parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--visualize_path", action="store_true", default=False, help="Visualize the path in the simulator.")

# navila/interactive specific arguments
parser.add_argument("--vlm_host", type=str, default="localhost", help="Host running the VLM server.")
parser.add_argument("--vlm_port", type=int, default=54321, help="Port for the VLM server.")
parser.add_argument(
    "--vlm_num_frames",
    type=int,
    default=8,
    help="Number of video frames to send to the VLM per query.",
)
parser.add_argument(
    "--auto_replan",
    action="store_true",
    help="Enable automatic replanning when progress is insufficient after a plan is executed.",
)
parser.add_argument(
    "--replan_delta",
    type=float,
    default=0.25,
    help="Plan 実行後に DistanceToGoal がこの値（メートル）以上改善していないと自動再計画します。",
)
parser.add_argument(
    "--save_map_frames",
    action="store_true",
    help="各プラン時に可視化フレームを logs/navila_events/maps に保存します。",
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
    help="Absolute or relative path to a USD file to load instead of the dataset scene.",
)
parser.add_argument(
    "--instruction",
    type=str,
    default=None,
    help="Instruction to send to the VLM. If omitted, the script prompts for input.",
)
parser.add_argument(
    "--instruction_file",
    type=str,
    default=None,
    help="Optional path to a text file containing the instruction string.",
)
parser.add_argument(
    "--instruction_mode",
    type=str,
    choices=["stdin", "socket"],
    default=None,
    help="Instruction input mode. Default: auto (stdin if TTY, otherwise socket server).",
)
parser.add_argument(
    "--instruction_host",
    type=str,
    default="127.0.0.1",
    help="Host/IP to bind the instruction socket server (when instruction_mode=socket).",
)
parser.add_argument(
    "--instruction_port",
    type=int,
    default=5557,
    help="Port for the instruction socket server (when instruction_mode=socket).",
)
parser.add_argument(
    "--r2r_data_path",
    type=str,
    default=os.path.join(REPO_ROOT, "isaaclab_exts", "omni.isaac.vlnce", "assets", "vln_ce_isaac_v1.json.gz"),
    help="Path to the R2R episode JSON (gzipped).",
)
parser.add_argument(
    "--episode_idx",
    type=int,
    default=0,
    help="Episode index to load. With --scene_id_override, this is the index within matching episodes.",
)
parser.add_argument(
    "--max_runtime",
    type=float,
    default=1200.0,
    help="Maximum simulation time in seconds before terminating.",
)

# RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "load_run", None) is None:
    parser.error("--load_run を指定してください（訓練済みポリシーの実行名）。")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.io import load_yaml
from isaaclab.utils import update_class_from_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.vlnce.utils import (
    ASSETS_DIR,
    RslRlVecEnvHistoryWrapper,
    VLNEnvWrapper,
    add_instruction_on_img,
    read_episodes,
)
from omni.isaac.vlnce.config import *  # noqa: F403,F401
from omni.isaac.vlnce.utils.measures import MeasureManager, PathLength, DistanceToGoal, Success, SPL

from navila_vla_utils import (
    ActionCommand,
    commands_cover_expected,
    commands_to_dicts,
    commands_to_velocity_plan,
    encode_images_to_base64,
    extract_distance_to_goal,
    parse_instruction_to_commands,
    parse_vlm_response,
    quantise_commands,
    sample_and_pad_images,
    save_map_image,
    summarize_commands,
)


def quat2eulers(q0: float, q1: float, q2: float, q3: float):
    """Converts a quaternion into roll, pitch, yaw."""
    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0**2 + q1**2 - q2**2 - q3**2)
    return roll, pitch, yaw


def normalize_scene_id(scene_path: str) -> str:
    """Return the scene identifier without directory or extension."""
    return os.path.splitext(os.path.basename(scene_path))[0]


def reset_start_pos_rot(env_cfg, instruction_episode):
    """Adjust environment configuration using the selected episode."""
    scene_id = normalize_scene_id(instruction_episode["scene_id"])

    if args_cli.scene_usd:
        usd_path = os.path.abspath(args_cli.scene_usd)
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}")
        env_cfg.scene.terrain.obj_filepath = usd_path
    else:
        default_usd = os.path.join(ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")
        if not os.path.exists(default_usd):
            raise FileNotFoundError(f"USD file not found: {default_usd}")
        env_cfg.scene.terrain.obj_filepath = default_usd

    start_pos = instruction_episode["start_position"]
    start_rot = instruction_episode["start_rotation"]
    goal_pos = instruction_episode["reference_path"][-1]
    env_cfg.scene.robot.init_state.rot = start_rot

    if "go2" in args_cli.task:
        z_offset = 0.4
    elif "h1" in args_cli.task:
        z_offset = 1.0
    elif "g1" in args_cli.task:
        z_offset = 0.8
    else:
        z_offset = 0.5
    env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2] + z_offset)
    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos

    env_cfg.scene.disk_1.init_state.pos = [start_pos[0], start_pos[1], start_pos[2] + 2.5]
    env_cfg.scene.disk_2.init_state.pos = [goal_pos[0], goal_pos[1], goal_pos[2] + 2.5]
    return env_cfg


def sample_images_and_send_to_vlm(
    image_list: List[Image.Image],
    vlm_host: str,
    vlm_port: int,
    query: str,
    num_frames: int,
) -> Optional[str]:
    """Send sampled images alongside the query to the VLM server."""
    if len(image_list) == 0:
        print("Did not receive any images.")
        return None

    sampled_images = sample_and_pad_images(image_list, num_frames=num_frames)
    encoded_images = encode_images_to_base64(sampled_images)

    history_frames = max(0, len(sampled_images) - 1)
    request_data = {
        "images": encoded_images,
        "query": query,
        "history_frames": history_frames,
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((vlm_host, vlm_port))
        data_bytes = json.dumps(request_data).encode()
        sock.sendall(len(data_bytes).to_bytes(8, "big"))
        sock.sendall(data_bytes)

        size_data = sock.recv(8)
        if not size_data:
            return None
        size = int.from_bytes(size_data, "big")

        response_data = b""
        while len(response_data) < size:
            packet = sock.recv(4096)
            if not packet:
                break
            response_data += packet
    if not response_data:
        return None
    response = json.loads(response_data.decode())
    return response


def log_vlm_decision(log_dir: str, record: dict) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "interactive.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")




def add_measurement(env, episode):
    measure_manager = MeasureManager()
    measure_names = ["PathLength", "DistanceToGoal", "Success", "SPL"]
    for measure_name in measure_names:
        measure = eval(measure_name)(env, episode, measure_manager)
        measure_manager.register_measure(measure)
    env.measure_manager = measure_manager


class InstructionProvider:
    """Utility that supplies instructions from stdin or a socket server."""

    def __init__(self):
        self.queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.stop_event = threading.Event()

        if args_cli.instruction_file:
            with open(args_cli.instruction_file, "r", encoding="utf-8") as file:
                text = file.read().strip()
                if text:
                    self.queue.put(text)
        elif args_cli.instruction:
            self.queue.put(args_cli.instruction.strip())

        resolved_mode = args_cli.instruction_mode
        if resolved_mode is None:
            resolved_mode = "stdin" if sys.stdin.isatty() else "socket"
        self.mode = resolved_mode

        if self.mode == "stdin":
            self.thread = threading.Thread(target=self._stdin_loop, name="InstructionStdin", daemon=True)
            self.thread.start()
            print("[INFO] Instruction mode: stdin. Enter commands in this terminal. Type 'exit' to quit.")
        else:
            self._server_socket: Optional[socket.socket] = None
            self.thread = threading.Thread(target=self._socket_loop, name="InstructionSocket", daemon=True)
            self.thread.start()
            print(
                f"[INFO] Instruction mode: socket. Connect via 'nc {args_cli.instruction_host} {args_cli.instruction_port}' "
                "and send commands. Type 'exit' to terminate."
            )

    def _stdin_loop(self):
        while not self.stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)
                continue
            text = line.strip()
            if not text:
                print("空行は無視しました。終了するには 'exit' と入力してください。")
                continue
            if text.lower() in {"exit", "quit"}:
                self.queue.put(None)
                break
            self.queue.put(text)

    def _socket_loop(self):
        host = args_cli.instruction_host
        port = args_cli.instruction_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen()
            self._server_socket = server
            while not self.stop_event.is_set():
                try:
                    server.settimeout(1.0)
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    buffer = b""
                    while not self.stop_event.is_set():
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
                            if text.lower() in {"exit", "quit"}:
                                self.queue.put(None)
                                self.stop_event.set()
                                conn.close()
                                break
                            self.queue.put(text)
                        if self.stop_event.is_set():
                            break

    def get_instruction(self) -> Optional[str]:
        try:
            instruction = self.queue.get(timeout=0.1)
            return instruction
        except queue.Empty:
            return ""

    def wait_for_instruction(self) -> Optional[str]:
        while not self.stop_event.is_set():
            try:
                instruction = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            return instruction
        return None

    def shutdown(self):
        self.stop_event.set()
        if self.mode == "socket" and self._server_socket:
            try:
                # Wake up accept()
                with socket.create_connection((args_cli.instruction_host, args_cli.instruction_port), timeout=0.5):
                    pass
            except Exception:
                pass


def build_instruction_provider() -> InstructionProvider:
    """Create an instruction provider instance."""
    if args_cli.instruction and args_cli.instruction_file:
        raise ValueError("--instruction と --instruction_file は同時に指定できません。")
    return InstructionProvider()


def _build_plan_from_commands(
    commands: List[ActionCommand],
    env,
) -> List[Tuple[torch.Tensor, int]]:
    velocity_plan = commands_to_velocity_plan(commands)
    step_duration = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    plan: List[Tuple[torch.Tensor, int]] = []
    for velocity, duration in velocity_plan:
        steps = 1
        if duration > 0:
            steps = max(1, int(math.ceil(duration / step_duration)))
        command_tensor = torch.tensor(velocity, device=env.unwrapped.device, dtype=torch.float32)
        plan.append((command_tensor, steps))
    return plan


def _decode_vlm_to_plan(
    vlm_text: str,
    env,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[ActionCommand]]:
    commands: List[ActionCommand] = parse_vlm_response(vlm_text)
    if not commands:
        return [], []
    commands = quantise_commands(commands)
    plan = _build_plan_from_commands(commands, env)
    return plan, commands


def query_vlm_and_prepare_command(
    instruction_text: str,
    image_observations: List[Image.Image],
    env,
) -> Optional[tuple[str, List[Tuple[torch.Tensor, int]], List[ActionCommand], bool]]:
    """Send instruction to VLM and convert its response into velocity commands."""
    response = sample_images_and_send_to_vlm(
        image_observations,
        args_cli.vlm_host,
        args_cli.vlm_port,
        instruction_text,
        args_cli.vlm_num_frames,
    )
    if response is None:
        print("VLM から応答が得られませんでした。停止します。")
        return None

    if isinstance(response, dict):
        vlm_text = response.get("response", "")
    else:
        vlm_text = str(response)
    if not vlm_text:
        print("VLM 応答が空です。停止します。")
        return None

    plan, commands = _decode_vlm_to_plan(vlm_text, env)
    fallback_used = False
    fallback_commands = parse_instruction_to_commands(instruction_text)
    if fallback_commands:
        fallback_commands = quantise_commands(fallback_commands)
        if not commands_cover_expected(fallback_commands, commands):
            print(
                "VLM 応答が指示と一致しないためフォールバックコマンドを使用します。"
            )
            commands = fallback_commands
            plan = _build_plan_from_commands(commands, env)
            fallback_used = True

    if not plan:
        print("VLM 応答から有効なコマンドを抽出できませんでした。")
        return None

    return vlm_text, plan, commands, fallback_used


def main():
    """IsaacSim interactive control using NaViLA and user-provided instructions."""
    r2r_path = args_cli.r2r_data_path
    if not os.path.isabs(r2r_path):
        r2r_path = os.path.join(REPO_ROOT, r2r_path)
    r2r_path = os.path.abspath(r2r_path)

    all_episodes = read_episodes(r2r_path)

    selected_dataset_idx = args_cli.episode_idx
    filtered_count = None

    if args_cli.scene_id_override:
        target_scene = args_cli.scene_id_override.strip()
        matching_indices = [
            idx for idx, ep in enumerate(all_episodes) if normalize_scene_id(ep["scene_id"]) == target_scene
        ]
        if not matching_indices:
            raise ValueError(
                f"No episodes found in dataset for scene_id '{target_scene}'. "
                "Please verify the ID or update the dataset."
            )
        filtered_count = len(matching_indices)
        if args_cli.episode_idx >= filtered_count:
            raise ValueError(
                f"--episode_idx={args_cli.episode_idx} exceeds available episodes ({filtered_count}) "
                f"for scene '{target_scene}'."
            )
        selected_dataset_idx = matching_indices[args_cli.episode_idx]
        print(
            f"[INFO] Using episode {args_cli.episode_idx} (dataset index {selected_dataset_idx}) "
            f"within scene '{target_scene}' ({filtered_count} available)."
        )

    if not (0 <= selected_dataset_idx < len(all_episodes)):
        raise ValueError("Episode index out of range.")

    episode = all_episodes[selected_dataset_idx]

    instruction_provider = build_instruction_provider()

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg = reset_start_pos_rot(env_cfg, episode)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli, play=True)
    log_root_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    assert os.path.exists(log_agent_cfg_file_path), f"Agent config file not found: {log_agent_cfg_file_path}"
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)

    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)  # type: ignore[name-defined]
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    add_measurement(env, episode)
    env = VLNEnvWrapper(
        env,
        policy,
        args_cli.task,
        episode,
        high_level_obs_key="camera_obs",
        measure_names=["PathLength", "DistanceToGoal", "Success", "SPL"],
    )

    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    _, _, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
    cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
    cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    obs, infos = env.reset()

    steps_per_image = 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    steps_per_viz_image = 0.1 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)

    rgb_obs = infos["observations"]["camera_obs"]
    init_frame = rgb_obs[0, :, :, :3].cpu().numpy()
    vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()

    MAX_IMAGE_HISTORY = 64
    image_observations: List[Image.Image] = [Image.fromarray(init_frame)]

    overlay_text = "Awaiting instruction..."
    init_cam_disp = init_frame.copy()
    init_viz_disp = vis_frame.copy()
    add_instruction_on_img(init_cam_disp, overlay_text)
    rgb_obses = [np.concatenate([init_cam_disp, init_viz_disp], axis=1)]

    instruction_video_root = os.path.join(PROJECT_ROOT, "..", "logs", "navila_events", "interactive_clips")
    os.makedirs(instruction_video_root, exist_ok=True)
    segment_frames: List[np.ndarray] = []
    segment_tag: Optional[str] = None
    recording_active = False
    instruction_counter = 0

    def flush_segment_video() -> None:
        nonlocal segment_frames, segment_tag, recording_active
        if segment_frames and segment_tag:
            output_path = os.path.join(instruction_video_root, f"{segment_tag}.mp4")
            try:
                with imageio.get_writer(output_path, fps=10) as writer:
                    for frame in segment_frames:
                        writer.append_data(frame)
                print(f"[INFO] Saved instruction clip: {output_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save clip {segment_tag}: {exc}")
        segment_frames = []
        segment_tag = None
        recording_active = False

    latest_cam_disp = init_cam_disp.copy()
    latest_viz_disp = init_viz_disp.copy()

    num_steps = 0
    start_time = time.time()

    vlm_response_cache = ""
    last_instruction_text = ""
    current_command = torch.zeros(env.action_space.shape[-1], device=env.unwrapped.device, dtype=torch.float32)
    command_steps_remaining = 0
    command_queue: List[Tuple[torch.Tensor, int]] = []
    stop_plan_active = False
    prompt_shown = False

    LOG_DIR = os.path.join(PROJECT_ROOT, "..", "logs", "navila_events")
    measurements = infos.get("measurements", {}) if isinstance(infos, dict) else {}
    last_distance = extract_distance_to_goal(measurements)
    PROGRESS_DELTA = args_cli.replan_delta
    plan_counter = 0
    current_plan_start_distance = last_distance
    current_plan_instruction: Optional[str] = None

    infos = {}

    while simulation_app.is_running():
        simulation_app.update()
        if command_steps_remaining <= 0:
            if command_queue:
                current_command, command_steps_remaining = command_queue.pop(0)
            else:
                instruction_text = instruction_provider.get_instruction()
                if instruction_text is None:
                    print("指示入力が終了したためシミュレーションを停止します。")
                    break
                if instruction_text:
                    if recording_active:
                        flush_segment_video()
                    instruction_counter += 1
                    prompt_shown = False  # reset prompt flag on new instruction
                    last_instruction_text = instruction_text
                    result = query_vlm_and_prepare_command(instruction_text, image_observations, env)
                    if result is None:
                        break
                    vlm_response_cache, plan, commands, fallback_used = result
                    command_queue.extend(plan)
                    stop_plan_active = all(cmd.kind == "stop" for cmd in commands)
                    current_command, command_steps_remaining = command_queue.pop(0)
                    plan_counter += 1
                    segment_tag = f"instruction_{instruction_counter:03d}"
                    segment_frames = []
                    recording_active = True
                    current_plan_start_distance = last_distance
                    current_plan_instruction = last_instruction_text
                    overlay_text = f"Instruction: {last_instruction_text}\nVLM: {vlm_response_cache}"
                    print(
                        f"VLM output: {vlm_response_cache}\n"
                        f"Queued segments: {len(plan)} | Executing first for {command_steps_remaining} steps\n"
                    )
                    print(f"Plan #{plan_counter}: {summarize_commands(commands)}")
                    log_vlm_decision(
                        LOG_DIR,
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "event": "user_instruction",
                            "instruction": instruction_text,
                            "vlm_output": vlm_response_cache,
                            "commands": commands_to_dicts(commands),
                            "fallback_used": fallback_used,
                            "command_segments": len(plan),
                            "distance_to_goal": last_distance,
                        },
                    )
                    if args_cli.save_map_frames:
                        save_map_image(
                            LOG_DIR,
                            latest_cam_disp,
                            latest_viz_disp,
                            f"plan_{plan_counter}",
                            subdir="maps_interactive",
                        )
                else:
                    # keep stepping with zero command while waiting
                    if recording_active:
                        flush_segment_video()
                    last_instruction_text = ""
                    current_plan_instruction = None
                    current_command.zero_()
                    overlay_text = "Awaiting instruction..."
                    if not prompt_shown and getattr(instruction_provider, "mode", None) == "stdin":
                        print("instruction:")
                        prompt_shown = True
                    command_steps_remaining = 1

        with torch.inference_mode():
            obs, _, done, infos = env.step(current_command)

        measurements = infos.get("measurements", measurements)

        if done or env.is_stop_called:
            print("環境から停止指示が出たため終了します。")
            break

        if time.time() - start_time > args_cli.max_runtime:
            print("最大実行時間に達したため終了します。")
            break

        current_distance = extract_distance_to_goal(measurements)
        if current_distance is not None:
            last_distance = current_distance

        if (
            args_cli.auto_replan
            and last_instruction_text
            and not command_queue
            and command_steps_remaining == 0
            and current_plan_instruction == last_instruction_text
        ):
            if (
                current_plan_start_distance is not None
                and current_distance is not None
                and current_plan_start_distance - current_distance < PROGRESS_DELTA
            ):
                print("進捗が不十分のため自動再計画します。")
                result = query_vlm_and_prepare_command(last_instruction_text, image_observations, env)
                if result is None:
                    break
                vlm_response_cache, plan, commands, fallback_used = result
                command_queue.extend(plan)
                stop_plan_active = all(cmd.kind == "stop" for cmd in commands)
                last_plan_had_stop = any(cmd.kind == "stop" for cmd in commands)
                if command_queue:
                    current_command, command_steps_remaining = command_queue.pop(0)
                plan_counter += 1
                current_plan_start_distance = current_distance
                overlay_text = f"Instruction: {last_instruction_text}\nVLM: {vlm_response_cache}"
                print(f"Replan #{plan_counter}: {summarize_commands(commands)}")
                log_vlm_decision(
                    LOG_DIR,
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "event": "auto_replan",
                        "instruction": last_instruction_text,
                        "vlm_output": vlm_response_cache,
                        "commands": commands_to_dicts(commands),
                        "fallback_used": fallback_used,
                        "command_segments": len(plan),
                        "distance_to_goal": current_distance,
                    },
                )
                if args_cli.save_map_frames:
                    save_map_image(
                        LOG_DIR,
                        latest_cam_disp,
                        latest_viz_disp,
                        f"replan_{plan_counter}",
                        subdir="maps_interactive",
                    )
                current_plan_instruction = last_instruction_text
                continue
            else:
                current_plan_instruction = None
                current_plan_start_distance = current_distance

        if not command_queue and command_steps_remaining == 0:
            current_plan_instruction = None
            current_plan_start_distance = current_distance if current_distance is not None else last_distance

        camera_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
        viz_frame_step = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()

        if num_steps % steps_per_image == 0:
            image_observations.append(Image.fromarray(camera_frame))
            if len(image_observations) > MAX_IMAGE_HISTORY:
                image_observations = image_observations[-MAX_IMAGE_HISTORY:]

        if num_steps % steps_per_viz_image == 0:
            cam_disp = camera_frame.copy()
            viz_disp = viz_frame_step.copy()
            if overlay_text:
                add_instruction_on_img(cam_disp, overlay_text)
                add_instruction_on_img(viz_disp, overlay_text)
            combined_frame = np.concatenate([cam_disp, viz_disp], axis=1)
            rgb_obses.append(combined_frame)
            if segment_tag is not None:
                segment_frames.append(combined_frame.astype(np.uint8))
            latest_cam_disp = cam_disp
            latest_viz_disp = viz_disp

        command_steps_remaining = max(command_steps_remaining - 1, 0)

        if stop_plan_active and not command_queue and command_steps_remaining == 0:
            env.set_stop_called(True)
            stop_plan_active = False


        num_steps += 1

    flush_segment_video()
    print("[INFO] Final measurements:", json.dumps(measurements, indent=2))

    instruction_provider.shutdown()
    env.close()

    if args_cli.visualize_path:
        os.makedirs("interactive_videos", exist_ok=True)
        writer = imageio.get_writer(
            f"interactive_videos/output_{int(episode['episode_id'])-1}.mp4",
            fps=10,
        )
        for frame in rgb_obses:
            frame = frame.astype(np.uint8)
            writer.append_data(frame)
        writer.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
