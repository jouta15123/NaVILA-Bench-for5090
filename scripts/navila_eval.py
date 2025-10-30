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
import socket
import queue
import threading
from datetime import datetime
from typing import List, Tuple, Optional

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
ISAACLAB_SITE_PACKAGES = os.path.join(os.path.dirname(ISAACLAB_SOURCE), "env_isaaclab", "lib", "python3.11", "site-packages")
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
parser.add_argument("--instruction", type=str, default=None, help="Custom instruction text. If provided, overrides episode instruction.")
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
    "--auto_replan",
    action="store_true",
    help="Enable automatic replanning when a plan finishes with insufficient progress.",
)
parser.add_argument(
    "--replan_delta",
    type=float,
    default=0.25,
    help="Plan 実行後に DistanceToGoal がこの値（メートル）以上改善していない場合に再計画します。",
)
parser.add_argument(
    "--save_map_frames",
    action="store_true",
    help="各プラン時の可視化フレームを logs/navila_events/maps に保存します。",
)
parser.add_argument(
    "--vlm_num_frames",
    type=int,
    default=8,
    help="Number of video frames to send to the VLM per query.",
)


# r2r argparse arguments
parser.add_argument("--episode_idx", type=int, default=0)

# RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner

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

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper
from omni.isaac.vlnce.utils.eval_utils import (
    get_vel_command,
    read_episodes,
    add_instruction_on_img,
    InstructionData,
)
from omni.isaac.vlnce.utils.measures import PathLength, DistanceToGoal, Success, SPL, OracleNavigationError, OracleSuccess, MeasureManager

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
    scene_id = os.path.splitext(os.path.basename(episode["scene_id"]))[0]
    env_cfg.scene.terrain.obj_filepath = os.path.join(ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")
    
    start_pos, start_rot, goal_pos = episode["start_position"], episode["start_rotation"], episode["reference_path"][-1]
    env_cfg.scene.robot.init_state.rot = start_rot

    if "go2" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+0.4)
    elif "h1" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+1.0)
    else:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+0.5)

    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos

    env_cfg.scene.disk_1.init_state.pos = ([start_pos[0], start_pos[1], start_pos[2] + 2.5])
    env_cfg.scene.disk_2.init_state.pos = ([goal_pos[0], goal_pos[1], goal_pos[2] + 2.5])

    return env_cfg


def add_measurement(env, episode):
    measure_manager = MeasureManager()
    measure_names = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    for measure_name in measure_names:
        measure = eval(measure_name)(env, episode, measure_manager)
        measure_manager.register_measure(measure)
    
    env.measure_manager = measure_manager
    return


def sample_images_and_send_to_vlm(image_list, vlm_host, vlm_port, query):
    if len(image_list) == 0:
        print("Did not receive any images.")
        return None

    sampled_images = sample_and_pad_images(image_list, num_frames=args_cli.vlm_num_frames)
    encoded_images = encode_images_to_base64(sampled_images)

    request_data = {
        "images": encoded_images,
        "query": query,
        "history_frames": max(0, len(sampled_images) - 1),
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((vlm_host, vlm_port))
        data_bytes = json.dumps(request_data).encode()
        s.sendall(len(data_bytes).to_bytes(8, "big"))
        s.sendall(data_bytes)

        size_data = s.recv(8)
        if not size_data:
            return None
        size = int.from_bytes(size_data, "big")

        response_data = b""
        while len(response_data) < size:
            packet = s.recv(4096)
            if not packet:
                break
            response_data += packet

        if not response_data:
            return None

        response = json.loads(response_data.decode())
        return response


def log_vlm_decision(log_dir: str, record: dict) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "eval.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


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


def _build_plan_from_commands(commands: List[ActionCommand], env) -> List[Tuple[torch.Tensor, int]]:
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


def _decode_vlm_to_plan(vlm_text: str, env) -> Tuple[List[Tuple[torch.Tensor, int]], List[ActionCommand]]:
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
    response = sample_images_and_send_to_vlm(
        image_observations,
        args_cli.vlm_host,
        args_cli.vlm_port,
        instruction_text,
    )
    if response is None:
        print("VLM から応答が得られませんでした。停止します。")
        return None

    vlm_text = response.get("response", "") if isinstance(response, dict) else str(response)
    if not vlm_text:
        print("VLM 応答が空です。停止します。")
        return None

    plan, commands = _decode_vlm_to_plan(vlm_text, env)
    fallback_used = False
    fallback_commands = parse_instruction_to_commands(instruction_text)
    if fallback_commands:
        fallback_commands = quantise_commands(fallback_commands)
        if not commands_cover_expected(fallback_commands, commands):
            print("VLM 応答が指示と一致しないためフォールバックコマンドを使用します。")
            commands = fallback_commands
            plan = _build_plan_from_commands(commands, env)
            fallback_used = True

    if not plan:
        print("VLM 応答から有効なコマンドを抽出できませんでした。")
        return None

    return vlm_text, plan, commands, fallback_used


def main():
    """IsaacSim Evaluation using NaViLA and trained low-level policy."""

    # read R2R test episodes
    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    all_episodes = read_episodes(r2r_data_path)
    episode = all_episodes[args_cli.episode_idx]

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    # reset the position and rotation of the robot
    env_cfg = reset_start_pos_rot(env_cfg, args_cli, episode)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli, play=True
    )

    # specify directory for logging experiments
    log_root_path = os.path.join(os.path.dirname(__file__),"../logs", "rsl_rl", agent_cfg.experiment_name)
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    # wrap around environment for rsl-rl
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    env = VLNEnvWrapper(env, policy, args_cli.task, episode, high_level_obs_key="camera_obs",
                        measure_names=all_measures)
    
    # set view pos and target
    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
    cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
    cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
    # set the camera view
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)
    
    # step with zeros actions to get the initial frame
    obs, infos = env.reset()

    # NaViLA training gets image observations each 0.5s, visualize every 0.1s
    steps_per_image = 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    steps_per_viz_image = 0.1 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)

    rgb_obs = infos["observations"]["camera_obs"]
    init_frame = rgb_obs[0, :, :, :3].cpu().numpy()
    # init_frame = cv2.rotate(init_frame, cv2.ROTATE_90_CLOCKWISE)
    
    # Set up instruction provider if interactive mode is requested
    # Interactive mode if: no instruction/instruction_file provided AND (instruction_mode is stdin OR stdin is TTY)
    use_interactive = (args_cli.instruction is None and 
                       args_cli.instruction_file is None and
                       (args_cli.instruction_mode == "stdin" or 
                        (args_cli.instruction_mode is None and sys.stdin.isatty())))
    
    if use_interactive:
        instruction_provider = build_instruction_provider()
        print("[INFO] Interactive mode enabled. Waiting for initial instruction...")
        # Wait for initial instruction
        initial_instruction = instruction_provider.wait_for_instruction()
        if initial_instruction is None:
            print("No instruction provided. Exiting.")
            return
        instruction = InstructionData(instruction_text=initial_instruction)
    else:
        instruction_provider = None
        # Use custom instruction if provided, otherwise use episode instruction
        if args_cli.instruction is not None:
            instruction_text = args_cli.instruction
            instruction = InstructionData(instruction_text=instruction_text)
        elif args_cli.instruction_file:
            with open(args_cli.instruction_file, "r", encoding="utf-8") as f:
                instruction_text = f.read().strip()
            instruction = InstructionData(instruction_text=instruction_text)
        else:
            instruction = InstructionData(**episode["instruction"])
    
    image_observations = []
    image_observations.append(Image.fromarray(init_frame))

    add_instruction_on_img(init_frame, instruction.instruction_text)
    vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    # vis_frame = cv2.rotate(vis_frame, cv2.ROTATE_90_CLOCKWISE)
    add_instruction_on_img(vis_frame, "")
    rgb_obses = [np.concatenate([init_frame, vis_frame], axis=1)]

    num_steps = 0
    target_steps = 0
    same_pos_count = 0
    prev_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    max_episode_steps = 100 * 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    vlm_response_cache = ""
    vlm_vel_commands = [0.0, 0.0, 0.0]
    env_steps_to_go = 0
    current_instruction = instruction.instruction_text

    # simulate environment
    while simulation_app.is_running():
        simulation_app.update()
        
        # Check for new instruction in interactive mode
        if use_interactive and instruction_provider:
            new_instruction = instruction_provider.get_instruction()
            if new_instruction is None:
                print("指示入力が終了したためシミュレーションを停止します。")
                break
            elif new_instruction:
                # New instruction received, update and reset target_steps to query VLM immediately
                current_instruction = new_instruction
                instruction = InstructionData(instruction_text=new_instruction)
                target_steps = num_steps  # Force immediate VLM query
                print(f"[INFO] New instruction: {new_instruction}")
        
        # run everything in inference mode
        with torch.inference_mode():
            if num_steps == target_steps:
                stream_output = sample_images_and_send_to_vlm(image_observations, args_cli.vlm_host, args_cli.vlm_port, current_instruction)
                if stream_output:
                    # Handle both string and dict responses
                    if isinstance(stream_output, dict):
                        vlm_text = stream_output.get("response", "")
                    else:
                        vlm_text = str(stream_output)
                    if vlm_text:
                        vlm_vel_commands, time_to_go = get_vel_command(vlm_text)
                        env_steps_to_go = int(time_to_go / (
                            env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
                        ))
                        target_steps = num_steps + env_steps_to_go
                        vlm_response_cache = vlm_text
                        print(f"VLM output: {vlm_text}\nVel Command: {vlm_vel_commands}, Env Steps to go: {env_steps_to_go}\n")
                    else:
                        print("VLM応答が空です。")
                        break
                else:
                    print("VLMからの応答が取得できませんでした。")
                    break

        obs, _, done, infos = env.step(torch.tensor(vlm_vel_commands, device=obs.device))

        if done or env.is_stop_called or num_steps > max_episode_steps:
            break

        cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_vel = np.linalg.norm(env.unwrapped.scene["robot"].data.root_vel_w[0].detach().cpu().numpy())
        if np.linalg.norm(cur_pos - prev_pos) < 0.01 and robot_vel < 0.01:
            same_pos_count += 1
        else:
            same_pos_count = 0
        prev_pos = cur_pos

        # Break out of the loop if the robot has stayed in the same location for 1000 steps
        if same_pos_count >= 1000:
            print("Robot has stayed in the same location for 1000 steps. Breaking out of the loop.")
            break

        if num_steps % steps_per_image == 0:
            curr_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            image_observations.append(Image.fromarray(curr_frame))
            # Keep only recent images
            MAX_IMAGE_HISTORY = 64
            if len(image_observations) > MAX_IMAGE_HISTORY:
                image_observations = image_observations[-MAX_IMAGE_HISTORY:]
            curr_frame_copy = curr_frame.copy()
            add_instruction_on_img(curr_frame_copy, instruction.instruction_text)
            
        if num_steps % steps_per_viz_image == 0:
            curr_vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
            add_instruction_on_img(curr_vis_frame, vlm_response_cache if vlm_response_cache else "")
            rgb_obses.append(np.concatenate([curr_frame_copy, curr_vis_frame], axis=1))

        num_steps += 1
        if env_steps_to_go == 0:
            env.set_stop_called(True)

    if instruction_provider:
        instruction_provider.shutdown()
    measurements = infos.get("measurements", {})

    result_dir = f"eval_results/{args_cli.task}_loco_{args_cli.load_run}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    measurement_dir = os.path.join(result_dir, "measurements")
    if not os.path.exists(measurement_dir):
        os.makedirs(measurement_dir)
    with open(f"{measurement_dir}/{int(episode['episode_id'])-1}.json", "w") as f:
        json.dump(measurements, f, indent=4)


    video_dir = os.path.join(result_dir, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    writer = imageio.get_writer(f"{video_dir}/output_{int(episode['episode_id'])-1}.mp4", fps=10)
    for frame in rgb_obses:
        frame = frame.astype(np.uint8)
        writer.append_data(frame)

    writer.close()

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
