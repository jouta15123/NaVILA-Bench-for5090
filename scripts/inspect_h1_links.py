#!/usr/bin/env python3
"""Dump H1 link / body names and positions after a single reset step.

Usage (inside the Isaac Sim container):

  /workspace/IsaacLab/isaaclab.sh -p scripts/inspect_h1_links.py \
    --task=h1_matterport_vision --num_envs=1 --episode_idx=0 --headless

This uses the same env config pathway as `navila_single_episode.py`,
but stops immediately after reset and prints the robot kinematic data.
"""

import argparse
import os
import sys

import gymnasium as gym

# ---------------------------------------------------------------------------
# Path setup (mirror navila_single_episode)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
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

# Local extensions (vlnce / matterport)
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

# ---------------------------------------------------------------------------
# App launcher (must be first Omni import)
# ---------------------------------------------------------------------------
try:
    from omni.isaac.lab.app import AppLauncher
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description="Inspect H1 links once after reset.")
    parser.add_argument("--task", type=str, default="h1_matterport_vision")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--scene_id_override", type=str, default=None)
    parser.add_argument("--scene_usd", type=str, default=None)
    parser.add_argument(
        "--load_run",
        type=str,
        default="2024-11-03_15-08-09_height_scan_obst",
        help="Only used to mirror start pose selection; policy is NOT loaded.",
    )

    # Isaac Sim launcher args (headless, gpu, fabric, etc.)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch simulator (headless by default)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # IsaacLab imports after launcher
    try:
        import omni.isaac.lab_tasks  # noqa: F401
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        from omni.isaac.lab.utils.io import load_yaml
        from omni.isaac.lab.utils import update_class_from_dict
    except ModuleNotFoundError:
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.io import load_yaml
        from isaaclab.utils import update_class_from_dict

    from omni.isaac.vlnce.utils import ASSETS_DIR
    from omni.isaac.vlnce.utils.eval_utils import read_episodes

    # Reuse helper functions from navila_single_episode (keeps logic in one place)
    # Minimal reimplementation of helpers to avoid importing navila_single_episode

    def select_episode(all_episodes, episode_idx: int, scene_id_override: str | None):
        if scene_id_override:
            target_scene = scene_id_override.strip()
            matching_indices = [
                idx
                for idx, ep in enumerate(all_episodes)
                if os.path.splitext(os.path.basename(ep["scene_id"]))[0] == target_scene
            ]
            if not matching_indices:
                raise ValueError(
                    f"No episodes found for scene_id '{target_scene}'"
                )
            if episode_idx >= len(matching_indices):
                raise ValueError(
                    f"--episode_idx={episode_idx} exceeds available episodes ({len(matching_indices)}) for scene '{target_scene}'"
                )
            return all_episodes[matching_indices[episode_idx]]

        if episode_idx >= len(all_episodes):
            raise ValueError(f"--episode_idx={episode_idx} exceeds total episodes ({len(all_episodes)})")
        return all_episodes[episode_idx]

    def reset_start_pos_rot(env_cfg, args_cli, episode):
        # Prefer direct USD override if provided
        if args_cli.scene_usd and os.path.isfile(args_cli.scene_usd):
            env_cfg.scene.terrain.obj_filepath = os.path.abspath(args_cli.scene_usd)
            scene_id = os.path.splitext(os.path.basename(env_cfg.scene.terrain.obj_filepath))[0]
        else:
            scene_id = os.path.splitext(os.path.basename(episode["scene_id"]))[0]
            env_cfg.scene.terrain.obj_filepath = os.path.join(ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")

        start_pos, start_rot, goal_pos = (
            episode["start_position"],
            episode["start_rotation"],
            episode["reference_path"][-1],
        )
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

    # ------------------------------------------------------------------
    # Dataset episode selection (to place the robot correctly)
    # ------------------------------------------------------------------
    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    all_episodes = read_episodes(r2r_data_path)
    episode = select_episode(all_episodes, args_cli.episode_idx, args_cli.scene_id_override)

    # Build env config and set start pose
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg = reset_start_pos_rot(env_cfg, args_cli, episode)

    # (Optional) keep viewer off in headless
    try:
        env_cfg.viewer.enable_camera_axes = False
    except Exception:
        pass

    # Instantiate env (no policy wrapper)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, infos = env.reset()

    # Grab robot data
    robot_data = env.unwrapped.scene["robot"].data
    to_np = lambda x: x.detach().cpu().numpy()

    print("\n=== H1 kinematic dump (env=0) ===")
    try:
        print("body_names ({}):".format(len(robot_data.body_names)))
        print(robot_data.body_names)
    except Exception:
        pass

    try:
        print("link_names ({}):".format(len(robot_data.link_names)))
        print(robot_data.link_names)
    except Exception:
        print("link_names not available on this build")

    try:
        print("root_pos_w:", to_np(robot_data.root_pos_w[0]))
        print("root_quat_w:", to_np(robot_data.root_quat_w[0]))
    except Exception:
        pass

    try:
        print("link_pos_w[0] (first env):\n", to_np(robot_data.link_pos_w[0]))
    except Exception as e:
        print(f"link_pos_w not available: {e}")

    # One physics step with zero action (best-effort) to confirm shapes
    try:
        act_shape = env.action_space.shape
        zero_action = env.action_space.sample() * 0.0
        zero_action = zero_action.reshape(act_shape)
        _ = env.step(zero_action)
        simulation_app.update()
        print("\n[OK] Stepped once with zero action; positions remain:")
        print("root_pos_w (post-step):", to_np(robot_data.root_pos_w[0]))
    except Exception as e:
        print(f"[WARN] Step skipped (could not build zero action): {e}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
