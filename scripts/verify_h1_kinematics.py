#!/usr/bin/env python3
"""
Robust verification script for H1 kinematics.
Ensures environment is registered and inspects body names/offsets.
"""

import sys
import os

# ---------------------------------------------------------------------------
# 1. Path Setup (Crucial for importing extensions)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /home/jouta/NaVILA-Bench
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Add Isaac Lab source
ISAACLAB_SOURCE = "/home/jouta/IsaacLab/source"
if os.path.isdir(ISAACLAB_SOURCE) and ISAACLAB_SOURCE not in sys.path:
    sys.path.append(ISAACLAB_SOURCE)
    sys.path.append(os.path.join(ISAACLAB_SOURCE, "isaaclab"))
    sys.path.append(os.path.join(ISAACLAB_SOURCE, "isaaclab_tasks"))
    sys.path.append(os.path.join(ISAACLAB_SOURCE, "isaaclab_assets"))

# Add Extensions explicitly
LEGGED_LOCO_PATH = os.path.join(PROJECT_ROOT, "legged-loco", "isaaclab_exts", "omni.isaac.leggedloco")
sys.path.append(LEGGED_LOCO_PATH)
VLN_PATH = os.path.join(PROJECT_ROOT, "isaaclab_exts", "omni.isaac.vlnce")
sys.path.append(VLN_PATH)
MATTERPORT_PATH = os.path.join(PROJECT_ROOT, "isaaclab_exts", "omni.isaac.matterport")
sys.path.append(MATTERPORT_PATH)


import argparse
import gymnasium as gym
import torch
import numpy as np

# ---------------------------------------------------------------------------
# 2. App Launcher
# ---------------------------------------------------------------------------
try:
    from isaaclab.app import AppLauncher
except ImportError:
    print("Could not import isaaclab.app. Checks sys.path:")
    print(sys.path)
    raise

# ---------------------------------------------------------------------------
# 3. Main Logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify H1 Kinematics")
    parser.add_argument("--task", type=str, default="h1_vision", help="Task name")
    parser.add_argument("--num_envs", type=int, default=1)
    
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch App
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 4. Import Isaac Lab modules AFTER app launch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg
    
    # 5. Import Extension Configs to Trigger Registration
    # This is key! Creating the config object or importing the module registers the gym envs.
    print("Importing extension configs...")
    try:
        import omni.isaac.leggedloco.config.h1
        print("Successfully imported omni.isaac.leggedloco.config.h1")
    except ImportError as e:
        print(f"Failed to import config module: {e}")
        # Try finding where it is
        print(f"Looking for omni.isaac.leggedloco in {sys.path}")
        
    # Check registration
    print(f"Checking if {args_cli.task} is registered...")
    if args_cli.task in gym.registry:
        print(f"SUCCESS: {args_cli.task} found in registry.")
    else:
        print(f"WARNING: {args_cli.task} NOT found in registry. Dump:")
        # print(list(gym.registry.keys()))

    # 6. Create Environment
    try:
        print(f"Creating environment: {args_cli.task}")
        env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
        
        # Override viewer for headless validation if needed
        # env_cfg.viewer.enable_camera_axes = False
        
        env = gym.make(args_cli.task, cfg=env_cfg)
        
        print("Resetting environment...")
        env.reset()
        
        # 7. Inspect Kinematics
        print("\n" + "="*50)
        print("H1 KINEMATICS INSPECTION")
        print("="*50)
        
        robot = env.unwrapped.scene["robot"]
        body_names = robot.data.body_names
        
        print(f"Total Bodies: {len(body_names)}")
        for i, name in enumerate(body_names):
            pos = robot.data.body_pos_w[0, i].cpu().numpy()
            print(f"{i:2d}: {name:<30} Pos: {pos}")
            
        # Check for specific links used in remapping
        target_links = ["torso_link", "right_elbow_link", "left_elbow_link"]
        print("\nChecking Remapping Targets:")
        for t in target_links:
            found = False
            for name in body_names:
                if t in name:
                    found = True
                    break
            status = "FOUND" if found else "MISSING"
            print(f"{t:<30}: {status}")

        print("="*50 + "\n")
        
        env.close()

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    simulation_app.close()

if __name__ == "__main__":
    main()
