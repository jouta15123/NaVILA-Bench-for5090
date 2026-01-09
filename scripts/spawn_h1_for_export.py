#!/usr/bin/env python3
"""Load H1 USD into Isaac Sim GUI for manual editing (no physics).

Usage (inside the Isaac Sim container):
  /workspace/IsaacLab/isaaclab.sh -p scripts/spawn_h1_for_export.py

Optional override:
  /workspace/IsaacLab/isaaclab.sh -p scripts/spawn_h1_for_export.py \
    --usd_path /workspace/NaVILA-Bench/assets/h1_custom/h1_with_hands.usd

After editing in the GUI, use File > Export As... to save your changes.
"""

import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Load H1 USD for manual editing (physics disabled).")
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="Override H1 USD path (e.g., local cached or custom USD).",
)
parser.add_argument(
    "--prim_path",
    type=str,
    default="/World/Robot",
    help="Prim path where the H1 asset should be spawned.",
)
# append AppLauncher cli args (headless, livestream, enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# USD / Omniverse imports (after app launch)
# -----------------------------------------------------------------------------
import omni.usd
from pxr import UsdGeom, Gf

# Get default H1 USD path from isaaclab_assets
from isaaclab_assets import H1_MINIMAL_CFG


def main():
    # Determine USD path
    if args_cli.usd_path:
        usd_path = args_cli.usd_path
    else:
        # Use default from H1_MINIMAL_CFG
        usd_path = H1_MINIMAL_CFG.spawn.usd_path
    
    print(f"[INFO] Loading USD: {usd_path}")
    print(f"[INFO] Target prim path: {args_cli.prim_path}")
    
    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()
    
    # Create a ground plane for reference (visual only)
    ground_path = "/World/Ground"
    UsdGeom.Mesh.Define(stage, ground_path)
    ground_prim = stage.GetPrimAtPath(ground_path)
    ground_mesh = UsdGeom.Mesh(ground_prim)
    # Simple quad for ground
    ground_mesh.GetPointsAttr().Set([
        Gf.Vec3f(-10, -10, 0), Gf.Vec3f(10, -10, 0),
        Gf.Vec3f(10, 10, 0), Gf.Vec3f(-10, 10, 0)
    ])
    ground_mesh.GetFaceVertexCountsAttr().Set([4])
    ground_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    
    # Add a dome light
    from pxr import UsdLux
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.GetIntensityAttr().Set(1000.0)
    
    # Add the H1 robot as a reference (no physics, just USD structure)
    robot_prim = stage.DefinePrim(args_cli.prim_path)
    robot_prim.GetReferences().AddReference(usd_path)
    
    print("[INFO] H1 loaded successfully!")
    print("[INFO] Physics is NOT running - robot will stay in place.")
    print("[INFO] You can now:")
    print("       - Navigate the Stage tree on the left")
    print("       - Add links/prims to head/hands")
    print("       - File > Export As... to save your changes")
    print("[INFO] Press Ctrl+C or close the window to exit.")
    
    # Keep GUI alive WITHOUT running physics
    while simulation_app.is_running():
        # Just update the app, don't step physics
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
