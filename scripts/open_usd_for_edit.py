#!/usr/bin/env python3
"""Open a USD in Isaac Sim GUI and keep the app running for manual editing/export.

Usage (inside the Isaac Sim container):
  /workspace/IsaacLab/isaaclab.sh -p scripts/open_usd_for_edit.py --usd_path /path/to/file.usd

If --usd_path is omitted, this will try to open the H1 minimal USD from Nucleus.
"""

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a USD in Isaac Sim GUI for manual editing.")
    parser.add_argument(
        "--usd_path",
        type=str,
        default=None,
        help="USD path to open (local path or omniverse:// URL).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch Isaac Sim GUI
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Open stage
    import omni.usd
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    usd_path = args_cli.usd_path
    if not usd_path:
        usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd"
        print(f"[INFO] --usd_path not set. Opening default: {usd_path}")

    print(f"[INFO] Opening stage: {usd_path}")
    try:
        omni.usd.get_context().open_stage(usd_path)
    except Exception as exc:
        print(f"[ERROR] Failed to open stage: {exc}")

    # Keep GUI alive for manual edits
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    main()
