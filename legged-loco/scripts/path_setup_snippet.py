import os
import sys

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
    from isaaclab.app import AppLauncher
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher
