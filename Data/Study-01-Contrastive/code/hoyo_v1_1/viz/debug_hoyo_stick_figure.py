#!/usr/bin/env python3
"""
Debug script to visualize HOYO dataset samples as stick figure animations.
Generates GIFs to verify preprocessing (centering, resampling, coordinate system).
"""

import argparse
import os
import sys

# Headless configuration
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
HOYO_ROOT = SCRIPT_DIR.parent
REPO_ROOT = HOYO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    INSTRUCTION_ONOMATOPEIA,
)

# HOYO Skeleton Connectivity (same as style_module)
# 0: Head, 1: Neck, 2: R-Shoulder, 3: R-Elbow, 4: R-Hand
# 5: L-Shoulder, 6: L-Elbow, 7: L-Hand
# 8: R-Hip, 9: R-Knee, 10: R-Ankle
# 11: L-Hip, 12: L-Knee, 13: L-Ankle

SKELETON_EDGES = [
    (0, 1),   # Head - Neck
    (1, 2),   # Neck - R-Shoulder
    (2, 3),   # R-Shoulder - R-Elbow
    (3, 4),   # R-Elbow - R-Hand
    (1, 5),   # Neck - L-Shoulder
    (5, 6),   # L-Shoulder - L-Elbow
    (6, 7),   # L-Elbow - L-Hand
    (1, 8),   # Neck - R-Hip (spine approximation)
    (8, 9),   # R-Hip - R-Knee
    (9, 10),  # R-Knee - R-Ankle
    (1, 11),  # Neck - L-Hip
    (11, 12), # L-Hip - L-Knee
    (12, 13), # L-Knee - L-Ankle
    (8, 11),  # R-Hip - L-Hip (pelvis)
]


def create_stick_figure_animation(
    data: np.ndarray,
    label: str,
    output_path: str,
    fps: int = 30,
    title_prefix: str = "",
):
    """
    Create a stick figure animation from HOYO-format data.
    
    Args:
        data: (T, 14, 2) array in HOYO coordinate system
              x: lateral (left positive), y: vertical (down positive)
        label: Style label for title
        output_path: Path to save GIF
        fps: Frames per second
        title_prefix: Additional prefix for title
    """
    T, J, C = data.shape
    assert J == 14 and C == 2, f"Expected (T, 14, 2), got {data.shape}"
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Compute axis limits from data
    x_all = data[..., 0]  # lateral (left positive)
    y_all = -data[..., 1]  # flip for display (up positive)
    
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    
    # Add padding
    pad_x = max(0.1, 0.15 * (x_max - x_min))
    pad_y = max(0.1, 0.15 * (y_max - y_min))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (left positive)')
    ax.set_ylabel('Y (up positive)')
    ax.grid(True, alpha=0.3)
    
    # Initialize plot elements
    scat = ax.scatter([], [], c='red', s=40, zorder=5)
    lines = [ax.plot([], [], 'b-', linewidth=2)[0] for _ in SKELETON_EDGES]
    
    # Highlight left/right sides with different colors
    # Right side (joints 2,3,4,8,9,10) = blue
    # Left side (joints 5,6,7,11,12,13) = green
    right_joints = [2, 3, 4, 8, 9, 10]
    left_joints = [5, 6, 7, 11, 12, 13]
    center_joints = [0, 1]
    
    scat_right = ax.scatter([], [], c='blue', s=40, zorder=5, label='Right')
    scat_left = ax.scatter([], [], c='green', s=40, zorder=5, label='Left')
    scat_center = ax.scatter([], [], c='red', s=50, zorder=6, label='Center')
    
    title = ax.set_title('')
    ax.legend(loc='upper right')
    
    def update(frame_idx):
        frame = data[frame_idx]  # (14, 2)
        x = frame[:, 0]
        y = -frame[:, 1]  # Flip Y for display
        
        # Update joint positions by side
        scat_right.set_offsets(np.c_[x[right_joints], y[right_joints]])
        scat_left.set_offsets(np.c_[x[left_joints], y[left_joints]])
        scat_center.set_offsets(np.c_[x[center_joints], y[center_joints]])
        
        # Update skeleton edges
        for line, (i, j) in zip(lines, SKELETON_EDGES):
            line.set_data([x[i], x[j]], [y[i], y[j]])
        
        title.set_text(f'{title_prefix}{label} - Frame {frame_idx}/{T}')
        
        return [scat_right, scat_left, scat_center, title] + lines
    
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=1000/fps)
    
    # Save GIF
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {output_path}")


def compute_energy_diagnostic(data: np.ndarray) -> dict:
    """
    trans_energy vs style_energy を計算して移動がスタイルより支配的か判断。
    
    Args:
        data: (T, 14, 2) センタリング前の生データ
    Returns:
        dict with trans_energy, style_energy, and recommendation
    """
    # HOYO joint indices
    R_HIP, L_HIP = 8, 11
    R_SHOULDER, L_SHOULDER = 2, 5
    R_HAND, L_HAND = 4, 7
    
    # Pelvis trajectory
    pelvis = 0.5 * (data[:, R_HIP, :] + data[:, L_HIP, :])  # (T, 2)
    pelvis_0 = pelvis[0:1, :]  # (1, 2)
    
    # trans_energy: 平均移動量（初期位置からのドリフト）
    trans_energy = np.mean(np.linalg.norm(pelvis - pelvis_0, axis=-1))
    
    # style_energy: 腕振り振幅（手-肩の距離の変動）
    r_arm_len = np.linalg.norm(data[:, R_HAND, :] - data[:, R_SHOULDER, :], axis=-1)
    l_arm_len = np.linalg.norm(data[:, L_HAND, :] - data[:, L_SHOULDER, :], axis=-1)
    style_energy = 0.5 * (np.std(r_arm_len) + np.std(l_arm_len))
    
    ratio = trans_energy / (style_energy + 1e-6)
    
    if ratio > 2.0:
        recommendation = "移動が支配的 → pelvis_mean or pelvis センタリング推奨"
    elif ratio > 0.5:
        recommendation = "バランス良い → first_frame_com で十分"
    else:
        recommendation = "スタイルが支配的 → first_frame_com がベスト"
    
    return {
        "trans_energy": trans_energy,
        "style_energy": style_energy,
        "ratio": ratio,
        "recommendation": recommendation,
    }


<<<<<<< ours
<<<<<<< ours
def _build_hoyo_candidates() -> dict:
    """Return candidate link substrings for HOYO joints, with fallbacks."""
    return {
        "head": ["head_marker", "head_link", "torso_link"],
        "neck": ["neck_marker", "torso_link"],
        "r_shoulder": ["right_shoulder_pitch_link"],
        "r_elbow": ["right_elbow_link"],
        "r_hand": ["right_hand_marker", "right_elbow_link"],
        "l_shoulder": ["left_shoulder_pitch_link"],
        "l_elbow": ["left_elbow_link"],
        "l_hand": ["left_hand_marker", "left_elbow_link"],
        "r_hip": ["right_hip_yaw_link"],
        "r_knee": ["right_knee_link"],
        "r_ankle": ["right_ankle_link"],
        "l_hip": ["left_hip_yaw_link"],
        "l_knee": ["left_knee_link"],
        "l_ankle": ["left_ankle_link"],
    }


def _load_mapping_overrides() -> dict | None:
    mapping_path = REPO_ROOT / "configs" / "h1_to_hoyo_mapping.json"
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _resolve_hoyo_indices(body_names: list[str]) -> tuple[np.ndarray, list[str]]:
    hoyo_order = [
        "head", "neck",
        "r_shoulder", "r_elbow", "r_hand",
        "l_shoulder", "l_elbow", "l_hand",
        "r_hip", "r_knee", "r_ankle",
        "l_hip", "l_knee", "l_ankle",
    ]
    candidates = _build_hoyo_candidates()
    overrides = _load_mapping_overrides()
    if overrides:
        for key, target in overrides.items():
            if key in candidates:
                if target not in candidates[key]:
                    candidates[key].insert(0, target)
            else:
                candidates[key] = [target]

    torso_idx = 0
    for i, name in enumerate(body_names):
        if "torso_link" in name:
            torso_idx = i
            break

    indices = []
    missing = []
    for key in hoyo_order:
        found_idx = None
        for target in candidates.get(key, ["torso_link"]):
            for i, name in enumerate(body_names):
                if target in name:
                    found_idx = i
                    break
            if found_idx is not None:
                break
        if found_idx is None:
            missing.append(f"{key}->{candidates.get(key, ['torso_link'])}")
            found_idx = torso_idx
        indices.append(found_idx)

    return np.array(indices, dtype=np.int64), missing


def run_usd_mode(args) -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab_assets import H1_MINIMAL_CFG

    @configclass
    class H1SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0"))
    scene_cfg = H1SceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot.spawn.usd_path = args.usd_path
    if args.disable_gravity:
        scene_cfg.robot.spawn.rigid_props.disable_gravity = True

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    scene.update(sim_dt)

    # Initialize a standing pose once
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()
    scene.update(sim_dt)

    body_names = list(robot.data.body_names)
    print(f"[USD] body_names ({len(body_names)}):")
    for name in body_names:
        print("  ", name)

    indices, missing = _resolve_hoyo_indices(body_names)
    if missing:
        print(f"[USD] Missing joints (fallback to torso): {missing}")

    history_2d = []
    for _ in range(args.frames):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        body_pos = robot.data.body_pos_w[0].detach().cpu().numpy()
        joints = body_pos[indices]  # (14, 3)
        # H1: Y=left, Z=up -> HOYO: x=left, y=down
        joints_2d = np.stack([joints[:, 1], -joints[:, 2]], axis=-1)
        history_2d.append(joints_2d)

    data = np.stack(history_2d, axis=0)
    output_dir = HOYO_ROOT / "viz" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.usd_path).stem
        output_path = str(output_dir / f"usd_{stem}.gif")

    create_stick_figure_animation(
        data,
        label="USD",
        output_path=output_path,
        fps=args.fps,
        title_prefix=f"USD ({Path(args.usd_path).name}) - ",
    )

    simulation_app.close()


=======
>>>>>>> theirs
=======
>>>>>>> theirs
def main():
    parser = argparse.ArgumentParser(description="Visualize HOYO dataset samples as stick figures")
    parser.add_argument("--label", type=str, default="すたすた", 
                        help="Onomatopoeia label to visualize")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index within the label")
    parser.add_argument("--target-len", type=int, default=100,
                        help="Target sequence length (frames)")
    parser.add_argument("--centering", type=str, default="first_frame_com",
                        choices=["pelvis", "pelvis_mean", "first_frame_pelvis", "first_frame_com"],
                        help="Centering mode (default: first_frame_com for less noise)")
    parser.add_argument("--src-fps", type=int, default=60,
                        help="Source FPS of HOYO data")
    parser.add_argument("--tgt-fps", type=int, default=50,
                        help="Target FPS (for resampling)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GIF path (default: hoyo_{label}_{idx}.gif)")
    parser.add_argument("--fps", type=int, default=25,
                        help="GIF playback FPS")
    parser.add_argument("--all-labels", action="store_true",
                        help="Generate one sample for each label")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Skip mean/std normalization for raw visualization")
    parser.add_argument("--view", type=str, default=None, choices=["front", "back"],
                        help="Filter by view (front/back). Default: all")
    parser.add_argument("--no-normalize-back", action="store_true",
                        help="Don't normalize back view to front (keep back as-is)")
    parser.add_argument("--pelvis-ema", type=float, default=0.9,
                        help="Pelvis EMA alpha (0=disabled, 0.8-0.95 recommended)")
    parser.add_argument("--no-pelvis-ema", action="store_true",
                        help="Disable pelvis EMA smoothing")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading HOYO dataset...")
    print(f"  target_len={args.target_len}, centering={args.centering}")
    print(f"  src_fps={args.src_fps}, tgt_fps={args.tgt_fps}")
    
    pelvis_ema = 0.0 if args.no_pelvis_ema else args.pelvis_ema
    dataset = HoyoInstructionDataset(
        root=HOYO_ROOT,
        target_labels=INSTRUCTION_ONOMATOPEIA,
        target_len=args.target_len,
        is_train=False,  # Use center crop for reproducibility
        use_aug=False,
        centering=args.centering,
        src_fps=args.src_fps,
        tgt_fps=args.tgt_fps,
        view_filter=args.view,
        normalize_back_to_front=not args.no_normalize_back,
        pelvis_ema_alpha=pelvis_ema,
    )
    
    # Don't apply standardization for visualization (unless explicitly wanted)
    if args.no_standardize:
        dataset.mean = None
        dataset.std = None
    
    output_dir = HOYO_ROOT / "viz" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all_labels:
        # Generate one sample for each label
        for label in INSTRUCTION_ONOMATOPEIA:
            samples = dataset.samples_by_label.get(label, [])
            if not samples:
                print(f"No samples for {label}, skipping")
                continue
            
            # Get first sample
            idx = 0
            for i, (lab, raw_idx) in enumerate(dataset._indices):
                if lab == label and raw_idx == 0:
                    idx = i
                    break
            
            data, label_id = dataset[idx]
            view_suffix = f"_{args.view}" if args.view else ""
            norm_suffix = "_raw" if args.no_normalize_back else ""
            output_path = str(output_dir / f"hoyo_{label}{view_suffix}{norm_suffix}_0.gif")
            
            print(f"\nProcessing {label}...")
            print(f"  Shape: {data.shape}")
            print(f"  X range: [{data[:,:,0].min():.3f}, {data[:,:,0].max():.3f}]")
            print(f"  Y range: [{data[:,:,1].min():.3f}, {data[:,:,1].max():.3f}]")
            
            view_str = args.view or "all"
            norm_str = "" if args.no_normalize_back else ", normalized"
            create_stick_figure_animation(
                data, label, output_path,
                fps=args.fps,
                title_prefix=f"HOYO ({view_str}{norm_str}, {args.centering}) - "
            )
    else:
        # Generate single sample
        label = args.label
        samples = dataset.samples_by_label.get(label, [])
        if not samples:
            print(f"No samples for label '{label}'")
            print(f"Available labels: {list(dataset.samples_by_label.keys())}")
            return
        
        if args.sample_idx >= len(samples):
            print(f"Sample index {args.sample_idx} out of range (max: {len(samples)-1})")
            return
        
        # Find the dataset index for this label/sample_idx
        target_idx = None
        for i, (lab, raw_idx) in enumerate(dataset._indices):
            if lab == label and raw_idx == args.sample_idx:
                target_idx = i
                break
        
        if target_idx is None:
            print(f"Could not find sample {args.sample_idx} for label {label}")
            return
        
        data, label_id = dataset[target_idx]
        
        # Energy diagnostic (using raw data before centering)
        raw_data = samples[args.sample_idx]  # (T_raw, 14, 2) 生データ
        # Use center crop like dataset does
        T_raw = raw_data.shape[0]
        if T_raw >= args.target_len:
            start = (T_raw - args.target_len) // 2
            raw_cropped = raw_data[start:start + args.target_len]
        else:
            raw_cropped = raw_data
        
        diag = compute_energy_diagnostic(raw_cropped)
        
        print(f"\nSample info:")
        print(f"  Label: {label}")
        print(f"  Shape: {data.shape}")
        print(f"  X range: [{data[:,:,0].min():.3f}, {data[:,:,0].max():.3f}]")
        print(f"  Y range: [{data[:,:,1].min():.3f}, {data[:,:,1].max():.3f}]")
        
        print(f"\nEnergy Diagnostic (centering method selection):")
        print(f"  trans_energy (移動量): {diag['trans_energy']:.4f}")
        print(f"  style_energy (腕振り): {diag['style_energy']:.4f}")
        print(f"  ratio (trans/style): {diag['ratio']:.2f}")
        print(f"  → {diag['recommendation']}")
        
        # Verify coordinate system
        # In HOYO front view (from viewer's perspective):
        # - Subject's right arm (joints 2,3,4) appears on LEFT side of image = x < 0
        # - Subject's left arm (joints 5,6,7) appears on RIGHT side of image = x > 0
        # This matches style_module where H1 Y (robot's left) = positive
        r_shoulder_x = data[:, 2, 0].mean()
        l_shoulder_x = data[:, 5, 0].mean()
        print(f"  R-Shoulder mean X: {r_shoulder_x:.3f} (subject's right, viewer's left, should be < 0)")
        print(f"  L-Shoulder mean X: {l_shoulder_x:.3f} (subject's left, viewer's right, should be > 0)")
        
        if r_shoulder_x > l_shoulder_x:
            print("  WARNING: Left/right may be inverted!")
        else:
            print("  OK: Left/right orientation looks correct")
        
        if args.output:
            output_path = args.output
        else:
            output_path = str(output_dir / f"hoyo_{label}_{args.sample_idx}.gif")
        
        view_str = args.view or "all"
        norm_str = "" if args.no_normalize_back else ", normalized"
        create_stick_figure_animation(
            data, label, output_path,
            fps=args.fps,
            title_prefix=f"HOYO ({view_str}{norm_str}, {args.centering}) - "
        )


if __name__ == "__main__":
    main()
