#!/usr/bin/env python3
"""
Utility functions for visualizing H1 -> HOYO 2D keypoint mapping as GIF animations.
Based on debug_stick_figure_anim.py logic.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    japanize_matplotlib = None


# HOYO Skeleton Connectivity (14 joints)
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
    (1, 8),   # Neck - R-Hip
    (8, 9),   # R-Hip - R-Knee
    (9, 10),  # R-Knee - R-Ankle
    (1, 11),  # Neck - L-Hip
    (11, 12), # L-Hip - L-Knee
    (12, 13), # L-Knee - L-Ankle
]

HOYO_JOINT_NAMES = [
    "Head", "Neck", "R-Shoulder", "R-Elbow", "R-Hand",
    "L-Shoulder", "L-Elbow", "L-Hand",
    "R-Hip", "R-Knee", "R-Ankle",
    "L-Hip", "L-Knee", "L-Ankle"
]


def save_hoyo_gif(
    history_2d: list[np.ndarray],
    out_path: str,
    fps: int = 30,
    title: str = "H1 → HOYO Mapping",
    figsize: tuple = (6, 6),
    stride: int = 1,
) -> bool:
    """
    Save HOYO 2D keypoints as an animated GIF.
    Based on debug_stick_figure_anim.py logic.
    
    Args:
        history_2d: List of (14, 2) arrays, one per frame.
        out_path: Output GIF path.
        fps: Frames per second.
        title: Plot title.
        figsize: Figure size.
        stride: Frame stride (skip frames for faster GIF).
    
    Returns:
        True if saved successfully, False otherwise.
    """
    if len(history_2d) == 0:
        print(f"[WARN] No frames to save for {out_path}")
        return False
    
    # Apply stride
    if stride > 1:
        history_2d = history_2d[::stride]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel("X (lateral, left=+)")
    ax.set_ylabel("Y (vertical, up=+)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    
    # Auto-scale axes from collected keypoints (same as debug_stick_figure_anim.py)
    all_data = np.stack(history_2d, axis=0)  # (T, 14, 2)
    x_all = all_data[..., 0]
    y_all = -all_data[..., 1]  # flip for display (up positive)
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    
    # Add padding to avoid clipping at edges
    pad_x = max(0.05, 0.1 * (x_max - x_min))
    pad_y = max(0.05, 0.1 * (y_max - y_min))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    
    # Initialize scatter and lines (same style as debug_stick_figure_anim.py)
    scat = ax.scatter([], [], c="r", s=20)
    lines = [ax.plot([], [], "b-")[0] for _ in SKELETON_EDGES]
    frame_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
    
    def update(frame_idx):
        data = history_2d[frame_idx]  # (14, 2)
        x = data[:, 0]
        y = -data[:, 1]  # Flip Y so Up is Positive for visualization
        scat.set_offsets(np.c_[x, y])
        for line, (i, j) in zip(lines, SKELETON_EDGES):
            line.set_data([x[i], x[j]], [y[i], y[j]])
        frame_text.set_text(f"Frame: {frame_idx * stride}")
        return [scat, frame_text] + lines
    
    ani = animation.FuncAnimation(fig, update, frames=len(history_2d), blit=True)
    
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        print(f"Saving GIF: {out_path}")
        ani.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save GIF: {e}")
        import traceback
        traceback.print_exc()
        plt.close(fig)
        return False


def save_hoyo_comparison_gif(
    h1_history: list[np.ndarray],
    hoyo_ref: np.ndarray,
    out_path: str,
    fps: int = 30,
    title: str = "H1 vs HOYO Reference",
    figsize: tuple = (12, 6),
    stride: int = 1,
    loop_hoyo: bool = True,
) -> bool:
    """
    Save side-by-side comparison of H1 mapping and HOYO reference.
    
    Args:
        h1_history: List of (14, 2) arrays from H1 robot.
        hoyo_ref: (T, 14, 2) array from HOYO dataset.
        out_path: Output GIF path.
        fps: Frames per second.
        title: Plot title.
        figsize: Figure size.
        stride: Frame stride.
        loop_hoyo: If True, loop HOYO reference to match H1 length.
    
    Returns:
        True if saved successfully, False otherwise.
    """
    if len(h1_history) == 0:
        print(f"[WARN] No H1 frames to save for {out_path}")
        return False
    
    # Apply stride
    if stride > 1:
        h1_history = h1_history[::stride]
    
    h1_len = len(h1_history)
    hoyo_len = len(hoyo_ref)
    
    # Handle length mismatch
    if loop_hoyo and h1_len > hoyo_len:
        # Loop HOYO reference to match H1 length
        repeats = (h1_len // hoyo_len) + 1
        hoyo_ref = np.tile(hoyo_ref, (repeats, 1, 1))[:h1_len]
    else:
        # Match to minimum length
        min_len = min(h1_len, hoyo_len)
        h1_history = h1_history[:min_len]
        hoyo_ref = hoyo_ref[:min_len]
    
    total_frames = len(h1_history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    ax1.set_title("H1 Robot (mapped)", fontsize=12, color="blue")
    ax1.set_xlabel("X (lateral)")
    ax1.set_ylabel("Y (vertical)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("HOYO Reference", fontsize=12, color="green")
    ax2.set_xlabel("X (lateral)")
    ax2.set_ylabel("Y (vertical)")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, alpha=0.3)
    
    # Auto-scale axes (use same range for both for fair comparison)
    h1_all = np.stack(h1_history, axis=0)
    hoyo_all = hoyo_ref[:len(h1_history)]
    all_data = np.concatenate([h1_all, hoyo_all], axis=0)
    
    x_all = all_data[..., 0]
    y_all = -all_data[..., 1]
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad_x = max(0.1, 0.15 * (x_max - x_min))
    pad_y = max(0.1, 0.15 * (y_max - y_min))
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
    
    # Initialize elements
    scat1 = ax1.scatter([], [], c="red", s=30, zorder=5)
    lines1 = [ax1.plot([], [], "b-", lw=2, zorder=4)[0] for _ in SKELETON_EDGES]
    
    scat2 = ax2.scatter([], [], c="limegreen", s=30, zorder=5)
    lines2 = [ax2.plot([], [], "darkgreen", lw=2, zorder=4)[0] for _ in SKELETON_EDGES]
    
    frame_text = fig.text(0.5, 0.02, "", ha='center', fontsize=10)
    
    def init():
        scat1.set_offsets(np.zeros((14, 2)))
        scat2.set_offsets(np.zeros((14, 2)))
        for line in lines1 + lines2:
            line.set_data([], [])
        frame_text.set_text("")
        return [scat1, scat2, frame_text] + lines1 + lines2
    
    def update(frame_idx):
        # H1
        data1 = h1_history[frame_idx]
        x1, y1 = data1[:, 0], -data1[:, 1]
        scat1.set_offsets(np.c_[x1, y1])
        for line, (i, j) in zip(lines1, SKELETON_EDGES):
            line.set_data([x1[i], x1[j]], [y1[i], y1[j]])
        
        # HOYO
        hoyo_frame_idx = frame_idx % len(hoyo_ref)
        data2 = hoyo_ref[hoyo_frame_idx]
        x2, y2 = data2[:, 0], -data2[:, 1]
        scat2.set_offsets(np.c_[x2, y2])
        for line, (i, j) in zip(lines2, SKELETON_EDGES):
            line.set_data([x2[i], x2[j]], [y2[i], y2[j]])
        
        frame_text.set_text(f"Frame: {frame_idx * stride} | HOYO: {hoyo_frame_idx}")
        return [scat1, scat2, frame_text] + lines1 + lines2
    
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init, blit=True
    )
    
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        print(f"Saving comparison GIF: {out_path}")
        ani.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save comparison GIF: {e}")
        import traceback
        traceback.print_exc()
        plt.close(fig)
        return False
