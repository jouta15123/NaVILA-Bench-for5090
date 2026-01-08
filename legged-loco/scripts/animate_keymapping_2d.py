#!/usr/bin/env python3
"""Animate HOYO keypoints (2D projection) from an NPZ recording."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

HOYO_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (2, 5),
    (8, 9), (9, 10),
    (11, 12), (12, 13),
    (8, 11),
    (1, 8), (1, 11),
]

VIEW_AXES = {
    "xz": (0, 2),
    "yz": (1, 2),
    "xy": (0, 1),
}


def _project(points: np.ndarray, view: str) -> np.ndarray:
    ax = VIEW_AXES[view]
    return points[..., [ax[0], ax[1]]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate HOYO keypoints (2D projection).")
    parser.add_argument("--input", type=str, required=True, help="Path to keypoints npz.")
    parser.add_argument("--out", type=str, required=True, help="Output mp4/gif path.")
    parser.add_argument("--view", type=str, choices=["xz", "yz", "xy"], default="xz")
    parser.add_argument("--center", type=str, choices=["none", "root", "first"], default="root")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--title", type=str, default="HOYO keypoints (world)")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    keypoints = data["keypoints_w"]  # (T, 14, 3)
    root_pos = data.get("root_pos_w", None)

    if args.center == "root" and root_pos is not None:
        keypoints = keypoints - root_pos[:, None, :]
    elif args.center == "first":
        keypoints = keypoints - keypoints[:1, :, :]

    # stride
    keypoints = keypoints[:: args.stride]

    points_2d = _project(keypoints, args.view)

    # axis limits
    min_xy = points_2d.min(axis=(0, 1))
    max_xy = points_2d.max(axis=(0, 1))
    pad = (max_xy - min_xy) * 0.1
    min_xy -= pad
    max_xy += pad

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(args.title)
    ax.set_xlim(min_xy[0], max_xy[0])
    ax.set_ylim(min_xy[1], max_xy[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    scat = ax.scatter([], [], s=30, c="#1f77b4")
    lines = [ax.plot([], [], lw=2, c="#333333")[0] for _ in HOYO_EDGES]

    def init():
        scat.set_offsets(np.zeros((14, 2)))
        for line in lines:
            line.set_data([], [])
        return [scat] + lines

    def update(frame_idx: int):
        pts = points_2d[frame_idx]
        scat.set_offsets(pts)
        for line, (i, j) in zip(lines, HOYO_EDGES):
            line.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
        return [scat] + lines

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(points_2d),
        interval=1000 / args.fps,
        blit=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(out_path, writer=writer)
    else:
        try:
            writer = animation.FFMpegWriter(fps=args.fps)
            anim.save(out_path, writer=writer)
        except Exception:
            # fallback to gif if ffmpeg is unavailable
            gif_path = out_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=args.fps)
            anim.save(gif_path, writer=writer)
            print(f"[WARN] ffmpeg unavailable. Saved GIF instead: {gif_path}")

    plt.close(fig)
    print(f"[INFO] Saved animation: {out_path}")


if __name__ == "__main__":
    main()
