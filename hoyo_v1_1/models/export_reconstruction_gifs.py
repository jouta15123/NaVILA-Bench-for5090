import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency
    imageio = None

# Ensure project root on sys.path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    INSTRUCTION_ONOMATOPEIA,
    apply_normalization_from_stats,
)
from hoyo_v1_1.models.train_motionclip_joint import (
    HOYO_ROOT,
    load_motionclip_full_model,
)
from hoyo_v1_1.models.export_hoyo_style_gifs import SKELETON_EDGES


# スタイル（オリジナル vs 再構成）を色で分ける
ORIG_STYLE = {
    "line_color": "#1f77b4",
    "line_width": 2.0,
    "marker_face_color": "#1f77b4",
    "marker_edge_color": "white",
    "marker_size": 4.0,
    "head_face_color": "#aec7e8",
    "head_edge_color": "#1f77b4",
}

RECON_STYLE = {
    "line_color": "#d62728",
    "line_width": 2.0,
    "marker_face_color": "#d62728",
    "marker_edge_color": "white",
    "marker_size": 4.0,
    "head_face_color": "#ff9896",
    "head_edge_color": "#d62728",
}


def _render_skeleton_with_style(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    head_radius: float,
    style: dict,
) -> None:
    """Draw a single HOYO skeleton with the given style."""
    max_idx = len(xs) - 1
    for u, v in SKELETON_EDGES:
        if u <= max_idx and v <= max_idx:
            ax.plot(
                [xs[u], xs[v]],
                [ys[u], ys[v]],
                color=style["line_color"],
                linewidth=style["line_width"],
                zorder=1,
            )

    ax.plot(
        xs,
        ys,
        "o",
        markerfacecolor=style["marker_face_color"],
        markeredgecolor=style["marker_edge_color"],
        markersize=style["marker_size"],
        markeredgewidth=1.0,
        zorder=2,
    )

    # 頭（index=0）を円で強調
    from matplotlib.patches import Circle

    head_x = xs[0]
    head_y = ys[0]
    head = Circle(
        (head_x, head_y),
        radius=head_radius,
        edgecolor=style["head_edge_color"],
        facecolor=style["head_face_color"],
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(head)


def _render_original_vs_recon_gif(
    label: str,
    orig: np.ndarray,
    recon: np.ndarray,
    out_path: Path,
    fps: int = 10,
) -> None:
    """
    1本のシーケンスについて、オリジナル vs 再構成を重ね描きしたGIFを出力する。

    orig, recon: (T, 14, 2) in [y, x] order (same as HOYO dataset representation)
    """
    if imageio is None:
        raise ImportError(
            "imageio がインストールされていません。例: pip install imageio"
        )

    assert orig.shape == recon.shape, f"Shape mismatch: {orig.shape} vs {recon.shape}"

    T, J, C = orig.shape
    assert J == 14 and C == 2

    # 全フレーム・両方を合わせて描画範囲を決める
    stacked = np.concatenate([orig, recon], axis=0)
    y_min = float(stacked[..., 0].min())
    y_max = float(stacked[..., 0].max())
    x_min = float(stacked[..., 1].min())
    x_max = float(stacked[..., 1].max())

    width = max(x_max - x_min, 1e-3)
    height = max(y_max - y_min, 1e-3)
    pad_x = 0.2 * width
    pad_y = 0.2 * height
    head_radius = 0.06 * height

    frames: List[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(4, 4))

    for t in range(T):
        ax.clear()

        coords_o = orig[t]  # (14, 2) [y, x]
        coords_r = recon[t]

        xs_o = coords_o[:, 1]
        ys_o = coords_o[:, 0]
        xs_r = coords_r[:, 1]
        ys_r = coords_r[:, 0]

        _render_skeleton_with_style(ax, xs_o, ys_o, head_radius, ORIG_STYLE)
        _render_skeleton_with_style(ax, xs_r, ys_r, head_radius * 0.95, RECON_STYLE)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}（青=元, 赤=再構成）", fontsize=10)

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.invert_yaxis()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)
        buf = np.roll(buf, -1, axis=2)
        image = buf[..., :3]
        frames.append(image)

    plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[GIF] Saved reconstruction animation for {label} to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export original vs reconstructed HOYO motions as GIFs."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(HOYO_ROOT / "joint_training_results" / "figures"),
        help="Directory to save GIF files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(HOYO_ROOT / "joint_training_results" / "checkpoints" / "motionclip_full_joint_best.pth"),
        help="Path to MotionCLIP full model checkpoint.",
    )
    parser.add_argument(
        "--normalization-stats",
        type=str,
        default=str(HOYO_ROOT / "joint_training_results" / "normalization_stats.json"),
        help="Path to normalization_stats.json used during training.",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=60,
        help="Temporal length (frames) to load from HOYOInstructionDataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the GIFs.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Subset of labels to export (default: all 11).",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=1,
        help="Number of sequences per label to export (first N).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    hoyo_root = HOYO_ROOT

    # 生データ（COM中心・正規化前）
    dataset_raw = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=args.target_len)

    # モデル入力用（学習と同じ正規化を適用）
    from copy import copy

    dataset_norm = copy(dataset_raw)
    dataset_norm.samples_by_label = {k: list(v) for k, v in dataset_raw.samples_by_label.items()}
    stats_path = Path(args.normalization_stats)
    apply_normalization_from_stats(dataset_norm, INSTRUCTION_ONOMATOPEIA, stats_path)

    # 正規化統計（逆変換に使用）
    import json

    with open(stats_path, "r") as f:
        stats = json.load(f)
    data_mean = np.asarray(stats["mean"], dtype=np.float32)
    data_std = np.asarray(stats["std"], dtype=np.float32)

    # モデル読み込み
    model, _ = load_motionclip_full_model(device=device, target_len=args.target_len)
    ckpt_path = Path(args.checkpoint)
    print(f"Loading MotionCLIP checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)

    target_labels = args.labels if args.labels is not None else list(INSTRUCTION_ONOMATOPEIA)

    with torch.no_grad():
        for lab in target_labels:
            if lab not in dataset_raw.samples_by_label:
                print(f"[WARN] Label {lab} not found in dataset, skip.")
                continue

            samples_raw = dataset_raw.samples_by_label[lab]
            samples_norm = dataset_norm.samples_by_label[lab]
            if not samples_raw:
                print(f"[WARN] No samples for label {lab}, skip.")
                continue

            n = min(args.samples_per_label, len(samples_raw))
            for idx in range(n):
                arr_raw = samples_raw[idx]  # (T, 14, 2) [y, x], COM-centered
                arr_norm = samples_norm[idx]

                # (T, 14, 2) -> (1, 14, 2, T)
                coords = arr_norm[np.newaxis, ...].transpose(0, 2, 3, 1)
                x = torch.from_numpy(coords).float().to(device)

                B, J, C, Tcur = x.shape
                mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
                lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)

                batch = {
                    "x": x,
                    "mask": mask,
                    "lengths": lengths,
                    "y": torch.zeros((B,), dtype=torch.long, device=device),
                }

                out = model(batch)
                rec_norm = out.get("output", out.get("rec", None))
                if rec_norm is None:
                    raise RuntimeError("Model output does not contain 'output' or 'rec'.")

                rec_norm_np = rec_norm.cpu().numpy()[0].transpose(2, 0, 1)  # (T, 14, 2)
                # 正規化を元に戻す
                rec_raw = rec_norm_np * data_std + data_mean

                # GIF を出力
                safe_lab = lab.replace("/", "_")
                out_name = f"recon_{safe_lab}_idx{idx}.gif"
                out_path = out_dir / out_name
                _render_original_vs_recon_gif(lab, arr_raw, rec_raw, out_path, fps=args.fps)


if __name__ == "__main__":
    main()




