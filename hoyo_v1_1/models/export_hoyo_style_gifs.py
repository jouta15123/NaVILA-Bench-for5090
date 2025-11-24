import argparse
import random
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語ラベル用
from matplotlib.patches import Circle


try:
    import imageio.v2 as imageio
except Exception as e:  # pragma: no cover - optional dependency
    imageio = None


# Ensure project root is on sys.path so that `hoyo_v1_1` can be imported
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    INSTRUCTION_ONOMATOPEIA,
)


# Coarse style groups (same定義 as joint training; 「通常」は含めない)
COARSE_GROUPS = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["とぼとぼ", "のろのろ"],
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}

COARSE_SLUGS = {
    "速い系": "fast",
    "遅い系": "slow",
    "重い系": "heavy",
    "ふらふら系": "wobbly",
}

# --- 視認性向上のためのスタイル定義 ---
STYLE_CONFIG = {
    "line_color": "#333333",      # ボーンの色（濃いグレー）
    "line_width": 2.5,            # ボーンの太さ
    "marker_face_color": "#d62728", # 関節の色（赤）
    "marker_edge_color": "white", # 関節の縁取り色
    "marker_size": 5,             # 関節の大きさ
    "head_face_color": "#ffcc99", # 頭の色（肌色）
    "head_edge_color": "#333333", # 頭の縁取り色
}

# ★ここをHOYOデータセットの定義に合わせて修正したで！
# HOYO定義: [頭(0), 首(1), 右肩(2), 右肘(3), 右手(4), 左肩(5), 左肘(6), 左手(7), 右腰(8), 右膝(9), 右足(10), 左腰(11), 左膝(12), 左足(13)]
SKELETON_EDGES = [
    # 上半身
    [0, 1],   # 頭 -> 首
    [1, 2],   # 首 -> 右肩
    [2, 3],   # 右肩 -> 右肘
    [3, 4],   # 右肘 -> 右手
    [1, 5],   # 首 -> 左肩
    [5, 6],   # 左肩 -> 左肘
    [6, 7],   # 左肘 -> 左手
    
    # 胴体（首から腰へ）
    [1, 8],   # 首 -> 右腰
    [1, 11],  # 首 -> 左腰
    [8, 11],  # 右腰 -> 左腰（骨盤を閉じる）

    # 下半身
    [8, 9],   # 右腰 -> 右膝
    [9, 10],  # 右膝 -> 右足
    [11, 12], # 左腰 -> 左膝
    [12, 13], # 左膝 -> 左足
]
# ------------------------------------


def _select_samples_for_group(
    dataset: HoyoInstructionDataset,
    coarse_label: str,
    n_samples: int,
    rng: random.Random,
) -> list[tuple[np.ndarray, str]]:
    """Pick up to n_samples sequences (T, 14, 2) for a given coarse style."""
    fine_labels = COARSE_GROUPS[coarse_label]
    all_samples: list[tuple[np.ndarray, str]] = []

    for fine_lab in fine_labels:
        for arr in dataset.samples_by_label.get(fine_lab, []):
            all_samples.append((arr, fine_lab))

    if not all_samples:
        return []

    if len(all_samples) <= n_samples:
        return all_samples

    return rng.sample(all_samples, n_samples)


def _render_skeleton(ax, xs, ys, head_radius):
    """
    HOYO仕様に合わせてスケルトンを描画する
    """
    
    # 1. ボーン（線）を描画
    max_idx = len(xs) - 1
    for u, v in SKELETON_EDGES:
        if u <= max_idx and v <= max_idx:
            ax.plot(
                [xs[u], xs[v]], 
                [ys[u], ys[v]], 
                color=STYLE_CONFIG["line_color"],
                linewidth=STYLE_CONFIG["line_width"],
                zorder=1
            )

    # 2. 関節（点）を描画
    ax.plot(
        xs,
        ys,
        "o",
        markerfacecolor=STYLE_CONFIG["marker_face_color"],
        markeredgecolor=STYLE_CONFIG["marker_edge_color"],
        markersize=STYLE_CONFIG["marker_size"],
        markeredgewidth=1.0,
        zorder=2
    )

    # 3. 頭を描画
    # HOYOでは index=0 が「頭」と決まっているので、座標探索せずに0番を使う
    head_x = xs[0]
    head_y = ys[0]
    
    head = Circle(
        (head_x, head_y),
        radius=head_radius,
        edgecolor=STYLE_CONFIG["head_edge_color"],
        facecolor=STYLE_CONFIG["head_face_color"],
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(head)


def _render_group_gif(
    coarse_label: str,
    samples: list[np.ndarray],
    out_path: Path,
    fps: int = 10,
) -> None:
    """Render複数サンプルを横並びにした簡易 2D アニメーション GIF."""
    if imageio is None:
        raise ImportError(
            "imageio がインストールされていません。"
            " 例: pip install imageio"
        )

    if not samples:
        print(f"[WARN] No samples for coarse label {coarse_label}, skip.")
        return

    T = samples[0].shape[0]
    n = len(samples)

    # 全サンプル・全フレームで座標範囲を集計
    stacked = np.concatenate(samples, axis=0)  # (n*T, 14, 2)
    x_min = stacked[..., 0].min()
    x_max = stacked[..., 0].max()
    y_min = stacked[..., 1].min()
    y_max = stacked[..., 1].max()

    width = max(x_max - x_min, 1e-3)
    height = max(y_max - y_min, 1e-3)

    head_radius = 0.08 * height
    dx = width * 2.0

    frames: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=(3 * n, 4))

    for t in range(T):
        ax.clear()

        for i, arr in enumerate(samples):
            coords = arr[t]  # (14, 2)
            # HOYOデータは [y, x] なので、plot用に xs=col1, ys=col0 にする
            xs = coords[:, 1] + i * dx
            ys = coords[:, 0]
            
            _render_skeleton(ax, xs, ys, head_radius)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(coarse_label)

        ax.set_xlim(x_min - width, x_min + dx * (n - 0.5))
        ax.set_ylim(y_min - 0.5 * height, y_max + 0.5 * height)
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
    print(f"[GIF] Saved {coarse_label} animation to {out_path}")


def _render_all_groups_gif(
    groups: dict[str, list[np.ndarray]],
    out_path: Path,
    fps: int = 10,
) -> None:
    """4スタイルすべてを 1 枚の GIF にまとめて描画する."""
    if imageio is None:
        raise ImportError(
            "imageio がインストールされていません。"
            " 例: pip install imageio"
        )

    all_arrays: list[np.ndarray] = []
    for arrs in groups.values():
        all_arrays.extend(arrs)
    if not all_arrays:
        print("[WARN] No samples for any group, skip combined GIF.")
        return

    T = all_arrays[0].shape[0]
    stacked = np.concatenate(all_arrays, axis=0)
    x_min = stacked[..., 0].min()
    x_max = stacked[..., 0].max()
    y_min = stacked[..., 1].min()
    y_max = stacked[..., 1].max()

    width = max(x_max - x_min, 1e-3)
    height = max(y_max - y_min, 1e-3)
    head_radius = 0.08 * height

    n_rows = len(groups)
    n_cols = max(len(v) for v in groups.values())

    dx = width * 2.0
    dy = height * 2.0

    frames: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=(3 * n_cols, 4 * n_rows))

    coarse_labels_order = list(COARSE_GROUPS.keys())
    row_labels = [lab for lab in coarse_labels_order if lab in groups]

    for t in range(T):
        ax.clear()

        for row_idx, coarse_label in enumerate(row_labels):
            arrs = groups[coarse_label]
            for col_idx, arr in enumerate(arrs):
                coords = arr[t]  # (14, 2)
                xs = coords[:, 1] + col_idx * dx
                ys = coords[:, 0] + row_idx * dy

                _render_skeleton(ax, xs, ys, head_radius)

            row_top_y = row_idx * dy - 0.4 * height
            row_center_x = x_min + dx * (n_cols - 1) / 2
            ax.text(
                row_center_x,
                row_top_y,
                coarse_label,
                fontsize=12,
                ha="center",
                va="bottom",
                color="black",
                weight="bold",
            )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(x_min - width, x_min + dx * (n_cols - 0.5))
        ax.set_ylim(y_min - 0.8 * height, y_min + dy * (n_rows - 0.5))
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
    print(f"[GIF] Saved ALL styles animation to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HOYO coarse 4-style walking animations as GIFs."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="tex/fig",
        help="Directory to save GIF files.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
        help="Number of sequences per coarse style to include in each GIF.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the GIFs.",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=60,
        help="Temporal length (frames) to load from HOYOInstructionDataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling sequences.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    hoyo_root = repo_root / "hoyo_v1_1"

    dataset = HoyoInstructionDataset(
        hoyo_root,
        INSTRUCTION_ONOMATOPEIA,
        target_len=args.target_len,
    )

    out_dir = Path(args.out_dir)

    groups: dict[str, list[np.ndarray]] = {}

    for coarse_label, slug in COARSE_SLUGS.items():
        selected = _select_samples_for_group(
            dataset,
            coarse_label,
            args.samples_per_class,
            rng,
        )
        if not selected:
            print(f"[WARN] No data for {coarse_label}, skipping.")
            continue

        arrays = [arr for (arr, _) in selected]
        groups[coarse_label] = arrays
        gif_path = out_dir / f"hoyo_{slug}_group.gif"
        _render_group_gif(coarse_label, arrays, gif_path, fps=args.fps)

    if groups:
        combined_path = out_dir / "hoyo_all_styles.gif"
        _render_all_groups_gif(groups, combined_path, fps=args.fps)


if __name__ == "__main__":
    main()