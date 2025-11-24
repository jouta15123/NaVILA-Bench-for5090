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
    "line_width": 2.0,            # ボーンの太さ
    "marker_face_color": "#d62728", # 関節の色（赤）
    "marker_edge_color": "white", # 関節の縁取り色
    "marker_size": 6,             # 関節の大きさ
    "head_face_color": "#ffcc99", # 頭の色（肌色）
    "head_edge_color": "#333333", # 頭の縁取り色
}
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

    # Assume all samples share same T
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

    # 顔の円の半径（全体の高さに対する比で決める）
    head_radius = 0.08 * height

    # 横方向オフセット（サンプル同士が重ならないように）
    dx = width * 2.0

    frames: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=(3 * n, 4))

    for t in range(T):
        ax.clear()

        for i, arr in enumerate(samples):
            coords = arr[t]  # (14, 2)
            # HOYO は (x, y) だが，見やすさのためプロット上では縦軸を第1成分に，
            # 横軸を第2成分に割り当てる（x, y を入れ替える）
            xs = coords[:, 1] + i * dx
            ys = coords[:, 0]

            # --- 描画スタイルの変更点 ---
            # 点と簡単な線でスティックっぽく描画
            ax.plot(
                xs,
                ys,
                "o-",
                color=STYLE_CONFIG["line_color"],
                linewidth=STYLE_CONFIG["line_width"],
                markerfacecolor=STYLE_CONFIG["marker_face_color"],
                markeredgecolor=STYLE_CONFIG["marker_edge_color"],
                markersize=STYLE_CONFIG["marker_size"],
                alpha=0.9,
                zorder=2
            )

            # プロット座標系（xs, ys）で一番上に来る点を「頭」とみなして円を描画
            head_idx = int(np.argmin(ys))
            head_x = xs[head_idx]
            head_y = ys[head_idx]
            head = Circle(
                (head_x, head_y),
                radius=head_radius,
                edgecolor=STYLE_CONFIG["head_edge_color"],
                facecolor=STYLE_CONFIG["head_face_color"], # 肌色に変更
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(head)
            # ------------------------

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(coarse_label)

        ax.set_xlim(x_min - width, x_min + dx * (n - 0.5))
        ax.set_ylim(y_min - 0.5 * height, y_max + 0.5 * height)
        # 画面上では頭が上，足が下になるように上下を反転
        ax.invert_yaxis()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # FigureCanvasAgg exposes tostring_argb; convert ARGB -> RGB
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)
        # Roll alpha channel to the end to get RGBA, then drop alpha
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

    # Flatten して座標レンジとフレーム数を取得
    all_arrays: list[np.ndarray] = []
    for arrs in groups.values():
        all_arrays.extend(arrs)
    if not all_arrays:
        print("[WARN] No samples for any group, skip combined GIF.")
        return

    T = all_arrays[0].shape[0]
    stacked = np.concatenate(all_arrays, axis=0)  # (G*n, T, 14, 2) → flatten (ここでは (N*T, 14, 2) 相当)
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

                # --- 描画スタイルの変更点 ---
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=STYLE_CONFIG["line_color"],
                    linewidth=STYLE_CONFIG["line_width"],
                    markerfacecolor=STYLE_CONFIG["marker_face_color"],
                    markeredgecolor=STYLE_CONFIG["marker_edge_color"],
                    markersize=STYLE_CONFIG["marker_size"],
                    alpha=0.9,
                    zorder=2
                )

                head_idx = int(np.argmin(ys))
                head_x = xs[head_idx]
                head_y = ys[head_idx]
                head = Circle(
                    (head_x, head_y),
                    radius=head_radius,
                    edgecolor=STYLE_CONFIG["head_edge_color"],
                    facecolor=STYLE_CONFIG["head_face_color"], # 肌色に変更
                    linewidth=1.5,
                    zorder=3,
                )
                ax.add_patch(head)
                # ------------------------

            # 各行の上（アニメーションの上）に coarse スタイル名を表示
            row_top_y = row_idx * dy - 0.4 * height  # 行の上側
            row_center_x = x_min + dx * (n_cols - 1) / 2  # 行の中央
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
        # キャプション用に上側を広げる
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

    # Load HOYO instruction dataset (11 fine labels, COM 中心化＋リサンプリング済み)
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

    # 4スタイルをまとめた比較用 GIF も出力
    if groups:
        combined_path = out_dir / "hoyo_all_styles.gif"
        _render_all_groups_gif(groups, combined_path, fps=args.fps)


if __name__ == "__main__":
    main()