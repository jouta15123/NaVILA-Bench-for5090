"""
HOYOデータセットの2Dスケルトンをアニメーション（GIF）で可視化するスクリプト

使い方の例:
  - instruction名で1本選んで可視化:
      python animate_hoyo_dataset.py --instruction "とぼとぼ" --out walk.gif --fps 60

  - 直接JSONを指定して可視化:
      python animate_hoyo_dataset.py --json ".\\hoyo_v1_1\\hoyo_v1_1\\000001_front.json" --out walk.gif --fps 60
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

matplotlib.use("Agg")


# 14関節の名前（参考）
HOYO_JOINT_NAMES = [
    "頭",
    "首",
    "右肩",
    "右肘",
    "右手",
    "左肩",
    "左肘",
    "左手",
    "右腰",
    "右膝",
    "右足",
    "左腰",
    "左膝",
    "左足",
]

# 描画用のボーン接続（左右と中心を意識）
# 中央: 頭-首, 肩帯(右肩-左肩), 骨盤(右腰-左腰)
# 右腕: 首-右肩-右肘-右手, 右脚: 首-右腰-右膝-右足
# 左腕: 首-左肩-左肘-左手, 左脚: 首-左腰-左膝-左足
HOYO_EDGES = [
    (0, 1),  # 頭-首
    (1, 2),
    (2, 3),
    (3, 4),  # 右腕
    (1, 5),
    (5, 6),
    (6, 7),  # 左腕
    (1, 8),
    (8, 9),
    (9, 10),  # 右脚
    (1, 11),
    (11, 12),
    (12, 13),  # 左脚
    (2, 5),  # 両肩を結ぶ
    (8, 11),  # 両腰を結ぶ
]
RIGHT_JOINTS = {2, 3, 4, 8, 9, 10}
LEFT_JOINTS = {5, 6, 7, 11, 12, 13}
CENTER_JOINTS = {0, 1}


def _load_hoyo_motion_from_json(json_path: Path):
    """
    指定JSONからメタデータと2Dスケルトン配列をロード
    返り値:
      metadata: dict（annotation 等を含む）
      motion: np.ndarray, 形状 (T, 14, 2) 予想
    """
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    pickle_path = json_path.parent / metadata["path"]
    with open(pickle_path, "rb") as f:
        motion = pickle.load(f)
    motion = np.asarray(motion)
    if motion.ndim != 3 or motion.shape[1] != 14 or motion.shape[2] != 2:
        raise ValueError(f"想定外の形状です: {motion.shape}（期待: (T,14,2)）")
    return metadata, motion


def _pick_sample_by_instruction(hoyo_dir: Path, instruction: str) -> Path:
    """
    指定オノマトペ(instruction)に一致する最初のサンプルJSONパスを返す
    見つからなければ例外
    """
    for json_file in sorted(hoyo_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("annotation", {}).get("instruction") == instruction:
            return json_file
    raise FileNotFoundError(f"'{instruction}' のサンプルが見つからへんかったで: {hoyo_dir}")


def animate_hoyo_sample(
    json_path: Path = None,
    instruction: str = None,
    hoyo_dir: Path = Path("hoyo_v1_1/hoyo_v1_1"),
    out_gif: Path = Path("hoyo_sample.gif"),
    fps: int = 30,
    linewidth: float = 3.0,
    joint_size: float = 20.0,
    invert_y: bool = True,
    swap_xy: bool = False,
):
    """
    HOYOの2DスケルトンをGIFでアニメ出力
      - json_path か instruction のどちらかでサンプルを指定
    """
    if json_path is None and instruction is None:
        raise ValueError("json_path か instruction のどっちかは指定してな。")
    if json_path is None:
        json_path = _pick_sample_by_instruction(Path(hoyo_dir), instruction)
    json_path = Path(json_path)

    metadata, motion = _load_hoyo_motion_from_json(json_path)
    T = motion.shape[0]
    # readme に従い、座標は [y, x] の順で格納されている
    ys = motion[:, :, 0]
    xs = motion[:, :, 1]
    # 念のため、XYを入れ替えて試したい場合用のオプション
    if swap_xy:
        xs, ys = ys, xs

    # 描画範囲を全フレームから自動決定（少しマージン）
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    pad_x = max(1e-6, 0.05 * (x_max - x_min))
    pad_y = max(1e-6, 0.05 * (y_max - y_min))

    # 色分け（右=赤、左=青、中心=黒）
    def edge_color(a, b):
        if a in RIGHT_JOINTS and b in RIGHT_JOINTS:
            return "#D14A61"  # red系
        if a in LEFT_JOINTS and b in LEFT_JOINTS:
            return "#3B73B9"  # blue系
        return "#333333"  # center/mix

    joint_colors = []
    for j in range(14):
        if j in RIGHT_JOINTS:
            joint_colors.append("#E57373")
        elif j in LEFT_JOINTS:
            joint_colors.append("#64B5F6")
        else:
            joint_colors.append("#212121")

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    ax.set_aspect("equal")
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    if invert_y:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    title = (
        f"{metadata.get('annotation', {}).get('instruction', '')} "
        f"(person={metadata.get('person')}, view={metadata.get('view')}, "
        f"len={metadata.get('length')})"
    )
    ax.set_title(title, fontsize=10)

    # アーティストを作成
    lines = []
    for (a, b) in HOYO_EDGES:
        ln, = ax.plot([], [], lw=linewidth, color=edge_color(a, b))
        lines.append(ln)
    # 初期化時の色数と点数の不一致を避けるため、1フレーム目で初期化
    scat = ax.scatter(xs[0], ys[0], s=joint_size, c=joint_colors, zorder=3)

    def init():
        for ln, (a, b) in zip(lines, HOYO_EDGES):
            ln.set_data([xs[0, a], xs[0, b]], [ys[0, a], ys[0, b]])
        scat.set_offsets(np.column_stack([xs[0], ys[0]]))
        return lines + [scat]

    def update(i):
        # エッジ更新
        for ln, (a, b) in zip(lines, HOYO_EDGES):
            ln.set_data([xs[i, a], xs[i, b]], [ys[i, a], ys[i, b]])
        # ジョイント更新
        scat.set_offsets(np.column_stack([xs[i], ys[i]]))
        return lines + [scat]

    interval_ms = 1000.0 / max(1, fps)
    ani = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval_ms, blit=True)

    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_gif), writer="pillow", fps=fps)
    plt.close(fig)

    print(f"\n✅ アニメGIFを保存したで: {out_gif}")
    return str(out_gif)


def main():
    parser = argparse.ArgumentParser(description="HOYO: 2Dスケルトンのアニメ可視化（GIF）")
    parser.add_argument(
        "--hoyo_dir",
        type=str,
        default="hoyo_v1_1/hoyo_v1_1",
        help="HOYOのディレクトリ（*.json と *.pickle がある階層）",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="可視化対象のJSONパス（これが優先される）",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="instruction で1本サンプルを選ぶ（例: とぼとぼ）",
    )
    parser.add_argument("--out", type=str, default="hoyo_sample.gif", help="出力GIFパス")
    parser.add_argument("--fps", type=int, default=30, help="GIFのFPS")
    parser.add_argument(
        "--swap_xy",
        action="store_true",
        help="XYを入れ替えて描画（デバッグ用・通常は不要）",
    )

    args = parser.parse_args()

    json_path = Path(args.json) if args.json else None
    animate_hoyo_sample(
        json_path=json_path,
        instruction=args.instruction,
        hoyo_dir=Path(args.hoyo_dir),
        out_gif=Path(args.out),
        fps=args.fps,
        swap_xy=args.swap_xy,
    )


if __name__ == "__main__":
    main()


