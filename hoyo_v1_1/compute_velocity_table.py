#!/usr/bin/env python3
"""
HOYO データセットから各オノマトペの相対歩行速度を計算し、速度テーブルを生成するスクリプト。

- 2D keypoints から身長スケール（head - mid_feet）を算出
- log(身長) の時間微分で「カメラに近づく」方向の相対速度を推定
- H1 m/s への変換は quantile マッピング（張り付き防止）。必要なら baseline も使える
"""

import json
import pickle
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_hoyo_motion(data_dir: Path, file_id: int):
    """指定 ID の HOYO モーションを読み込む"""
    json_path = data_dir / f"{file_id}.json"
    pickle_path = data_dir / f"{file_id}.pickle"

    if not json_path.exists() or not pickle_path.exists():
        return None, None

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(pickle_path, "rb") as f:
        motion = pickle.load(f)

    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 3 or motion.shape[1] != 14 or motion.shape[2] != 2:
        return meta, None

    return meta, motion


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

def _smooth_trajectory(traj: np.ndarray, window: int) -> np.ndarray:
    """移動平均で軽く平滑化."""
    if window <= 1:
        return traj
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(traj, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(traj)
    for i in range(traj.shape[1]):
        smoothed[:, i] = np.convolve(padded[:, i], kernel, mode="valid")
    return smoothed


def compute_height_ref(motion: np.ndarray, height_percentile: float = 90.0) -> float:
    """
    身長っぽいスケールを推定（ロバスト）。
    頭(0) と 足(10, 13) の距離を取り、percentile で基準化。
    """
    head_pos = motion[:, 0, :]
    feet_pos = 0.5 * (motion[:, 10, :] + motion[:, 13, :])
    heights = np.linalg.norm(head_pos - feet_pos, axis=-1)

    heights = heights[np.isfinite(heights)]
    if heights.size == 0:
        return 0.0

    h_ref = float(np.percentile(heights, height_percentile))
    return h_ref


def compute_scale_speed_per_s(
    motion: np.ndarray,
    fps: float = 60.0,
    smooth_window: int = 5,
    height_percentile: float = 90.0,
    min_height_ratio: float = 0.2,
) -> float:
    """
    スケール変化（log身長）から奥行き方向の相対速度を推定。
    近づくほど head-feet の距離が大きくなる前提。
    """
    T = motion.shape[0]
    if T < 2:
        return 0.0

    # 身長スケール（head - mid_feet）
    head_pos = motion[:, 0, :]
    feet_pos = 0.5 * (motion[:, 10, :] + motion[:, 13, :])
    heights = np.linalg.norm(head_pos - feet_pos, axis=-1)

    heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
    if heights.size == 0:
        return 0.0

    # 極端な欠損値を避けるため、基準の一定比率以下は除外
    h_ref = float(np.percentile(heights[heights > 0], height_percentile)) if np.any(heights > 0) else 0.0
    if h_ref <= 0.0:
        return 0.0
    min_h = max(1e-6, h_ref * float(min_height_ratio))
    heights = np.clip(heights, min_h, None)

    # 身長スケールを平滑化してから log 変換
    if smooth_window > 1:
        heights = _smooth_trajectory(heights[:, None], smooth_window).squeeze(-1)
    log_h = np.log(heights).astype(np.float32)

    # 全体トレンドの傾き（線形回帰）で速度を推定
    log_h = log_h[np.isfinite(log_h)]
    if log_h.size < 2:
        return 0.0
    t = np.arange(log_h.size, dtype=np.float32) / float(fps)
    A = np.stack([t, np.ones_like(t)], axis=1)
    slope, _ = np.linalg.lstsq(A, log_h, rcond=None)[0]
    speed = float(slope)
    if not np.isfinite(speed):
        return 0.0
    return float(max(0.0, speed))


def quantile_map(values: dict[str, float], q_lo: float, q_hi: float, t_lo: float, t_hi: float):
    """
    dict の値（raw）を quantile で線形変換して目標レンジ [t_lo, t_hi] に合わせる。
    """
    labels = list(values.keys())
    raw = np.array([values[k] for k in labels], dtype=np.float32)

    if raw.size == 0:
        return {}

    lo, hi = np.percentile(raw, [q_lo, q_hi])
    lo = float(lo); hi = float(hi)

    if abs(hi - lo) < 1e-8:
        a = 1.0
        b = 0.0
    else:
        a = (t_hi - t_lo) / (hi - lo)
        b = t_lo - a * lo

    mapped = {}
    for k in labels:
        mapped[k] = float(a * values[k] + b)
    return mapped


def main():
    parser = argparse.ArgumentParser(description="Compute HOYO speed table from 2D keypoints")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--height-percentile", type=float, default=90.0)
    parser.add_argument("--min-height-ratio", type=float, default=0.2,
                        help="scaleモード時の最小身長比（欠損対策）")

    # H1変換モード
    parser.add_argument("--map-mode", type=str, default="quantile",
                        choices=["quantile", "baseline", "relative"],
                        help="マッピング方法: quantile(おすすめ) / baseline / relative(基準=1.0)")
    # quantile mapping params
    parser.add_argument("--q-lo", type=float, default=20.0)
    parser.add_argument("--q-hi", type=float, default=80.0)
    parser.add_argument("--t-lo", type=float, default=0.25)
    parser.add_argument("--t-hi", type=float, default=0.85)

    # baseline mapping params
    parser.add_argument("--baseline-label", type=str, default="通常")
    parser.add_argument("--baseline-speed", type=float, default=0.5, help="Baseline speed in m/s")
    parser.add_argument("--relative-to", type=str, default="baseline",
                        choices=["baseline", "max", "percentile"],
                        help="relative の基準: baseline(指定ラベル) / max(最速=1.0) / percentile")
    parser.add_argument("--relative-percentile", type=float, default=95.0,
                        help="relative_to=percentile の基準パーセンタイル (例: 95)")
    parser.add_argument("--relative-digits", type=int, default=3,
                        help="relative 出力の小数点桁数")

    # clamp
    parser.add_argument("--min-speed", type=float, default=0.15)
    parser.add_argument("--max-speed", type=float, default=1.0)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    velocities_by_label = defaultdict(list)

    json_files = list(data_dir.glob("*.json"))
    file_ids = sorted([int(f.stem) for f in json_files])

    print(f"Loading {len(file_ids)} HOYO files... (fps={args.fps}, smooth={args.smooth_window}, "
          f"h_pctl={args.height_percentile}, map={args.map_mode})")

    for file_id in file_ids:
        meta, motion = load_hoyo_motion(data_dir, file_id)
        if meta is None or motion is None:
            continue

        annotation = meta.get("annotation", {})
        instruction = annotation.get("instruction")
        if not instruction:
            continue

        v = compute_scale_speed_per_s(
            motion,
            fps=args.fps,
            smooth_window=args.smooth_window,
            height_percentile=args.height_percentile,
            min_height_ratio=args.min_height_ratio,
        )
        # 変な値を除外
        if np.isfinite(v) and v >= 0.0:
            velocities_by_label[instruction].append(float(v))

    print("\n" + "=" * 60)
    print("HOYO 歩行速度分析（height/sec, 前方: 総移動/時間）")
    print("=" * 60)

    speed_table = {}
    for label in sorted(velocities_by_label.keys()):
        vels = np.array(velocities_by_label[label], dtype=np.float32)
        if vels.size == 0:
            continue
        mean_vel = float(np.mean(vels))
        std_vel = float(np.std(vels))
        min_vel = float(np.min(vels))
        max_vel = float(np.max(vels))

        print(f"{label:12s}: mean={mean_vel:.3f}, std={std_vel:.3f}, "
              f"range=[{min_vel:.3f}, {max_vel:.3f}], n={vels.size}")

        rep_vel = float(np.median(vels))
        speed_table[label] = float(round(rep_vel, 6))  # rawは丸めすぎない

    print("\n" + "=" * 60)
    if args.map_mode == "relative":
        print("相対速度テーブル（基準=1.0）")
    else:
        print("H1 ロボット用速度テーブル（m/s）")
    print("=" * 60)

    if args.map_mode == "baseline":
        baseline_label = args.baseline_label
        baseline_speed = float(args.baseline_speed)
        base = float(speed_table.get(baseline_label, 0.0))
        scale_factor = (baseline_speed / base) if base > 1e-8 else 1.0
        mapped = {k: v * scale_factor for k, v in speed_table.items()}
        print(f"[INFO] baseline mapping: label='{baseline_label}', "
              f"raw={base:.6f} -> {baseline_speed:.2f} m/s, scale={scale_factor:.3f}")
    elif args.map_mode == "relative":
        if args.relative_to == "max":
            base = float(max(speed_table.values())) if speed_table else 0.0
            if base > 1e-8:
                mapped = {k: v / base for k, v in speed_table.items()}
            else:
                mapped = {k: 0.0 for k in speed_table}
            print(f"[INFO] relative mapping: max raw={base:.6f} -> 1.00x")
        elif args.relative_to == "percentile":
            vals = np.array(list(speed_table.values()), dtype=np.float32) if speed_table else np.array([])
            vals = vals[vals > 1e-8]
            base = float(np.percentile(vals, args.relative_percentile)) if vals.size else 0.0
            if base > 1e-8:
                mapped = {k: v / base for k, v in speed_table.items()}
            else:
                mapped = {k: 0.0 for k in speed_table}
            print(f"[INFO] relative mapping: p{args.relative_percentile:.0f} raw={base:.6f} -> 1.00x")
        else:
            baseline_label = args.baseline_label
            base = float(speed_table.get(baseline_label, 0.0))
            if base > 1e-8:
                mapped = {k: v / base for k, v in speed_table.items()}
            else:
                mapped = {k: 0.0 for k in speed_table}
            print(f"[INFO] relative mapping: label='{baseline_label}', raw={base:.6f} -> 1.00x")
    else:
        mapped = quantile_map(
            speed_table,
            q_lo=float(args.q_lo),
            q_hi=float(args.q_hi),
            t_lo=float(args.t_lo),
            t_hi=float(args.t_hi),
        )
        print(f"[INFO] quantile mapping: q{args.q_lo:.0f}->{args.t_lo:.2f} m/s, "
              f"q{args.q_hi:.0f}->{args.t_hi:.2f} m/s")

    if args.map_mode == "relative":
        # relative: clampしない（単位なしの倍率）
        h1_speed_table = {}
        for label, vel in mapped.items():
            rel = float(round(vel, args.relative_digits))
            h1_speed_table[label] = rel
            print(f"{label:12s}: {rel:.{args.relative_digits}f} x")
        output_path = Path(__file__).parent.parent / "configs" / "style_speed_table_relative.json"
    else:
        # clamp & round
        h1_speed_table = {}
        for label, vel in mapped.items():
            h1_vel = float(np.clip(vel, args.min_speed, args.max_speed))
            h1_speed_table[label] = float(round(h1_vel, 2))
            print(f"{label:12s}: {h1_vel:.2f} m/s")
        output_path = Path(__file__).parent.parent / "configs" / "style_speed_table_auto.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(h1_speed_table, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Speed table saved to: {output_path}")


if __name__ == "__main__":
    main()
