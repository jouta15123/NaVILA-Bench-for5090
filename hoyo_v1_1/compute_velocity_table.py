#!/usr/bin/env python3
"""
HOYO データセットから各オノマトペの平均歩行速度を計算し、
速度テーブルを生成するスクリプト。

移動量（X 方向の変位）をフレーム数で割って速度を推定する。
HOYO のフレームレートは 30fps と仮定。
"""

import json
import os
import pickle
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


def compute_velocity(motion: np.ndarray, fps: float = 30.0):
    """
    モーションから歩行速度を計算。
    
    Args:
        motion: (T, 14, 2) 形状の 2D スケルトン
        fps: フレームレート
    
    Returns:
        velocity: 推定歩行速度 (ピクセル/秒 → 正規化後は無次元)
    """
    T = motion.shape[0]
    if T < 2:
        return 0.0
    
    # 重心の軌跡を取得
    # HOYO は [y, x] 形式（画像座標系）
    center_traj = motion.mean(axis=1)  # (T, 2)
    
    # X 方向の移動量（歩行方向）
    # index 1 が X（画像の横方向）
    x_start = center_traj[0, 1]
    x_end = center_traj[-1, 1]
    x_displacement = abs(x_end - x_start)
    
    # 時間
    duration = T / fps
    
    # 速度（ピクセル/秒）
    velocity = x_displacement / duration if duration > 0 else 0.0
    
    return velocity


def compute_normalized_velocity(motion: np.ndarray, fps: float = 30.0):
    """
    身長で正規化した速度を計算（身長/秒）
    """
    T = motion.shape[0]
    if T < 2:
        return 0.0
    
    # HOYO は [y, x] 形式
    # 頭(0) と 足(10, 13) の距離で身長を推定
    head_pos = motion[:, 0, :]  # (T, 2)
    feet_pos = 0.5 * (motion[:, 10, :] + motion[:, 13, :])  # (T, 2)
    heights = np.linalg.norm(head_pos - feet_pos, axis=-1)  # (T,)
    mean_height = heights.mean()
    
    if mean_height < 1e-6:
        return 0.0
    
    # 重心の X 方向移動量
    center_traj = motion.mean(axis=1)
    x_displacement = abs(center_traj[-1, 1] - center_traj[0, 1])
    
    # 身長で正規化した移動量
    normalized_displacement = x_displacement / mean_height
    
    # 時間
    duration = T / fps
    
    # 速度（身長/秒）
    velocity = normalized_displacement / duration if duration > 0 else 0.0
    
    return velocity


def main():
    data_dir = Path(__file__).parent / "data"
    
    # 全ファイルを読み込んでオノマトペごとに集計
    velocities_by_label = defaultdict(list)
    
    # ID の範囲を調べる
    json_files = list(data_dir.glob("*.json"))
    file_ids = sorted([int(f.stem) for f in json_files])
    
    print(f"Loading {len(file_ids)} HOYO files...")
    
    for file_id in file_ids:
        meta, motion = load_hoyo_motion(data_dir, file_id)
        if meta is None or motion is None:
            continue
        
        # instruction ラベルを取得
        annotation = meta.get("annotation", {})
        instruction = annotation.get("instruction")
        if not instruction:
            continue
        
        # 速度を計算
        velocity = compute_normalized_velocity(motion)
        velocities_by_label[instruction].append(velocity)
    
    # 統計を計算
    print("\n" + "=" * 60)
    print("HOYO 歩行速度分析（身長/秒）")
    print("=" * 60)
    
    speed_table = {}
    
    for label in sorted(velocities_by_label.keys()):
        vels = velocities_by_label[label]
        mean_vel = np.mean(vels)
        std_vel = np.std(vels)
        min_vel = np.min(vels)
        max_vel = np.max(vels)
        
        print(f"{label:12s}: mean={mean_vel:.3f}, std={std_vel:.3f}, "
              f"range=[{min_vel:.3f}, {max_vel:.3f}], n={len(vels)}")
        
        speed_table[label] = round(mean_vel, 3)
    
    # H1 ロボット用にスケーリング
    # H1 の歩行速度は 0.3〜0.8 m/s くらいが自然
    # HOYO の身長/秒 → H1 の m/s への変換係数を決める
    # 
    # 仮定: 「通常」= 0.5 m/s として、他をスケール
    print("\n" + "=" * 60)
    print("H1 ロボット用速度テーブル（m/s）")
    print("=" * 60)
    
    baseline_label = "通常"
    baseline_speed = 0.5  # m/s
    
    if baseline_label in speed_table and speed_table[baseline_label] > 0:
        scale_factor = baseline_speed / speed_table[baseline_label]
    else:
        scale_factor = 1.0
    
    h1_speed_table = {}
    for label, vel in speed_table.items():
        h1_vel = vel * scale_factor
        # クランプ
        h1_vel = max(0.15, min(1.0, h1_vel))
        h1_speed_table[label] = float(round(h1_vel, 2))  # float32 → float
        print(f"{label:12s}: {h1_vel:.2f} m/s")
    
    # JSON 出力
    output_path = Path(__file__).parent.parent / "configs" / "style_speed_table_auto.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(h1_speed_table, f, indent=4, ensure_ascii=False)
    
    print(f"\n[INFO] Speed table saved to: {output_path}")


if __name__ == "__main__":
    main()
