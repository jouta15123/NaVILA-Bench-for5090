#!/usr/bin/env python3
"""
HOYOデータセット内の教師モーション同士のDTW誤差を計算し、ベースラインを確立する。
"""

import sys
import os
import numpy as np
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
NAVILA_ROOT = REPO_ROOT.parent

sys.path.insert(0, str(NAVILA_ROOT))

from hoyo_v1_1.models.common import HoyoInstructionDataset, INSTRUCTION_ONOMATOPEIA


def _frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: (14, 2) -> mean L2 distance across joints"""
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band: int = 10) -> float:
    """DTW distance with Sakoe-Chiba band constraint."""
    T = seq_a.shape[0]
    U = seq_b.shape[0]
    dp = np.full((T + 1, U + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        j_start = max(1, i - band)
        j_end = min(U, i + band)
        for j in range(j_start, j_end + 1):
            cost = _frame_distance(seq_a[i - 1], seq_b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[T, U] / max(1, T + U))


def compute_baseline_errors():
    """各オノマトペについて、同じラベル内の教師モーション同士のDTW誤差を計算"""

    hoyo_root = NAVILA_ROOT / "hoyo_v1_1"

    dataset = HoyoInstructionDataset(
        root=hoyo_root,
        target_labels=INSTRUCTION_ONOMATOPEIA,
        target_len=60,
        is_train=False,
        use_aug=False,
    )

    print("=" * 60)
    print("HOYO教師モーション同士のDTW誤差（ベースライン）")
    print("=" * 60)
    print()

    results = {}

    for label in INSTRUCTION_ONOMATOPEIA:
        # 同じラベルのサンプルを複数取得
        samples = []
        for _ in range(10):  # 10サンプル取得
            try:
                sample = dataset.get_sample(label)  # (T, 14, 2)
                samples.append(sample)
            except Exception as e:
                print(f"[WARN] Failed to get sample for {label}: {e}")
                break

        if len(samples) < 2:
            print(f"{label}: サンプル不足（{len(samples)}サンプル）")
            continue

        # 同じラベル内のペアでDTW誤差を計算
        errors = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                err = dtw_distance(samples[i], samples[j])
                errors.append(err)

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        min_err = np.min(errors)
        max_err = np.max(errors)

        results[label] = {
            "mean": mean_err,
            "std": std_err,
            "min": min_err,
            "max": max_err,
            "n_pairs": len(errors),
        }

        print(f"{label}: mean={mean_err:.4f}, std={std_err:.4f}, min={min_err:.4f}, max={max_err:.4f} (n={len(errors)})")

    print()
    print("=" * 60)
    print("統計サマリー")
    print("=" * 60)

    all_means = [r["mean"] for r in results.values()]
    print(f"全オノマトペの平均誤差: {np.mean(all_means):.4f}")
    print(f"全オノマトペの誤差範囲: {np.min(all_means):.4f} ~ {np.max(all_means):.4f}")

    return results


if __name__ == "__main__":
    compute_baseline_errors()
