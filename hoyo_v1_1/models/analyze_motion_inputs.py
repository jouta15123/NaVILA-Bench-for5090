import argparse
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np

# Make project root importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    INSTRUCTION_ONOMATOPEIA,
)


def compute_basic_stats(dataset: HoyoInstructionDataset, labels: List[str]) -> Dict[str, dict]:
    """
    Compute per-label basic statistics of the motion inputs:
    - mean / std of coordinates
    - mean per-frame speed (L2 over joints)
    - sequence length distribution
    """
    stats: Dict[str, dict] = {}

    for lab in labels:
        samples = dataset.samples_by_label.get(lab, [])
        if not samples:
            stats[lab] = {"n": 0}
            continue

        coords_all = np.stack(samples, axis=0)  # (N, T, J, C)
        N, T, J, C = coords_all.shape

        # basic position stats
        mean = coords_all.mean(axis=(0, 1, 2))  # (C,)
        std = coords_all.std(axis=(0, 1, 2))  # (C,)

        # per-frame speed: ||x_t+1 - x_t|| over joints, averaged
        diffs = coords_all[:, 1:, :, :] - coords_all[:, :-1, :, :]  # (N, T-1, J, C)
        speed = np.linalg.norm(diffs, axis=-1).mean(axis=(0, 2))  # (T-1,)
        avg_speed = float(speed.mean())

        stats[lab] = {
            "n": int(N),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "avg_speed": avg_speed,
            "T": int(T),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze HOYO motion inputs going into MotionCLIP.")
    parser.add_argument(
        "--root",
        type=str,
        default="hoyo_v1_1",
        help="Root directory of HOYO data (where data/*.json, *.pickle live).",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=60,
        help="Target sequence length used in training.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    dataset = HoyoInstructionDataset(root, INSTRUCTION_ONOMATOPEIA, target_len=args.target_len)

    stats = compute_basic_stats(dataset, INSTRUCTION_ONOMATOPEIA)
    print("=== Motion input statistics (before normalization) ===")
    for lab in INSTRUCTION_ONOMATOPEIA:
        s = stats.get(lab, {})
        if not s or s.get("n", 0) == 0:
            print(f"- {lab}: no samples")
            continue
        print(
            f"- {lab}: n={s['n']}, T={s['T']}, "
            f"mean={np.round(s['mean'], 4)}, std={np.round(s['std'], 4)}, "
            f"avg_speed={s['avg_speed']:.4f}"
        )


if __name__ == "__main__":
    main()


