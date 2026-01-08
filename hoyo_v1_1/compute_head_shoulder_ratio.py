#!/usr/bin/env python3
"""Compute HOYO head-to-shoulder distance ratio after canonical scaling.

This script loads HOYO pickles, applies the same preprocessing used in
training (scale by head-feet distance, swap [y,x] -> [x,y], root at frame0),
and reports the average head-shoulder distance as a fraction of head-feet.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


ONOMATOPEIA = {
    "通常",
    "すたすた",
    "せかせか",
    "てくてく",
    "どっしどっし",
    "とぼとぼ",
    "のしのし",
    "のろのろ",
    "ぶらぶら",
    "よたよた",
    "よろよろ",
}


def _apply_horizontal_flip(coords: np.ndarray) -> np.ndarray:
    flipped = coords.copy()
    flipped[..., 0] *= -1
    pairs = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)]
    for r, l in pairs:
        tmp = flipped[:, r, :].copy()
        flipped[:, r, :] = flipped[:, l, :]
        flipped[:, l, :] = tmp
    return flipped


def _load_raw_and_scale(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        arr = pickle.load(f)  # (T, 14, 2) [y, x]
    arr = arr.astype(np.float32)

    # Centering for scale calc
    com = arr.mean(axis=1, keepdims=True)
    arr_centered = arr - com

    # Head-feet scale (matches current preprocessing)
    head_pos = arr_centered[:, 0, :]
    feet_pos = 0.5 * (arr_centered[:, 10, :] + arr_centered[:, 13, :])
    dists = np.linalg.norm(head_pos - feet_pos, axis=-1)
    scale = float(dists.mean())
    if scale < 1e-6:
        scale = 1.0

    # Swap [y, x] -> [x, y] and root at initial frame
    arr_swapped = arr[..., ::-1]
    initial_com = arr_swapped[0].mean(axis=0)
    arr_rooted = arr_swapped - initial_com

    return arr_rooted / scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute HOYO head-shoulder ratio.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--view-filter", type=str, choices=["front", "back"], default=None)
    parser.add_argument("--no-back-normalize", action="store_true", help="Disable back->front normalization.")
    parser.add_argument("--all-labels", action="store_true", help="Include non-onomatopoeia labels if present.")
    args = parser.parse_args()

    root = Path(args.root)
    data_dir = root / "data"
    if not data_dir.exists():
        data_dir = root

    json_files = sorted(data_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {data_dir}")

    head_shoulder = []
    head_feet = []
    total_sequences = 0
    total_frames = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = data["annotation"]["instruction"]
        if not args.all_labels and inst not in ONOMATOPEIA:
            continue
        if args.view_filter is not None and data.get("view") != args.view_filter:
            continue
        pkl_path = root / data["path"]
        if not pkl_path.exists():
            continue

        coords = _load_raw_and_scale(pkl_path)
        if (not args.no_back_normalize) and data.get("view") == "back":
            coords = _apply_horizontal_flip(coords)

        head = coords[:, 0, :]
        r_sh = coords[:, 2, :]
        l_sh = coords[:, 5, :]
        shoulder_mid = 0.5 * (r_sh + l_sh)
        feet_mid = 0.5 * (coords[:, 10, :] + coords[:, 13, :])

        d_hs = np.linalg.norm(head - shoulder_mid, axis=-1)
        d_hf = np.linalg.norm(head - feet_mid, axis=-1)

        head_shoulder.append(d_hs)
        head_feet.append(d_hf)
        total_sequences += 1
        total_frames += len(d_hs)

    if total_frames == 0:
        raise RuntimeError("No frames collected. Check filters or dataset paths.")

    head_shoulder = np.concatenate(head_shoulder)
    head_feet = np.concatenate(head_feet)

    hs_mean = float(head_shoulder.mean())
    hs_std = float(head_shoulder.std())
    hf_mean = float(head_feet.mean())
    hf_std = float(head_feet.std())
    ratio = hs_mean / hf_mean if hf_mean > 1e-6 else float("nan")

    print("=== HOYO head-shoulder ratio (canonical scaled) ===")
    print(f"sequences: {total_sequences}, frames: {total_frames}")
    print(f"head-shoulder mean: {hs_mean:.4f} (std {hs_std:.4f})")
    print(f"head-feet mean:     {hf_mean:.4f} (std {hf_std:.4f})")
    print(f"ratio (hs/hf):      {ratio:.4f}")


if __name__ == "__main__":
    main()
