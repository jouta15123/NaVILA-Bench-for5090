#!/usr/bin/env python3
"""Regenerate HOYO confusion or margin heatmap from eval_motion JSON with dataset-speed label order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


DATASET_SPEED_ORDER = [
    "すたすた",
    "せかせか",
    "通常",
    "てくてく",
    "どっしどっし",
    "ぶらぶら",
    "のしのし",
    "よたよた",
    "のろのろ",
    "とぼとぼ",
    "よろよろ",
]
ORDER_INDEX = {label: idx for idx, label in enumerate(DATASET_SPEED_ORDER)}


def ordered_labels(labels: list[str]) -> list[str]:
    unique = list(dict.fromkeys(labels))
    original_index = {label: idx for idx, label in enumerate(unique)}
    return sorted(
        unique,
        key=lambda label: (ORDER_INDEX.get(label, len(DATASET_SPEED_ORDER)), original_index[label]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--mode", choices=["confusion", "margin"], default="confusion")
    parser.add_argument("--sim-key", default="hoyo_similarity_centered_mean")
    parser.add_argument("--title", default="H1 vs HOYO Similarity Matrix (Centered Cos)")
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No results found in {args.input_json}")

    row_labels = ordered_labels([str(r.get("onomatopoeia", "")) for r in results if r.get("onomatopoeia")])
    sim_dicts = [r.get(args.sim_key) or {} for r in results]
    col_label_candidates: list[str] = []
    for sim in sim_dicts:
        col_label_candidates.extend([str(k) for k in sim.keys()])
    col_labels = ordered_labels(col_label_candidates)

    row_index = {label: i for i, label in enumerate(row_labels)}
    col_index = {label: j for j, label in enumerate(col_labels)}
    sim_matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float32)

    for r in results:
        row_label = r.get("onomatopoeia")
        if row_label not in row_index:
            continue
        i = row_index[row_label]
        sim = r.get(args.sim_key) or {}
        for label, val in sim.items():
            j = col_index.get(label)
            if j is not None:
                sim_matrix[i, j] = float(val)

    if args.mode == "confusion":
        matrix = np.where(np.isfinite(sim_matrix), sim_matrix, 0.0)
        vmin, vmax = -1.0, 1.0
        cbar_label = "Cosine Similarity"
    else:
        margin = np.full_like(sim_matrix, np.nan)
        for i, row_label in enumerate(row_labels):
            diag_j = col_index.get(row_label)
            if diag_j is None:
                continue
            diag_val = sim_matrix[i, diag_j]
            if not np.isfinite(diag_val):
                continue
            row = sim_matrix[i, :]
            valid = np.isfinite(row)
            margin[i, valid] = diag_val - row[valid]
        matrix = margin
        max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        if max_abs < 1e-6:
            max_abs = 1.0
        vmin, vmax = -max_abs, max_abs
        cbar_label = "Cosine Margin (diag - col)"

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.8), max(6, len(row_labels) * 0.6)))
    if args.mode == "margin":
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.cm.RdYlGn
        cmap.set_bad(color="#cccccc")
        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=8)
                continue
            color = "white" if abs(val) > 0.5 else "black"
            weight = "bold" if row_labels[i] == col_labels[j] else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight=weight)

    ax.set_title(args.title, fontsize=12)
    ax.set_xlabel("HOYO Reference Style", fontsize=11)
    ax.set_ylabel("H1 Executed Style", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10)
    plt.tight_layout()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.output_png}")


if __name__ == "__main__":
    main()
