#!/usr/bin/env python3
"""
Plot z_cos_ref_shift_sweep from style_evaluation_results.json.

Example:
  python legged-loco/scripts/plot_z_cos_ref_shift.py \
    --input eval_results/style_per_onomatopoeia/style_evaluation_results.json \
    --out_dir eval_results/style_per_onomatopoeia/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _iter_styles(data: list[dict], only: set[str] | None) -> Iterable[dict]:
    for item in data:
        style = item.get("onomatopoeia")
        if not style:
            continue
        if only and style not in only:
            continue
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot z_cos_ref_shift_sweep curves.")
    parser.add_argument("--input", type=str, required=True, help="Path to style_evaluation_results.json")
    parser.add_argument("--out_dir", type=str, default="eval_results/plots", help="Output directory for PNGs")
    parser.add_argument("--style", type=str, default=None, help="Comma-separated style filter")
    parser.add_argument("--combine", action="store_true", help="Plot all styles on one figure")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of per-style summaries.")

    only = None
    if args.style:
        only = {s.strip() for s in args.style.split(",") if s.strip()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    if args.combine:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for item in _iter_styles(data, only):
            sweep = item.get("z_cos_ref_shift_sweep") or []
            if not sweep:
                continue
            shifts = [int(d["shift"]) for d in sweep]
            means = [float(d["mean"]) for d in sweep]
            ax.plot(shifts, means, marker="o", label=item["onomatopoeia"])
            plotted = True
        if not plotted:
            print("[WARN] No z_cos_ref_shift_sweep data found.")
            return
        ax.set_xlabel("Reference shift (frames)")
        ax.set_ylabel("Cos similarity (mean)")
        ax.set_title("ZCos vs HOYO reference shift")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        out_path = out_dir / "z_cos_ref_shift_combined.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"[INFO] Saved: {out_path}")
        return

    for item in _iter_styles(data, only):
        sweep = item.get("z_cos_ref_shift_sweep") or []
        if not sweep:
            continue
        shifts = [int(d["shift"]) for d in sweep]
        means = [float(d["mean"]) for d in sweep]
        stds = [float(d.get("std", 0.0)) for d in sweep]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(shifts, means, marker="o")
        ax.fill_between(
            shifts,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
        )
        ax.set_xlabel("Reference shift (frames)")
        ax.set_ylabel("Cos similarity (mean)")
        ax.set_title(f"{item['onomatopoeia']} - ZCos shift sweep")
        ax.grid(True, alpha=0.3)
        out_path = out_dir / f"z_cos_ref_shift_{_sanitize_name(item['onomatopoeia'])}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
