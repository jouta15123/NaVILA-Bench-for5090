#!/usr/bin/env python3
"""
Plot retrieval curves (t2m/m2t) from retrieval_metrics.jsonl.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def load_metrics(path: Path) -> Dict[str, List[float]]:
    data = {
        "step": [],
        "t2m_r1": [],
        "m2t_r1": [],
        "t2m_r5": [],
        "m2t_r5": [],
        "avg_r1": [],
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record.get("step", 0))
            t2m = record.get("t2m", {})
            m2t = record.get("m2t", {})
            t2m_r1 = float(t2m.get("R@1", 0.0))
            m2t_r1 = float(m2t.get("R@1", 0.0))
            t2m_r5 = float(t2m.get("R@5", 0.0))
            m2t_r5 = float(m2t.get("R@5", 0.0))
            avg_r1 = 0.5 * (t2m_r1 + m2t_r1)

            data["step"].append(step)
            data["t2m_r1"].append(t2m_r1)
            data["m2t_r1"].append(m2t_r1)
            data["t2m_r5"].append(t2m_r5)
            data["m2t_r5"].append(m2t_r5)
            data["avg_r1"].append(avg_r1)
    return data


def save_plot(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retrieval curves from jsonl logs.")
    parser.add_argument("--metrics", type=str, required=True, help="Path to retrieval_metrics.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    parser.add_argument("--title", type=str, default="Retrieval Curves")
    parser.add_argument("--show-r5", action="store_true", help="Also plot R@5 curves")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    data = load_metrics(metrics_path)
    if not data["step"]:
        raise ValueError("No metrics found in file.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["step"], data["t2m_r1"], label="t2m R@1", linewidth=2)
    ax.plot(data["step"], data["m2t_r1"], label="m2t R@1", linewidth=2)
    ax.plot(data["step"], data["avg_r1"], label="avg R@1", linewidth=2, linestyle="--")
    if args.show_r5:
        ax.plot(data["step"], data["t2m_r5"], label="t2m R@5", linewidth=1.5, alpha=0.7)
        ax.plot(data["step"], data["m2t_r5"], label="m2t R@5", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Recall")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot(fig, Path(args.out))


if __name__ == "__main__":
    main()
