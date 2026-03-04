#!/usr/bin/env python3
"""
Plot training loss curves for a single MotionCLIP joint-training run.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def parse_log_file(log_path: Path) -> Dict[str, List[float]]:
    data = {
        "step": [],
        "total_loss": [],
        "vae_loss": [],
        "contrastive_loss": [],
        "train_acc": [],
        "test_step": [],
        "test_acc1": [],
        "test_acc3": [],
        "test_mpjpe": [],
    }

    train_pattern = re.compile(
        r"\[Step (\d+)\] loss_total=(\d+\.\d+) "
        r"\(vae=(\d+\.\d+), cont=(\d+\.\d+)\), train_acc@1=(\d+\.\d+)"
    )
    test_pattern = re.compile(
        r"\[TEST\] Acc@1: (\d+\.\d+) \| Acc@3: (\d+\.\d+) \| MPJPE: (\d+\.\d+)"
    )

    current_step = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            train_match = train_pattern.search(line)
            if train_match:
                step = int(train_match.group(1))
                data["step"].append(step)
                data["total_loss"].append(float(train_match.group(2)))
                data["vae_loss"].append(float(train_match.group(3)))
                data["contrastive_loss"].append(float(train_match.group(4)))
                data["train_acc"].append(float(train_match.group(5)))
                current_step = step

            test_match = test_pattern.search(line)
            if test_match and current_step is not None:
                data["test_step"].append(current_step)
                data["test_acc1"].append(float(test_match.group(1)))
                data["test_acc3"].append(float(test_match.group(2)))
                data["test_mpjpe"].append(float(test_match.group(3)))

    return data


def save_plot(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss curves for a single run.")
    parser.add_argument("--log", type=str, required=True, help="Path to train log file.")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Training Loss Curves")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    data = parse_log_file(log_path)
    if not data["step"]:
        raise ValueError("No training steps found in log (pattern mismatch).")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    ax1.plot(data["step"], data["total_loss"], label="total", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(data["step"], data["vae_loss"], label="vae", linewidth=2)
    ax2.plot(data["step"], data["contrastive_loss"], label="contrastive", linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Components")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    save_plot(fig, Path(args.out))


if __name__ == "__main__":
    main()
