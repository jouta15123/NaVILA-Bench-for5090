import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_PATTERN = re.compile(
    r"\[Step\s+(\d+)\]\s+loss_total=([0-9.]+)\s+\(vae=([0-9.]+),\s+cont=([0-9.]+)\),\s+train_acc@1=([0-9.]+)"
)


def parse_log(log_path: Path):
    steps = []
    loss_total = []
    loss_vae = []
    loss_cont = []
    acc1 = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = STEP_PATTERN.search(line)
            if not m:
                continue
            step = int(m.group(1))
            lt = float(m.group(2))
            lv = float(m.group(3))
            lc = float(m.group(4))
            a1 = float(m.group(5))

            steps.append(step)
            loss_total.append(lt)
            loss_vae.append(lv)
            loss_cont.append(lc)
            acc1.append(a1)

    return steps, loss_total, loss_vae, loss_cont, acc1


def main():
    parser = argparse.ArgumentParser(description="Plot training losses from HOYO MotionCLIP joint log.")
    parser.add_argument(
        "--log",
        type=str,
        default="hoyo_v1_1/joint_training_results/logs/train_log_20251124_143824.txt",
        help="Path to train_log_*.txt.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="hoyo_v1_1/joint_training_results/figures/train_loss.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    steps, loss_total, loss_vae, loss_cont, acc1 = parse_log(log_path)
    if not steps:
        raise RuntimeError("No [Step ...] lines found in the log; nothing to plot.")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(steps, loss_total, label="total", color="C0")
    ax1.plot(steps, loss_vae, label="vae", color="C1", linestyle="--")
    ax1.plot(steps, loss_cont, label="contrastive", color="C2", linestyle="-.")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training loss plot to {out_path}")


if __name__ == "__main__":
    main()


