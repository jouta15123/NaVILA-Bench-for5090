#!/usr/bin/env python3
"""
Plot confusion matrix from evaluate_retrieval.py JSON output.
Supports Japanese labels (optional) and row-normalized view.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Optional, for Japanese labels
    import japanize_matplotlib  # noqa: F401
except Exception:
    japanize_matplotlib = None  # type: ignore

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path, normalize: bool, title: str):
    if normalize:
        cm_plot = normalize_rows(cm)
        fmt = ".2f"
    else:
        cm_plot = cm.astype(int)
        fmt = "d"

    plt.figure(figsize=(8, 7))

    if sns is not None:
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
        )
    else:
        plt.imshow(cm_plot, cmap="Blues")
        plt.colorbar()
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                plt.text(j, i, format(cm_plot[i, j], fmt), ha="center", va="center", fontsize=8)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrix from JSON.")
    parser.add_argument("--confusion-json", type=str, required=True, help="Path to confusion_fine.json")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    parser.add_argument("--normalize", action="store_true", help="Row-normalize confusion matrix")
    parser.add_argument("--title", type=str, default="Confusion Matrix", help="Plot title")
    args = parser.parse_args()

    json_path = Path(args.confusion_json)
    if not json_path.exists():
        raise FileNotFoundError(f"Confusion JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [str(l) for l in data.get("labels", [])]
    cm = np.array(data.get("confusion", []), dtype=float)
    if cm.size == 0:
        raise ValueError("Confusion matrix is empty.")

    plot_confusion(cm, labels, Path(args.out), normalize=args.normalize, title=args.title)
    print(f"Saved confusion matrix to: {args.out}")


if __name__ == "__main__":
    main()
