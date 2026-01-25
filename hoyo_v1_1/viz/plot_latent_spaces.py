#!/usr/bin/env python3
"""
Latent space visualization for MotionCLIP (PCA/UMAP + H1 overlay).

Usage:
  python hoyo_v1_1/viz/plot_latent_spaces.py \
    --snapshot hoyo_v1_1/joint_training_results/sarashina_full_fixed/latent_snapshot_final.npz \
    --h1-latents eval_results/style_per_onomatopoeia/h1_latents_*.npz \
    --out-dir hoyo_v1_1/joint_training_results/visualizations \
    --umap-n-neighbors 15 --umap-min-dist 0.1
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:  # Optional, for Japanese labels
    import japanize_matplotlib  # noqa: F401
except Exception:
    japanize_matplotlib = None  # type: ignore


def pca_2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA via SVD. Returns (proj_2d, mean, components)."""
    x_mean = x.mean(axis=0, keepdims=True)
    x_center = x - x_mean
    _, _, vt = np.linalg.svd(x_center, full_matrices=False)
    components = vt[:2]
    x_2d = x_center @ components.T
    return x_2d, x_mean, components


def umap_2d(x: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> np.ndarray:
    try:
        import umap
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "UMAP is not available. Install with `pip install umap-learn`."
        ) from exc
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(x)


def _parse_list(csv: str) -> list[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]


def load_snapshot(snapshot_path: Path, splits: Iterable[str] | None = None):
    data = np.load(snapshot_path, allow_pickle=True)
    z_m = data["z_m"]
    labels_idx = data["labels_idx"].astype(int)
    label_list = [str(l) for l in data["label_list"]]
    splits_arr = data.get("splits", None)

    if splits_arr is not None and splits:
        splits_arr = np.asarray(splits_arr).astype(str)
        mask = np.isin(splits_arr, list(splits))
        if not mask.any():
            raise ValueError(f"No samples for splits={splits} in snapshot: {snapshot_path}")
        z_m = z_m[mask]
        labels_idx = labels_idx[mask]

    return z_m, labels_idx, label_list


def _normalize_label_list(labels: np.ndarray) -> list[str]:
    return [str(l) for l in labels]


def load_h1_latents(h1_path: Path, target_label_list: list[str]):
    data = np.load(h1_path, allow_pickle=True)
    if "z_h1" in data:
        z_h1 = data["z_h1"]
    elif "z" in data:
        z_h1 = data["z"]
    else:
        raise KeyError("H1 latents file must contain 'z_h1' or 'z'.")

    if "labels_idx" in data and "label_list" in data:
        labels_idx = data["labels_idx"].astype(int)
        label_list = _normalize_label_list(data["label_list"])
        if label_list != target_label_list:
            label_to_idx = {lab: i for i, lab in enumerate(target_label_list)}
            mapped = []
            for li in labels_idx:
                lab = label_list[int(li)]
                mapped.append(label_to_idx.get(lab, -1))
            labels_idx = np.asarray(mapped, dtype=int)
    elif "labels" in data:
        labels = _normalize_label_list(data["labels"])
        label_to_idx = {lab: i for i, lab in enumerate(target_label_list)}
        labels_idx = np.asarray([label_to_idx.get(lab, -1) for lab in labels], dtype=int)
    else:
        raise KeyError("H1 latents file must contain labels_idx+label_list or labels.")

    valid = labels_idx >= 0
    if not valid.any():
        raise ValueError("No valid H1 labels matched target label list.")
    return z_h1[valid], labels_idx[valid]


def build_label_colors(label_list: list[str], neutral_labels: set[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab20")
    colors = {}
    color_i = 0
    for lab in label_list:
        if lab in neutral_labels:
            colors[lab] = "#4a4a4a"
        else:
            colors[lab] = cmap(color_i % cmap.N)
            color_i += 1
    return colors


def scatter_by_label(
    ax,
    coords: np.ndarray,
    labels_idx: np.ndarray,
    label_list: list[str],
    colors: dict[str, str],
    neutral_labels: set[str],
    marker: str = "o",
    neutral_marker: str = "s",
    size: float = 12.0,
    alpha: float = 0.6,
    label_suffix: str | None = None,
    show_labels: bool = True,
):
    for idx, lab in enumerate(label_list):
        mask = labels_idx == idx
        if not np.any(mask):
            continue
        label = None
        if show_labels:
            label = f"{lab}{label_suffix}" if label_suffix else lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            marker=neutral_marker if lab in neutral_labels else marker,
            color=colors.get(lab, "#7f7f7f"),
            label=label,
        )


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MotionCLIP latent space (PCA/UMAP + H1 overlay).")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to latent_snapshot_final.npz")
    parser.add_argument("--h1-latents", type=str, default="", help="Optional H1 latents .npz (from eval script).")
    parser.add_argument("--out-dir", type=str, default="hoyo_v1_1/joint_training_results/visualizations")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for output filenames.")
    parser.add_argument("--splits", type=str, default="", help="Comma-separated splits to include (e.g., test).")
    parser.add_argument("--neutral-labels", type=str, default="通常,neutral", help="Comma-separated neutral labels.")
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    splits = _parse_list(args.splits) if args.splits else None
    z_m, labels_idx, label_list = load_snapshot(snapshot_path, splits=splits)

    neutral_labels = set(_parse_list(args.neutral_labels))
    colors = build_label_colors(label_list, neutral_labels)

    out_dir = Path(args.out_dir)
    prefix = f"{args.prefix}_" if args.prefix else ""

    # --- PCA 2D ---
    pca_2d_coords, mean_vec, components = pca_2d(z_m)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_by_label(
        ax,
        pca_2d_coords,
        labels_idx,
        label_list,
        colors,
        neutral_labels,
        marker="o",
        neutral_marker="s",
        size=args.point_size,
        alpha=args.alpha,
    )
    ax.set_title("PCA 2D (Motion Latents)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_plot(fig, out_dir / f"{prefix}pca_2d.png")

    # --- UMAP 2D ---
    umap_coords = umap_2d(
        z_m,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_by_label(
        ax,
        umap_coords,
        labels_idx,
        label_list,
        colors,
        neutral_labels,
        marker="o",
        neutral_marker="s",
        size=args.point_size,
        alpha=args.alpha,
    )
    ax.set_title(
        f"UMAP 2D (n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist})"
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_plot(fig, out_dir / f"{prefix}umap_2d.png")

    # --- PCA overlay with H1 latents ---
    if args.h1_latents:
        h1_path = Path(args.h1_latents)
        if not h1_path.exists():
            raise FileNotFoundError(f"H1 latents file not found: {h1_path}")
        z_h1, h1_labels_idx = load_h1_latents(h1_path, label_list)
        h1_coords = (z_h1 - mean_vec) @ components.T

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_by_label(
            ax,
            pca_2d_coords,
            labels_idx,
            label_list,
            colors,
            neutral_labels,
            marker="o",
            neutral_marker="s",
            size=args.point_size,
            alpha=args.alpha,
        )
        scatter_by_label(
            ax,
            h1_coords,
            h1_labels_idx,
            label_list,
            colors,
            neutral_labels,
            marker="x",
            neutral_marker="x",
            size=args.point_size * 1.2,
            alpha=0.8,
            label_suffix=" (H1)",
            show_labels=False,
        )
        ax.set_title("PCA 2D (HOYO) + H1 Rollout Overlay")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        label_legend = ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
        ax.add_artist(label_legend)
        marker_handles = [
            Line2D([0], [0], marker="o", color="w", label="HOYO (teacher)", markerfacecolor="#777777", markersize=8),
            Line2D([0], [0], marker="x", color="#333333", label="H1 rollout", markersize=8),
        ]
        ax.legend(handles=marker_handles, loc="lower right", fontsize=8, frameon=True)

        fig.tight_layout()
        save_plot(fig, out_dir / f"{prefix}pca_h1_overlay.png")

    print(f"Saved PCA/UMAP plots to: {out_dir}")
    if args.h1_latents:
        print(f"Saved H1 overlay plot to: {out_dir / f'{prefix}pca_h1_overlay.png'}")


if __name__ == "__main__":
    main()
