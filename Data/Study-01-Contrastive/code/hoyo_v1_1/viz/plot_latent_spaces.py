#!/usr/bin/env python3
"""
Latent space visualization for MotionCLIP (PCA/UMAP + H1 overlay).

Usage:
  python hoyo_v1_1/viz/plot_latent_spaces.py \
    --snapshot hoyo_v1_1/joint_training_results/sarashina_full_fixed/latent_snapshot_final.npz \
    --h1-latents eval_results/style_per_onomatopoeia/h1_latents_*.npz \
    --out-dir hoyo_v1_1/joint_training_results/visualizations \
    --umap-n-neighbors 15 --umap-min-dist 0.1 \
    --show-text-prototypes \
    --label-mode coarse-with-normal \
    --unknown-words hoyo_v1_1/data/unknown_words_coarse.txt \
    --sem-proj hoyo_v1_1/joint_training_results/sarashina_full_fixed/checkpoints/sem_proj_joint_best.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import japanize_matplotlib  # noqa: F401


# Coarse style groups (same as train_motionclip_joint.py / evaluate_retrieval.py)
COARSE_GROUPS = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["通常", "neutral", "とぼとぼ", "のろのろ"],
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}
COARSE_LABELS = list(COARSE_GROUPS.keys())
COARSE_COLOR_MAP = {
    "速い系": "#1f77b4",  # blue
    "遅い系": "#ff7f0e",  # orange
    "重い系": "#d62728",  # red
    "ふらふら系": "#2ca02c",  # green
}

# Coarse groups with '通常' separated
COARSE_WITH_NORMAL_GROUPS = {
    "通常": ["通常", "neutral"],
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["とぼとぼ", "のろのろ"],
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}
COARSE_WITH_NORMAL_LABELS = list(COARSE_WITH_NORMAL_GROUPS.keys())
COARSE_WITH_NORMAL_COLOR_MAP = {
    "通常": "#7f7f7f",  # gray
    "速い系": "#1f77b4",  # blue
    "遅い系": "#ff7f0e",  # orange
    "重い系": "#d62728",  # red
    "ふらふら系": "#2ca02c",  # green
}

UNKNOWN_GROUP_ALIASES = {
    "速い": "速い系",
    "遅い": "遅い系",
    "重い": "重い系",
    "ふらふら": "ふらふら系",
    "通常": "通常",
    "neutral": "通常",
}


def pca_2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA via SVD. Returns (proj_2d, mean, components, explained_ratio)."""
    x_mean = x.mean(axis=0, keepdims=True)
    x_center = x - x_mean
    _, s, vt = np.linalg.svd(x_center, full_matrices=False)
    components = vt[:2]
    x_2d = x_center @ components.T
    variances = (s ** 2) / max(1, x_center.shape[0] - 1)
    explained_ratio = variances / variances.sum()
    return x_2d, x_mean, components, explained_ratio


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


def load_snapshot(
    snapshot_path: Path, splits: Iterable[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    data = np.load(snapshot_path, allow_pickle=True)
    z_m = data["z_m"]
    labels_idx = data["labels_idx"].astype(int)
    label_list = [str(l) for l in data["label_list"]]
    z_s_cls = data.get("z_s_cls", None)
    splits_arr = data.get("splits", None)

    if splits_arr is not None and splits:
        splits_arr = np.asarray(splits_arr).astype(str)
        mask = np.isin(splits_arr, list(splits))
        if not mask.any():
            raise ValueError(f"No samples for splits={splits} in snapshot: {snapshot_path}")
        z_m = z_m[mask]
        labels_idx = labels_idx[mask]

    return z_m, labels_idx, label_list, z_s_cls


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


def build_label_colors(
    label_list: list[str],
    neutral_labels: set[str],
    color_map: dict[str, str] | None = None,
) -> dict[str, str]:
    cmap = plt.get_cmap("tab20")
    colors = {}
    color_i = 0
    for lab in label_list:
        if lab in neutral_labels:
            colors[lab] = "#4a4a4a"
        elif color_map and lab in color_map:
            colors[lab] = color_map[lab]
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


def add_color_key(
    ax,
    label_list: list[str],
    colors: dict[str, str],
    title: str = "Color key",
    x: float = 0.02,
    y: float = 0.98,
    line_step: float = 0.055,
):
    ax.text(
        x,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )
    for idx, lab in enumerate(label_list):
        ax.text(
            x,
            y - (idx + 1) * line_step,
            lab,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=colors.get(lab, "#333333"),
        )


def load_unknown_words(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Unknown words file not found: {path}")
    items: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace("：", ":").replace("、", ",")
        if ":" in line:
            group, rest = line.split(":", 1)
            group = group.strip()
            words = [w.strip() for w in rest.split(",") if w.strip()]
        elif "\t" in line:
            group, word = line.split("\t", 1)
            group = group.strip()
            words = [word.strip()] if word.strip() else []
        else:
            group = ""
            words = [line]
        for w in words:
            items.append((w, group))
    if not items:
        raise ValueError(f"No unknown words parsed from: {path}")
    return items


def normalize_unknown_group(label: str) -> str:
    label = label.strip()
    if label in UNKNOWN_GROUP_ALIASES:
        return UNKNOWN_GROUP_ALIASES[label]
    if label.endswith("系"):
        return label
    return label


def build_fine_to_group_mapping(
    label_list: list[str],
    group_map: dict[str, list[str]],
    group_labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    if len(label_list) == len(group_labels) and all(lab in group_labels for lab in label_list):
        return np.arange(len(label_list), dtype=int), list(label_list)
    label_to_group: dict[str, int] = {}
    for group_idx, (group_label, fine_labels) in enumerate(group_map.items()):
        for fine_lab in fine_labels:
            label_to_group[fine_lab] = group_idx
    mapped = []
    unknown = []
    for lab in label_list:
        if lab in label_to_group:
            mapped.append(label_to_group[lab])
        elif lab in group_labels:
            mapped.append(group_labels.index(lab))
        else:
            mapped.append(-1)
            unknown.append(lab)
    if unknown:
        raise ValueError(f"Unknown labels for group mapping: {unknown}")
    return np.asarray(mapped, dtype=int), list(group_labels)


def build_group_prototypes(
    z_s_cls: np.ndarray,
    label_list: list[str],
    group_map: dict[str, list[str]],
    group_labels: list[str],
) -> np.ndarray:
    if z_s_cls.shape[0] == len(group_labels):
        return z_s_cls
    if z_s_cls.shape[0] != len(label_list):
        raise ValueError(
            "z_s_cls rows must match label_list length "
            f"({z_s_cls.shape[0]} vs {len(label_list)})."
        )
    label_to_vec = {label_list[i]: z_s_cls[i] for i in range(len(label_list))}
    grouped = []
    for group_label in group_labels:
        fine_labels = group_map[group_label]
        vecs = [label_to_vec[lab] for lab in fine_labels if lab in label_to_vec]
        if not vecs:
            raise ValueError(f"No fine labels found for group: {group_label}")
        grouped.append(np.mean(np.stack(vecs, axis=0), axis=0))
    return np.stack(grouped, axis=0)


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep legends that sit outside the axes.
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
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
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP plot generation (useful when umap-learn is not installed).",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        choices=["fine", "coarse", "coarse-with-normal"],
        default="fine",
        help=(
            "Label granularity for color/legend "
            "(fine=original labels, coarse=4 groups, coarse-with-normal=5 groups)."
        ),
    )
    parser.add_argument(
        "--show-text-prototypes",
        action="store_true",
        help="Overlay text prototypes (z_s_cls) on the PCA plot.",
    )
    parser.add_argument(
        "--unknown-words",
        type=str,
        default="",
        help="Path to unknown words list (e.g., '速い: すいすい, さっさ').",
    )
    parser.add_argument(
        "--sem-proj",
        type=str,
        default="",
        help="Path to sem_proj checkpoint for unknown word embeddings.",
    )
    parser.add_argument(
        "--hide-unknown-labels",
        action="store_true",
        help="Do not render text labels next to unknown-word markers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    splits = _parse_list(args.splits) if args.splits else None
    z_m, labels_idx, label_list, z_s_cls = load_snapshot(snapshot_path, splits=splits)
    fine_label_list = list(label_list)
    fine_to_group = None
    group_map = None
    group_labels = None
    group_color_map = None

    if args.label_mode != "fine":
        if args.label_mode == "coarse":
            group_map = COARSE_GROUPS
            group_labels = COARSE_LABELS
            group_color_map = COARSE_COLOR_MAP
        else:
            group_map = COARSE_WITH_NORMAL_GROUPS
            group_labels = COARSE_WITH_NORMAL_LABELS
            group_color_map = COARSE_WITH_NORMAL_COLOR_MAP

        fine_to_group, group_labels = build_fine_to_group_mapping(label_list, group_map, group_labels)
        labels_idx = fine_to_group[labels_idx]
        label_list = group_labels
        if args.show_text_prototypes:
            if z_s_cls is None:
                raise KeyError("Snapshot does not contain z_s_cls required for --show-text-prototypes.")
            z_s_cls = build_group_prototypes(z_s_cls, fine_label_list, group_map, group_labels)

    if args.show_text_prototypes and z_s_cls is None:
        raise KeyError("Snapshot does not contain z_s_cls required for --show-text-prototypes.")
    if args.show_text_prototypes and args.label_mode == "fine":
        if z_s_cls.shape[0] != len(label_list):
            raise ValueError(
                "z_s_cls rows must match label_list length "
                f"({z_s_cls.shape[0]} vs {len(label_list)})."
            )

    neutral_labels = set(_parse_list(args.neutral_labels))
    color_map = group_color_map if args.label_mode != "fine" else None
    colors = build_label_colors(label_list, neutral_labels, color_map=color_map)

    out_dir = Path(args.out_dir)
    prefix = f"{args.prefix}_" if args.prefix else ""

    # --- PCA 2D ---
    pca_2d_coords, mean_vec, components, explained_ratio = pca_2d(z_m)
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
    if args.show_text_prototypes:
        proto_labels_idx = np.arange(len(label_list), dtype=int)
        text_coords = (z_s_cls - mean_vec) @ components.T
        scatter_by_label(
            ax,
            text_coords,
            proto_labels_idx,
            label_list,
            colors,
            neutral_labels,
            marker="*",
            neutral_marker="*",
            size=args.point_size * 1.8,
            alpha=min(0.9, args.alpha + 0.2),
            show_labels=False,
        )
    unknown_coords = None
    if args.unknown_words:
        if not args.sem_proj:
            raise ValueError("--sem-proj is required when --unknown-words is specified.")
        unknown_items = load_unknown_words(Path(args.unknown_words))
        unknown_words = [w for w, _ in unknown_items]
        unknown_groups = [normalize_unknown_group(g) for _, g in unknown_items]
        unknown_color_map = (
            COARSE_WITH_NORMAL_COLOR_MAP
            if any(g == "通常" for g in unknown_groups)
            else COARSE_COLOR_MAP
        )
        unknown_colors = []
        unknown_labels = []
        for word, group in zip(unknown_words, unknown_groups):
            if not group:
                raise ValueError(f"Unknown word '{word}' has no group label.")
            if group not in unknown_color_map:
                raise ValueError(f"Unknown group '{group}' for word '{word}'.")
            unknown_colors.append(unknown_color_map[group])
            unknown_labels.append(f"{word}({group})")

        import torch
        import torch.nn.functional as F

        from hoyo_v1_1.models.common import encode_semantics_sarashina

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sem_emb = encode_semantics_sarashina(unknown_words, device=device)
        d_sem = sem_emb.shape[1]
        d_motion = z_m.shape[1]
        sem_proj = torch.nn.Linear(d_sem, d_motion, bias=False).to(device)
        sem_proj.load_state_dict(torch.load(Path(args.sem_proj), map_location=device))
        sem_proj.eval()
        with torch.no_grad():
            z_unknown = sem_proj(sem_emb)
            z_unknown = F.normalize(z_unknown, dim=-1).cpu().numpy()
        unknown_coords = (z_unknown - mean_vec) @ components.T
        ax.scatter(
            unknown_coords[:, 0],
            unknown_coords[:, 1],
            s=args.point_size * 2.0,
            marker="X",
            color=unknown_colors,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
        )
        if not args.hide_unknown_labels:
            for (x, y), label in zip(unknown_coords, unknown_labels):
                ax.text(x + 0.01, y + 0.01, label, fontsize=9, color="black")
    if args.label_mode != "fine":
        add_color_key(ax, label_list, colors, title="Colors (group labels)")
    if args.label_mode != "fine":
        ax.set_title("Latent Space Visualization (PCA)")
    else:
        ax.set_title("PCA 2D (Motion Latents)")
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    label_legend = ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    label_legend.set_title("Labels (motion)")
    ax.add_artist(label_legend)
    marker_handles = []
    if args.show_text_prototypes:
        marker_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Motion samples (color=label)",
                    markerfacecolor="#777777",
                    markersize=8,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="#333333",
                    label="Text prototypes",
                    markerfacecolor="#333333",
                    markersize=10,
                ),
            ]
        )
    if unknown_coords is not None:
        marker_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="#333333",
                label="Unknown words",
                markerfacecolor="#bbbbbb",
                markersize=9,
            )
        )
    if marker_handles:
        ax.legend(handles=marker_handles, loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    save_plot(fig, out_dir / f"{prefix}pca_2d.png")

    # --- UMAP 2D ---
    if args.skip_umap:
        print("[Info] Skipping UMAP plot (--skip-umap).")
    else:
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
        if args.label_mode != "fine":
            add_color_key(ax, label_list, colors, title="Colors (group labels)")
        ax.set_title(
            f"UMAP 2D (n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist})"
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        label_legend = ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
        label_legend.set_title("Labels (motion)")
        fig.tight_layout()
        save_plot(fig, out_dir / f"{prefix}umap_2d.png")

    # --- PCA overlay with H1 latents ---
    if args.h1_latents:
        h1_path = Path(args.h1_latents)
        if not h1_path.exists():
            raise FileNotFoundError(f"H1 latents file not found: {h1_path}")
        z_h1, h1_labels_idx = load_h1_latents(h1_path, fine_label_list)
        if fine_to_group is not None:
            h1_labels_idx = fine_to_group[h1_labels_idx]
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
        ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")

        label_legend = ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
        label_legend.set_title("Labels (motion)")
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
