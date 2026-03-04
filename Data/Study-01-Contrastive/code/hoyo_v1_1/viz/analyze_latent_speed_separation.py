#!/usr/bin/env python3
"""
Analyze whether speed information is recoverable from MotionCLIP latent embeddings.

This script compares two run checkpoints by:
1) Re-extracting motion latents on the same HOYO sample set
2) Computing per-sample scale-speed from 2D keypoints
3) Evaluating sample-level speed recoverability (regression)
4) Evaluating 3-band speed-bin separability (classification)
5) Running permutation tests for regression/classification significance
6) Exporting optional auxiliary analyses (label-order, semantic classes)
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hoyo_v1_1.compute_velocity_table import compute_scale_speed_per_s
from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    INSTRUCTION_ONOMATOPEIA,
    apply_normalization_from_stats,
)
from hoyo_v1_1.models.train_motionclip_joint import load_motionclip_full_model


DEFAULT_NEW_RUN = (
    "20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_"
    "lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full"
)
DEFAULT_OLD_RUN = "20260109_optuna_trial1_full"
DEFAULT_OUT_DIR = "docs/experiments/assets/20260207_motionclip_speed_latent"

DEFAULT_RUN_CONFIG = {
    "target_len": 100,
    "centering": "first_frame_com",
    "view_filter": None,
    "normalize_back_to_front": True,
    "sem_encoder": "sarashina",
}


@dataclass
class SampleMeta:
    sample_id: int
    label: str
    label_idx: int
    speed_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze speed separability in MotionCLIP latent space.")
    parser.add_argument("--run-a", type=str, default=DEFAULT_NEW_RUN, help="Run name for model A (new).")
    parser.add_argument("--run-b", type=str, default=DEFAULT_OLD_RUN, help="Run name for model B (old).")
    parser.add_argument("--hoyo-root", type=str, default="hoyo_v1_1", help="Path to HOYO root directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=3, help="Quantile speed bins for core probe classification.")
    parser.add_argument("--n-perm", type=int, default=1000, help="Number of permutations for p-value.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--speed-fps", type=float, default=60.0)
    parser.add_argument("--speed-smooth-window", type=int, default=5)
    parser.add_argument("--speed-height-percentile", type=float, default=90.0)
    parser.add_argument("--speed-min-height-ratio", type=float, default=0.2)

    parser.add_argument("--use-relative-speed-table", type=str, choices=["on", "off"], default="on")
    parser.add_argument("--speed-table-baseline-label", type=str, default="通常")
    parser.add_argument("--semantic-fast-min", type=float, default=0.8)
    parser.add_argument("--semantic-mid-min", type=float, default=0.4)
    parser.add_argument("--semantic-mid-max", type=float, default=0.6)
    parser.add_argument("--semantic-slow-max", type=float, default=0.35)
    parser.add_argument(
        "--primary-metric",
        type=str,
        choices=["sample_rho", "label_rank_rho"],
        default="sample_rho",
    )

    parser.add_argument("--pass-rho", type=float, default=0.35)
    parser.add_argument("--pass-macro-f1", type=float, default=0.50)
    parser.add_argument("--pass-p", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def resolve_run_config(run_dir: Path) -> dict[str, Any]:
    cfg = dict(DEFAULT_RUN_CONFIG)
    rc_path = run_dir / "run_config.json"
    if not rc_path.exists():
        return cfg
    loaded = json.loads(rc_path.read_text(encoding="utf-8"))
    for key in ("target_len", "centering", "view_filter", "normalize_back_to_front", "sem_encoder"):
        if key in loaded and loaded[key] is not None:
            cfg[key] = loaded[key]
    return cfg


def load_dataset_for_run(hoyo_root: Path, run_cfg: dict[str, Any], stats_path: Path) -> HoyoInstructionDataset:
    dataset = HoyoInstructionDataset(
        root=hoyo_root,
        target_labels=INSTRUCTION_ONOMATOPEIA,
        target_len=int(run_cfg["target_len"]),
        is_train=False,
        use_aug=False,
        view_filter=run_cfg.get("view_filter", None),
        normalize_back_to_front=bool(run_cfg.get("normalize_back_to_front", True)),
        centering=str(run_cfg.get("centering", "first_frame_com")),
    )
    apply_normalization_from_stats(dataset, stats_path)
    return dataset


def collect_sample_metadata(
    dataset: HoyoInstructionDataset,
    hoyo_root: Path,
    fps: float,
    smooth_window: int,
    height_percentile: float,
    min_height_ratio: float,
) -> list[SampleMeta]:
    data_dir = hoyo_root / "data"
    json_files = sorted(data_dir.glob("*.json"), key=lambda p: int(p.stem))
    target_labels = set(dataset.target_labels)

    metadata: list[SampleMeta] = []
    labels_in_scan: list[str] = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        label = meta.get("annotation", {}).get("instruction")
        if label not in target_labels:
            continue
        if dataset.view_filter is not None and meta.get("view") != dataset.view_filter:
            continue

        pkl_path = hoyo_root / meta.get("path", "")
        if not pkl_path.exists():
            continue

        coords = dataset._load_raw_and_scale(pkl_path)
        if coords.shape[0] < dataset.min_length:
            continue

        with open(pkl_path, "rb") as f:
            motion = pickle.load(f)
        motion = np.asarray(motion, dtype=np.float32)
        speed = compute_scale_speed_per_s(
            motion,
            fps=fps,
            smooth_window=smooth_window,
            height_percentile=height_percentile,
            min_height_ratio=min_height_ratio,
        )

        label_idx = dataset.label_to_id[label]
        sample_id = int(json_path.stem)
        metadata.append(
            SampleMeta(
                sample_id=sample_id,
                label=label,
                label_idx=label_idx,
                speed_scale=float(speed),
            )
        )
        labels_in_scan.append(label)

    ds_labels = [lab for lab, _ in dataset._indices]
    if len(metadata) != len(dataset):
        raise RuntimeError(
            f"Sample count mismatch: metadata={len(metadata)} dataset={len(dataset)}. "
            "Filtering logic mismatch."
        )
    if ds_labels != labels_in_scan:
        raise RuntimeError("Sample order mismatch between dataset and metadata scan.")

    return metadata


@torch.no_grad()
def extract_latents(
    dataset: HoyoInstructionDataset,
    run_dir: Path,
    target_len: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model, _ = load_motionclip_full_model(device, target_len=target_len)
    ckpt_path = run_dir / "checkpoints" / "motionclip_full_joint_best.pth"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "motionclip_full_joint_final.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found in {run_dir / 'checkpoints'}")

    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys when loading {ckpt_path.name}: {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys when loading {ckpt_path.name}: {len(unexpected)}")

    model.eval()

    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for x_batch, y_batch in loader:
        x_batch = x_batch.permute(0, 2, 3, 1).to(device)  # (B, 14, 2, T)
        y_batch = y_batch.to(device)
        bsz = x_batch.shape[0]
        mask = torch.ones((bsz, target_len), dtype=torch.bool, device=device)
        lengths = torch.full((bsz,), target_len, dtype=torch.long, device=device)
        batch = {"x": x_batch, "mask": mask, "lengths": lengths, "y": y_batch}
        out = model(batch)
        z_m = F.normalize(out["mu"], dim=-1).detach().cpu().numpy()
        zs.append(z_m)
        ys.append(y_batch.detach().cpu().numpy())

    z_all = np.concatenate(zs, axis=0)
    y_all = np.concatenate(ys, axis=0).astype(np.int64)
    return z_all, y_all


def compute_speed_bins(speed: np.ndarray, n_bins: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    if speed.ndim != 1:
        raise ValueError("speed must be a 1D array")
    if speed.size < n_bins:
        raise ValueError(f"Need at least n_bins samples: len(speed)={speed.size}, n_bins={n_bins}")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(speed, quantiles)

    # right=True keeps exact-threshold values (e.g. speed==0) in lower bins.
    bins = np.digitize(speed, bins=edges, right=True).astype(int)

    # Quantile edges can still collapse classes when many ties exist.
    # Fall back to stable rank buckets to guarantee exactly n_bins non-empty classes.
    if np.unique(bins).shape[0] != n_bins:
        order = np.argsort(speed, kind="mergesort")
        bins = np.empty_like(order, dtype=np.int64)
        for b_idx, idx_chunk in enumerate(np.array_split(order, n_bins)):
            bins[idx_chunk] = b_idx

    return bins, edges


def build_label_speed_table_from_hoyo(
    sample_meta: list[SampleMeta],
    baseline_label: str,
    use_relative: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for meta in sample_meta:
        grouped[meta.label].append(float(meta.speed_scale))

    raw: dict[str, float] = {}
    for label in INSTRUCTION_ONOMATOPEIA:
        values = grouped.get(label, [])
        if values:
            raw[label] = float(np.median(np.asarray(values, dtype=np.float32)))

    if not use_relative:
        return raw, dict(raw)

    baseline = raw.get(baseline_label, 0.0)
    if baseline <= 1e-8:
        fallback = max(raw.values()) if raw else 1.0
        baseline = float(fallback if fallback > 1e-8 else 1.0)
        print(
            f"[Warn] baseline label '{baseline_label}' not usable ({raw.get(baseline_label)}). "
            f"Fallback baseline={baseline:.6f}."
        )
    relative = {label: float(value / baseline) for label, value in raw.items()}
    return raw, relative


def regression_cv_predictions(z: np.ndarray, speed: np.ndarray, cv_folds: int, seed: int) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, z, speed, cv=cv, n_jobs=1)
    return pred


def fit_regression_model(z: np.ndarray, speed: np.ndarray):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(z, speed)
    return model


def classification_cv_predictions(z: np.ndarray, bins: np.ndarray, cv_folds: int, seed: int) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, z, bins, cv=cv, n_jobs=1)
    return pred


def evaluate_regression(z: np.ndarray, speed: np.ndarray, cv_folds: int, seed: int) -> tuple[dict[str, float], np.ndarray]:
    pred = regression_cv_predictions(z, speed, cv_folds=cv_folds, seed=seed)
    rho, _ = spearmanr(speed, pred)
    metrics = {
        "r2": float(r2_score(speed, pred)),
        "mae": float(mean_absolute_error(speed, pred)),
        "spearman_rho": float(rho),
    }
    return metrics, pred


def evaluate_classification(
    z: np.ndarray, bins: np.ndarray, cv_folds: int, seed: int
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    pred = classification_cv_predictions(z, bins, cv_folds=cv_folds, seed=seed)
    labels = np.arange(int(max(np.max(bins), np.max(pred))) + 1)
    metrics = {
        "macro_f1": float(f1_score(bins, pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(bins, pred)),
    }
    cm = confusion_matrix(bins, pred, labels=labels)
    return metrics, pred, cm


def permutation_test_regression(
    z: np.ndarray,
    speed: np.ndarray,
    observed_rho: float,
    cv_folds: int,
    seed: int,
    n_perm: int,
) -> float:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_perm):
        y_perm = rng.permutation(speed)
        pred = regression_cv_predictions(z, y_perm, cv_folds=cv_folds, seed=seed)
        rho, _ = spearmanr(y_perm, pred)
        stats.append(float(rho))
    stats_arr = np.asarray(stats, dtype=np.float64)
    p = (float(np.sum(stats_arr >= observed_rho)) + 1.0) / (n_perm + 1.0)
    return p


def permutation_test_classification(
    z: np.ndarray,
    bins: np.ndarray,
    observed_f1: float,
    cv_folds: int,
    seed: int,
    n_perm: int,
) -> float:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_perm):
        y_perm = rng.permutation(bins)
        pred = classification_cv_predictions(z, y_perm, cv_folds=cv_folds, seed=seed)
        f1 = f1_score(y_perm, pred, average="macro")
        stats.append(float(f1))
    stats_arr = np.asarray(stats, dtype=np.float64)
    p = (float(np.sum(stats_arr >= observed_f1)) + 1.0) / (n_perm + 1.0)
    return p


def permutation_test_label_order(
    gt: np.ndarray,
    pred: np.ndarray,
    observed_rho: float,
    n_perm: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_perm):
        gt_perm = rng.permutation(gt)
        rho, _ = spearmanr(gt_perm, pred)
        stats.append(float(rho))
    stats_arr = np.asarray(stats, dtype=np.float64)
    p = (float(np.sum(stats_arr >= observed_rho)) + 1.0) / (n_perm + 1.0)
    return p


def knn_speed_consistency(z: np.ndarray, speed: np.ndarray, k: int) -> dict[str, float]:
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sims = z_norm @ z_norm.T
    n = z.shape[0]
    abs_diffs = np.abs(speed[:, None] - speed[None, :])
    global_mean = float(abs_diffs[np.triu_indices(n, k=1)].mean())

    nn_diffs = []
    for i in range(n):
        order = np.argsort(-sims[i])
        order = order[order != i][:k]
        nn_diffs.append(float(np.abs(speed[order] - speed[i]).mean()))
    nn_mean = float(np.mean(nn_diffs))
    ratio = nn_mean / max(global_mean, 1e-8)
    return {
        "knn_speed_absdiff_mean": nn_mean,
        "global_speed_absdiff_mean": global_mean,
        "knn_vs_global_ratio": ratio,
    }


def assign_semantic_speed_class(
    value: float,
    fast_min: float,
    mid_min: float,
    mid_max: float,
    slow_max: float,
) -> int:
    # 0=slow, 1=mid, 2=fast
    if value >= fast_min:
        return 2
    if mid_min <= value <= mid_max:
        return 1
    if value <= slow_max:
        return 0

    # Fallback for uncovered gaps to keep all samples classifiable.
    if value > mid_max and value < fast_min:
        return 1
    if value > slow_max and value < mid_min:
        return 1
    return 1


def build_semantic_bins_from_labels(
    labels: np.ndarray,
    label_speed_relative: dict[str, float],
    fast_min: float,
    mid_min: float,
    mid_max: float,
    slow_max: float,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    label_to_class: dict[str, int] = {}
    for label in INSTRUCTION_ONOMATOPEIA:
        if label not in label_speed_relative:
            continue
        label_to_class[label] = assign_semantic_speed_class(
            float(label_speed_relative[label]), fast_min=fast_min, mid_min=mid_min, mid_max=mid_max, slow_max=slow_max
        )

    bins = np.asarray([label_to_class[str(label)] for label in labels], dtype=np.int64)
    counts = {"slow": int(np.sum(bins == 0)), "mid": int(np.sum(bins == 1)), "fast": int(np.sum(bins == 2))}
    return bins, label_to_class, counts


def compute_label_order_consistency(
    z: np.ndarray,
    labels_idx: np.ndarray,
    label_speed_target: dict[str, float],
    reg_model,
    n_perm: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    ordered_labels: list[str] = []
    centroids: list[np.ndarray] = []
    gt_speed: list[float] = []

    for label in INSTRUCTION_ONOMATOPEIA:
        if label not in label_speed_target:
            continue
        idx = INSTRUCTION_ONOMATOPEIA.index(label)
        mask = labels_idx == idx
        if not np.any(mask):
            continue
        ordered_labels.append(label)
        centroids.append(z[mask].mean(axis=0))
        gt_speed.append(float(label_speed_target[label]))

    if len(ordered_labels) < 3:
        raise RuntimeError("Need at least 3 labels for rank-based order consistency.")

    centroid_matrix = np.stack(centroids, axis=0)
    pred_speed = reg_model.predict(centroid_matrix).astype(np.float64)
    gt_arr = np.asarray(gt_speed, dtype=np.float64)

    rho, _ = spearmanr(gt_arr, pred_speed)
    p = permutation_test_label_order(gt_arr, pred_speed, float(rho), n_perm=n_perm, seed=seed)

    gt_rank = np.argsort(np.argsort(-gt_arr)) + 1
    pred_rank = np.argsort(np.argsort(-pred_speed)) + 1
    rank_delta = pred_rank - gt_rank

    detail_rows = []
    for i, label in enumerate(ordered_labels):
        detail_rows.append(
            {
                "label": label,
                "gt_speed": float(gt_arr[i]),
                "pred_speed": float(pred_speed[i]),
                "gt_rank": int(gt_rank[i]),
                "pred_rank": int(pred_rank[i]),
                "rank_delta": int(rank_delta[i]),
            }
        )

    metrics = {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_labels": int(len(ordered_labels)),
        "gt_speed_by_label": {label: float(gt_arr[i]) for i, label in enumerate(ordered_labels)},
        "pred_speed_by_label": {label: float(pred_speed[i]) for i, label in enumerate(ordered_labels)},
    }
    details = {
        "rows": detail_rows,
    }
    return metrics, details, gt_arr, pred_speed, ordered_labels


def save_samples_csv(
    out_path: Path,
    sample_meta: list[SampleMeta],
    speed_bins: np.ndarray,
    semantic_bins: np.ndarray,
    z: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_id", "label", "label_idx", "speed_scale", "speed_bin", "semantic_speed_class"]
        header.extend([f"z_{i}" for i in range(z.shape[1])])
        writer.writerow(header)
        for meta, speed_bin, sem_bin, vec in zip(sample_meta, speed_bins, semantic_bins, z):
            row = [
                meta.sample_id,
                meta.label,
                meta.label_idx,
                meta.speed_scale,
                int(speed_bin),
                int(sem_bin),
            ]
            row.extend([float(v) for v in vec])
            writer.writerow(row)


def plot_pca_speed(z: np.ndarray, speed: np.ndarray, out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(z)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=speed, cmap="viridis", s=20, alpha=0.8)
    ax.set_title("PCA 2D (color = speed_scale)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("speed_scale")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pca_labels(z: np.ndarray, labels_idx: np.ndarray, out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(z)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(INSTRUCTION_ONOMATOPEIA):
        mask = labels_idx == idx
        if not np.any(mask):
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, color=cmap(idx % cmap.N), label=label)
    ax.set_title("PCA 2D (color = onomatopoeia)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_speed_distribution(sample_meta: list[SampleMeta], out_path: Path) -> None:
    grouped: dict[str, list[float]] = {label: [] for label in INSTRUCTION_ONOMATOPEIA}
    for m in sample_meta:
        grouped[m.label].append(m.speed_scale)
    labels = [label for label in INSTRUCTION_ONOMATOPEIA if grouped[label]]
    data = [grouped[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_title("Speed Scale Distribution per Onomatopoeia")
    ax.set_ylabel("speed_scale")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path, class_names: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names[: cm.shape[1]])
    ax.set_yticklabels(class_names[: cm.shape[0]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_label_centroid_similarity(
    z: np.ndarray,
    labels_idx: np.ndarray,
    sample_meta: list[SampleMeta],
    out_path: Path,
) -> None:
    label_to_speed: dict[str, list[float]] = {label: [] for label in INSTRUCTION_ONOMATOPEIA}
    for m in sample_meta:
        label_to_speed[m.label].append(m.speed_scale)
    label_speed_mean = {
        label: float(np.mean(vals)) for label, vals in label_to_speed.items() if len(vals) > 0
    }
    ordered = sorted(label_speed_mean.keys(), key=lambda x: label_speed_mean[x], reverse=True)

    centroids = []
    for label in ordered:
        idx = INSTRUCTION_ONOMATOPEIA.index(label)
        mask = labels_idx == idx
        if not np.any(mask):
            continue
        c = z[mask].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids.append(c)
    cmat = np.stack(centroids, axis=0)
    sim = cmat @ cmat.T

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ordered, fontsize=8)
    ax.set_title("Label Centroid Cosine Similarity (sample-speed ordered)")
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_label_order_scatter(gt: np.ndarray, pred: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gt, pred, s=42, alpha=0.85)
    for x, y, label in zip(gt, pred, labels):
        ax.text(x, y, label, fontsize=8, alpha=0.9)
    ax.set_xlabel("GT relative speed (label table)")
    ax.set_ylabel("Predicted speed from label centroid")
    ax.set_title("Label-level Speed Order Consistency")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_")


def run_single_analysis(
    run_name: str,
    hoyo_root: Path,
    out_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = hoyo_root / "joint_training_results" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    stats_path = run_dir / "normalization_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    run_cfg = resolve_run_config(run_dir)
    dataset = load_dataset_for_run(hoyo_root, run_cfg, stats_path)
    sample_meta = collect_sample_metadata(
        dataset=dataset,
        hoyo_root=hoyo_root,
        fps=args.speed_fps,
        smooth_window=args.speed_smooth_window,
        height_percentile=args.speed_height_percentile,
        min_height_ratio=args.speed_min_height_ratio,
    )
    z, labels_idx = extract_latents(
        dataset=dataset,
        run_dir=run_dir,
        target_len=int(run_cfg["target_len"]),
        device=device,
        batch_size=args.batch_size,
    )

    expected_labels = np.asarray([m.label_idx for m in sample_meta], dtype=np.int64)
    if not np.array_equal(labels_idx, expected_labels):
        raise RuntimeError("Extracted latent labels do not match metadata labels.")

    speed = np.asarray([m.speed_scale for m in sample_meta], dtype=np.float32)
    labels_name = np.asarray([m.label for m in sample_meta], dtype=object)

    use_relative = args.use_relative_speed_table == "on"
    label_speed_raw, label_speed_target = build_label_speed_table_from_hoyo(
        sample_meta,
        baseline_label=args.speed_table_baseline_label,
        use_relative=use_relative,
    )

    reg_metrics, reg_pred = evaluate_regression(z, speed, cv_folds=args.cv_folds, seed=args.seed)
    reg_model = fit_regression_model(z, speed)

    label_rank_metrics, label_rank_details, label_gt_arr, label_pred_arr, label_order = compute_label_order_consistency(
        z=z,
        labels_idx=labels_idx,
        label_speed_target=label_speed_target,
        reg_model=reg_model,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    speed_bins, speed_bin_edges = compute_speed_bins(speed, n_bins=args.n_bins)
    cls_metrics, cls_pred, cls_cm = evaluate_classification(
        z,
        speed_bins,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )

    semantic_bins, label_to_semantic_class, semantic_counts = build_semantic_bins_from_labels(
        labels=labels_name,
        label_speed_relative=label_speed_target,
        fast_min=args.semantic_fast_min,
        mid_min=args.semantic_mid_min,
        mid_max=args.semantic_mid_max,
        slow_max=args.semantic_slow_max,
    )
    sem_cls_metrics, sem_cls_pred, sem_cls_cm = evaluate_classification(
        z,
        semantic_bins,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )

    knn_metrics = knn_speed_consistency(z, speed, k=args.knn_k)

    p_reg = permutation_test_regression(
        z,
        speed,
        observed_rho=reg_metrics["spearman_rho"],
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_perm=args.n_perm,
    )
    p_cls = permutation_test_classification(
        z,
        speed_bins,
        observed_f1=cls_metrics["macro_f1"],
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_perm=args.n_perm,
    )
    p_sem_cls = permutation_test_classification(
        z,
        semantic_bins,
        observed_f1=sem_cls_metrics["macro_f1"],
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_perm=args.n_perm,
    )

    primary_value = (
        float(label_rank_metrics["spearman_rho"])
        if args.primary_metric == "label_rank_rho"
        else float(reg_metrics["spearman_rho"])
    )
    primary_p = (
        float(label_rank_metrics["p_value"])
        if args.primary_metric == "label_rank_rho"
        else float(p_reg)
    )

    pass_check = (
        primary_value >= args.pass_rho
        and cls_metrics["macro_f1"] >= args.pass_macro_f1
        and primary_p < args.pass_p
        and p_cls < args.pass_p
    )

    metrics = {
        "run_name": run_name,
        "n_samples": int(z.shape[0]),
        "n_dim": int(z.shape[1]),
        "primary_metric_name": args.primary_metric,
        "speed_target_type": "relative_label_table" if use_relative else "raw_label_median",
        "regression": reg_metrics,
        "classification": {
            **cls_metrics,
            "speed_bin_edges": [float(v) for v in speed_bin_edges],
            "n_bins": int(args.n_bins),
        },
        "label_rank": label_rank_metrics,
        "semantic_classification": {
            **sem_cls_metrics,
            "class_counts": semantic_counts,
            "label_to_class": label_to_semantic_class,
            "class_name_map": {"0": "slow", "1": "mid", "2": "fast"},
            "thresholds": {
                "fast_min": float(args.semantic_fast_min),
                "mid_min": float(args.semantic_mid_min),
                "mid_max": float(args.semantic_mid_max),
                "slow_max": float(args.semantic_slow_max),
            },
        },
        "knn": knn_metrics,
        "permutation_p": {
            "spearman_rho_p": float(p_reg),
            "macro_f1_p": float(p_cls),
            "label_rank_rho_p": float(label_rank_metrics["p_value"]),
            "semantic_macro_f1_p": float(p_sem_cls),
        },
        "thresholds": {
            "rho": float(args.pass_rho),
            "macro_f1": float(args.pass_macro_f1),
            "p": float(args.pass_p),
        },
        "pass_speed_reflection": bool(pass_check),
    }

    run_out = out_root / f"run_{sanitize_name(run_name)}"
    run_out.mkdir(parents=True, exist_ok=True)

    (run_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_out / "label_speed_table_raw.json").write_text(
        json.dumps(label_speed_raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_out / "label_speed_table_relative.json").write_text(
        json.dumps(label_speed_target, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_out / "label_order_metrics.json").write_text(
        json.dumps(label_rank_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rank_csv = run_out / "label_order_rank_table.csv"
    with open(rank_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "gt_speed", "pred_speed", "gt_rank", "pred_rank", "rank_delta"],
        )
        writer.writeheader()
        for row in label_rank_details["rows"]:
            writer.writerow(row)

    save_samples_csv(
        run_out / "samples.csv",
        sample_meta=sample_meta,
        speed_bins=speed_bins,
        semantic_bins=semantic_bins,
        z=z,
    )
    np.save(run_out / "regression_oof_pred.npy", reg_pred)
    np.save(run_out / "classification_oof_pred.npy", cls_pred)
    np.save(run_out / "semantic_classification_oof_pred.npy", sem_cls_pred)
    np.save(run_out / "speed_bins.npy", speed_bins)
    np.save(run_out / "semantic_speed_bins.npy", semantic_bins)

    plot_pca_speed(z, speed, run_out / "pca_speed.png")
    plot_pca_labels(z, labels_idx, run_out / "pca_label.png")
    plot_speed_distribution(sample_meta, run_out / "speed_distribution_by_label.png")
    plot_confusion(
        cls_cm,
        run_out / "speed_bin_confusion.png",
        class_names=["low", "mid", "high"],
        title="3-bin Speed Probe Confusion",
    )
    plot_confusion(
        sem_cls_cm,
        run_out / "semantic_speed_confusion.png",
        class_names=["slow", "mid", "fast"],
        title="Semantic Speed-Class Probe Confusion (Aux)",
    )
    plot_label_centroid_similarity(z, labels_idx, sample_meta, run_out / "label_centroid_similarity_speed_order.png")
    plot_label_order_scatter(label_gt_arr, label_pred_arr, label_order, run_out / "label_order_scatter.png")

    sample_ids = np.asarray([m.sample_id for m in sample_meta], dtype=np.int64)
    return {
        "run_name": run_name,
        "sample_ids": sample_ids,
        "labels": labels_name,
        "speed": speed,
        "metrics": metrics,
        "out_dir": run_out,
    }


def write_comparison_outputs(
    out_root: Path,
    run_a: dict[str, Any],
    run_b: dict[str, Any],
) -> None:
    a = run_a["metrics"]
    b = run_b["metrics"]

    rows = [
        ("sample_spearman_rho", a["regression"]["spearman_rho"], b["regression"]["spearman_rho"]),
        ("regression_r2", a["regression"]["r2"], b["regression"]["r2"]),
        ("regression_mae", a["regression"]["mae"], b["regression"]["mae"]),
        (
            "probe_macro_f1",
            a["classification"]["macro_f1"],
            b["classification"]["macro_f1"],
        ),
        (
            "probe_balanced_acc",
            a["classification"]["balanced_acc"],
            b["classification"]["balanced_acc"],
        ),
        ("perm_p_rho", a["permutation_p"]["spearman_rho_p"], b["permutation_p"]["spearman_rho_p"]),
        ("perm_p_macro_f1", a["permutation_p"]["macro_f1_p"], b["permutation_p"]["macro_f1_p"]),
        ("label_rank_rho_aux", a["label_rank"]["spearman_rho"], b["label_rank"]["spearman_rho"]),
        ("perm_p_label_rank", a["permutation_p"]["label_rank_rho_p"], b["permutation_p"]["label_rank_rho_p"]),
        ("semantic_macro_f1_aux", a["semantic_classification"]["macro_f1"], b["semantic_classification"]["macro_f1"]),
        ("perm_p_semantic_f1", a["permutation_p"]["semantic_macro_f1_p"], b["permutation_p"]["semantic_macro_f1_p"]),
        ("knn_vs_global_ratio", a["knn"]["knn_vs_global_ratio"], b["knn"]["knn_vs_global_ratio"]),
    ]

    csv_path = out_root / "comparison_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", run_a["run_name"], run_b["run_name"], "delta_a_minus_b"])
        for metric, va, vb in rows:
            writer.writerow([metric, va, vb, va - vb])

    pass_a = bool(a["pass_speed_reflection"])
    pass_b = bool(b["pass_speed_reflection"])
    primary_name = a["primary_metric_name"]
    pa = a["regression"]["spearman_rho"] if primary_name == "sample_rho" else a["label_rank"]["spearman_rho"]
    pb = b["regression"]["spearman_rho"] if primary_name == "sample_rho" else b["label_rank"]["spearman_rho"]
    primary_better = run_a["run_name"] if pa >= pb else run_b["run_name"]

    md_lines = [
        "# MotionCLIP Latent Speed Separation Comparison",
        "",
        "## Runs",
        f"- A (new): `{run_a['run_name']}`",
        f"- B (old): `{run_b['run_name']}`",
        "",
        "## Primary Decision Metric",
        f"- primary_metric_name: `{primary_name}`",
        f"- better_by_primary: `{primary_better}`",
        "",
        "## Key Metrics",
        "",
        f"| Metric | {run_a['run_name']} | {run_b['run_name']} | Delta (A-B) |",
        "|---|---:|---:|---:|",
    ]
    for metric, va, vb in rows:
        md_lines.append(f"| {metric} | {va:.6f} | {vb:.6f} | {(va - vb):.6f} |")

    md_lines.extend(
        [
            "",
            "## Threshold Decision",
            f"- A pass: `{pass_a}`",
            f"- B pass: `{pass_b}`",
            "",
            "## Probe Bin Edges (Quantile 3-bin)",
            f"- A: `{a['classification']['speed_bin_edges']}`",
            f"- B: `{b['classification']['speed_bin_edges']}`",
            "",
            "## Semantic Class Counts (Aux)",
            f"- A: `{a['semantic_classification']['class_counts']}`",
            f"- B: `{b['semantic_classification']['class_counts']}`",
            "",
            "## Output Paths",
            f"- A outputs: `{run_a['out_dir']}`",
            f"- B outputs: `{run_b['out_dir']}`",
            f"- Comparison CSV: `{csv_path}`",
        ]
    )
    (out_root / "comparison_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hoyo_root = (REPO_ROOT / args.hoyo_root).resolve()
    out_root = (REPO_ROOT / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    print(f"[Info] Run A: {args.run_a}")
    print(f"[Info] Run B: {args.run_b}")

    run_a = run_single_analysis(args.run_a, hoyo_root, out_root, args, device)
    run_b = run_single_analysis(args.run_b, hoyo_root, out_root, args, device)

    if not np.array_equal(run_a["sample_ids"], run_b["sample_ids"]):
        raise RuntimeError("Run A/B sample IDs are not aligned. Comparison must use identical samples.")
    if not np.array_equal(run_a["labels"], run_b["labels"]):
        raise RuntimeError("Run A/B label order mismatch.")
    if not np.allclose(run_a["speed"], run_b["speed"], atol=1e-8):
        raise RuntimeError("Run A/B speed vectors mismatch.")

    write_comparison_outputs(out_root, run_a, run_b)
    print(f"[Done] Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
