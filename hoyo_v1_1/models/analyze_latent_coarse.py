import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def compute_cluster_stats(
    z_m: np.ndarray,
    labels_idx: np.ndarray,
    label_list: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute per-cluster mean / variance and class counts.

    Returns a dict[label] -> {"mean": (D,), "var": (D,), "count": int}
    """
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    num_labels = len(label_list)
    for k in range(num_labels):
        name = str(label_list[k])
        mask = labels_idx == k
        if not np.any(mask):
            continue
        z_k = z_m[mask]
        mean = z_k.mean(axis=0)
        var = z_k.var(axis=0)
        stats[name] = {
            "mean": mean,
            "var": var,
            "count": np.array([z_k.shape[0]], dtype=np.int64),
        }
    return stats


def compute_distance_matrices(
    means: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a dict[label] -> mean vector, compute:
      - label_list: array of labels (ordered)
      - euclidean_dists[i, j]
      - cosine_dists[i, j] = 1 - cosine_sim
    """
    labels = sorted(means.keys())
    mu = np.stack([means[l] for l in labels], axis=0)  # (L, D)

    # Euclidean distances
    diff = mu[:, None, :] - mu[None, :, :]  # (L, L, D)
    euclid = np.linalg.norm(diff, axis=-1)  # (L, L)

    # Cosine distances
    mu_norm = mu / np.linalg.norm(mu, axis=-1, keepdims=True)
    cos_sim = mu_norm @ mu_norm.T
    cos_dist = 1.0 - cos_sim
    return np.array(labels), euclid, cos_dist


def supervised_fisher_ratio(
    z_m: np.ndarray,
    labels_idx: np.ndarray,
    num_labels: int,
) -> float:
    """
    粗い指標として，全体分散とクラス内分散の比を 1 スカラーで返す．

    - 全体分散: z_m 全体の各次元の分散を平均したもの
    - クラス内分散: 各クラスの分散を平均したもの
    """
    overall_var = z_m.var(axis=0).mean()
    within_vars = []
    for k in range(num_labels):
        mask = labels_idx == k
        if not np.any(mask):
            continue
        within_vars.append(z_m[mask].var(axis=0).mean())
    if not within_vars:
        return float("nan")
    within_mean = float(np.mean(within_vars))
    if within_mean == 0.0:
        return float("inf")
    return float(overall_var / within_mean)


def latent_editing_direction(
    stats: Dict[str, Dict[str, np.ndarray]],
    src_label: str,
    dst_label: str,
) -> np.ndarray:
    """
    Compute a simple editing direction v = mu_dst - mu_src.
    """
    if src_label not in stats or dst_label not in stats:
        raise ValueError(f"Editing direction requires both {src_label} and {dst_label} in stats.")
    return stats[dst_label]["mean"] - stats[src_label]["mean"]


def analyze_editing_effects(
    z_m: np.ndarray,
    labels_idx: np.ndarray,
    label_list: np.ndarray,
    z_s_cls: np.ndarray,
    src_label: str,
    dst_label: str,
    alphas=(0.0, 0.5, 1.0, 2.0),
) -> str:
    """
    For a given source and destination label, apply z' = z + alpha * v
    on source-class samples and report how the cosine similarity to
    semantic prototypes changes as alpha increases.
    """
    name_to_idx = {str(l): i for i, l in enumerate(label_list)}
    if src_label not in name_to_idx or dst_label not in name_to_idx:
        return f"[latent_editing] labels {src_label} / {dst_label} not found in label_list; skip.\n"

    stats = compute_cluster_stats(z_m, labels_idx, label_list)
    v = latent_editing_direction(stats, src_label, dst_label)

    src_idx = name_to_idx[src_label]
    src_mask = labels_idx == src_idx
    if not np.any(src_mask):
        return f"[latent_editing] no samples for label {src_label}; skip.\n"
    z_src = z_m[src_mask]  # (Ns, D)

    # semantic prototypes are assumed already normalized
    z_sem = z_s_cls  # (L, D)
    lines = []
    lines.append(f"[latent_editing] src={src_label}, dst={dst_label}")
    for alpha in alphas:
        z_edit = z_src + alpha * v
        # normalize edited vectors
        z_edit = z_edit / np.linalg.norm(z_edit, axis=-1, keepdims=True)
        cos = z_edit @ z_sem.T  # (Ns, L)
        cos_mean = cos.mean(axis=0)  # (L,)
        best_idx = int(np.argmax(cos_mean))
        best_label = str(label_list[best_idx])
        lines.append(
            f"  alpha={alpha:.2f}: mean cos per class="
            + ", ".join(
                f"{str(label_list[i])}={cos_mean[i]:.3f}" for i in range(len(label_list))
            )
            + f" -> argmax={best_label}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze coarse 4-style motion latents.")
    parser.add_argument(
        "--snapshot",
        type=str,
        default="hoyo_v1_1/joint_training_results/latent_snapshot_final.npz",
        help="Path to latent snapshot .npz file produced by train_motionclip_joint.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="hoyo_v1_1/joint_training_results/latent_analysis",
        help="Directory to save analysis outputs (CSV / TXT).",
    )
    parser.add_argument(
        "--src-label",
        type=str,
        default=None,
        help="Source label name for latent editing (e.g., '遅い系'). "
        "If not provided, tries the first label in label_list.",
    )
    parser.add_argument(
        "--dst-label",
        type=str,
        default=None,
        help="Destination label name for latent editing (e.g., '速い系'). "
        "If not provided, tries the last label in label_list.",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(snapshot_path, allow_pickle=True)
    z_m = data["z_m"]  # (N, D)
    labels_idx = data["labels_idx"]  # (N,)
    splits = data["splits"]  # (N,)
    label_list = data["label_list"]  # (L,)
    z_s_cls = data["z_s_cls"]  # (L, D)

    # 1) cluster stats
    stats = compute_cluster_stats(z_m, labels_idx, label_list)
    stats_csv = out_dir / "cluster_stats.csv"
    with stats_csv.open("w", encoding="utf-8") as f:
        header = ["label", "count", "mean_norm", "mean_var"]
        f.write(",".join(header) + "\n")
        for name in sorted(stats.keys()):
            s = stats[name]
            mean = s["mean"]
            var = s["var"]
            count = int(s["count"][0])
            mean_norm = float(np.linalg.norm(mean))
            mean_var = float(var.mean())
            f.write(f"{name},{count},{mean_norm:.6f},{mean_var:.6f}\n")

    # 2) distance matrices
    label_names, euclid, cos_dist = compute_distance_matrices(
        {k: v["mean"] for k, v in stats.items()}
    )
    dist_euclid_csv = out_dir / "class_dist_euclidean.csv"
    dist_cosine_csv = out_dir / "class_dist_cosine.csv"
    with dist_euclid_csv.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(label_names) + "\n")
        for i, name in enumerate(label_names):
            row = ",".join(f"{euclid[i, j]:.6f}" for j in range(len(label_names)))
            f.write(f"{name},{row}\n")
    with dist_cosine_csv.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(label_names) + "\n")
        for i, name in enumerate(label_names):
            row = ",".join(f"{cos_dist[i, j]:.6f}" for j in range(len(label_names)))
            f.write(f"{name},{row}\n")

    # 3) simple Fisher-like ratio
    fisher = supervised_fisher_ratio(z_m, labels_idx, num_labels=len(label_list))
    fisher_txt = out_dir / "fisher_ratio.txt"
    with fisher_txt.open("w", encoding="utf-8") as f:
        f.write(f"supervised Fisher-like ratio (overall_var / within_var_mean) = {fisher:.6f}\n")

    # 4) latent editing (semantic side only, no decoder)
    if args.src_label is None:
        src_label = str(label_list[0])
    else:
        src_label = args.src_label
    if args.dst_label is None:
        dst_label = str(label_list[-1])
    else:
        dst_label = args.dst_label

    editing_report = analyze_editing_effects(
        z_m=z_m,
        labels_idx=labels_idx,
        label_list=label_list,
        z_s_cls=z_s_cls,
        src_label=src_label,
        dst_label=dst_label,
    )
    editing_txt = out_dir / "latent_editing.txt"
    with editing_txt.open("w", encoding="utf-8") as f:
        f.write(editing_report)

    # 5) basic split statistics
    split_txt = out_dir / "split_counts.txt"
    with split_txt.open("w", encoding="utf-8") as f:
        unique_splits = sorted(set(splits.tolist()))
        for split_name in unique_splits:
            mask = splits == split_name
            f.write(f"{split_name}: {int(mask.sum())} samples\n")
            for k, lab in enumerate(label_list):
                m2 = mask & (labels_idx == k)
                if np.any(m2):
                    f.write(f"  {lab}: {int(m2.sum())}\n")

    print(f"[analyze_latent_coarse] wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()


