import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


def pca_2d(x: np.ndarray):
    """
    Simple PCA to 2D using SVD (no external dependencies).
    Returns projected points, mean vector, and top-2 principal directions.
    """
    x_mean = x.mean(axis=0, keepdims=True)
    x_center = x - x_mean
    # x_center = U S Vt, rows of Vt are principal directions
    _, _, Vt = np.linalg.svd(x_center, full_matrices=False)
    components = Vt[:2]  # (2, D)
    x_2d = x_center @ components.T  # (N, 2)
    return x_2d, x_mean, components


def main():
    parser = argparse.ArgumentParser(description="Visualize motion latents (PCA 2D scatter).")
    parser.add_argument(
        "--snapshot",
        type=str,
        default="hoyo_v1_1/joint_training_results/latent_snapshot_final.npz",
        help="Path to latent snapshot .npz file produced by train_motionclip_joint.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="latent_pca.png",
        help="Output PNG path for the visualization.",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    data = np.load(snapshot_path, allow_pickle=True)
    z_m = data["z_m"]  # (N, D)
    labels_idx = data["labels_idx"]  # (N,)
    splits = data["splits"]  # (N,)
    label_list = data["label_list"]  # (L,)
    z_s_cls = data["z_s_cls"]  # (L, D)

    # PCA to 2D on motion latents
    z_2d, mean_vec, components = pca_2d(z_m)
    # Project semantic prototypes into the same 2D space
    z_s_center = z_s_cls - mean_vec
    z_s_2d = z_s_center @ components.T  # (L, 2)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for visualization. "
            "Please install it with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8, 6))
    label_list = [str(l) for l in label_list]

    # Plot motion latents per split / label
    markers = {"train": "o", "test": "x"}
    for split_name in sorted(set(splits)):
        split_mask = splits == split_name
        for lab_idx, lab in enumerate(label_list):
            mask = split_mask & (labels_idx == lab_idx)
            if not np.any(mask):
                continue
            plt.scatter(
                z_2d[mask, 0],
                z_2d[mask, 1],
                s=10,
                alpha=0.6,
                marker=markers.get(split_name, "."),
                label=f"{lab} ({split_name})",
            )

    # Plot semantic prototypes as large stars
    plt.scatter(
        z_s_2d[:, 0],
        z_s_2d[:, 1],
        s=120,
        marker="*",
        edgecolors="k",
        facecolors="none",
        linewidths=1.0,
        label="sem prototypes",
    )

    plt.title("Motion / Semantic Latent Space (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=8)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved latent visualization to {out_path}")


if __name__ == "__main__":
    main()


