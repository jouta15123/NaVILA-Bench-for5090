import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import japanize_matplotlib


# Make project root importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    INSTRUCTION_ONOMATOPEIA,
    encode_semantics_sarashina,
    apply_normalization_from_stats,
)
from hoyo_v1_1.models import train_motionclip_joint as joint_mod


def pca_2d(x: np.ndarray):
    """
    Simple PCA to 2D using SVD (no external dependencies).
    Returns projected points, mean vector, and top-2 principal directions.
    """
    x_mean = x.mean(axis=0, keepdims=True)
    x_center = x - x_mean
    _, _, Vt = np.linalg.svd(x_center, full_matrices=False)
    components = Vt[:2]  # (2, D)
    x_2d = x_center @ components.T  # (N, 2)
    return x_2d, x_mean, components


def main():
    parser = argparse.ArgumentParser(description="Evaluate best joint model: PCA + confusion matrix.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(joint_mod.HOYO_ROOT / "joint_training_results"),
        help="Directory where checkpoints and outputs are stored.",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=60,
        help="Target sequence length used in training.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_dir / "normalization_stats.json"
    full_ckpt = ckpt_dir / "motionclip_full_joint_best.pth"
    sem_proj_ckpt = ckpt_dir / "sem_proj_joint_best.pth"

    if not full_ckpt.exists():
        raise FileNotFoundError(f"Best full model checkpoint not found: {full_ckpt}")
    if not sem_proj_ckpt.exists():
        raise FileNotFoundError(f"Best sem_proj checkpoint not found: {sem_proj_ckpt}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using checkpoints:\n  full={full_ckpt}\n  sem_proj={sem_proj_ckpt}")

    # Load dataset (full, without split) and apply same normalization as training
    hoyo_root = joint_mod.HOYO_ROOT
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=args.target_len)
    apply_normalization_from_stats(dataset, INSTRUCTION_ONOMATOPEIA, stats_path)

    # Load model and best weights
    model, _ = joint_mod.load_motionclip_full_model(device, target_len=args.target_len)
    state_full = torch.load(full_ckpt, map_location=device)
    model.load_state_dict(state_full)
    model.eval()

    # Build and load sem_proj
    sem_emb = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)
    d_motion = model.latent_dim
    d_sem = sem_emb.shape[1]

    sem_proj = torch.nn.Linear(d_sem, d_motion, bias=False).to(device)
    sem_proj.load_state_dict(torch.load(sem_proj_ckpt, map_location=device))
    sem_proj.eval()

    with torch.no_grad():
        z_s_cls = sem_proj(sem_emb)  # (L, D)
        z_s_cls = F.normalize(z_s_cls, dim=-1).cpu().numpy()

    # Collect motion latents for all samples
    zs = []
    ys = []
    for lab_idx, lab in enumerate(INSTRUCTION_ONOMATOPEIA):
        samples = dataset.samples_by_label.get(lab, [])
        for arr in samples:
            # arr: (T, 14, 2) -> (1, J, C, T)
            coords = arr[np.newaxis, ...].transpose(0, 2, 3, 1)
            x = torch.from_numpy(coords).to(device)

            B, _, _, Tcur = x.shape
            mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
            lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)

            batch = {
                "x": x,
                "mask": mask,
                "lengths": lengths,
                "y": torch.zeros((B,), dtype=torch.long, device=device),
            }

            with torch.no_grad():
                out = model(batch)
                z_m = out["mu"]
                z_m = F.normalize(z_m, dim=-1).squeeze(0).cpu().numpy()

            zs.append(z_m)
            ys.append(lab_idx)

    if not zs:
        print("No samples found in dataset; nothing to evaluate.")
        return

    Z = np.stack(zs, axis=0)  # (N, D)
    y_true = np.asarray(ys, dtype=np.int64)  # (N,)
    labels = np.asarray(INSTRUCTION_ONOMATOPEIA)
    num_classes = len(labels)

    # Predictions via semantic prototypes
    logits = Z @ z_s_cls.T  # (N, L)
    y_pred = logits.argmax(axis=1)
    overall_acc = float((y_pred == y_true).mean())

    # Top-3 accuracy
    top3_idx = np.argsort(-logits, axis=1)[:, :3]
    correct_top3 = np.any(top3_idx == y_true[:, None], axis=1)
    top3_acc = float(correct_top3.mean())

    # Confusion matrix (true x pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_acc = []
    for i in range(num_classes):
        total_i = cm[i].sum()
        acc_i = float(cm[i, i] / total_i) if total_i > 0 else 0.0
        per_class_acc.append(acc_i)

    print("=== Overall classification (best checkpoint) ===")
    print(f"Overall Acc@1: {overall_acc:.3f}")
    print(f"Overall Acc@3: {top3_acc:.3f}")
    print("Per-class Acc@1:")
    for lab, acc_i in zip(labels, per_class_acc):
        print(f"  {lab}: {acc_i:.3f}")

    # Save confusion matrix as a heatmap
    # Normalize confusion matrix row-wise for visualization
    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_norm,
        np.where(row_sums == 0, 1.0, row_sums),
        out=np.zeros_like(cm_norm),
        where=row_sums != 0,
    )

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(num_classes), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(num_classes), labels, fontsize=8)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (normalized)")
    plt.tight_layout()
    cm_path = fig_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    print(f"Saved confusion matrix plot to {cm_path}")
    plt.close()

    # PCA joint space (motion + semantic prototypes)
    Z_all = np.concatenate([Z, z_s_cls], axis=0)
    Z_2d, mean_vec, components = pca_2d(Z_all)
    Z_m_2d = Z_2d[: Z.shape[0]]
    Z_s_2d = Z_2d[Z.shape[0] :]

    plt.figure(figsize=(8, 6))
    # motion latents
    for i, lab in enumerate(labels):
        mask = y_true == i
        if not np.any(mask):
            continue
        plt.scatter(
            Z_m_2d[mask, 0],
            Z_m_2d[mask, 1],
            s=10,
            alpha=0.6,
            label=f"{lab}",
        )

    # semantic prototypes
    plt.scatter(
        Z_s_2d[:, 0],
        Z_s_2d[:, 1],
        s=120,
        marker="*",
        edgecolors="k",
        facecolors="none",
        linewidths=1.0,
        label="sem prototypes",
    )

    plt.title("Joint Motion / Semantic Latent Space (PCA 2D, best checkpoint)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=8)
    plt.tight_layout()
    pca_path = fig_dir / "joint_space_pca.png"
    plt.savefig(pca_path, dpi=200)
    print(f"Saved joint PCA plot to {pca_path}")
    plt.close()


if __name__ == "__main__":
    main()


