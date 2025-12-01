import os
import sys
import json
import inspect
from collections import namedtuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path
import torch.nn.functional as F
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Compatibility patches for old libraries (e.g., chumpy, smplx) on Python 3.11+.
# 1) chumpy still uses inspect.getargspec, which was removed in 3.11.
# 2) chumpy imports deprecated NumPy aliases (np.int, np.float, ...).
# We recreate these shims here so that MotionCLIP can be loaded.
# ---------------------------------------------------------------------------
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _compat_getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _compat_getargspec  # type: ignore[attr-defined]

_legacy_numpy_types = [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
]
for _name, _type in _legacy_numpy_types:
    if not hasattr(np, _name):
        setattr(np, _name, _type)

# Import MotionCLIP dynamically
ROOT = Path(__file__).resolve().parents[1]
MOTIONCLIP_ROOT = ROOT / "MotionCLIP"
if str(MOTIONCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONCLIP_ROOT))

from src.models.get_model import get_model as motionclip_get_model

# Reuse existing components
try:
    from hoyo_v1_1.hoyo_sem_motion_contrastive_motionclip import (
        HoyoInstructionDataset,
        encode_semantics_sarashina,
        INSTRUCTION_ONOMATOPEIA
    )
    from hoyo_v1_1.train_motionclip_joint import load_motionclip_full_model
except ImportError:
    from hoyo_sem_motion_contrastive_motionclip import (
        HoyoInstructionDataset,
        encode_semantics_sarashina,
        INSTRUCTION_ONOMATOPEIA
    )
    from train_motionclip_joint import load_motionclip_full_model

def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    cov = x_centered.T @ x_centered / (x_centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    W = eigvecs[:, idx]
    x_2d = x_centered @ W
    return x_2d

def visualize_joint_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    hoyo_root = ROOT / "hoyo_v1_1"
    res_dir = hoyo_root / "joint_training_results"
    ckpt_dir = res_dir / "checkpoints"
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not res_dir.exists():
        print(f"Error: Results directory {res_dir} not found. Run training first.")
        return

    # 1. Load Model & Projector
    target_len = 60
    model, params = load_motionclip_full_model(device, target_len)

    # Prefer FINAL / BEST checkpoints if available
    model_candidates = [
        "motionclip_full_joint_final.pth",
        "motionclip_full_joint_best.pth",
        "motionclip_full_joint.pth",
    ]
    model_path = None
    for name in model_candidates:
        p_new = ckpt_dir / name
        p_old = res_dir / name
        if p_new.exists():
            model_path = p_new
            break
        if p_old.exists():
            model_path = p_old
            break
    if model_path is None:
        print("Error: No joint-trained MotionCLIP checkpoint found.")
        return
    print(f"Loading MotionCLIP model from: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded Joint Trained MotionCLIP Model.")

    # Load Projector (prefer FINAL / BEST if available)
    sem_proj_candidates = [
        "sem_proj_joint_final.pth",
        "sem_proj_joint_best.pth",
        "sem_proj_joint.pth",
    ]
    sem_proj_path = None
    for name in sem_proj_candidates:
        p_new = ckpt_dir / name
        p_old = res_dir / name
        if p_new.exists():
            sem_proj_path = p_new
            break
        if p_old.exists():
            sem_proj_path = p_old
            break
    if sem_proj_path is None:
        print("Error: No sem_proj checkpoint found.")
        return
    print(f"Loading semantic projector from: {sem_proj_path}")
    d_motion = model.latent_dim
    # Dummy encode to get dim
    sem_emb_dummy = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA[:1], device)
    d_sem = sem_emb_dummy.shape[1]
    
    sem_proj = torch.nn.Linear(d_sem, d_motion, bias=False).to(device)
    sem_proj.load_state_dict(torch.load(sem_proj_path, map_location=device))
    sem_proj.eval()
    print("Loaded Semantic Projector.")
    
    # Load Normalization Stats
    with open(res_dir / "normalization_stats.json", "r") as f:
        stats = json.load(f)
        data_mean = np.array(stats["mean"])
        data_std = np.array(stats["std"])

    # 2. Prepare Data
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=target_len)
    
    # Normalize
    for lab in INSTRUCTION_ONOMATOPEIA:
        new_samples = []
        for arr in dataset.samples_by_label[lab]:
            norm_arr = (arr - data_mean) / data_std
            new_samples.append(norm_arr)
        dataset.samples_by_label[lab] = new_samples

    # 3. Compute Embeddings (Motion & Text)
    sem_emb_all = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device)
    with torch.no_grad():
        z_text = sem_proj(sem_emb_all)
        z_text = F.normalize(z_text, dim=-1).cpu().numpy()
        
    all_z_motion = []
    all_labels = []
    all_recons = []
    all_originals = []

    # Reconstruction error stats (per fine label)
    mpjpe_sums = {lab: 0.0 for lab in INSTRUCTION_ONOMATOPEIA}
    mpjpe_counts = {lab: 0 for lab in INSTRUCTION_ONOMATOPEIA}
    
    print("Computing motion embeddings...")
    with torch.no_grad():
        for lab in INSTRUCTION_ONOMATOPEIA:
            samples = dataset.samples_by_label[lab]
            # Pick a few samples for recon viz
            for i, arr in enumerate(samples):
                # (T, 14, 2) -> (1, 14, 2, T)
                x_np = arr.transpose(1, 2, 0)[None, ...]
                x = torch.from_numpy(x_np).to(device).float()
                
                batch = {
                    "x": x,
                    "mask": torch.ones((1, target_len), dtype=torch.bool, device=device),
                    "lengths": torch.full((1,), target_len, dtype=torch.long, device=device),
                    "y": torch.zeros((1,), dtype=torch.long, device=device)
                }
                
                out = model(batch)
                z = out["mu"]
                z = F.normalize(z, dim=-1)

                # Motion latent for joint PCA
                all_z_motion.append(z.cpu().numpy()[0])
                all_labels.append(lab)

                # Reconstruction in normalized HOYO座標系 (T, 14, 2)
                rec_norm = out["output"].cpu().numpy()[0].transpose(2, 0, 1)
                orig_norm = arr  # 既に (T, 14, 2) で正規化済み

                # MPJPE（2D）: 各フレーム・各関節での L2 距離の平均
                diff = rec_norm - orig_norm
                mpjpe = np.linalg.norm(diff, axis=2).mean()
                mpjpe_sums[lab] += float(mpjpe)
                mpjpe_counts[lab] += 1

                # 可視化用に最初のサンプルだけ保存（実座標に戻す）
                if i < 1:
                    rec = rec_norm * data_std + data_mean
                    orig = orig_norm * data_std + data_mean
                    all_recons.append(rec)
                    all_originals.append(orig)

    all_z_motion = np.stack(all_z_motion, axis=0)

    # ------ Print reconstruction stats (fine + coarse) ------
    print("\n[Reconstruction MPJPE (2D, normalized units)]")
    for lab in INSTRUCTION_ONOMATOPEIA:
        if mpjpe_counts[lab] == 0:
            continue
        mean_err = mpjpe_sums[lab] / mpjpe_counts[lab]
        print(f"  {lab}: {mean_err:.4f}  (N={mpjpe_counts[lab]})")

    # Coarse 4-style aggregation
    COARSE_GROUPS = {
        "速い系": ["すたすた", "せかせか", "てくてく"],
        "遅い系": ["とぼとぼ", "のろのろ"],
        "重い系": ["どっしどっし", "のしのし"],
        "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
    }
    print("\n[Reconstruction MPJPE aggregated per coarse style]")
    for coarse, fines in COARSE_GROUPS.items():
        num = 0
        denom = 0.0
        for fl in fines:
            if mpjpe_counts.get(fl, 0) > 0:
                num += mpjpe_counts[fl]
                denom += mpjpe_sums[fl]
        if num == 0:
            continue
        print(f"  {coarse}: {denom/num:.4f}  (N={num})")
    
    # 4. Visualization 1: Latent Space (PCA)
    # Combine Text and Motion for joint plot
    n_text = len(INSTRUCTION_ONOMATOPEIA)
    n_motion = len(all_z_motion)
    
    combined_feats = np.concatenate([z_text, all_z_motion], axis=0)
    combined_pca = pca_2d(combined_feats)
    
    pca_text = combined_pca[:n_text]
    pca_motion = combined_pca[n_text:]
    
    plt.figure(figsize=(10, 10))
    
    # Plot Motion points
    unique_labels = INSTRUCTION_ONOMATOPEIA
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, lab in enumerate(unique_labels):
        indices = [j for j, l in enumerate(all_labels) if l == lab]
        plt.scatter(pca_motion[indices, 0], pca_motion[indices, 1], 
                   color=colors[i], alpha=0.5, s=20, label=f"{lab} (Motion)")
        
        # Plot Text Anchor
        plt.scatter(pca_text[i, 0], pca_text[i, 1], 
                   color=colors[i], marker="X", s=200, edgecolors='black', linewidth=1.5)
        plt.text(pca_text[i, 0], pca_text[i, 1], lab, fontsize=12, weight='bold')

    plt.title("Joint Latent Space: Motion (dots) vs Text (X)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(fig_dir / "joint_space_pca.png", dpi=150)
    plt.close()
    print(f"Saved PCA plot to {fig_dir / 'joint_space_pca.png'}")

    # 5. Visualization 2: Reconstruction Quality
    # Plot trajectories of original vs reconstruction for a few classes.
    #  - 統一スケールで 2 関節 (J0, J7) の軌跡を比較
    #  - 開始 / 終了点をマーカーで明示して、どこからどこまで歩いたかを可視化
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()

    # 全サンプル共通の表示レンジ（比較しやすくするため）
    xs_all = []
    ys_all = []
    for arr in all_originals:
        xs_all.append(arr[:, :, 0].reshape(-1))
        ys_all.append(arr[:, :, 1].reshape(-1))
    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)
    x_min, x_max = xs_all.min(), xs_all.max()
    y_min, y_max = ys_all.min(), ys_all.max()

    for i, lab in enumerate(unique_labels[:12]):  # Limit to first 12
        if i >= len(axes):
            break
        ax = axes[i]

        orig = all_originals[i]  # (T, 14, 2)
        rec = all_recons[i]      # (T, 14, 2)

        # Original (solid lines)
        ax.plot(orig[:, 0, 0], orig[:, 0, 1], "b-", alpha=0.6, label="Orig J0")
        ax.plot(orig[:, 7, 0], orig[:, 7, 1], "c-", alpha=0.6, label="Orig J7")

        # Reconstruction (dashed lines)
        ax.plot(rec[:, 0, 0], rec[:, 0, 1], "r--", alpha=0.8, label="Rec J0")
        ax.plot(rec[:, 7, 0], rec[:, 7, 1], "m--", alpha=0.8, label="Rec J7")

        # Start / end markers（J0 のみ）
        ax.scatter(orig[0, 0, 0], orig[0, 0, 1], c="k", s=20, marker="o")
        ax.scatter(orig[-1, 0, 0], orig[-1, 0, 1], c="k", s=20, marker="x")

        ax.set_title(f"Reconstruction: {lab}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        # 凡例は左上のサブプロットだけに表示して、図全体のゴチャゴチャを減らす
        if i == 0:
            ax.legend(fontsize="small", loc="upper left")

    plt.tight_layout()
    plt.savefig(fig_dir / "reconstruction_check.png", dpi=150)
    plt.close()
    print(f"Saved Reconstruction plot to {fig_dir / 'reconstruction_check.png'}")
    
    # 6. Metric: Alignment Accuracy (Top-1 Retrieval)
    # Motion -> Text
    scores = all_z_motion @ z_text.T # (N_motion, N_text)
    preds = np.argmax(scores, axis=1)
    
    gt_indices = [INSTRUCTION_ONOMATOPEIA.index(l) for l in all_labels]
    correct = (preds == gt_indices).sum()
    acc = correct / len(all_labels)
    
    print(f"Alignment Accuracy (Motion -> Text): {acc:.2%}")
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt_indices, preds)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Acc={acc:.2%})")
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45)
    plt.yticks(tick_marks, unique_labels)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix to {fig_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    visualize_joint_results()

