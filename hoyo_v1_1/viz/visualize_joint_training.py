import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path
import torch.nn.functional as F

# Import MotionCLIP dynamically
REPO_ROOT = Path(__file__).resolve().parents[2]
HOYO_ROOT = REPO_ROOT / "hoyo_v1_1"
MOTIONCLIP_ROOT = REPO_ROOT / "MotionCLIP"
if str(MOTIONCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONCLIP_ROOT))

from src.models.get_model import get_model as motionclip_get_model

# Local modules
sys.path.append(str(REPO_ROOT))
from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    encode_semantics_sarashina,
    INSTRUCTION_ONOMATOPEIA,
)
from hoyo_v1_1.models.train_motionclip_joint import load_motionclip_full_model

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
    
    hoyo_root = HOYO_ROOT
    res_dir = hoyo_root / "joint_training_results"
    if not res_dir.exists():
        print(f"Error: Results directory {res_dir} not found. Run training first.")
        return

    # 1. Load Model & Projector
    target_len = 60
    model, params = load_motionclip_full_model(device, target_len)
    
    model_path = res_dir / "motionclip_full_joint.pth"
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found.")
        return
        
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded Joint Trained MotionCLIP Model.")

    # Load Projector
    sem_proj_path = res_dir / "sem_proj_joint.pth"
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
                
                all_z_motion.append(z.cpu().numpy()[0])
                all_labels.append(lab)
                
                if i < 1: # Save first sample of each label for recon check
                    # Denormalize
                    rec = out["output"].cpu().numpy()[0].transpose(2, 0, 1) # (T, 14, 2)
                    rec = rec * data_std + data_mean
                    orig = arr * data_std + data_mean
                    all_recons.append(rec)
                    all_originals.append(orig)

    all_z_motion = np.stack(all_z_motion, axis=0)
    
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
    plt.savefig(res_dir / "joint_space_pca.png", dpi=150)
    plt.close()
    print(f"Saved PCA plot to {res_dir / 'joint_space_pca.png'}")

    # 5. Visualization 2: Reconstruction Quality
    # Plot trajectories of original vs reconstruction for a few classes
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, lab in enumerate(unique_labels[:12]): # Limit to first 12
        if i >= len(axes): break
        ax = axes[i]
        
        orig = all_originals[i]
        rec = all_recons[i]
        
        # Plot simple trajectory of one joint (e.g., Root or Foot)
        # shape: (T, 14, 2) -> Take joint 0 (Hips?) and joint 3 (LeftFoot?)
        # Assuming COCO-17 like layout, but indices might vary. Just picking joint 0 and 7 (extremities)
        
        # Original
        ax.plot(orig[:, 0, 0], orig[:, 0, 1], 'b-', alpha=0.5, label='Orig J0')
        ax.plot(orig[:, 7, 0], orig[:, 7, 1], 'c-', alpha=0.5, label='Orig J7')
        
        # Recon
        ax.plot(rec[:, 0, 0], rec[:, 0, 1], 'r--', label='Rec J0')
        ax.plot(rec[:, 7, 0], rec[:, 7, 1], 'm--', label='Rec J7')
        
        ax.set_title(f"Reconstruction: {lab}")
        ax.legend(fontsize='small')
        ax.axis('equal')
        
    plt.tight_layout()
    plt.savefig(res_dir / "reconstruction_check.png", dpi=150)
    plt.close()
    print(f"Saved Reconstruction plot to {res_dir / 'reconstruction_check.png'}")
    
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
    plt.savefig(res_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix to {res_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    visualize_joint_results()

