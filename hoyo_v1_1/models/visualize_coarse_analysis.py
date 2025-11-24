import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # for Japanese labels
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Coarse styles definitions
COARSE_GROUPS = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["とぼとぼ", "のろのろ"],  # 「通常」はスタイルなしとして除外
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}
COARSE_LABELS = list(COARSE_GROUPS.keys())

# Map fine labels (11) to coarse labels (4)
# Must match the order in INSTRUCTION_ONOMATOPEIA
FINE_LABELS = [
    "通常", "すたすた", "せかせか", "てくてく", "どっしどっし", 
    "とぼとぼ", "のしのし", "のろのろ", "ぶらぶら", "よたよた", "よろよろ"
]
FINE_TO_COARSE_IDX = {}
for i, fine in enumerate(FINE_LABELS):
    for c_idx, coarse in enumerate(COARSE_LABELS):
        if fine in COARSE_GROUPS[coarse]:
            FINE_TO_COARSE_IDX[i] = c_idx
            break

def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Coarse 4-Style Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")

def plot_scatter(z, y, method_name, labels, out_path, prototypes=None):
    plt.figure(figsize=(8, 6))
    
    # Plot samples
    for i, label in enumerate(labels):
        mask = (y == i)
        if not np.any(mask): continue
        plt.scatter(z[mask, 0], z[mask, 1], label=label, s=15, alpha=0.6)
        
    # Plot prototypes if available
    if prototypes is not None:
        plt.scatter(prototypes[:, 0], prototypes[:, 1], 
                   c='black', marker='*', s=200, label='Prototypes', edgecolors='white')

    plt.title(f"Latent Space Visualization ({method_name})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {method_name} plot to {out_path}")

def plot_cosine_distances(z_m, z_s_cls, y_true, labels, out_path):
    # z_m: (N, D), z_s_cls: (K, D)
    # Compute cosine similarity for each sample against ALL prototypes
    # sims[i, k] = cos(z_m[i], proto[k])
    sims = z_m @ z_s_cls.T  # (N, K)
    
    # Gather correct vs incorrect similarities
    data = []
    for i in range(len(z_m)):
        true_cls = y_true[i]
        for k in range(len(labels)):
            sim = sims[i, k]
            is_target = (k == true_cls)
            data.append({
                "Similarity": sim,
                "Type": "Target" if is_target else "Others",
                "Class": labels[true_cls]
            })
            
    import pandas as pd
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Class', y='Similarity', hue='Type', 
                order=labels, palette={"Target": "green", "Others": "gray"})
    plt.title("Cosine Similarity: Target Prototype vs Others")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved cosine distance boxplot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load snapshot
    data = np.load(args.snapshot, allow_pickle=True)
    z_m = data["z_m"]
    labels_idx = data["labels_idx"]
    label_list = [str(l) for l in data["label_list"]]

    # Decide whether labels_idx は fine(11) か coarse(4) か
    if len(label_list) == len(COARSE_LABELS) and all(a == b for a, b in zip(label_list, COARSE_LABELS)):
        # すでに coarse 4 ラベルで保存されているケース（今回の joint_coarse）
        labels_idx_coarse = labels_idx
    else:
        # fine 11 ラベルから coarse 4 ラベルへマッピング
        labels_idx_coarse = np.array([FINE_TO_COARSE_IDX[i] for i in labels_idx])
    
    # Prototypes (semantic)
    z_s_cls = data["z_s_cls"]  # (K, D)
    
    # If z_s_cls is already coarse (K=4), use it directly
    if z_s_cls.shape[0] == len(COARSE_LABELS):
        z_s_coarse = z_s_cls
    # If z_s_cls is fine (K=11), aggregate to coarse
    elif z_s_cls.shape[0] == len(FINE_LABELS):
        z_s_coarse_list = []
        for c_lab in COARSE_LABELS:
            fine_indices = [i for i, f in enumerate(FINE_LABELS) if f in COARSE_GROUPS[c_lab]]
            vecs = z_s_cls[fine_indices]
            mean_vec = np.mean(vecs, axis=0)
            z_s_coarse_list.append(mean_vec)
        z_s_coarse = np.stack(z_s_coarse_list, axis=0)
    else:
        raise ValueError(f"Unexpected shape for z_s_cls: {z_s_cls.shape}")

    z_s_coarse = normalize(z_s_coarse)
    
    # 1. Confusion Matrix (Coarse)
    # Predict by nearest coarse prototype
    sims = z_m @ z_s_coarse.T
    preds = sims.argmax(axis=1)
    plot_confusion_matrix(labels_idx_coarse, preds, COARSE_LABELS, out_dir / "confusion_matrix_coarse.png")
    
    # 2. PCA & t-SNE
    # Combine z_m and prototypes for projection
    z_all = np.concatenate([z_m, z_s_coarse], axis=0)
    n_samples = len(z_m)
    
    # PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_all)
    plot_scatter(z_pca[:n_samples], labels_idx_coarse, "PCA", COARSE_LABELS, 
                 out_dir / "pca_coarse.png", prototypes=z_pca[n_samples:])
    
    # t-SNE (only if enough samples)
    if n_samples > 50:
        tsne = TSNE(n_components=2, perplexity=min(30, n_samples//5), random_state=42)
        z_tsne = tsne.fit_transform(z_all)
        plot_scatter(z_tsne[:n_samples], labels_idx_coarse, "t-SNE", COARSE_LABELS,
                     out_dir / "tsne_coarse.png", prototypes=z_tsne[n_samples:])
        
    # 3. Cosine Similarity Boxplot
    plot_cosine_distances(z_m, z_s_coarse, labels_idx_coarse, COARSE_LABELS, 
                          out_dir / "cosine_similarity_boxplot.png")

if __name__ == "__main__":
    main()

