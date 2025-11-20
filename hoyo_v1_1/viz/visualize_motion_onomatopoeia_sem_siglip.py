import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, SiglipModel

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "hoyo_v1_1" / "viz" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOTION_ONOMATOPOEIA = [
    "通常",
    "すたすた",
    "せかせか",
    "てくてく",
    "どっしどっし",
    "とぼとぼ",
    "のしのし",
    "のろのろ",
    "ぶらぶら",
    "よたよた",
    "よろよろ",
]


def encode_siglip(texts: List[str], device: torch.device, model_id: str = "google/siglip-base-patch16-224") -> torch.Tensor:
    """
    Encode Japanese texts using SigLIP text tower.
    Returns L2-normalized embeddings of shape (len(texts), D).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = SiglipModel.from_pretrained(model_id).to(device)
    model.eval()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        embeddings = F.normalize(outputs, dim=-1)

    return embeddings


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


def plot_motion_onoma(coords: np.ndarray, labels: List[str], out_path: Path):
    os.makedirs(out_path.parent, exist_ok=True)

    cat_map: Dict[str, str] = {}
    fast = {"すたすた", "せかせか", "てくてく"}
    slow = {"のろのろ", "とぼとぼ"}
    heavy = {"どっしどっし", "のしのし"}
    unstable = {"よたよた", "よろよろ"}
    aimless = {"ぶらぶら"}

    for w in labels:
        if w in fast:
            cat_map[w] = "速い"
        elif w in slow:
            cat_map[w] = "遅い"
        elif w in heavy:
            cat_map[w] = "重い"
        elif w in unstable:
            cat_map[w] = "ふらつき"
        elif w in aimless:
            cat_map[w] = "ぶらぶら系"
        else:
            cat_map[w] = "その他"

    cat_to_color = {
        "速い": "tab:blue",
        "遅い": "tab:orange",
        "重い": "tab:red",
        "ふらつき": "tab:green",
        "ぶらぶら系": "tab:purple",
        "その他": "gray",
    }

    plt.figure(figsize=(6, 6))
    for (x, y), label in zip(coords, labels):
        color = cat_to_color.get(cat_map[label], "gray")
        plt.scatter(x, y, c=color, s=40)
        plt.text(x + 0.02, y + 0.02, label, fontsize=10)

    plt.title("移動系オノマトペ（SigLIP 埋め込み, PCA2D）")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=cat)
        for cat, color in cat_to_color.items()
    ]
    plt.legend(handles=handles, title="ざっくりカテゴリ", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    texts = [f"{w}と歩いている。" if w != "通常" else "普通に歩いている。" for w in MOTION_ONOMATOPOEIA]
    print("Texts:")
    for t in texts:
        print(" ", t)

    emb = encode_siglip(texts, device=device).cpu().numpy()
    coords = pca_2d(emb)

    out_path = OUTPUT_DIR / "hoyo_motion_onomatopoeia_sem_pca_siglip.png"
    plot_motion_onoma(coords, MOTION_ONOMATOPOEIA, out_path)
    print("\nSaved SigLIP onomatopoeia semantic plot to:", out_path)


if __name__ == "__main__":
    main()

