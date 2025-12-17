#!/usr/bin/env python3
"""
PCA解析スクリプト: Sarashina vs SigLIP のオノマトペ埋め込み比較

出力:
- 各PCの寄与率（Explained Variance Ratio）
- 各オノマトペのPC座標
- カテゴリ（速い系、遅い系、重い系、ふらふら系）の重心
- 軸解釈レポート
- 比較プロット（Sarashina vs SigLIP サブプロット）
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "hoyo_v1_1" / "viz" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# オノマトペリスト（HOYO instruction 11語）
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

# カテゴリ分け（train_motionclip_joint.py の COARSE_GROUPS と同じ）
COARSE_GROUPS: Dict[str, List[str]] = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["とぼとぼ", "のろのろ"],
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}

# カテゴリごとの色
CATEGORY_COLORS = {
    "速い系": "tab:blue",
    "遅い系": "tab:orange",
    "重い系": "tab:red",
    "ふらふら系": "tab:green",
    "その他": "gray",
}


@dataclass
class PCAResult:
    """PCA解析結果を格納するデータクラス"""
    coords_2d: np.ndarray          # (N, 2) - PC1, PC2 座標
    explained_variance: np.ndarray  # 全PCの寄与率
    eigenvectors: np.ndarray       # 固有ベクトル（主成分方向）
    mean: np.ndarray               # 元データの平均
    labels: List[str]              # ラベル（オノマトペ）
    encoder_name: str              # エンコーダ名


def get_category(word: str) -> str:
    """オノマトペからカテゴリを取得"""
    for cat, words in COARSE_GROUPS.items():
        if word in words:
            return cat
    return "その他"


def encode_sarashina(texts: List[str], device: torch.device) -> np.ndarray:
    """Sarashina でテキストをエンコード"""
    model_id = "sbintuitions/sarashina-embedding-v2-1b"
    model = SentenceTransformer(model_id, device=str(device))
    emb = model.encode(texts, convert_to_tensor=True, device=device)
    emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def encode_siglip(texts: List[str], device: torch.device) -> np.ndarray:
    """SigLIP でテキストをエンコード"""
    from transformers import AutoTokenizer, SiglipTextModel

    model_id = "google/siglip-base-patch16-256-multilingual"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_model = SiglipTextModel.from_pretrained(model_id).to(device)
    text_model.eval()

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = text_model(**encoded)

    # CLS token (first token) の埋め込みを使用
    emb = out.last_hidden_state[:, 0, :]
    emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def pca_full(x: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCAを実行し、詳細な結果を返す
    
    Returns:
        coords: 変換後の座標 (N, n_components)
        explained_variance_ratio: 各PCの寄与率
        eigenvectors: 固有ベクトル（列が主成分方向）
        mean: データの平均
    """
    x = x.astype(np.float64)
    n_samples, n_features = x.shape
    
    if n_components is None:
        n_components = min(n_samples, n_features)
    
    # 中心化
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean
    
    # 共分散行列
    cov = x_centered.T @ x_centered / (n_samples - 1)
    
    # 固有値分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # 降順にソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 寄与率
    total_var = eigvals.sum()
    explained_variance_ratio = eigvals / total_var
    
    # 変換
    W = eigvecs[:, :n_components]
    coords = x_centered @ W
    
    return coords, explained_variance_ratio, eigvecs, x_mean


def compute_category_centroids(coords: np.ndarray, labels: List[str]) -> Dict[str, np.ndarray]:
    """カテゴリごとの重心を計算"""
    centroids = {}
    for cat in COARSE_GROUPS.keys():
        members = COARSE_GROUPS[cat]
        indices = [i for i, lab in enumerate(labels) if lab in members]
        if indices:
            centroids[cat] = coords[indices].mean(axis=0)
    return centroids


def interpret_axis(coords: np.ndarray, labels: List[str], axis_idx: int) -> Dict[str, float]:
    """
    PC軸の解釈を試みる
    各カテゴリの平均位置を計算し、その軸上での並びを返す
    """
    axis_values = coords[:, axis_idx]
    
    category_means = {}
    for cat, members in COARSE_GROUPS.items():
        indices = [i for i, lab in enumerate(labels) if lab in members]
        if indices:
            category_means[cat] = float(np.mean(axis_values[indices]))
    
    return category_means


def analyze_embeddings(device: torch.device) -> Tuple[PCAResult, PCAResult]:
    """Sarashina と SigLIP の埋め込みをPCA解析"""
    
    # テキスト準備（「〜と歩いている。」テンプレート）
    texts = [
        f"{w}と歩いている。" if w != "通常" else "普通に歩いている。"
        for w in MOTION_ONOMATOPOEIA
    ]
    
    print("=== テキスト一覧 ===")
    for t in texts:
        print(f"  {t}")
    print()
    
    # Sarashina エンコード
    print("Encoding with Sarashina...")
    emb_sarashina = encode_sarashina(texts, device)
    print(f"  Embedding shape: {emb_sarashina.shape}")
    
    # SigLIP エンコード
    print("Encoding with SigLIP...")
    emb_siglip = encode_siglip(texts, device)
    print(f"  Embedding shape: {emb_siglip.shape}")
    
    # PCA (Sarashina)
    print("\nPCA on Sarashina embeddings...")
    coords_s, var_s, eigvec_s, mean_s = pca_full(emb_sarashina)
    result_sarashina = PCAResult(
        coords_2d=coords_s[:, :2],
        explained_variance=var_s,
        eigenvectors=eigvec_s,
        mean=mean_s,
        labels=MOTION_ONOMATOPOEIA,
        encoder_name="Sarashina",
    )
    
    # PCA (SigLIP)
    print("PCA on SigLIP embeddings...")
    coords_g, var_g, eigvec_g, mean_g = pca_full(emb_siglip)
    result_siglip = PCAResult(
        coords_2d=coords_g[:, :2],
        explained_variance=var_g,
        eigenvectors=eigvec_g,
        mean=mean_g,
        labels=MOTION_ONOMATOPOEIA,
        encoder_name="SigLIP",
    )
    
    return result_sarashina, result_siglip


def plot_comparison(result_s: PCAResult, result_g: PCAResult, out_path: Path):
    """Sarashina vs SigLIP の比較プロット"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, result in zip(axes, [result_s, result_g]):
        coords = result.coords_2d
        labels = result.labels
        var = result.explained_variance
        
        # カテゴリごとにプロット
        for i, (x, y) in enumerate(coords):
            lab = labels[i]
            cat = get_category(lab)
            color = CATEGORY_COLORS.get(cat, "gray")
            ax.scatter(x, y, c=color, s=80, zorder=3)
            ax.annotate(
                lab,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                weight="bold" if cat != "その他" else "normal",
            )
        
        # カテゴリ重心をプロット
        centroids = compute_category_centroids(coords, labels)
        for cat, centroid in centroids.items():
            color = CATEGORY_COLORS[cat]
            ax.scatter(
                centroid[0], centroid[1],
                c=color, s=200, marker="X", edgecolors="black", linewidth=2,
                alpha=0.7, zorder=2, label=f"{cat} 重心"
            )
        
        # 軸ラベルに寄与率を表示
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=12)
        ax.set_title(f"{result.encoder_name} 埋め込みのPCA", fontsize=14)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    
    # 共通の凡例
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=10, label=cat)
        for cat, color in CATEGORY_COLORS.items()
        if cat != "その他"
    ]
    handles.append(
        plt.Line2D([0], [0], marker="X", linestyle="", color="black", markersize=12, label="カテゴリ重心")
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {out_path}")


def generate_report(result_s: PCAResult, result_g: PCAResult, out_path: Path):
    """軸解釈レポートを生成"""
    
    lines = []
    lines.append("=" * 60)
    lines.append("PCA軸解析レポート: オノマトペ埋め込みの主成分解釈")
    lines.append("=" * 60)
    lines.append("")
    
    for result in [result_s, result_g]:
        lines.append(f"### {result.encoder_name} ###")
        lines.append("-" * 40)
        
        # 寄与率
        lines.append("\n■ 寄与率（Explained Variance Ratio）:")
        cumsum = 0.0
        for i, var in enumerate(result.explained_variance[:5]):
            cumsum += var
            lines.append(f"   PC{i+1}: {var*100:5.2f}%  (累積: {cumsum*100:5.2f}%)")
        
        # PC座標
        lines.append("\n■ 各オノマトペのPC座標:")
        lines.append(f"   {'オノマトペ':<12} {'PC1':>8} {'PC2':>8}  カテゴリ")
        lines.append("   " + "-" * 45)
        for i, lab in enumerate(result.labels):
            x, y = result.coords_2d[i]
            cat = get_category(lab)
            lines.append(f"   {lab:<12} {x:>8.4f} {y:>8.4f}  ({cat})")
        
        # カテゴリ重心
        centroids = compute_category_centroids(result.coords_2d, result.labels)
        lines.append("\n■ カテゴリ重心:")
        lines.append(f"   {'カテゴリ':<12} {'PC1':>8} {'PC2':>8}")
        lines.append("   " + "-" * 30)
        for cat, centroid in centroids.items():
            lines.append(f"   {cat:<12} {centroid[0]:>8.4f} {centroid[1]:>8.4f}")
        
        # 軸解釈
        lines.append("\n■ 軸解釈（カテゴリ平均位置）:")
        for pc_idx in range(2):
            cat_means = interpret_axis(result.coords_2d, result.labels, pc_idx)
            # ソートして表示
            sorted_cats = sorted(cat_means.items(), key=lambda x: x[1])
            lines.append(f"\n   PC{pc_idx+1}軸 (低 → 高):")
            for cat, val in sorted_cats:
                lines.append(f"      {val:+.4f}: {cat}")
        
        lines.append("\n")
    
    # 解釈のヒント
    lines.append("=" * 60)
    lines.append("■ 軸解釈のヒント")
    lines.append("=" * 60)
    lines.append("""
PC軸の意味は、カテゴリの配置から推測できます：

- 「速い系」と「遅い系」が対極に位置する軸 → 速度軸
- 「ふらふら系」が他と分離している軸 → 安定性軸
- 「重い系」が特定方向に寄っている軸 → 重さ/力強さ軸

上記のカテゴリ平均位置を見て、どのPCがどの質感を捉えているか
解釈してみてください。
""")
    
    report_text = "\n".join(lines)
    
    # ファイル出力
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    # ターミナルにも出力
    print("\n" + report_text)
    print(f"\nSaved report to: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 解析実行
    result_sarashina, result_siglip = analyze_embeddings(device)
    
    # 比較プロット
    plot_path = OUTPUT_DIR / "pca_comparison_sarashina_siglip.png"
    plot_comparison(result_sarashina, result_siglip, plot_path)
    
    # レポート生成
    report_path = OUTPUT_DIR / "pca_axis_interpretation.txt"
    generate_report(result_sarashina, result_siglip, report_path)
    
    print("\n=== 解析完了 ===")
    print(f"プロット: {plot_path}")
    print(f"レポート: {report_path}")


if __name__ == "__main__":
    main()


