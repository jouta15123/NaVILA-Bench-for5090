#!/usr/bin/env python3
"""
比較可視化スクリプト: SigLIP vs Sarashina × Fine vs Coarse

複数の学習結果（latent_snapshot_final.npz）を読み込み、
2x2 サブプロットで比較可視化を行う。

使用例:
    python compare_encoder_results.py --results-dir /path/to/joint_training_results
    
    python compare_encoder_results.py \
        --snapshots sarashina_fine/latent_snapshot_final.npz \
                    sarashina_coarse/latent_snapshot_final.npz \
                    siglip_fine/latent_snapshot_final.npz \
                    siglip_coarse/latent_snapshot_final.npz
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score


# ラベル定義
FINE_LABELS = [
    "通常", "すたすた", "せかせか", "てくてく", "どっしどっし",
    "とぼとぼ", "のしのし", "のろのろ", "ぶらぶら", "よたよた", "よろよろ"
]

COARSE_GROUPS = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["通常", "とぼとぼ", "のろのろ"],  # 「通常」を「遅い系」として扱う
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}
COARSE_LABELS = list(COARSE_GROUPS.keys())

LABEL_TO_COARSE = {}
for coarse_name, fine_list in COARSE_GROUPS.items():
    LABEL_TO_COARSE[coarse_name] = coarse_name
    for fine in fine_list:
        LABEL_TO_COARSE[fine] = coarse_name

# カラーマップ
COARSE_COLORS = {
    "速い系": "#1f77b4",      # blue
    "遅い系": "#ff7f0e",      # orange
    "重い系": "#d62728",      # red
    "ふらふら系": "#2ca02c",  # green
}

FALLBACK_COLOR = "#7f7f7f"


@dataclass
class SnapshotData:
    """学習結果のスナップショットデータ"""
    z_m: np.ndarray           # モーション潜在ベクトル (N, D)
    labels_idx: np.ndarray    # ラベルインデックス (N,)
    label_list: List[str]     # ラベル名リスト
    z_s_cls: np.ndarray       # セマンティックプロトタイプ (K, D)
    splits: Optional[np.ndarray] = None  # train/test 分割情報
    name: str = ""            # 識別名（例: "sarashina_fine"）


def normalize(x: np.ndarray) -> np.ndarray:
    """L2正規化"""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_snapshot(path: Path, name: str = "") -> SnapshotData:
    """スナップショットファイルを読み込む"""
    data = np.load(path, allow_pickle=True)
    
    return SnapshotData(
        z_m=data["z_m"],
        labels_idx=data["labels_idx"],
        label_list=[str(l) for l in data["label_list"]],
        z_s_cls=data["z_s_cls"],
        splits=data.get("splits", None),
        name=name or path.parent.name,
    )


def filter_snapshot_by_splits(snapshot: SnapshotData, allowed_splits: Tuple[str, ...]) -> SnapshotData:
    """train/testなどの分割でフィルタリング"""
    if snapshot.splits is None or not allowed_splits:
        return snapshot

    split_arr = np.asarray(snapshot.splits).astype(str)
    mask = np.isin(split_arr, list(allowed_splits))
    if not mask.any():
        raise ValueError(f"Snapshot '{snapshot.name}' has no samples for splits {allowed_splits}")

    suffix = "+".join(sorted(set(allowed_splits)))
    new_name = snapshot.name if suffix in ("train+test", "test+train") else f"{snapshot.name} ({suffix})"

    return SnapshotData(
        z_m=snapshot.z_m[mask],
        labels_idx=snapshot.labels_idx[mask],
        label_list=snapshot.label_list,
        z_s_cls=snapshot.z_s_cls,
        splits=split_arr[mask],
        name=new_name,
    )


def detect_label_mode(snapshot: SnapshotData) -> str:
    """Fine (11) か Coarse (4) かを判定"""
    if len(snapshot.label_list) == len(COARSE_LABELS):
        if all(a == b for a, b in zip(snapshot.label_list, COARSE_LABELS)):
            return "coarse"
    return "fine"


def _map_samples_to_coarse(snapshot: SnapshotData) -> np.ndarray:
    """各サンプルを4クラスラベルにマッピング"""
    coarse_indices = []
    label_list = list(snapshot.label_list)
    for lbl_idx in snapshot.labels_idx:
        label_name = label_list[int(lbl_idx)]
        coarse_name = LABEL_TO_COARSE.get(label_name)
        if coarse_name is None:
            raise ValueError(f"Label '{label_name}' is not assigned to any coarse group.")
        coarse_indices.append(COARSE_LABELS.index(coarse_name))
    return np.asarray(coarse_indices, dtype=np.int64)


def _compute_centroids(z_vectors: np.ndarray, labels_idx: np.ndarray, num_classes: int) -> np.ndarray:
    """ラベルごとにモーション潜在の重心を計算"""
    centroids = np.zeros((num_classes, z_vectors.shape[1]), dtype=np.float32)
    for cls in range(num_classes):
        mask = labels_idx == cls
        if mask.any():
            centroid = z_vectors[mask].mean(axis=0, keepdims=True)
            centroids[cls] = normalize(centroid)[0]
    return centroids


def convert_to_coarse(snapshot: SnapshotData) -> Tuple[np.ndarray, np.ndarray]:
    """Fine/Coarseを問わず、Coarse 4クラス基準へマッピング（モーション重心ベース）"""
    labels_coarse = _map_samples_to_coarse(snapshot)
    z_m = normalize(snapshot.z_m)
    coarse_centroids = _compute_centroids(z_m, labels_coarse, len(COARSE_LABELS))
    return labels_coarse, coarse_centroids


def get_fine_centroids(snapshot: SnapshotData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fineラベル用にモーション潜在の重心を計算"""
    z_m = normalize(snapshot.z_m)
    labels_fine = snapshot.labels_idx.astype(int)
    centroids = _compute_centroids(z_m, labels_fine, len(snapshot.label_list))
    return z_m, centroids, labels_fine


def compute_metrics(snapshot: SnapshotData) -> Dict[str, float]:
    """各種メトリクスを計算（モーション重心ベース）"""
    z_m = normalize(snapshot.z_m)
    labels_idx, z_centroids = convert_to_coarse(snapshot)
    
    # Cosine similarity で予測
    sims = z_m @ z_centroids.T
    preds = sims.argmax(axis=1)
    
    # Accuracy
    acc = (preds == labels_idx).mean()
    
    # Top-3 Accuracy
    if sims.shape[1] >= 3:
        top3 = np.argsort(sims, axis=1)[:, -3:]
        acc_top3 = np.mean([labels_idx[i] in top3[i] for i in range(len(labels_idx))])
    else:
        acc_top3 = acc
    
    # Silhouette Score
    try:
        sil = silhouette_score(z_m, labels_idx)
    except Exception:
        sil = 0.0
    
    return {
        "accuracy": acc,
        "accuracy_top3": acc_top3,
        "silhouette": sil,
        "n_samples": len(z_m),
        "n_classes": len(COARSE_LABELS),
    }


def plot_pca_comparison(snapshots: List[SnapshotData], out_path: Path):
    """1x2 PCA 比較プロット（Sarashina vs SigLIP）
    
    モーションサンプルを4クラス（Coarse）で色分けして比較する。
    """
    n_snapshots = min(len(snapshots), 2)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(7 * n_snapshots, 6))
    
    if n_snapshots == 1:
        axes = [axes]
    
    titles = ["Sarashina", "SigLIP"]
    
    for idx, snapshot in enumerate(snapshots[:2]):
        ax = axes[idx]
        
        # PCA 実行（モーションサンプルのみでfit）
        z_m, z_fine_centroids, labels_fine = get_fine_centroids(snapshot)
        pca = PCA(n_components=2)
        z_pca_samples = pca.fit_transform(z_m)
        
        pc1_var = pca.explained_variance_ratio_[0] * 100
        pc2_var = pca.explained_variance_ratio_[1] * 100
        
        # Coarseラベルに変換して色分け
        labels_coarse, _ = convert_to_coarse(snapshot)
        
        # サンプルをCoarseグループごとにプロット
        plotted_labels = set()
        for coarse_idx, coarse_name in enumerate(COARSE_LABELS):
            mask = labels_coarse == coarse_idx
            if mask.sum() == 0:
                continue
            c = COARSE_COLORS.get(coarse_name, FALLBACK_COLOR)
            ax.scatter(
                z_pca_samples[mask, 0],
                z_pca_samples[mask, 1],
                c=c,
                label=coarse_name if coarse_name not in plotted_labels else None,
                s=20,
                alpha=0.6,
            )
            plotted_labels.add(coarse_name)
        
        # メトリクス計算（提供クラス数 / Coarse 4クラス）
        sims_fine = z_m @ z_fine_centroids.T
        preds_fine = sims_fine.argmax(axis=1)
        acc_fine = (preds_fine == labels_fine).mean()
        metrics = compute_metrics(snapshot)
        label_mode = detect_label_mode(snapshot)
        fine_label_desc = f"{len(snapshot.label_list)}クラス"
        
        ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=11)
        ax.set_title(
            f"{titles[idx]} ({label_mode})\n{fine_label_desc}: {acc_fine:.1%} / 4クラス: {metrics['accuracy']:.1%}",
            fontsize=12,
        )
        ax.grid(alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA comparison to: {out_path}")


def plot_confusion_comparison(snapshots: List[SnapshotData], out_path: Path):
    """2x2 Confusion Matrix 比較プロット
    
    上段: 11クラス（Fine）分類
    下段: 4クラス（Coarse）分類
    左列: Sarashina
    右列: SigLIP
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 最大2つのスナップショットを想定（Sarashina, SigLIP）
    for col_idx, snapshot in enumerate(snapshots[:2]):
        z_m, z_fine_centroids, labels_fine = get_fine_centroids(snapshot)
        
        # === 上段: 11クラス（Fine）混同行列 ===
        ax_fine = axes[0, col_idx]
        
        # Fine評価: 元のラベルとプロトタイプをそのまま使用
        sims_fine = z_m @ z_fine_centroids.T
        preds_fine = sims_fine.argmax(axis=1)
        fine_label_count = len(snapshot.label_list)
        cm_fine = confusion_matrix(
            labels_fine,
            preds_fine,
            labels=np.arange(fine_label_count),
            normalize="true",
        )
        cm_fine = np.nan_to_num(cm_fine)
        fine_labels_short = [name[:4] for name in snapshot.label_list]
        
        sns.heatmap(
            cm_fine,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=fine_labels_short,
            yticklabels=fine_labels_short,
            ax=ax_fine,
            annot_kws={"size": 7},
        )
        ax_fine.set_xlabel('Predicted', fontsize=10)
        ax_fine.set_ylabel('True', fontsize=10)
        ax_fine.tick_params(axis='both', labelsize=8)
        
        # 精度計算
        acc_fine = (preds_fine == labels_fine).mean()
        encoder_name = "Sarashina" if col_idx == 0 else "SigLIP"
        ax_fine.set_title(f'{encoder_name} - {fine_label_count}クラス (Acc: {acc_fine:.1%})', fontsize=12)
        
        # === 下段: 4クラス（Coarse）混同行列 ===
        ax_coarse = axes[1, col_idx]
        
        # Coarse変換
        labels_coarse, z_coarse_centroids = convert_to_coarse(snapshot)
        sims_coarse = z_m @ z_coarse_centroids.T
        preds_coarse = sims_coarse.argmax(axis=1)
        
        # 4クラスの混同行列
        cm_coarse = confusion_matrix(
            labels_coarse,
            preds_coarse,
            labels=np.arange(len(COARSE_LABELS)),
            normalize="true",
        )
        cm_coarse = np.nan_to_num(cm_coarse)
        
        sns.heatmap(
            cm_coarse,
            annot=True,
            fmt='.2f',
            cmap='Oranges',
            xticklabels=COARSE_LABELS,
            yticklabels=COARSE_LABELS,
            ax=ax_coarse,
            annot_kws={'size': 10},
        )
        ax_coarse.set_xlabel('Predicted', fontsize=10)
        ax_coarse.set_ylabel('True', fontsize=10)
        ax_coarse.tick_params(axis='both', labelsize=9)
        
        # 精度計算
        acc_coarse = (preds_coarse == labels_coarse).mean()
        ax_coarse.set_title(f'{encoder_name} - 4クラス (Acc: {acc_coarse:.1%})', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix comparison to: {out_path}")


def plot_metrics_comparison(snapshots: List[SnapshotData], out_path: Path):
    """メトリクス比較バープロット"""
    names = []
    accs = []
    sils = []
    
    for snapshot in snapshots:
        metrics = compute_metrics(snapshot)
        names.append(snapshot.name.replace("_", "\n"))
        accs.append(metrics["accuracy"])
        sils.append(metrics["silhouette"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(names))
    width = 0.6
    colors = ['#1f77b4' if 'sarashina' in s.name.lower() else '#ff7f0e' for s in snapshots]
    
    # Accuracy
    bars1 = axes[0].bar(x, accs, width, color=colors)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Classification Accuracy (Coarse 4-class)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance (25%)')
    axes[0].legend()
    
    # バーの上に値を表示
    for bar, val in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
    
    # Silhouette Score
    bars2 = axes[1].bar(x, sils, width, color=colors)
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (Cluster Separation)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylim(-0.2, 0.5)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars2, sils):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to: {out_path}")


def generate_summary_report(snapshots: List[SnapshotData], out_path: Path):
    """サマリーレポートを生成"""
    lines = []
    lines.append("=" * 70)
    lines.append("SigLIP vs Sarashina 比較実験サマリー")
    lines.append("=" * 70)
    lines.append("")
    
    # 各実験の結果
    for snapshot in snapshots:
        mode = detect_label_mode(snapshot)
        metrics = compute_metrics(snapshot)
        split_summary = "N/A"
        if snapshot.splits is not None:
            unique_splits = sorted(set(np.asarray(snapshot.splits).astype(str)))
            split_summary = ", ".join(unique_splits)
        
        lines.append(f"### {snapshot.name} ###")
        lines.append("-" * 50)
        lines.append(f"  Split(s): {split_summary}")
        lines.append(f"  Label Mode: {mode} ({len(snapshot.label_list)} classes)")
        lines.append(f"  Samples: {metrics['n_samples']}")
        lines.append(f"  Accuracy (Coarse 4-class): {metrics['accuracy']:.1%}")
        lines.append(f"  Accuracy Top-3: {metrics['accuracy_top3']:.1%}")
        lines.append(f"  Silhouette Score: {metrics['silhouette']:.4f}")
        lines.append("")
    
    # 比較サマリー
    lines.append("=" * 70)
    lines.append("比較サマリー")
    lines.append("=" * 70)
    
    # Sarashina vs SigLIP
    sarashina_results = [s for s in snapshots if "sarashina" in s.name.lower()]
    siglip_results = [s for s in snapshots if "siglip" in s.name.lower()]
    
    if sarashina_results and siglip_results:
        sar_acc = np.mean([compute_metrics(s)["accuracy"] for s in sarashina_results])
        sig_acc = np.mean([compute_metrics(s)["accuracy"] for s in siglip_results])
        
        lines.append(f"\nAverage Accuracy:")
        lines.append(f"  Sarashina: {sar_acc:.1%}")
        lines.append(f"  SigLIP:    {sig_acc:.1%}")
        
        if sar_acc > sig_acc:
            lines.append(f"\n→ Sarashina が {(sar_acc - sig_acc)*100:.1f}pp 優位")
        else:
            lines.append(f"\n→ SigLIP が {(sig_acc - sar_acc)*100:.1f}pp 優位")
    
    # Fine vs Coarse
    fine_results = [s for s in snapshots if "fine" in s.name.lower()]
    coarse_results = [s for s in snapshots if "coarse" in s.name.lower()]
    
    if fine_results and coarse_results:
        fine_acc = np.mean([compute_metrics(s)["accuracy"] for s in fine_results])
        coarse_acc = np.mean([compute_metrics(s)["accuracy"] for s in coarse_results])
        
        lines.append(f"\nLabel Mode Comparison:")
        lines.append(f"  Fine (11→4 mapped): {fine_acc:.1%}")
        lines.append(f"  Coarse (4 direct):  {coarse_acc:.1%}")
    
    report = "\n".join(lines)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n" + report)
    print(f"\nSaved summary report to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare SigLIP vs Sarashina training results")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Base directory containing experiment subdirs (e.g., joint_training_results)")
    parser.add_argument("--snapshots", type=str, nargs="+", default=None,
                       help="List of snapshot file paths")
    parser.add_argument("--out-dir", type=str, default=None,
                       help="Output directory for visualizations")
    parser.add_argument("--split-mode", choices=["test", "train", "both"], default="test",
                        help="Which split(s) to visualize (default: test)")
    args = parser.parse_args()
    
    # スナップショットファイルを収集
    snapshots = []
    
    if args.snapshots:
        for path in args.snapshots:
            p = Path(path)
            if p.exists():
                snapshots.append(load_snapshot(p))
    elif args.results_dir:
        base_dir = Path(args.results_dir)
        # 期待される順序で探索
        expected_names = ["sarashina_fine", "sarashina_coarse", "siglip_fine", "siglip_coarse"]
        for name in expected_names:
            snapshot_path = base_dir / name / "latent_snapshot_final.npz"
            if snapshot_path.exists():
                snapshots.append(load_snapshot(snapshot_path, name))
                print(f"Loaded: {snapshot_path}")
            else:
                print(f"Not found: {snapshot_path}")
    else:
        # デフォルトパス
        base_dir = Path(__file__).resolve().parents[1] / "joint_training_results"
        expected_names = ["sarashina_fine", "sarashina_coarse", "siglip_fine", "siglip_coarse"]
        for name in expected_names:
            snapshot_path = base_dir / name / "latent_snapshot_final.npz"
            if snapshot_path.exists():
                snapshots.append(load_snapshot(snapshot_path, name))
                print(f"Loaded: {snapshot_path}")
    
    if not snapshots:
        print("Error: No snapshot files found!")
        print("Please run training first or specify --snapshots or --results-dir")
        return
    split_options = {
        "test": ("test",),
        "train": ("train",),
        "both": (),
    }
    allowed_splits = split_options[args.split_mode]
    if allowed_splits:
        snapshots = [filter_snapshot_by_splits(s, allowed_splits) for s in snapshots]

    print(f"\nLoaded {len(snapshots)} snapshots")
    
    # 出力ディレクトリ
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent / "outputs" / "encoder_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 可視化実行
    print("\nGenerating visualizations...")
    
    plot_pca_comparison(snapshots, out_dir / "pca_comparison_2x2.png")
    plot_confusion_comparison(snapshots, out_dir / "confusion_comparison_2x2.png")
    plot_metrics_comparison(snapshots, out_dir / "metrics_comparison.png")
    generate_summary_report(snapshots, out_dir / "summary_report.txt")
    
    print(f"\n=== Done! All outputs saved to: {out_dir} ===")


if __name__ == "__main__":
    main()
