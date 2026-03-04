#!/usr/bin/env python3
"""
損失曲線・精度曲線の可視化スクリプト
SarashinaとSigLIPエンコーダの学習ログを比較プロット
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib

# 日本語フォント設定 (英語ラベルのみ使用するためコメントアウト)
# matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans CJK JP', 'sans-serif']
plt.style.use('seaborn-v0_8-whitegrid')


def parse_log_file(log_path: Path) -> Dict[str, List]:
    """
    学習ログファイルをパースして各種メトリクスを抽出
    """
    data = {
        "step": [],
        "total_loss": [],
        "vae_loss": [],
        "contrastive_loss": [],
        "train_acc": [],
        "test_step": [],
        "test_acc1": [],
        "test_acc3": [],
        "test_mpjpe": [],
    }
    
    # Train log pattern
    # [Step 00100] loss_total=1.0461 (vae=0.2432, cont=1.6058), train_acc@1=0.094, ...
    train_pattern = re.compile(
        r'\[Step (\d+)\] loss_total=(\d+\.\d+) \(vae=(\d+\.\d+), cont=(\d+\.\d+)\), train_acc@1=(\d+\.\d+)'
    )
    
    # Test log pattern
    # [TEST] Acc@1: 0.039 | Acc@3: 0.148 | MPJPE: 0.3188
    test_pattern = re.compile(
        r'\[TEST\] Acc@1: (\d+\.\d+) \| Acc@3: (\d+\.\d+) \| MPJPE: (\d+\.\d+)'
    )
    
    # Evaluating step pattern
    eval_step_pattern = re.compile(r'Evaluating on Test Set...')
    
    current_step = None
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # Parse train metrics
        train_match = train_pattern.search(line)
        if train_match:
            step = int(train_match.group(1))
            data["step"].append(step)
            data["total_loss"].append(float(train_match.group(2)))
            data["vae_loss"].append(float(train_match.group(3)))
            data["contrastive_loss"].append(float(train_match.group(4)))
            data["train_acc"].append(float(train_match.group(5)))
            current_step = step
        
        # Parse test metrics
        test_match = test_pattern.search(line)
        if test_match and current_step is not None:
            data["test_step"].append(current_step)
            data["test_acc1"].append(float(test_match.group(1)))
            data["test_acc3"].append(float(test_match.group(2)))
            data["test_mpjpe"].append(float(test_match.group(3)))
    
    return data


def plot_training_comparison(
    sarashina_log: Path,
    siglip_log: Path,
    output_dir: Path,
):
    """
    SarashinaとSigLIPの学習曲線を比較プロット
    """
    # Parse logs
    sarashina_data = parse_log_file(sarashina_log)
    siglip_data = parse_log_file(siglip_log)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'sarashina': '#2E86AB',  # Blue
        'siglip': '#A23B72',      # Pink/Magenta
    }
    
    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(sarashina_data["step"], sarashina_data["total_loss"], 
             color=colors['sarashina'], label='Sarashina', linewidth=2, alpha=0.8)
    ax1.plot(siglip_data["step"], siglip_data["total_loss"], 
             color=colors['siglip'], label='SigLIP', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Total Loss', fontsize=11)
    ax1.set_title('Total Loss Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Components (VAE + Contrastive)
    ax2 = axes[0, 1]
    ax2.plot(sarashina_data["step"], sarashina_data["vae_loss"], 
             color=colors['sarashina'], linestyle='-', label='Sarashina VAE', linewidth=2, alpha=0.8)
    ax2.plot(sarashina_data["step"], sarashina_data["contrastive_loss"], 
             color=colors['sarashina'], linestyle='--', label='Sarashina Contrastive', linewidth=2, alpha=0.8)
    ax2.plot(siglip_data["step"], siglip_data["vae_loss"], 
             color=colors['siglip'], linestyle='-', label='SigLIP VAE', linewidth=2, alpha=0.8)
    ax2.plot(siglip_data["step"], siglip_data["contrastive_loss"], 
             color=colors['siglip'], linestyle='--', label='SigLIP Contrastive', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Loss Components (VAE / Contrastive)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Train Accuracy
    ax3 = axes[1, 0]
    ax3.plot(sarashina_data["step"], sarashina_data["train_acc"], 
             color=colors['sarashina'], label='Sarashina', linewidth=2, alpha=0.8)
    ax3.plot(siglip_data["step"], siglip_data["train_acc"], 
             color=colors['siglip'], label='SigLIP', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Train Accuracy (Top-1)', fontsize=11)
    ax3.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
    # 4. Test Accuracy (Top-1)
    ax4 = axes[1, 1]
    ax4.plot(sarashina_data["test_step"], sarashina_data["test_acc1"], 
             color=colors['sarashina'], marker='o', markersize=5, label='Sarashina', linewidth=2, alpha=0.8)
    ax4.plot(siglip_data["test_step"], siglip_data["test_acc1"], 
             color=colors['siglip'], marker='s', markersize=5, label='SigLIP', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Test Accuracy (Top-1)', fontsize=11)
    ax4.set_title('Test Accuracy (11-class)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.0)
    
    # Final accuracies annotation
    sarashina_final = sarashina_data["test_acc1"][-1] if sarashina_data["test_acc1"] else 0
    siglip_final = siglip_data["test_acc1"][-1] if siglip_data["test_acc1"] else 0
    sarashina_best = max(sarashina_data["test_acc1"]) if sarashina_data["test_acc1"] else 0
    siglip_best = max(siglip_data["test_acc1"]) if siglip_data["test_acc1"] else 0
    
    ax4.axhline(y=sarashina_best, color=colors['sarashina'], linestyle=':', alpha=0.6)
    ax4.axhline(y=siglip_best, color=colors['siglip'], linestyle=':', alpha=0.6)
    ax4.text(5000, sarashina_best + 0.02, f'Best: {sarashina_best:.1%}', 
             color=colors['sarashina'], fontsize=9, ha='right')
    ax4.text(5000, siglip_best - 0.05, f'Best: {siglip_best:.1%}', 
             color=colors['siglip'], fontsize=9, ha='right')
    
    plt.suptitle('SigLIP vs Sarashina Text Encoder Training Curves\n(11-class Fine-grained, Full Stage Training)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "training_curves_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def plot_test_metrics_detail(
    sarashina_log: Path,
    siglip_log: Path,
    output_dir: Path,
):
    """
    テストメトリクスの詳細比較（Top-1, Top-3, MPJPE）
    """
    sarashina_data = parse_log_file(sarashina_log)
    siglip_data = parse_log_file(siglip_log)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {
        'sarashina': '#2E86AB',
        'siglip': '#A23B72',
    }
    
    # Top-1 Accuracy
    ax1 = axes[0]
    ax1.plot(sarashina_data["test_step"], sarashina_data["test_acc1"], 
             color=colors['sarashina'], marker='o', markersize=4, label='Sarashina', linewidth=2)
    ax1.plot(siglip_data["test_step"], siglip_data["test_acc1"], 
             color=colors['siglip'], marker='s', markersize=4, label='SigLIP', linewidth=2)
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Test Accuracy (Top-1)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Top-3 Accuracy
    ax2 = axes[1]
    ax2.plot(sarashina_data["test_step"], sarashina_data["test_acc3"], 
             color=colors['sarashina'], marker='o', markersize=4, label='Sarashina', linewidth=2)
    ax2.plot(siglip_data["test_step"], siglip_data["test_acc3"], 
             color=colors['siglip'], marker='s', markersize=4, label='SigLIP', linewidth=2)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Test Accuracy (Top-3)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # MPJPE
    ax3 = axes[2]
    ax3.plot(sarashina_data["test_step"], sarashina_data["test_mpjpe"], 
             color=colors['sarashina'], marker='o', markersize=4, label='Sarashina', linewidth=2)
    ax3.plot(siglip_data["test_step"], siglip_data["test_mpjpe"], 
             color=colors['siglip'], marker='s', markersize=4, label='SigLIP', linewidth=2)
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('MPJPE', fontsize=11)
    ax3.set_title('Reconstruction Error (MPJPE)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Test Metrics Detailed Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "test_metrics_detail.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def parse_args() -> argparse.Namespace:
    """
    CLI引数を定義し、デフォルトで直近（2025-12-04）のログを参照する。
    """
    base_dir = Path(__file__).resolve().parents[1] / "joint_training_results"
    default_sarashina = base_dir / "sarashina_full_fixed" / "logs" / "train_log_20251204_150735.txt"
    default_siglip = base_dir / "siglip_full_fixed" / "logs" / "train_log_20251204_151008.txt"
    default_out = Path(__file__).resolve().parent / "outputs" / "encoder_comparison_full"

    parser = argparse.ArgumentParser(
        description="Plot loss/accuracy curves for Sarashina vs SigLIP text encoders."
    )
    parser.add_argument("--sarashina-log", type=Path, default=default_sarashina,
                        help=f"Path to Sarashina log (default: {default_sarashina})")
    parser.add_argument("--siglip-log", type=Path, default=default_siglip,
                        help=f"Path to SigLIP log (default: {default_siglip})")
    parser.add_argument("--output-dir", type=Path, default=default_out,
                        help=f"Directory to save plots (default: {default_out})")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.sarashina_log.exists():
        print(f"Error: Sarashina log not found: {args.sarashina_log}")
        sys.exit(1)
    if not args.siglip_log.exists():
        print(f"Error: SigLIP log not found: {args.siglip_log}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_training_comparison(args.sarashina_log, args.siglip_log, args.output_dir)
    plot_test_metrics_detail(args.sarashina_log, args.siglip_log, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
