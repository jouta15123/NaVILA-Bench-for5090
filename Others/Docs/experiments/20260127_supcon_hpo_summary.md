# Optuna Hyperparameter Optimization Report (SupCon)

**Date:** 2026-01-27  
**Experiment:** MotionCLIP Joint Training with Supervised Contrastive Learning (SupCon)  
**Method:** Optuna (TPE Sampler)  
**Trials:** 30 (Trial 0-29)

---

## 1. 目的

MotionCLIPの対照学習（SupCon）において、Latent空間の整合性（Retrieval精度）を最大化するためのハイパーパラメータを探索する。特に、以下のバランスを決定することを目的とする：
- **対照学習の強さ** (`lambda_contrastive`) vs **VAE再構成** (`lambda_vae`)
- **温度パラメータ** (`temp`) の最適値

## 2. 探索設定

### 探索空間 (推測値含む)

コード (`hoyo_v1_1/models/optuna_motionclip.py`) および実行結果から、以下の範囲で探索が行われた。

| Parameter | Range | Scale | Note |
|-----------|-------|-------|------|
| `lambda_contrastive` | ~0.5 - 3.0 | log | 対照学習の重み |
| `lambda_vae` | ~0.05 - 0.5 | log | VAE再構成ロスの重み |
| `temp` | 0.02 - 0.2 | log | 温度調整パラメータ |
| `lr` | 1e-5 - 3e-4 | log | 学習率 |
| `weight_decay` | 1e-4 - 1e-2 | log | 重み減衰 |
| `full_steps` | 2000 - 7000 | int | Freeze後の学習ステップ数 |

### 固定設定 (Fixed Parameters)

- **Dataset**: HOYO (Onomatopoeia-Motion Dataset)
- **Batch Size**: 32
- **Encoder**: Sarashina (Japanese BERT)
- **Contrastive Mode**: SupCon (Supervised Contrastive Learning)
- **Evaluation Metric**: `avg_r@1` (= (M2T R@1 + T2M R@1) / 2)

---

## 3. 実験結果

### Best Trial (Trial 1)

**Score (Avg R@1):** **0.611**  
(Text-to-Motion R@1: 0.818, Motion-to-Text R@1: 0.403)

| Parameter | Best Value |
|-----------|------------|
| `lambda_contrastive` | **2.36** |
| `lambda_vae` | **0.057** |
| `temp` | **0.021** |
| `lr` | 7.73e-5 |
| `weight_decay` | 6.8e-4 |
| `full_steps` | 7000 |

### Top Trials 比較

| Rank | Trial | Avg R@1 | λ_cont | λ_vae | temp | 特徴 |
|------|-------|---------|--------|-------|------|------|
| **1** | **#1** | **0.611** | **2.36** | **0.057** | **0.021** | **高Cont / 低VAE / 低Temp** |
| 2 | #12 | 0.573 | 1.09 | 0.078 | 0.020 | Cont中程度でも低Tempなら高精度 |
| 3 | #13 | 0.557 | 2.20 | 0.072 | 0.059 | Temp高めだがBalance良 |
| 4 | #27 | 0.528 | 1.28 | 0.230 | 0.053 | VAE高めでも健闘 |
| 5 | #7 | 0.437 | 2.55 | 0.054 | 0.041 | - |

---

## 4. 傾向分析 (Insights)

### 1. 温度パラメータ (`temp`) の影響
- 上位Trial (#1, #12) は共通して **`temp ≈ 0.02` (探索下限)** を選択している。
- 温度が低いほど類似度分布が鋭敏になり、Contrastive Lossによる分離能力が最大化される傾向がある。

### 2. ロスのバランス (`lambda_contrastive` vs `lambda_vae`)
- **「高Contrastive + 低VAE」** の組み合わせがRetrieval精度には有利。
    - `lambda_contrastive` > 2.0
    - `lambda_vae` < 0.1
- VAE項を強くしすぎると (`lambda_vae` > 0.2)、埋め込み空間の自由度が再構成に割かれ、モダリティ間のAlignment (Retrieval) が低下するトレードオフが見られる。

---

## 5. 本番実験への反映方針

Optunaの結果は「Retrieval精度の最大化」に特化しているため、学習が不安定化したり生成品質 (MPJPE) が犠牲になっている可能性がある。
したがって、論文用の本番実験では以下の方針をとる。

1.  **Best設定の採用**: Trial 1 をベースライン (`proposed`) とする。
2.  **再現性の確認**: Seedを変えて学習し、偶然ではないことを示す。
3.  **感度分析 (Sensitivity Analysis)**:
    - 特に影響の大きかった **`temp`** についてアブレーションを行い、パラメータ選定の妥当性を主張する。
    - 必要に応じて `lambda_vae` を強めた設定 (RL安定性重視) と比較する。

### 決定した本番用固定ハイパラ

```python
config = {
    "lambda_contrastive": 2.3,
    "lambda_vae": 0.06,
    "temp": 0.02,
    "lr": 7e-5,
    "weight_decay": 7e-4,
    "batch_size": 32,  # Optuna条件を維持
}
```
