# SigLIP vs Sarashina テキストエンコーダ比較実験レポート

**実験日**: 2025年12月6日  
**目的**: 日本語オノマトペ（歩行スタイル）とモーションの対照学習において、テキストエンコーダの選択が精度に与える影響を調査

---

## 1. 実験概要

### 1.1 比較対象

| テキストエンコーダ | 説明 |
|------------------|------|
| **Sarashina** | 日本語特化の文埋め込みモデル (`sbintuitions/sarashina-embedding-v2-1b`) |
| **SigLIP** | 多言語対応の Vision-Language モデルのテキストエンコーダ (`google/siglip-base-patch16-256-multilingual`) |

### 1.2 評価軸

| 評価軸 | クラス数 | 目的 |
|--------|---------|------|
| **11クラス評価** | 11 | 個別オノマトペの細かい識別能力 |
| **4クラス評価** | 4 | 粗いスタイル群（速い系、遅い系、重い系、ふらふら系）の分類能力 |

**重要**: 学習は常に **11個のオノマトペ埋め込み** で行い、評価時に両方の軸で精度を測定。
4クラス評価では、11クラスのプロトタイプを平均してCoarseプロトタイプを作成し、ラベルもマッピング。

### 1.3 学習設定

```
stage: full (エンコーダ・デコーダ両方を学習)
steps: 5000
lr: 5e-05
lr_encoder: 2e-05
lr_decoder: 2e-05
lambda_vae: 1.0
lambda_contrastive: 0.5
batch_size: 32
temperature: 0.07
seed: 42
```

### 1.4 損失関数

学習には以下の2つの損失関数を組み合わせて使用：

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{vae}} \cdot \mathcal{L}_{\text{VAE}} + \lambda_{\text{cont}} \cdot \mathcal{L}_{\text{SupCon}}
$$

#### 1.4.1 VAE損失 ($\mathcal{L}_{\text{VAE}}$)

MotionCLIPのVAEアーキテクチャによるモーション再構成損失。入力モーションを潜在空間にエンコードし、デコードした結果との差を最小化する。

- **再構成損失**: 入力と出力の差（L2距離）
- **速度損失**: フレーム間の動き（速度）の再現性
- **KL正則化**: 潜在空間の正則化

#### 1.4.2 Supervised Contrastive損失 ($\mathcal{L}_{\text{SupCon}}$)

モーション潜在ベクトルとテキスト埋め込みの対照学習。同じラベルを持つサンプル同士を近づけ、異なるラベルのサンプルを遠ざける。

$$
\mathbf{F} = [\mathbf{z}_{\text{motion}}; \mathbf{e}_{\text{text}}] \in \mathbb{R}^{(N+M) \times d}
$$

$$
\mathbf{S} = \exp(\tau) \cdot \mathbf{F} \mathbf{F}^\top
$$

ここで、$\mathbf{z}_{\text{motion}}$ はモーション潜在ベクトル、$\mathbf{e}_{\text{text}}$ はテキスト埋め込み、$\tau$ は学習可能な logit_scale パラメータ。

- **正例（Positive）**: 同じオノマトペラベルを持つペア（モーション-モーション、テキスト-テキスト、モーション-テキスト）
- **負例（Negative）**: 異なるオノマトペラベルを持つペア
- **温度パラメータ**: `temp=0.07`（学習可能なlogit_scaleで制御）

この損失により、テキストエンコーダの埋め込み空間とモーションエンコーダの潜在空間が整列し、オノマトペによるモーション検索・生成が可能になる。

---

## 2. データセット

### 2.1 HoYo Dataset 概要

| 項目 | 値 |
|------|-----|
| 総サンプル数 | **292** |
| オノマトペ種類 | **11** |
| データ形式 | 14関節 × 2座標 (2D) |
| フレームレート | 可変長（60フレームにクロップ） |

### 2.2 オノマトペ別サンプル数

| オノマトペ | サンプル数 | Train | Test |
|-----------|-----------|-------|------|
| 通常 | 32 | 25 | 7 |
| すたすた | 32 | 25 | 7 |
| せかせか | 22 | 17 | 5 |
| てくてく | 22 | 17 | 5 |
| どっしどっし | 32 | 25 | 7 |
| とぼとぼ | 22 | 17 | 5 |
| のしのし | 22 | 17 | 5 |
| のろのろ | 32 | 25 | 7 |
| ぶらぶら | 22 | 17 | 5 |
| よたよた | 22 | 17 | 5 |
| よろよろ | 32 | 25 | 7 |
| **合計** | **292** | **227** | **65** |

### 2.3 Coarse (4クラス) グルーピング

| グループ | 含まれるオノマトペ | サンプル数 |
|---------|-------------------|-----------|
| 速い系 | すたすた, せかせか, てくてく | 76 |
| 遅い系 | とぼとぼ, のろのろ | 54 |
| 重い系 | どっしどっし, のしのし | 54 |
| ふらふら系 | ぶらぶら, よたよた, よろよろ | 76 |

※「通常」は4クラス評価時に「遅い系」として扱う（計86サンプル）

---

## 3. 実験結果

### 3.1 定量評価

| エンコーダ | 評価軸 | Accuracy | Top-3 Accuracy | Silhouette Score |
|-----------|--------|----------|----------------|------------------|
| **Sarashina** | 11クラス | **44.6%** | - | 0.073 |
| SigLIP | 11クラス | 24.6% | - | 0.085 |
| **Sarashina** | 4クラス | **67.7%** | 92.3% | 0.073 |
| SigLIP | 4クラス | 60.0% | 95.4% | 0.085 |

※ 評価はテストセット（65サンプル）に対して実施。11クラスおよび4クラスの分類精度を算出。

### 3.2 エンコーダ比較サマリー

| 評価軸 | Sarashina | SigLIP | 差分 | 優位 |
|--------|-----------|--------|------|------|
| **11クラス（細かい識別）** | **44.6%** | 24.6% | **+20.0pp** | **Sarashina** |
| **4クラス（粗い識別）** | **67.7%** | 60.0% | **+7.7pp** | **Sarashina** |

### 3.3 主要な発見

1. **細かい識別（11クラス）では Sarashina が大幅に優位**
   - **+20.0pp** の差（44.6% vs 24.6%）で、日本語オノマトペの微妙なニュアンスを捉える能力に明確な差
   - SigLIP は 24.6% とチャンスレベル（約9%）を上回るものの、実用には不十分
   - 混同行列を見ると、Sarashina は「すたすた」(86%)、「てくてく」(80%)、「よろよろ」(71%) で高精度を達成
   - SigLIP は「すたすた」(71%)、「てくてく」(60%) 以外は識別が困難

2. **粗い識別（4クラス）でも Sarashina が優位**
   - **+7.7pp** の差（67.7% vs 60.0%）で一貫して Sarashina が優位
   - 混同行列を見ると、Sarashina は「速い系」で 88%、「ふらふら系」で 76% と高精度
   - SigLIP は「速い系」で 94% と高いが、「遅い系」を「速い系」と誤分類する傾向（58%）
   - 両者とも「重い系」の識別が課題（Sarashina: 50%, SigLIP: 33%）

3. **Top-3 Accuracy は高いが、両者に差がある**
   - Sarashina: 92.3% vs SigLIP: 95.4%
   - 正解が Top-3 に入る確率は両モデルとも高く、相対的な順序関係は学習できている
   - SigLIP は Top-3 では Sarashina をわずかに上回る

4. **Silhouette Score は SigLIP がやや優位**
   - SigLIP: 0.085 vs Sarashina: 0.073
   - クラスタ分離の観点では SigLIP の方が潜在空間の構造がやや良好
   - ただし精度には直結せず、Sarashina の方が高精度を達成

---

## 4. 可視化結果

### 4.1 潜在空間のPCA可視化

各エンコーダにおける、モーション潜在空間の2次元PCA投影。
- 点: モーションサンプル（色は4つのCoarseスタイル群）
- プロトタイプ可視化は廃止し、純粋にサンプル分布のみで比較（テキスト埋め込みとの整列度はクラス重心で評価）

![PCA Comparison](hoyo_v1_1/viz/outputs/encoder_comparison_full/pca_comparison_2x2.png)

**観察:**
- 両エンコーダとも、各スタイル群のモーションがある程度まとまって分布
- Sarashina（PC1: 22.0%, PC2: 17.5%）と SigLIP（PC1: 28.4%, PC2: 15.7%）で主成分の説明力に差がある
- PCA 可視化上では両者とも4クラスの境界が明確ではないが、Sarashina は精度 44.6%/67.7% を達成している

### 4.2 混同行列

上段: 11クラス（Fine）分類
下段: 4クラス（Coarse）分類

![Confusion Matrix Comparison](hoyo_v1_1/viz/outputs/encoder_comparison_full/confusion_comparison_2x2.png)

### 4.3 メトリクス比較

![Metrics Comparison](hoyo_v1_1/viz/outputs/encoder_comparison_full/metrics_comparison.png)


---

## 5. 考察

### 5.1 Sarashina の強み

日本語オノマトペは**音象徴（sound symbolism）** に基づく表現であり、微妙な音の違いが意味のニュアンスを変える：
- 「すたすた」→ 軽快で速い
- 「せかせか」→ 急いでいる、焦っている

Sarashina は日本語に特化した事前学習により、これらの**微妙な意味の違い**をベクトル空間上で適切に表現できていると考えられる。11クラスでの **+20.0pp** の大幅な優位性はこれを裏付ける。

混同行列を見ると、Sarashina は一部のオノマトペで高い正解率を達成している：
- 「すたすた」で 86%、「てくてく」で 80% の高精度
- 「よろよろ」で 71% の識別
- 一方で「のしのし」「よたよた」など識別が難しいクラスも存在

### 5.2 SigLIP の特徴

SigLIP は多言語モデルとして、**視覚的・概念的な特徴**を捉える傾向がある。
- Silhouette Score が高い（0.085 vs 0.073）ことから、潜在空間の幾何構造は良好
- しかし精度では Sarashina に劣り、特に11クラス識別で 24.6% と大きく差がつく
- 混同行列を見ると、「せかせか」→「すたすた」や「のしのし」→「どっしどっし」など、同じ粗粒度グループ内での誤分類が多い
- 「速い系」の識別は得意（94%）だが、「遅い系」「重い系」で誤分類が多い

### 5.3 実用上の示唆

| ユースケース | 推奨エンコーダ |
|-------------|---------------|
| 細かいオノマトペ指示（「すたすたで歩いて」vs「せかせかで歩いて」） | **Sarashina**（+20.0pp優位） |
| 粗いスタイル指示（「速く歩いて」「ゆっくり歩いて」） | **Sarashina**（+7.7pp優位） |
| 多言語対応が必要な場合 | SigLIP |

---

## 6. 結論

本実験により、以下の知見が得られた：

1. **細かい識別（11クラス）**: Sarashina が **+20.0pp 優位**（44.6% vs 24.6%）
2. **粗い識別（4クラス）**: Sarashina が **+7.7pp 優位**（67.7% vs 60.0%）
3. **推奨**:
   - 日本語オノマトペを用いたモーション制御には **Sarashina** を使用
   - 多言語対応が必要な場合のみ SigLIP を検討

**結論**: 日本語オノマトペとモーションの対照学習において、日本語特化モデル Sarashina が多言語モデル SigLIP に対して明確な優位性を示した。特に11クラスの細かい識別では +20.0pp という大幅な差があり、日本語オノマトペの音象徴的なニュアンスを捉える能力に大きな差がある。日本語オノマトペを用いたモーション生成・制御システムでは Sarashina の使用を強く推奨する。

---

## 付録A: 実験環境

- **GPU**: NVIDIA RTX (24GB VRAM)
- **Python**: 3.11
- **PyTorch**: 2.x
- **MotionCLIP**: paper-model checkpoint
- **データセット**: HoYo Dataset（11種類オノマトペ、歩行モーション）

## 付録B: 実行コマンド

### B.1 学習コマンド

**Sarashina エンコーダ:**
```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/models/train_motionclip_joint.py \
  --stage full \
  --sem-encoder sarashina \
  --label-mode fine \
  --steps 5000 \
  --batch-size 32 \
  --lr 5e-5 \
  --lr-encoder 2e-5 \
  --lr-decoder 2e-5 \
  --lambda-vae 1.0 \
  --lambda-contrastive 0.5 \
  --temp 0.07 \
  --log-interval 100 \
  --eval-interval 200 \
  --seed 42 \
  --run-name sarashina_full_fixed
```

**SigLIP エンコーダ:**
```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/models/train_motionclip_joint.py \
  --stage full \
  --sem-encoder siglip \
  --label-mode fine \
  --steps 5000 \
  --batch-size 32 \
  --lr 5e-5 \
  --lr-encoder 2e-5 \
  --lr-decoder 2e-5 \
  --lambda-vae 1.0 \
  --lambda-contrastive 0.5 \
  --temp 0.07 \
  --log-interval 100 \
  --eval-interval 200 \
  --seed 42 \
  --run-name siglip_full_fixed
```

### B.2 可視化コマンド

```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/compare_encoder_results.py \
  --snapshots hoyo_v1_1/joint_training_results/sarashina_full_fixed/latent_snapshot_final.npz \
              hoyo_v1_1/joint_training_results/siglip_full_fixed/latent_snapshot_final.npz \
  --out-dir hoyo_v1_1/viz/outputs/encoder_comparison_full
```

### B.3 学習スクリプトの主要オプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--stage` | 学習段階 (freeze/encoder/full) | freeze |
| `--sem-encoder` | テキストエンコーダ (sarashina/siglip) | sarashina |
| `--label-mode` | ラベル粒度 (fine/coarse) | fine |
| `--steps` | 学習ステップ数 | 3000 |
| `--batch-size` | バッチサイズ | 32 |
| `--lr` | プロジェクタ学習率 | 1e-5 |
| `--lr-encoder` | エンコーダ学習率 | 1e-5 |
| `--lr-decoder` | デコーダ学習率 | 1e-5 |

| `--lambda-vae` | VAE損失の重み | 1.0 |
| `--lambda-contrastive` | 対照損失の重み | 0.1 |
| `--temp` | 温度パラメータ | 0.07 |
| `--seed` | 乱数シード | 42 |
| `--run-name` | 実験名（出力ディレクトリ名） | タイムスタンプ |

## ファイル構成

```
encoder_comparison_full/
├── analysis_report.md              # 本レポート
├── pca_comparison_2x2.png          # PCA可視化
├── confusion_comparison_2x2.png    # 混同行列
├── metrics_comparison.png          # メトリクス比較
└── summary_report.txt              # テキストサマリー
```

## 関連ファイル

```
hoyo_v1_1/
├── models/
│   ├── train_motionclip_joint.py   # 学習スクリプト
│   └── common.py                   # データセット・ユーティリティ
├── viz/
│   └── compare_encoder_results.py  # 可視化スクリプト
├── joint_training_results/
│   ├── sarashina_full_fixed/       # Sarashina学習結果
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── latent_snapshot_final.npz
│   └── siglip_full_fixed/          # SigLIP学習結果
│       ├── checkpoints/
│       ├── logs/
│       └── latent_snapshot_final.npz
└── data/                           # HoYoデータセット (292サンプル)
```
