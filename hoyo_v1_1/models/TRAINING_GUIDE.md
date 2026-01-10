# HOYO + MotionCLIP 学習ガイド

## クイックスタート（最小構成）

まずは簡単な設定で試してみたい場合：

```bash
cd /home/jouta/NaVILA-Bench/hoyo_v1_1/models

# Freezeステージのみ（プロジェクタだけ学習、約10-20分）
python train_motionclip_joint.py \
  --stage freeze \
  --steps 1000 \
  --batch-size 32 \
  --run-name quick_test
```

---

## 推奨学習戦略（3段階）

段階的に学習することで、安定した収束と良い性能が期待できます。

### ステージ1: Freeze（プロジェクタのみ学習）

MotionCLIPのエンコーダ・デコーダを凍結し、セマンティックプロジェクタ（`sem_proj`）と温度パラメータ（`logit_scale`）だけを学習します。

```bash
cd /home/jouta/NaVILA-Bench/hoyo_v1_1/models

python train_motionclip_joint.py \
  --stage freeze \
  --steps 2000 \
  --batch-size 32 \
  --lr 1e-4 \
  --temp 0.07 \
  --lambda-contrastive 1.0 \
  --lambda-vae 0.0 \
  --label-mode fine \
  --sem-encoder sarashina \
  --contrastive-mode supcon \
  --log-interval 100 \
  --eval-interval 200 \
  --run-name freeze_sarashina_supcon \
  --wandb \
  --wandb-project hoyo_motion \
  --wandb-group freeze_stage
```

**ポイント:**
- `--lambda-vae 0.0`: VAE損失は無効（プロジェクタだけ学習）
- `--lr 1e-4`: プロジェクタは比較的大きめの学習率でOK
- データ拡張: 現在は無効（固定）

### ステージ2: Encoder（エンコーダも学習）

ステージ1の最良モデルをロードし、エンコーダも学習対象に加えます。

```bash
python train_motionclip_joint.py \
  --stage encoder \
  --steps 3000 \
  --batch-size 32 \
  --lr 1e-5 \
  --lr-encoder 1e-5 \
  --temp 0.07 \
  --lambda-contrastive 0.5 \
  --lambda-vae 1.0 \
  --label-mode fine \
  --sem-encoder sarashina \
  --contrastive-mode supcon \
  --log-interval 100 \
  --eval-interval 200 \
  --run-name encoder_sarashina_supcon \
  --wandb \
  --wandb-project hoyo_motion \
  --wandb-group encoder_stage
```

**ポイント:**
- ステージ1の `sem_proj_joint_best.pth` と `logit_scale_joint_best.pt` が自動的にロードされる
- `--lambda-vae 1.0`: VAE損失を有効化（再構成品質を保つ）
- `--lambda-contrastive 0.5`: コントラスティブ損失の重みを下げる（バランス調整）

### ステージ3: Full（全パラメータ学習）

エンコーダ・デコーダの全パラメータを学習します。

```bash
python train_motionclip_joint.py \
  --stage full \
  --steps 4000 \
  --batch-size 24 \
  --lr 1e-5 \
  --lr-encoder 5e-6 \
  --lr-decoder 5e-6 \
  --temp 0.07 \
  --lambda-contrastive 0.3 \
  --lambda-vae 1.0 \
  --label-mode fine \
  --sem-encoder sarashina \
  --contrastive-mode supcon \
  --log-interval 100 \
  --eval-interval 200 \
  --run-name full_sarashina_supcon \
  --wandb \
  --wandb-project hoyo_motion \
  --wandb-group full_stage
```

**ポイント:**
- `--batch-size 24`: メモリ使用量が増えるので少し小さめに
- `--lr-decoder 5e-6`: デコーダは慎重に学習（小さめの学習率）
- `--lambda-contrastive 0.3`: さらにバランス調整

---

## その他の推奨設定

### Coarse Label（4スタイル群）での学習

細かい11ラベルではなく、4つのスタイル群で学習する場合：

```bash
python train_motionclip_joint.py \
  --stage freeze \
  --steps 2000 \
  --batch-size 32 \
  --lr 1e-4 \
  --label-mode coarse \
  --sem-encoder sarashina \
  --contrastive-mode supcon \
  --run-name freeze_coarse \
  --wandb
```

### SigLIP エンコーダを使用

Sarashinaの代わりにSigLIPを使う場合：

```bash
python train_motionclip_joint.py \
  --stage freeze \
  --steps 2000 \
  --batch-size 32 \
  --lr 1e-4 \
  --sem-encoder siglip \
  --run-name freeze_siglip \
  --wandb
```

### CLIP-style CE損失を使用

Supervised Contrastiveの代わりにCLIP-style Cross-Entropyを使う場合：

```bash
python train_motionclip_joint.py \
  --stage freeze \
  --steps 2000 \
  --batch-size 32 \
  --lr 1e-4 \
  --contrastive-mode clip_ce \
  --run-name freeze_clipce \
  --wandb
```

---

## チェックポイントの確認

各ステージの学習後、以下のファイルが保存されます：

```
joint_training_results/{run_name}/checkpoints/
├── motionclip_encoder_joint_best.pth      # エンコーダ（最良）
├── sem_proj_joint_best.pth                # プロジェクタ（最良）
├── logit_scale_joint_best.pt              # 温度パラメータ（最良）
├── motionclip_full_joint_best.pth         # フルモデル（最良）
├── motionclip_encoder_joint_final.pth     # エンコーダ（最終）
├── sem_proj_joint_final.pth               # プロジェクタ（最終）
└── motionclip_full_joint_final.pth        # フルモデル（最終）
```

**最良モデル（best）**: テストセットでのAcc@1が最高だった時点のモデル  
**最終モデル（final）**: 学習終了時点のモデル

---

## トラブルシューティング

### メモリ不足（OOM）が発生する場合

- `--batch-size` を小さくする（例: 32 → 16 → 8）
- `--stage full` では特に注意（デコーダも学習するため）

### 学習が不安定な場合

- `--lambda-contrastive` を小さくする（例: 1.0 → 0.5 → 0.3）
- `--lr` や `--lr-encoder` を小さくする
- `--temp` を調整（デフォルト0.07から、0.05〜0.1の範囲で試す）

### 収束が遅い場合

- `--steps` を増やす
- `--lr` を少し大きくする（ただし不安定になる可能性あり）
- データ拡張は現在無効（固定）

---

## 実行例（完全版）

**推奨パターン: 同じ `--run-name` で3ステージを順番に実行**

```bash
# 実験名を定義（環境変数で管理すると便利）
export EXP_NAME="exp001_sarashina_supcon"

# 1. Freezeステージ
python train_motionclip_joint.py \
  --stage freeze \
  --steps 2000 \
  --batch-size 32 \
  --lr 1e-4 \
  --lambda-contrastive 1.0 \
  --lambda-vae 0.0 \
  --label-mode fine \
  --sem-encoder sarashina \
  --contrastive-mode supcon \
  --run-name ${EXP_NAME} \
  --wandb \
  --wandb-project hoyo_motion

# 2. Encoderステージ（freezeの結果を自動ロード）
python train_motionclip_joint.py \
  --stage encoder \
  --steps 3000 \
  --batch-size 32 \
  --lr 1e-5 \
  --lr-encoder 1e-5 \
  --lambda-contrastive 0.5 \
  --lambda-vae 1.0 \
  --label-mode fine \
  --sem-encoder sarashina \
  --contrastive-mode supcon \
  --run-name ${EXP_NAME} \
  --wandb \
  --wandb-project hoyo_motion

# 3. Fullステージ（encoderの結果を自動ロード）
python train_motionclip_joint.py \
  --stage full \
  --steps 4000 \
  --batch-size 24 \
  --lr 1e-5 \
  --lr-encoder 5e-6 \
  --lr-decoder 5e-6 \
  --lambda-contrastive 0.3 \
  --lambda-vae 1.0 \
  --label-mode fine \
  --sem-encoder sarashina \
  --contrastive-mode supcon \
  --run-name ${EXP_NAME} \
  --wandb \
  --wandb-project hoyo_motion
```

**結果の保存場所:**
```
hoyo_v1_1/joint_training_results/${EXP_NAME}/
├── checkpoints/          # 各ステージのチェックポイント
├── logs/                 # 学習ログ
└── latent_snapshot_final.npz  # 可視化用の潜在空間スナップショット
```

**注意**: 
- 各ステージは同じ `out_dir/checkpoints/` 内で前ステージのチェックポイントを探します
- **同じ `--run-name` を使う場合**: 各ステージを順番に実行すれば、自動的に前ステージの `sem_proj` と `logit_scale` がロードされます
- **異なる `--run-name` を使う場合**: 手動でチェックポイントをコピーするか、`--run-name` を統一してください

**推奨**: 実験ごとに一意の `--run-name` を使い、3ステージすべて同じ名前で実行するのが最も簡単です。
