# Contrastive+VAE (fine) 可視化結果まとめ

最終更新: 2026-01-28

## 対象 run

- **RUN_NAME**: `20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full`
- **出力先**: `hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/`

---

## 1. 学習曲線（Loss）

**生成物**:
- `loss_curves.png`

**生成コマンド**:
```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/plot_training_curves_single.py \
  --log hoyo_v1_1/joint_training_results/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/logs/train_log_20260127_221732.txt \
  --out hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/loss_curves.png \
  --title "Loss Curves (20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full)"
```

---

## 2. Retrieval 指標（t2m / m2t）

**生成物**:
- `retrieval_curves.png`

**生成コマンド**:
```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/plot_retrieval_curves.py \
  --metrics hoyo_v1_1/joint_training_results/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/retrieval_metrics.jsonl \
  --out hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/retrieval_curves.png \
  --title "Retrieval R@1 (20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full)"
```

---

## 3. 埋め込み空間の可視化

**生成物**:
- `contrastive_fine_pca_2d.png`

**生成コマンド**:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/plot_latent_spaces.py \
  --snapshot hoyo_v1_1/joint_training_results/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/latent_snapshot_final.npz \
  --out-dir hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full \
  --prefix contrastive_fine \
  --label-mode fine \
  --show-text-prototypes \
  --skip-umap
```

**補足**:
- UMAP は `umap-learn` が未インストールのためスキップ。

---

## 4. 混同行列（test split）

**生成物**:
- `confusion_fine_test.json`
- `confusion_fine_test.png`
- `confusion_fine_test_norm.png`

**生成コマンド**:
```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/evaluate_retrieval.py \
  --snapshot hoyo_v1_1/joint_training_results/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/latent_snapshot_final.npz \
  --splits test \
  --mode fine \
  --confusion \
  --confusion-out hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/confusion_fine_test.json

/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/plot_confusion_from_json.py \
  --confusion-json hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/confusion_fine_test.json \
  --out hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/confusion_fine_test.png

/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/plot_confusion_from_json.py \
  --confusion-json hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/confusion_fine_test.json \
  --out hoyo_v1_1/viz/outputs/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/confusion_fine_test_norm.png \
  --normalize
```

**test split の数値（出力ログ）**:
- t2m R@1 = 0.5455
- m2t R@1 = 0.4194
- confusion Acc@1 = 0.4194
- silhouette = 0.0136

---

## 変更点メモ

- `hoyo_v1_1/viz/plot_latent_spaces.py`: `--skip-umap` を追加（UMAP未導入環境で使用可能に）
- `hoyo_v1_1/viz/plot_confusion_from_json.py`: 未正規化時の `fmt='d'` エラー修正（float -> int 変換）
- 新規:
  - `hoyo_v1_1/viz/plot_training_curves_single.py`
  - `hoyo_v1_1/viz/plot_retrieval_curves.py`
