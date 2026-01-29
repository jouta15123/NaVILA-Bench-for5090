# Optuna探索 → 本番運用 方針まとめ（Contrastive+VAE / SupCon）

最終更新: 2026-01-28

## 1. 何を目的にOptunaを回したか

- **目的**: 対照学習（SupCon）における Retrieval 精度（`avg_r@1`）を最大化する設定の“当たり”を探索する。
- **位置づけ**: Optuna は **設計の補助**。最終比較は固定設定で公平に行う。

参照:
- `docs/experiments/20260127_supcon_hpo_summary.md`
- `docs/experiments/20260127_contrastive_vae_experiment_plan.md`

---

## 2. Optunaで探索した範囲（実装ベース）

使用スクリプト: `hoyo_v1_1/models/optuna_motionclip.py`

### 探索対象（full stage）
- `lr`: **1e-5 ～ 3e-4**（log）
- `weight_decay`: **1e-6 ～ 1e-2**（log）
- `temp`: **0.02 ～ 0.2**（log）
- `steps`（full stage）: **min～max を 500 step刻み**（指定時のみ探索）
- `lambda_contrastive` / `lambda_vae`: `--search-lambda` 使用時のみ探索
  - `lambda_contrastive`: **0.1 ～ 2.0**（log）
  - `lambda_vae`: **0.1 ～ 2.0**（log）
  - ただし `lambda_contrastive >= lambda_vae` を満たすよう制約

### 固定条件
- `batch_size=32`
- `label_mode=fine`
- `contrastive_mode=supcon`
- `best_metric=avg_r@1`
- freeze/full 構成（freeze step は試行設定に従う）

**備考**:
- 目的関数は `avg_r@1 + silhouette_weight * silhouette`（`silhouette_weight` 既定 0.1）。
- 実際の探索範囲は Optuna 実行時の CLI 引数で上書き可能。

---

## 3. Optunaで見えた傾向（実験結果）

`docs/experiments/20260127_supcon_hpo_summary.md` の結果から以下が見えた:

- **低温度（temp ≈ 0.02）に張り付きやすい**
- **Contrastive を強め、VAE を弱める方が Retrieval に有利**
- **高精度だが境界付近に寄る**（探索範囲の限界や過最適の可能性）

代表的なベスト例（HPO summary より）:
- `lambda_contrastive ≈ 2.36`
- `lambda_vae ≈ 0.057`
- `temp ≈ 0.021`
- `lr ≈ 7.7e-5`
- `weight_decay ≈ 6.8e-4`
- `full_steps ≈ 7000`

---

## 4. それを踏まえた「本番運用」方針

### なぜ Optuna の best をそのまま本番にしないか
- **境界値に寄る傾向**（探索範囲の影響 / たまたまの可能性）
- **seed 依存のブレが大きい**ため、再現性の裏取りが必要
- **公平な比較**を成立させるには、固定プロトコルが必須

### 本番で採用した運用（固定プロトコル）
- **label_mode=fine 固定**
- **freeze → full** の 2段構成
  - `freeze=2000` / `full=8000`
  - full は freeze ckpt から **必ず継承**（run名分離 + `--init-from-run`）
- **評価 / best ルール統一**
  - `best_metric=avg_r@1`
  - `eval_interval=200` / `log_interval=100`
  - `EarlyStop` 無効
- **Scheduler**: ReduceLROnPlateau
  - `plateau_patience=3` / `plateau_factor=0.5`
  - **監視指標は best と同一**（`avg_r@1`）
- **W&B**: `project=supcon_hoyo_main`（実験単位で group 分割）
- **seed**: 3 seeds 以上で再現性を担保

### 本番で行う比較の方針
- **Optuna ベストを起点に**
  - `temp`, `lambda_vae`, `lambda_cont` を**固定条件で感度分析**
- **Ablation で寄与を切り分け**
  - temp sweep（最優先）
  - λ_vae / λ_cont の 3点比較

---

## 5. 実験運用の狙い（結論）

- **Optunaで大きな傾向を掴み**、
- **固定プロトコル + 複数 seed + アブレーション**で
  - **再現性**
  - **要素の寄与**
  - **公平な比較**
  を論文として担保する。

---

## 付記: 参考スクリプト

- Optuna: `hoyo_v1_1/models/optuna_motionclip.py`
- 本番・一括スイープ: `run_contrastive_vae_sweeps.sh`
- 集約: `scripts/experiments/summarize_motionclip_runs.py`
