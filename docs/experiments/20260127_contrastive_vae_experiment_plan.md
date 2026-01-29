# Contrastive + VAE 実験計画（論文用プロトコル）

最終更新: 2026-01-27

## 目的と主指標

- 目的: 提案法（Contrastive + VAE）の有効性と、効いている要素（temp / λ_vae / λ_cont）を切り分ける。
- 主指標: `avg_r@1`
- 付随指標: `avg_r@5`, `avg_r@10`(可能なら `acc@1`, `silhouette` も併記)

## 実験の流れ：Optunaで探索 → パラメータ固定で検証

論文用の実験では、以下の2段階で進める。

### Phase 1: Optunaによるハイパーパラメータ探索

まず、Optuna(TPE)で `temp`, `lambda_contrastive`, `lambda_vae` を広く探索し、Retrieval精度(`avg_r@1`)を最大化する設定を見つける。30試行の結果、以下の傾向が見えた:

- **temp は低いほど良い**(上位Trialは全て `temp ≈ 0.02`)
- **Contrastive強め + VAE弱め** が Retrieval には有利(`λ_cont > 2.0`, `λ_vae < 0.1`)

### Phase 2: 固定設定での再現性確認とアブレーション

Optunaの結果はあくまで「探索中のベスト」なので、論文で主張するには以下が必要:

1. **再現性の確認**  
   ベスト設定(Trial 1)を固定し、Seedを変えて複数回学習する。これで「たまたま良かった」のではなく、安定して高精度が出ることを示す。

2. **パラメータの寄与を切り分ける(Ablation)**  
   特に影響が大きかった `temp` について、値を振って性能変化を見る。ベースラインが固定されているので、`temp` だけの効果を公平に比較できる。

3. **考察のための材料を揃える**  
   「なぜこの設定が良いのか」を議論するには、周辺の設定との比較が要る。固定ベースラインがあることで、各要素(temp / λ_vae / λ_cont)の役割を深く考察できる。


## Protocol Lock（比較の固定条件）

比較実験は以下を固定する。

- バッチ: `batch_size=32`
- 学習長: `steps` 固定（epoch 固定は使わない）
- ラベル粒度: **fine 固定**（`--label-mode=fine`）
- scheduler: `ReduceLROnPlateau`（`--scheduler plateau` が既定）
- scheduler詳細: `plateau_patience=3`, `plateau_factor=0.5` を明示して固定
- scheduler 監視指標: **best 指標と同一**（`avg_r@1` を使う）
- 温度: **`temp` は必ず明示指定**（デフォルトに依存しない）
- ログ間隔: `--log-interval=100`
- 評価間隔: `--eval-interval=200`
- EarlyStop: 無効（`--early-stop-patience=0`）
- best 採用ルール: `--best-metric=avg_r@1`
- 2段構成の steps: `freeze=2000 -> full=8000`
- freeze/full の関係: **run名を分けても full は freeze を必ず引き継ぐ**
  - full は `--init-from-run <freeze_run_name>` で初期化する
  - run名は `..._freeze` / `..._full` を採用する

## 重要な実装メモ（修正済み）

- 以前は Plateau scheduler が `acc@1` を監視していた。
- 現在は **best 指標（score）を監視**するよう修正済み。
  - 対象: `hoyo_v1_1/models/train_motionclip_joint.py`
- `--init-from-run` を追加し、別run名でも freeze の ckpt を明示的に引き継げるようにした。
  - 対象: `hoyo_v1_1/models/train_motionclip_joint.py`

## W&B（固定方針）

- project: `supcon_hoyo_main`
- group: 実験単位で分ける（例: `temp_sweep_v1`, `lambda_vae_v1` など）
- 既定値は各 run スクリプトに入れてあり、環境変数で上書き可能。

例:

```bash
WANDB_PROJECT=supcon_hoyo_main \
WANDB_GROUP=temp_sweep_v1 \
bash run_all_experiments_supcon_main.sh
```

## 論文向けログ運用（おすすめ導線）

### 1) seedをまとめて回す（run_configに痕跡を残す）

run スクリプトは `run_config.json` に seed / git_commit / steps / best ルールを保存する。

例（production系）:

```bash
SEEDS=41,42,43 \
RUN_PREFIX=20260127_prod_supcon_v1 \
WANDB_PROJECT=supcon_hoyo_main \
WANDB_GROUP=prod_supcon_seed3 \
bash run_production_supcon_runs.sh
```

例（supcon_main系, coarse固定）:

```bash
SEEDS=41,42,43 \
RUN_PREFIX=20260127_supcon_main \
WANDB_PROJECT=supcon_hoyo_main \
WANDB_GROUP=supcon_main_seed3 \
bash run_all_experiments_supcon_main.sh
```

### 2) best step抽出 + seed集約（mean±std）を自動化

既存の集約スクリプトで、per-run と seed集約の両方を出力できる。

例（production系の集約）:

```bash
python3 scripts/experiments/summarize_motionclip_runs.py \
  --prefix 20260127_prod_supcon_v1 \
  --require-substring _full \
  --name prod_supcon_v1_seed3
```

例（supcon_main系の集約）:

```bash
python3 scripts/experiments/summarize_motionclip_runs.py \
  --prefix 20260127_supcon_main \
  --require-substring _full \
  --name supcon_main_seed3
```

出力（論文に貼りやすい形）:

- per-run: `.../<name>.csv`, `.../<name>.md`
- seed集約: `.../<name>_agg.csv`, `.../<name>_agg.md`

### 3) temp / λ_vae / λ_cont を一括で回す

以下のスクリプトで、計画にある 3 つのスイープをまとめて実行できる。

- `run_contrastive_vae_sweeps.sh`

例（sarashina, seed=3本で一気に）:

```bash
SEEDS=41,42,43 \
RUN_PREFIX=20260127_contrastive_vae \
WANDB_PROJECT=supcon_hoyo_main \
WANDB_GROUP=contrastive_vae_seed3 \
bash run_contrastive_vae_sweeps.sh
```

集約（_fullのみ）:

```bash
python3 scripts/experiments/summarize_motionclip_runs.py \
  --prefix 20260127_contrastive_vae \
  --require-substring _full \
  --name contrastive_vae_seed3
```

## アブレーション計画（最小で刺す）

### 1) temp 感度（最優先）

- 他を固定して temp のみ sweep する。
- 推奨点: `0.02 / 0.03 / 0.05 / 0.07 / 0.10`（必要なら 0.15）
- 出すもの:
  - `avg_r@1 (mean±std)`
  - 学習曲線（val `avg_r@1` と LR 推移）

### 2) λ_vae 感度（3点）

- temp はベスト帯に固定。
- 推奨点: `0.03 / 0.06 / 0.12`

### 3) λ_cont 感度（3点）

- temp / λ_vae を固定。
- 推奨点: `1.0 / 2.0 / 3.0`
- 制約: `λ_cont >= λ_vae`

### 4) 境界外ベストの反証（低コスト）

- Optuna が境界に張り付いた場合のみ、近傍を 2-3 点だけ追加確認する。

## ベースライン比較（同一プロトコル）

- VAEのみ（contrastive=0）
- contrastiveのみ（VAE=0）
- 提案法（両方あり）

すべてで steps / scheduler / eval_interval / best ルールを揃える。

## 実行順（迷わない順）

1. アンカー run を 1本
2. temp sweep
3. λ_vae 3点
4. λ_cont 3点
5. 必要なら境界外ミニ sweep
6. 最終設定 + baseline を seeds 複数

## run スクリプトの現状（チェック済み）

以下のスクリプトは、`supcon_hoyo_main` / `avg_r@1` / `eval_interval=200` に揃えてある。

- `run_all_experiments.sh`
- `run_all_experiments_full.sh`
- `run_all_experiments_supcon_main.sh`（freeze=2000 -> full=8000, run名分離 + init-from-run）
- `run_production_supcon_runs.sh`（run名分離 + init-from-run）
- `run_contrastive_vae_sweeps.sh`（temp / λ_vae / λ_cont を一括スイープ）

補足:

- `train_motionclip_joint.py` の既定 `--eval-interval` は 200。
- `--early-stop-patience` の既定は 0（無効）。
- split は seed に依存して固定される（main 冒頭で seed 固定済み）。

## 再現性ログ（最低限）

- seed
- commit hash
- split（seed と split ratio）
- 主要ハイパラ（temp / λ_vae / λ_cont / steps / scheduler / eval_interval）
