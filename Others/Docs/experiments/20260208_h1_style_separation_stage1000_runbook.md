# Style分離強化 Runbook（Stage1=1000）

## 目的
- 新encoder環境で、`legacy` と `hardneg`、`residual` と `fullft` の差を2段階で比較する。

## 今までとの違い（重要）
- スタイル報酬を `legacy` だけでなく `hardneg` でも学習できるようにした。
- Stage1 を `1500` から `1000` に短縮した。
- 比較アームに `fullft_newenc`（Arm-C）を明示追加した。

### スタイル報酬の変更点
- 旧（legacy）:
  $$
  r_{\text{legacy}}
  = \beta_{\text{text}}\,r_{\text{text}}
  + \beta_{\text{teacher}}\,r_{\text{teacher}}
  $$
- 新（hardneg）:
  - 負例類似度 \(r_{\text{neg}}\)（現在ラベル以外の centroid との最大コサイン類似度）を使い、
  $$
  p
  = \max\!\left(0,\; r_{\text{neg}} - r_{\text{teacher}} + m\right)
  $$
  $$
  r_{\text{hardneg}}
  = r_{\text{pos}} - \lambda_{\text{neg}}\,p
  $$
  ここで、
  $$
  r_{\text{pos}}
  = \beta_{\text{text}}\,r_{\text{text}}
  + \beta_{\text{teacher}}\,r_{\text{teacher}},
  \quad
  m=\texttt{style\_neg\_margin},
  \quad
  \lambda_{\text{neg}}=\texttt{style\_neg\_weight}
  $$
- 切替は `train.py` のCLIで実施:
  - `--style_reward_mode {legacy,hardneg}`
  - `--style_neg_weight`
  - `--style_neg_margin`
- 可観測性として以下をログ出力:
  - `metrics/style_neg_sim`
  - `metrics/style_neg_penalty`
  - `metrics/style_reward_mode`

## 固定条件
- `seed=42`
- `style_weight=10.0`
- `style_centroid_mode=centroid`
- `history_length=9`
- `terrain=flat`
- hardneg: `style_neg_weight=0.4`, `style_neg_margin=0.03`
- 評価速度: `lin_vel_x=0.3`, `lin_vel_y=0.0`, `ang_vel_z=0.0`
- W&B: `entity=jouta15123-osaka-univercity`, `project=RL_new_style_reward`

## Arm定義
1. `armA`: `h1_vision_without_speedinput_exp_c_fixed03_res08_newenc` + `legacy`
2. `armB`: `h1_vision_without_speedinput_exp_c_fixed03_res08_newenc` + `hardneg`
3. `armC`: `h1_vision_without_speedinput_exp_c_fixed03_fullft_newenc` + `hardneg`

## run名規約
- Stage1: `s1_{arm}_seed42`
- Stage2: `s2_{arm}_seed42`

## Stage-1: 表現空間の予備診断（推奨）

hardnegの前提（centroid分離）を先に確認する。ここでは安全側の確認として
実運用値 `m=0.03` より厳しい `m=0.05` を使う。

```bash
cd /workspace/NaVILA-Bench

mkdir -p docs/experiments/assets/20260208_style_sep_precheck

# new encoder
/home/jouta/venvs/motionclip/bin/python scripts/experiments/analyze_centroid_headroom.py \
  --snapshot hoyo_v1_1/joint_training_results/20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full/latent_snapshot_final.npz \
  --margin 0.05 \
  --split test \
  --output-json docs/experiments/assets/20260208_style_sep_precheck/headroom_newenc_m005.json

# old encoder (比較用)
/home/jouta/venvs/motionclip/bin/python scripts/experiments/analyze_centroid_headroom.py \
  --snapshot hoyo_v1_1/joint_training_results/20260109_optuna_trial1_full/latent_snapshot_final.npz \
  --margin 0.05 \
  --split test \
  --output-json docs/experiments/assets/20260208_style_sep_precheck/headroom_oldenc_m005.json
```

判断の目安:
- `label_margin_unsafe_count` が少ないほど、`m=0.05` の hardneg が成立しやすい
- `penalty_trigger_rate` が高すぎる場合は、実運用の `m=0.03` へ下げるか、まずlegacyで回す

## 実行（Docker /workspace想定, Stage0スキップ版）

```bash
cd /workspace/NaVILA-Bench

export WANDB_ENTITY=jouta15123-osaka-univercity
export LOG_PROJECT_NAME=RL_new_style_reward

export STYLE_WEIGHT=10.0
export STYLE_NEG_WEIGHT=0.4
export STYLE_NEG_MARGIN=0.03

# Stage1: train (1000 iter x 3 arms)
bash scripts/experiments/run_h1_style_separation.sh stage1-train

# Stage1: eval (model_999.pt x 3 arms)
bash scripts/experiments/run_h1_style_separation.sh stage1-eval
```

## Stage2選抜（自動）

複合ゲート（Baseline=`armA` 比）:
- `mean_cos_centroid` 改善 `>= +0.05`
- margin offdiag mean 改善 `>= +0.01`
- 速度MAE悪化 `<= 0.03`
- `mean_joint_error` 悪化 `<= 0.5`

通過が2arm未満のときは fallback score で上位2armを選抜。

```bash
cd /workspace/NaVILA-Bench

python3 scripts/experiments/select_stage2_arms.py \
  --arm-result armA=eval_results/motion/style_sep_stage1/armA/model_999 \
  --arm-result armB=eval_results/motion/style_sep_stage1/armB/model_999 \
  --arm-result armC=eval_results/motion/style_sep_stage1/armC/model_999 \
  --baseline armA \
  --target-speed 0.3 \
  --output-json eval_results/motion/style_sep_stage1/selection_stage2.json \
  --output-selected eval_results/motion/style_sep_stage1/selected_arms.txt
```

## Stage2実行（上位2arm）

`selected_arms.txt` に `armB,armC` のようなCSVが出るので、それを `STAGE2_ARMS` に渡す。

```bash
cd /workspace/NaVILA-Bench

STAGE2_ARMS="$(cat eval_results/motion/style_sep_stage1/selected_arms.txt)" \
  bash scripts/experiments/run_h1_style_separation.sh stage2-train

STAGE2_ARMS="$(cat eval_results/motion/style_sep_stage1/selected_arms.txt)" \
  bash scripts/experiments/run_h1_style_separation.sh stage2-eval
```

## 出力先
- Stage1 eval: `eval_results/motion/style_sep_stage1/...`
- Stage2 eval: `eval_results/motion/style_sep_stage2/...`
- 選抜結果: `eval_results/motion/style_sep_stage1/selection_stage2.json`
- Stage2投入arm: `eval_results/motion/style_sep_stage1/selected_arms.txt`
