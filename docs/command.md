# 強化学習 訓練・評価コマンド集

## 概要

H1ロボットのスタイル歩行学習に関する訓練・評価コマンドをまとめています。

---

## 環境変数

```bash
# WandB設定
export WANDB_ENTITY=jouta15123-osaka-univercity

# スタイルモジュール設定
export STYLE_RUN_NAME=20260109_optuna_trial1_full    # MotionCLIP学習済みモデル
export STYLE_CENTROID_MODE=random                    # random or fixed
```

---

## 訓練コマンド

### 基本タスク: `h1_vision_heading_fixed`

報酬設定:
- `track_lin_vel_xy_exp`: 0.5
- `style_tracking`: 5.0
- `flat_orientation_l2`: -0.2

```bash
WANDB_ENTITY=jouta15123-osaka-univercity \
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/train.py \
  --task h1_vision_heading_fixed \
  --num_envs 2048 \
  --max_iterations 3000 \
  --save_interval 200 \
  --history_length 9 \
  --run_name <RUN_NAME> \
  --terrain flat \
  --headless \
  --logger wandb \
  --log_project_name StyleWalker_RL_fixed \
  --experiment_name <EXPERIMENT_NAME> \
  --style_beta_text 0.0 \
  --style_beta_teacher_motion 1.0
```

### 実験A: `h1_vision_heading_fixed_exp_a`

報酬設定:
- `track_lin_vel_xy_exp`: 0.5
- `style_tracking`: 2.0 (減少)
- `flat_orientation_l2`: -0.5 (増加)

```bash
WANDB_ENTITY=jouta15123-osaka-univercity \
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/train.py \
  --task h1_vision_heading_fixed_exp_a \
  --num_envs 2048 \
  --max_iterations 3000 \
  --save_interval 200 \
  --history_length 9 \
  --run_name exp_a_style2_orient05 \
  --terrain flat \
  --headless \
  --logger wandb \
  --log_project_name StyleWalker_RL_fixed \
  --experiment_name style_exp_a \
  --style_beta_text 0.0 \
  --style_beta_teacher_motion 1.0
```

### 速度コマンド固定: `h1_vision_without_speedinput`

目的:
- 速度コマンドを固定し、スタイル主導の速度変化を観察
- heading/速度追従報酬を抑えた設定

```bash
WANDB_ENTITY=jouta15123-osaka-univercity \
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/train.py \
  --task h1_vision_without_speedinput \
  --num_envs 2048 \
  --max_iterations 3000 \
  --save_interval 200 \
  --history_length 9 \
  --run_name <RUN_NAME> \
  --terrain flat \
  --headless \
  --logger wandb \
  --log_project_name StyleWalker_RL_fixed \
  --experiment_name <EXPERIMENT_NAME> \
  --style_beta_text 0.0 \
  --style_beta_teacher_motion 1.0
```

### 実験B: 速度追従強化 `h1_vision_without_speedinput_exp_b`（2026-01-18）

報酬設定:
- `track_lin_vel_xy_exp`: 0.5（0.1 → 0.5）
- `style_tracking`: 2.0
- `flat_orientation_l2`: -0.5
- `heading_tracking`: 0.2

コマンド:
```bash
WANDB_ENTITY=jouta15123-osaka-univercity \
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/train.py \
  --task h1_vision_without_speedinput_exp_b \
  --num_envs 2048 \
  --max_iterations 3000 \
  --save_interval 200 \
  --history_length 9 \
  --run_name exp_b_track05 \
  --terrain flat \
  --headless \
  --logger wandb \
  --log_project_name StyleWalker_RL_fixed \
  --experiment_name style_exp_b \
  --style_beta_text 0.0 \
  --style_beta_teacher_motion 1.0
```

### 実験B-0.5固定: 速度コマンド固定 `h1_vision_without_speedinput_exp_b_fixed05`（2026-01-19）

変更点:
- base_velocity: lin_vel_x = 0.5 固定
- 報酬設定は ExpB と同じ

コマンド:
```bash
WANDB_ENTITY=jouta15123-osaka-univercity \
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/train.py \
  --task h1_vision_without_speedinput_exp_b_fixed05 \
  --num_envs 2048 \
  --max_iterations 3000 \
  --save_interval 200 \
  --history_length 9 \
  --run_name exp_b_cmd05 \
  --terrain flat \
  --headless \
  --logger wandb \
  --log_project_name StyleWalker_RL_fixed \
  --experiment_name style_exp_b \
  --style_beta_text 0.0 \
  --style_beta_teacher_motion 1.0
```

### 訓練オプション一覧

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--task` | タスク名 | 必須 |
| `--num_envs` | 並列環境数 | 4096 |
| `--max_iterations` | 最大イテレーション数 | 1500 |
| `--save_interval` | チェックポイント保存間隔 | 50 |
| `--history_length` | 観測履歴の長さ | 9 |
| `--run_name` | 実行名（ログディレクトリ名に使用） | 自動生成 |
| `--terrain` | 地形タイプ (flat, rough) | flat |
| `--headless` | GUI無しで実行 | False |
| `--logger` | ロガー (tensorboard, wandb) | tensorboard |
| `--style_beta_text` | テキスト埋め込みの重み | 0.0 |
| `--style_beta_teacher_motion` | ティーチャーモーション埋め込みの重み | 1.0 |

---

## 評価コマンド

### MotionCLIP 学習後 PCA 可視化（unknown words 付き）

用途:
- motion latent の分布を PCA 2D で可視化
- 未知語（学習データ外）を Sarashina + sem_proj で射影して重ね描画

```bash
python hoyo_v1_1/viz/plot_latent_spaces.py \
  --snapshot hoyo_v1_1/joint_training_results/sarashina_full_fixed/latent_snapshot_final.npz \
  --out-dir hoyo_v1_1/joint_training_results/visualizations \
  --label-mode coarse-with-normal \
  --unknown-words hoyo_v1_1/data/unknown_words_coarse.txt \
  --sem-proj hoyo_v1_1/joint_training_results/sarashina_full_fixed/checkpoints/sem_proj_joint_best.pth
```

補足:
- `--label-mode coarse-with-normal` で「通常」を独立カテゴリとして表示
- unknown words は `hoyo_v1_1/data/unknown_words_coarse.txt` を編集すれば反映される

### Exp-C 評価（eval_motion.py / 新encoder使用）

新しい対照学習encoderを使用した評価。

```bash
STYLE_RUN_NAME=20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_motion.py \
  --task h1_vision_without_speedinput_exp_c_fixed03 \
  --load_run 2026-01-28_h1_vision_without_speedinput_exp_c_fixed03_exp_c_cmd03_seed42_dup1 \
  --checkpoint model_2000.pt \
  --num_envs 16 \
  --eval_steps 500 \
  --terrain flat \
  --log_reward_terms \
  --headless
```

### Exp-C 評価（video保存 / Docker）

video 保存時は `--enable_cameras` が必要。

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_motion.py \
  --task h1_vision_without_speedinput_exp_c_fixed03 \
  --load_run 2026-02-01_h1_vision_without_speedinput_exp_c_fixed03_exp_c_cmd03_stylew4_seed42 \
  --checkpoint model_2999.pt \
  --num_envs 16 \
  --eval_steps 500 \
  --terrain flat \
  --log_reward_terms \
  --video \
  --video_length 500 \
  --enable_cameras \
  --headless
```

### 比較doc用: 既存JSONから混同行列を速度順で再生成

`eval_motion.py` の評価を再実行せず、既存の `eval_motion_*.json` から
混同行列画像だけを速度順（すたすた→…→よろよろ）で再生成する。

```bash
/home/jouta/venvs/motionclip/bin/python scripts/experiments/regenerate_confusion_heatmap_from_json.py \
  --input-json eval_results/motion/20260201_compare/old_dup1_m2000/eval_motion_20260201_091804.json \
  --output-png docs/experiments/assets/20260201_h1_vision_style_reward_comparison/old_dup1_m2000_confusion_heatmap_centered_speed_order_20260207.png \
  --title "H1 vs HOYO Similarity Matrix (Centered Cos, Speed Order)"

/home/jouta/venvs/motionclip/bin/python scripts/experiments/regenerate_confusion_heatmap_from_json.py \
  --input-json eval_results/motion/eval_motion_20260203_000603.json \
  --output-png docs/experiments/assets/20260201_h1_vision_style_reward_comparison/new_stylew4_dup1_m2000_confusion_heatmap_centered_speed_order_20260207.png \
  --title "H1 vs HOYO Similarity Matrix (Centered Cos, Speed Order)"
```

### 比較doc用: 既存JSONからマージン行列を速度順で再生成（主指標）

`M[i,j] = S[i,i] - S[i,j]`（`S`: centered cosine）を画像化する。

```bash
/home/jouta/venvs/motionclip/bin/python scripts/experiments/regenerate_confusion_heatmap_from_json.py \
  --mode margin \
  --sim-key hoyo_similarity_centered_mean \
  --input-json eval_results/motion/20260201_compare/old_dup1_m2000/eval_motion_20260201_091804.json \
  --output-png docs/experiments/assets/20260201_h1_vision_style_reward_comparison/old_dup1_m2000_margin_heatmap_centered_speed_order_20260207.png \
  --title "H1 vs HOYO Margin Matrix (Centered Cos, Speed Order)"

/home/jouta/venvs/motionclip/bin/python scripts/experiments/regenerate_confusion_heatmap_from_json.py \
  --mode margin \
  --sim-key hoyo_similarity_centered_mean \
  --input-json eval_results/motion/eval_motion_20260203_000603.json \
  --output-png docs/experiments/assets/20260201_h1_vision_style_reward_comparison/new_stylew4_dup1_m2000_margin_heatmap_centered_speed_order_20260207.png \
  --title "H1 vs HOYO Margin Matrix (Centered Cos, Speed Order)"

/home/jouta/venvs/motionclip/bin/python scripts/experiments/regenerate_confusion_heatmap_from_json.py \
  --mode margin \
  --sim-key hoyo_similarity_centered_mean \
  --input-json eval_results/motion/eval_motion_20260202_082729.json \
  --output-png docs/experiments/assets/20260201_h1_vision_style_reward_comparison/stylew6_m2999_margin_heatmap_centered_speed_order_20260207.png \
  --title "H1 vs HOYO Margin Matrix (Centered Cos, Speed Order)"
```

## タスク一覧

| タスク名 | 説明 | style_tracking | flat_orientation_l2 |
|----------|------|----------------|---------------------|
| `h1_vision_heading_fixed` | 基本設定 | 5.0 | -0.2 |
| `h1_vision_heading_fixed_exp_a` | 姿勢ペナルティ強化 | 2.0 | -0.5 |
| `h1_vision_without_speedinput` | 速度コマンド固定 + heading抑制 | 2.0 | -0.5 |
| `h1_vision_without_speedinput_exp_b` | 速度コマンド固定 + 速度追従強化 | 2.0 | -0.5 |
| `h1_vision_without_speedinput_exp_b_fixed05` | 速度コマンド0.5固定 + 速度追従強化 | 2.0 | -0.5 |

タスク設定ファイル: `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/config/h1/h1_low_vision_cfg.py`

---

## 学習済みモデルのパス

```
/workspace/NaVILA-Bench/logs/rsl_rl/h1_vision_rough/<RUN_DIRECTORY_NAME>/
├── model_*.pt          # チェックポイント
├── env.yaml            # 環境設定
└── params/
    └── agent.yaml      # エージェント設定
```

Base Policy（事前学習済み）:
```
/workspace/NaVILA-Bench/logs/rsl_rl/h1_vision_rough/2024-11-03_15-08-09_height_scan_obst/model_4999_pad877_256.pt
```

---

## トラブルシューティング

### ロボットが全く歩けない
- `--no_base_policy` を誤って指定していないか確認
- Base Policyのパスが正しいか確認
- 評価ログで「Base policy will be loaded」が表示されているか確認

### 評価時に環境設定が適用されない
- `--task` を学習時と同じに設定
- `--use_log_env` が有効か確認（デフォルト True）

### スタイルの差が出ない
- 速度コマンドが固定されている可能性
- `--base_velocity_mode style_table` で速度テーブルを使用
