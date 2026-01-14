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

### オノマトペ別評価

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_style_per_onomatopoeia.py \
  --task h1_vision_heading_fixed \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint model_2999.pt \
  --num_envs 1 \
  --eval_steps 300 \
  --compute_hoyo_error
```

### 評価オプション一覧

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--task` | タスク名 | 必須 |
| `--load_run` | 学習済みモデルのディレクトリ名 | 必須 |
| `--checkpoint` | チェックポイントファイル名 | model_*.pt |
| `--num_envs` | 環境数（通常1で評価） | 1 |
| `--eval_steps` | 評価ステップ数 | 300 |
| `--video` | MP4動画を保存 | False |
| `--video_length` | 動画の長さ（ステップ数） | 500 |
| `--no_base_policy` | Base Policyを無効化 | False (デフォルトでロード) |
| `--compute_hoyo_error` | HOYOエラーを計算 | False |
| `--output_dir` | 結果出力先 | eval_results/style_per_onomatopoeia |
| `--base_velocity_mode` | 速度モード (fixed, style_table) | fixed |
| `--base_velocity_table` | 速度テーブルJSONパス | None |
| `--save_hoyo_gif` | HOYO 2Dマッピングを GIF で保存 | False |
| `--save_hoyo_comparison_gif` | H1とHOYOリファレンスの比較GIF | False |
| `--hoyo_gif_fps` | GIF の FPS | 50 |
| `--hoyo_gif_stride` | GIF フレームストライド | 2 |

### ✅ 修正済み: Base Policy のロード

**Base Policyはデフォルトでロードされるようになりました。**

ResidualActorCritic アーキテクチャでは：
- 学習時: Base Policy + Style Residual で学習
- 評価時: Base Policy が自動ロードされる ✅

無効化したい場合のみ `--no_base_policy` を指定してください。

### ✅ 修正済み: yaw リセット

**heading_fixed タスクでは yaw=0 で初期化されるようになりました（訓練と同じ）。**

---

## 評価例

### 基本評価（h1_vision_heading_fixed で学習したモデル）

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_style_per_onomatopoeia.py \
  --task h1_vision_heading_fixed \
  --load_run 2026-01-09_15-16-31_trial_h1_vision_style_flat \
  --checkpoint model_1999.pt \
  --num_envs 1 \
  --eval_steps 300 \
  --compute_hoyo_error
```

### Exp-A 評価（h1_vision_heading_fixed_exp_a で学習したモデル）

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_style_per_onomatopoeia.py \
  --task h1_vision_heading_fixed_exp_a \
  --load_run 2026-01-11_14-09-32_exp_a_style2_orient05 \
  --checkpoint model_2999.pt \
  --num_envs 1 \
  --eval_steps 300 \
  --compute_hoyo_error
```

### スタイル別速度テーブルを使用した評価

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_style_per_onomatopoeia.py \
  --task h1_vision_heading_fixed \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint model_2999.pt \
  --num_envs 1 \
  --eval_steps 300 \
  --compute_hoyo_error \
  --base_velocity_mode style_table \
  --base_velocity_table configs/style_speed_table_auto.json
```

### MP4 動画を保存（全オノマトペ）

`--video` を付けると各オノマトペごとに MP4 が保存されます。  
動画長は `--video_length`（ステップ数）で指定し、**50 FPS 固定**で保存されます。

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_style_per_onomatopoeia.py \
  --task h1_vision_heading_fixed \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint model_2999.pt \
  --num_envs 1 \
  --video \
  --video_length 500
```

動画は `eval_results/style_per_onomatopoeia/videos/` に保存されます。

### HOYO マッピングを GIF で可視化

H1ロボットのポーズがHOYO 2D座標にどう変換されているか確認できます。

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_style_per_onomatopoeia.py \
  --task h1_vision_heading_fixed \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint model_2999.pt \
  --num_envs 1 \
  --eval_steps 300 \
  --save_hoyo_gif \
  --hoyo_gif_fps 30 \
  --hoyo_gif_stride 2
```

GIFは `eval_results/style_per_onomatopoeia/hoyo_gifs/` に保存されます。

### H1 と HOYO リファレンスの比較 GIF

H1ロボットの動作と、HOYOデータセットの教師データを並べて比較できます。

```bash
STYLE_RUN_NAME=20260109_optuna_trial1_full \
STYLE_CENTROID_MODE=random \
/workspace/IsaacLab/isaaclab.sh -p legged-loco/scripts/eval_style_per_onomatopoeia.py \
  --task h1_vision_heading_fixed \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint model_2999.pt \
  --num_envs 1 \
  --eval_steps 300 \
  --save_hoyo_comparison_gif \
  --hoyo_gif_fps 30 \
  --hoyo_gif_stride 2
```

比較GIFは `eval_results/style_per_onomatopoeia/hoyo_comparison_gifs/` に保存されます。
- 左: H1ロボットのマッピング（青）
- 右: HOYOリファレンス（緑）

---

## タスク一覧

| タスク名 | 説明 | style_tracking | flat_orientation_l2 |
|----------|------|----------------|---------------------|
| `h1_vision_heading_fixed` | 基本設定 | 5.0 | -0.2 |
| `h1_vision_heading_fixed_exp_a` | 姿勢ペナルティ強化 | 2.0 | -0.5 |

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
