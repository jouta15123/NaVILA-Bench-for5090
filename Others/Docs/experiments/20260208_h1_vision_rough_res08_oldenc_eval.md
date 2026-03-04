# H1 Vision Rough Res08 Old Encoder Eval (2026-02-08)

## 目的
- `h1_vision_rough_res08_oldenc` run を `eval_motion.py` で動画付き評価し、定量結果と観察結果を記録する。

## 評価対象
- run: `logs/rsl_rl/h1_vision_rough_res08_oldenc/2026-02-07_h1_vision_without_speedinput_exp_c_fixed03_res08_oldenc_2026-02-07_exp_c_fixed03_res08_oldenc_seed42_seed42`
- task: `h1_vision_without_speedinput_exp_c_fixed03_res08_oldenc`
- checkpoint: `model_2500.pt`
- 実行日: 2026-02-08

## 評価条件
- `num_envs=16`
- `eval_steps=500`
- `terrain=flat`
- `lin_vel_x=0.3`, `lin_vel_y=0.0`, `ang_vel_z=0.0`
- `--log_reward_terms --video --video_length 500 --enable_cameras --headless`

## 出力ファイル
- JSON: `eval_results/motion/eval_motion_20260208_062544.json`
- heatmap:
  - `eval_results/motion/confusion_heatmap_centered_20260208_062544.png`
  - `eval_results/motion/confusion_heatmap_raw_20260208_062544.png`
  - `eval_results/motion/hoyo_pairwise_heatmap_20260208_062544.png`
  - `eval_results/motion/margin_heatmap_centered_20260208_062544.png`
  - `eval_results/motion/teacher_margin_heatmap_20260208_062544.png`
- 動画: `eval_results/motion/videos/20260208_062544/*.mp4`（11本）

## 全体サマリ
- 速度追従は崩壊: `cmd=0.3` に対し `mean_velocity_x` は 11スタイル平均 `0.0352`（MAE `0.2648`）。
- `fall_rate` は全スタイル `0.0`、`mean_episode_length` は全スタイル `500.0`。
- ただし動画観察では姿勢が崩れており、実用的な歩容ではない（ユーザー所見）。
- `mean_action_sq` は 11スタイル平均 `420.10`（min `301.09`, max `493.24`）で大きい。
- スタイル識別は弱い:
  - `hoyo_centered_top1 = 1/11`
  - `teacher_margin_top1 = 1/11`
- ログ警告:
  - `HOYO embeddings are too similar (off-diag mean=0.991)` が出力され、encoder 側の feature collapse 疑い。

## スタイル別主要値（抜粋）
| style | mean_velocity_x | mean_style_score | mean_cos_centroid | mean_joint_error |
| --- | ---: | ---: | ---: | ---: |
| 通常 | 0.0421 | 0.0000 | 0.5736 | 3.6271 |
| すたすた | 0.0398 | -0.1812 | 0.3228 | 4.5572 |
| せかせか | 0.0433 | -0.3617 | 0.3841 | 3.6557 |
| てくてく | 0.0294 | +0.0918 | 0.6743 | 3.7470 |
| どっしどっし | 0.0309 | +0.0572 | 0.7725 | 3.6833 |
| とぼとぼ | 0.0336 | +0.0449 | 0.6647 | 4.4908 |
| のしのし | 0.0325 | +0.0870 | 0.7453 | 4.5525 |
| のろのろ | 0.0336 | -0.0264 | 0.6210 | 3.3766 |
| ぶらぶら | 0.0330 | -0.0022 | 0.6638 | 4.1634 |
| よたよた | 0.0341 | -0.0525 | 0.6217 | 4.3420 |
| よろよろ | 0.0350 | +0.0079 | 0.6764 | 3.5048 |

## 混同行列
### H1 vs HOYO Similarity Matrix (Centered Cos)
![confusion-centered](../../eval_results/motion/confusion_heatmap_centered_20260208_062544.png)

### H1 vs HOYO Similarity Matrix (Raw Cos)
![confusion-raw](../../eval_results/motion/confusion_heatmap_raw_20260208_062544.png)

### Teacher Margin Matrix
![teacher-margin](../../eval_results/motion/teacher_margin_heatmap_20260208_062544.png)

## 結論
- 本 run は「転倒しないが歩けていない」状態で、姿勢の崩れも大きく、現状では利用困難。
- 速度追従（0.3への到達）と姿勢安定性の両面で、再学習または設定見直しが必要。
