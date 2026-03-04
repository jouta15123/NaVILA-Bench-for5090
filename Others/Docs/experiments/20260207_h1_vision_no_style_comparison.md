# H1 Vision No-Style vs Style Comparison (2026-02-07)

## 目的
- `style_tracking=0.0`（no-style）と `style_tracking=6.0`（styleあり）を同条件で比較し、
  速度・混同行列・その他指標を整理する。

## 比較対象
- no-style run: `logs/rsl_rl/h1_vision_rough/2026-02-06_h1_vision_without_speedinput_exp_c_fixed03_exp_c_cmd03_no_style_seed42`
- style run: `logs/rsl_rl/h1_vision_rough/2026-02-01_h1_vision_without_speedinput_exp_c_fixed03_exp_c_cmd03_stylew4_seed42`
- checkpoint: `model_2999.pt`
- task: `h1_vision_without_speedinput_exp_c_fixed03`
- eval条件: `num_envs=16`, `eval_steps=500`, `terrain=flat`, `--log_reward_terms`
- 実行日: 2026-02-07

## 出力ファイル
- no-style JSON (`cmd=0.5`): `eval_results/motion/20260207_no_style_compare/no_style_m2999/eval_motion_20260207_040351.json`
- no-style JSON (`cmd=0.3`, 学習条件合わせ): `eval_results/motion/20260207_no_style_compare/no_style_m2999_cmd03/eval_motion_20260207_041753.json`
- no-style JSON (`cmd=0.3`, 強制buffer更新): `eval_results/motion/20260207_no_style_compare/no_style_force_m2999_cmd03/eval_motion_20260207_050323.json`
- style JSON: `eval_results/motion/20260207_no_style_compare/style_m2999/eval_motion_20260207_040548.json`
- style JSON (`cmd=0.3`, 学習条件合わせ): `eval_results/motion/20260207_no_style_compare/style_m2999_cmd03/eval_motion_20260207_052311.json`
- no-style マージン混同行列 (centered, `cmd=0.3`, 強制buffer更新): `eval_results/motion/20260207_no_style_compare/no_style_force_m2999_cmd03/margin_heatmap_centered_20260207_050323.png`
- style マージン混同行列 (centered, `cmd=0.3`): `eval_results/motion/20260207_no_style_compare/style_m2999_cmd03/margin_heatmap_centered_20260207_052311.png`

## 重要な追記（2026-02-07）
- no-style / style run の学習時 `base_velocity.lin_vel_x` は `0.3` 固定（`params/env.yaml`）。
- ただし `eval_motion.py` は `--lin_vel_x` 未指定時に `0.5` を上書きする。
- 先の no-style `fall_rate=100%` は **`cmd=0.5` の評価条件**で発生した値。
- 同じ checkpoint を `cmd=0.3` で再評価すると、全11スタイルで `fall_rate=0%` / `episode_count=0` を確認。

## 全体サマリ
| 指標 | no-style (`cmd=0.5`) | no-style (`cmd=0.3`) | styleあり (`cmd=0.5`) |
| --- | ---: | ---: | ---: |
| mean velocity_x (11style平均) | 0.7213 | 0.3048 | 0.5027 |
| velocity MAE to command | 0.2213 (to 0.5) | 0.0048 (to 0.3) | 0.0223 (to 0.5) |
| velocity range (max-min) | 0.0494 | 0.0058 | 0.0745 |
| fall_rate 平均 | 1.0000 | 0.0000 | 0.0000 |
| mean_episode_length 平均 | 215.7 | 500.0 | 500.0 |
| mean joint_error (L2) | 1.3309 | 1.3309 | 3.7735 |
| mean_action_sq | 22.6848 | 6.9020 | 22.1789 |
| mean cos_centroid | N/A | N/A | 0.6144 |
| mean style_score | N/A | N/A | 0.0074 |
| centered margin offdiag mean (diag-col) | N/A | +0.0004 (force) | +0.0091 |
| centered margin top1一致率 | N/A | 1/11 (=0.0909, force) | 1/11 (=0.0909) |

## 速度の比較
- no-style の `cmd=0.5` 評価では `velocity_x` が 0.689〜0.739 と大きく上振れし、転倒が発生。
- no-style の `cmd=0.3` 評価では 0.302〜0.308 に収まり、学習時コマンドにほぼ一致。
- styleあり (`cmd=0.5`) は 0.457〜0.532 で安定し、こちらも転倒は発生しない。

## マージン混同行列の比較
### no-style
- 通常モードでは `cos_centroid_count=0`, `style_score_count=0`（全11スタイル）で `confusion_heatmap_*` は未生成。
- `--force_style_buffer_update` を使うと no-style でも埋め込みが計算され、マージン混同行列を生成可能。
- 強制モード実測: 全11スタイルで `cos_centroid_count=640`, `style_score_count=640`。
- 強制モードの centered margin 指標: `offdiag mean=+0.00042`, `min=-1.773`, `top1一致率=1/11`。

![no-style-centered-margin-forced](../../eval_results/motion/20260207_no_style_compare/no_style_force_m2999_cmd03/margin_heatmap_centered_20260207_050323.png)

### styleあり
- 同条件比較のため `cmd=0.3` で評価した centered margin を使用。
- （注）全体サマリ表の style 列は `cmd=0.5` 集計で、本節のみ `cmd=0.3` の同条件比較を示す。
- 行ごとの top1 が **全スタイルで `どっしどっし` 列** に集中。
- centered margin 指標: `offdiag mean=+0.01049`, `min=-1.606`, `top1一致率=1/11`。

![style-centered-margin-cmd03](../../eval_results/motion/20260207_no_style_compare/style_m2999_cmd03/margin_heatmap_centered_20260207_052311.png)

## その他評価
### 報酬寄与（平均）
- no-style (`cmd=0.5`):
  - `rate^mag` top1: `termination_penalty` 53.74%（11/11スタイル）
  - `rate^adv` top1: `termination_penalty` 98.00%（11/11スタイル）
- no-style (`cmd=0.3`):
  - `rate^mag` top1: `track_lin_vel_xy_exp` 100%（11/11スタイル）
  - `rate^adv` top1: `heading_tracking` 10/11（残り1/11は `action_rate_l2`）
- styleあり:
  - `rate^mag` top1: `style_tracking` 53.14%（9/11スタイル）
  - `rate^adv` top1: `style_tracking` 59.86%（8/11スタイル）

### 代表的な解釈
- no-style の「崩壊」は `cmd=0.5` への条件変更で起きた現象で、学習条件 `cmd=0.3` では再現しない。
- styleありは安定歩行は維持でき、style関連指標も算出可能だが、ラベル分離は `どっしどっし` への吸着が強い。

### スタイル別比較（主要値）
| style | vel_no(0.5) | vel_no(0.3) | vel_style(0.5) | fall_no(0.5) | fall_no(0.3) | fall_style(0.5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 通常 | 0.7367 | 0.3044 | 0.4912 | 1.0 | 0.0 | 0.0 |
| すたすた | 0.6894 | 0.3057 | 0.4571 | 1.0 | 0.0 | 0.0 |
| せかせか | 0.6941 | 0.3081 | 0.4602 | 1.0 | 0.0 | 0.0 |
| てくてく | 0.6956 | 0.3056 | 0.4836 | 1.0 | 0.0 | 0.0 |
| どっしどっし | 0.7349 | 0.3049 | 0.5228 | 1.0 | 0.0 | 0.0 |
| とぼとぼ | 0.7343 | 0.3033 | 0.5180 | 1.0 | 0.0 | 0.0 |
| のしのし | 0.7253 | 0.3023 | 0.5201 | 1.0 | 0.0 | 0.0 |
| のろのろ | 0.7211 | 0.3031 | 0.5032 | 1.0 | 0.0 | 0.0 |
| ぶらぶら | 0.7389 | 0.3068 | 0.5182 | 1.0 | 0.0 | 0.0 |
| よたよた | 0.7296 | 0.3046 | 0.5235 | 1.0 | 0.0 | 0.0 |
| よろよろ | 0.7345 | 0.3040 | 0.5317 | 1.0 | 0.0 | 0.0 |

## 考察
- no-style は学習条件 (`cmd=0.3`) では転倒しない。したがって「no-styleだから転倒する」という結論は誤り。
- `fall_rate=100%` は `cmd=0.5` へ外挿したときの失敗モードであり、**速度条件差の影響**が支配的。
- margin 混同行列で見ても style/no-style ともに top1 は `1/11` で、`どっしどっし` への吸着傾向は変わらない。
- styleありは安定性回復に寄与しているが、マージン分離（`offdiag mean`）は改善が小さく、スタイル条件付けの識別性能はまだ不足。
- no-style の `cos/style` 未算出は `cmd=0.3` でも継続しているため、別要因（embedding算出経路）として切り分ける必要がある。
- no-style のマージン混同行列は強制モードで可視化可能になったため、今後は「通常モード」と「強制モード」を分けて記録する。

## 次にやるべきこと
1. margin 指標（`offdiag mean`, `row-best margin`）を継続記録し、`どっしどっし` 吸着の緩和を追跡する。
2. no-style の `cos_centroid_count=0` 原因を `get_current_motion_embedding()` 側で追跡し、転倒以外の欠損要因を特定する。
3. `cmd=0.3` と `cmd=0.5` を分けて記録し、速度外挿耐性とスタイル分離性能を独立評価する。
