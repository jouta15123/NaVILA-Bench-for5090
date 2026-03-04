# Hardneg前提チェック（Headroom予備実験）

- 実行日: 2026-02-08
- 目的: hardneg報酬の前提（centroid分離）が成立しているかを、学習前に確認する
- スクリプト: `scripts/experiments/analyze_centroid_headroom.py`
- 判定margin: `m=0.05`
- split: `test`

## 対象
1. new encoder  
   `20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full`
2. old encoder  
   `20260109_optuna_trial1_full`

## 主要結果

| encoder | top1 | headroom_mean | headroom_q10 | penalty_trigger_rate | penalty_mean | centroid_offdiag_mean | unsafe_labels (m=0.05) |
|---|---:|---:|---:|---:|---:|---:|---:|
| new | 0.6935 | 0.1509 | -0.2840 | 0.3548 | 0.0780 | 0.1212 | 5 / 11 |
| old | 0.7258 | 0.0393 | -0.0656 | 0.5161 | 0.0346 | 0.5722 | 11 / 11 |

## 解釈

- new encoder は old encoder より centroid分離が明確に改善している（offdiag mean: `0.1212` vs `0.5722`）。
- ただし new encoder でも `m=0.05` では unsafe label が `5/11` あり、hardnegが過剰に発火する可能性が残る。
- old encoder は `m=0.05` を満たせる見込みが低く、hardneg前提としては不利。

## 判断（今回の実験向け）

- 予定通り new encoder で進めるのは妥当。
- ただし hardnegの安定化のため、以下を推奨:
  1. Stage0/Stage1初期ログで `style_neg_penalty` の発火率を確認する
  2. 発火率が高すぎる場合は `style_neg_margin` を `0.05 -> 0.02~0.03` に下げる
  3. 比較の公平性のため、margin変更時は Arm-B/C の両方で同値に固定する

## 出力ファイル

- `docs/experiments/assets/20260208_style_sep_precheck/headroom_newenc_m005.json`
- `docs/experiments/assets/20260208_style_sep_precheck/headroom_oldenc_m005.json`

