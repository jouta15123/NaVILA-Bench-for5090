# Data (Minimal Reproducibility Set)

このディレクトリは、研究再現に必要な最小コードと設定を整理したものです。

- `Study-01-Contrastive/`: 対照学習（MotionCLIP系）
- `Study-02-RL/`: RL評価・学習関連（legged-loco系）
- `Common/`: 共通設定・共通実験スクリプト

## Quick Repro Steps

1. `Common/configs/` の設定を確認
2. `Study-01-Contrastive/README.md` または `Study-02-RL/README.md` の入口スクリプトを実行
3. 出力はこのブランチでは追跡しない（`logs/`, `eval_results/`, `wandb/`）

## Removed Outputs

容量削減のため、学習ログ・評価結果・中間出力は除外しています。
必要な結果は外部バックアップから復元してください。
