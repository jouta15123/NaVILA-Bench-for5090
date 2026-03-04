# NaVILA-Bench Research Archive

このブランチ（`archive/research-20260304`）は、研究アーカイブ用途に再編した軽量構成です。
開発用の大容量生成物（`logs/`, `eval_results/`, `wandb/` など）は含めていません。

## 主要ディレクトリ

- `Thesis/`: 卒論・概要資料（TeX/図/最終PDF）
- `Paper/`: 論文メモ・下書き
- `Data/`: 最小再現セット（対照学習/RL/共通設定）
- `Others/`: 研究運用ドキュメント

## 参照マップ

旧パスから新パスへの対応は [`MIGRATION_MAP.md`](./MIGRATION_MAP.md) を参照してください。

## 最小再現の入口

- Contrastive: `Data/Study-01-Contrastive/README.md`
- RL: `Data/Study-02-RL/README.md`
- 共通設定: `Data/Common/README.md`

## 含めていないもの

以下は容量削減のためこのブランチには含めていません。

- `logs/`, `eval_results/`, `wandb/`
- `hoyo_v1_1/joint_training_results/`, `hoyo_v1_1/log/`, `hoyo_v1_1/proto_training_results/`
- `legged-loco/logs/`, `legged-loco/wandb/`, `legged-loco/eval_results/`
- `isaaclab_exts/omni.isaac.vlnce/assets/`, `_extracted/`

必要な生成物は外部バックアップ（ローカル保管・共有ストレージ）から復元してください。
