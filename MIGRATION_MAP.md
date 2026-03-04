# Migration Map (Old -> New)

| Old Path | New Path |
|---|---|
| `Sasaki_BachelorThesis_2025/` | `Thesis/BachelorThesis_2025/` |
| `Sasaki_abst/` | `Thesis/BachelorAbstract_2025/` |
| `paper/` | `Paper/Notes/` |
| `hoyo_v1_1/models/` | `Data/Study-01-Contrastive/code/hoyo_v1_1/models/` |
| `hoyo_v1_1/viz/` | `Data/Study-01-Contrastive/code/hoyo_v1_1/viz/` |
| `legged-loco/scripts/` | `Data/Study-02-RL/code/legged-loco/scripts/` |
| `legged-loco/src/` | `Data/Study-02-RL/code/legged-loco/src/` |
| `configs/` | `Data/Common/configs/` |
| `scripts/experiments/` | `Data/Common/scripts/experiments/` |
| `docs/command.md` and major docs under `docs/` | `Others/Docs/` |

## Policy Notes

- 論文フォルダは `Thesis/` 配下へ移動し、内側 `.git` は追跡対象外としています。
- 生成物（ログ・評価結果・W&B・中間出力）はアーカイブ方針により除外しています。
