# 実験設定サマリ (Phase 1: NaVILA + H1 Vision)

現在の学習設定（`h1_low_vision_cfg.py`）の概要です。

## 1. アルゴリズム (Algorithm)
**Residual Policy (残差ポリシー)** を採用し、事前学習済みの歩行動作を維持しながら、スタイルのみを追加学習します。

*   **ポリシー構造**:
    *   $a_{total} = \pi_{base}(o_{base}) + \lambda \cdot \pi_{residual}(o_{base}, z_{style})$
    *   **Base Policy**: 事前学習済みモデル（重み固定）
    *   **Residual Policy**: 新規学習（スタイル条件付き）
    *   **Residual Scale ($\lambda$)**: `0.1` (初期設定)
*   **学習手法**: PPO (Proximal Policy Optimization)

## 2. ベースモデル (Base Model)
*   **Checkpoint**: `/home/jouta/NaVILA-Bench/logs/rsl_rl/h1_vision_rough/2024-11-03_15-08-09_height_scan_obst/model_4999.pt`
*   **入力**: 視覚情報（Height Scan）を含む歩行ポリシー

## 3. 観測空間 (Observation Space)
`policy` グループ（Actorへの入力）には、基本観測に加え **スタイル潜在変数** が追加されています。

| 観測項目 | 次元 | 説明 | 備考 |
| :--- | :--- | :--- | :--- |
| `base_lin_vel` | 3 | ベース並進速度 | |
| `base_ang_vel` | 3 | ベース角速度 | |
| `projected_gravity` | 3 | 重力ベクトル射影 | |
| `velocity_commands` | 3 | 指示速度 (x, y, yaw) | |
| `joint_pos` | 19 | 関節位置 (相対) | |
| `joint_vel` | 19 | 関節速度 | |
| `actions` | 19 | 前ステップの行動 | |
| `height_scan` | N | 地形ハイトスキャン | Vision対応 |
| **`style_latents`** | **512** | **オノマトペ/テキスト潜在変数** | **今回追加** |

## 4. 報酬関数 (Reward Function)
基本的な歩行報酬 (`H1Rewards`) に、スタイル追従報酬を追加しています。

### 追加・変更された報酬
| 報酬項 | 重み (Weight) | 説明 | パラメータ |
| :--- | :--- | :--- | :--- |
| **`style_tracking`** | **0.5** | スタイル追従報酬 | `beta_text=0.5`<br>`beta_centroid=0.5` |
| `feet_stumble` | -0.5 | 足のつまずきペナルティ | `CustomH1Rewards` で再定義 |

#### `style_tracking` の詳細
現在のエージェントの動作履歴（過去60ステップ）をMotionCLIPエンコーダで潜在変数 $z_{agent}$ に変換し、指示テキストとの類似度を計算します。

```math
R_{style} = \beta_{text} \cdot \text{cos}(z_{agent}, z_{text}) + \beta_{centroid} \cdot \text{cos}(z_{agent}, z_{centroid})
```

*   **$z_{agent}$**: 現在の歩行動作（正規化済み）
*   **$z_{text}$**: 指示オノマトペ（例：「のしのし」）のテキスト埋め込み
*   **$z_{centroid}$**: 該当クラスの重心ベクトル（学習の安定化用）
*   **$\text{cos}(a, b)$**: コサイン類似度 ($a \cdot b$)

### 基本報酬 (H1Rewards から継承)
*   `track_lin_vel_xy_exp`: 平面速度追従
*   `track_ang_vel_z_exp`: 旋回速度追従
*   `lin_vel_z_l2`: 上下動抑制 (ペナルティ)
*   `ang_vel_xy_l2`: 傾き抑制 (ペナルティ)
*   その他、トルク最小化、関節速度抑制などの正則化項

## 5. その他設定
*   **タスク名**: `h1_vision`
*   **スタイルコマンド**: `StyleCommandGeneratorCfg` (ランダムなオノマトペ/テキスト埋め込みを生成)
