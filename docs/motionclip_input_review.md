# MotionCLIP入力レビュー（StyleModule 前処理〜z_m算出）

## 対象範囲

- `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/leggedloco/mdp/style_module.py`
  - `update_buffer`
  - `_get_hoyo_joints_from_h1`
  - `_prepare_centered_2d`
  - `encode_buffer`
- `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/leggedloco/mdp/commands/style_command_generator.py`
  - `StyleCommandGeneratorCfg.coord_mode`
- `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/config/h1/h1_low_vision_cfg.py`
  - `H1VisionRoughEnvCfg_Legacy` の `coord_mode`

## パイプライン概要（入力〜z_m）

1. **H1リンク→HOYO 14関節へのマッピング**
   - `update_buffer` 内で `_get_hoyo_joints_from_h1` を呼び出し、
     `body_pos_w (B, N, 3)` を `hoyo_3d (B, 14, 3)` に変換。
   - head/neck/hand など不足リンクはオフセット推定。

2. **バッファ更新**
   - `motion_buffer`: `(B, T, 14, 3)` を時系列でシフト更新。
   - `heading_buffer`: `(B, T)` にyawを蓄積。
   - `T = buffer_len`（デフォルト100フレーム = 約2秒@50Hz）。

3. **前処理（_prepare_centered_2d）**
   - `coord_mode` に応じて 3D→2Dへ投影し、正規化・標準化。

4. **MotionCLIPへ入力**
   - `encode_buffer` で `_prepare_centered_2d(apply_yaw_correction=True)` を呼び出し、
     `(B, T, 14, 2)` を `(B, 14, 2, T)` へ変換。
   - MotionCLIP出力 `mu` を正規化して `z_m` を得る。

## MotionCLIPへの「データの入れ方」（入力仕様）

`style_module.py` の `encode_buffer()` が実際に MotionCLIP へ渡しているバッチ内容。

- **入力テンソル `x`**
  - 形状: `(B, 14, 2, T)`
  - 生成: `_prepare_centered_2d()` で `(B, T, 14, 2)` を作り、`permute(0, 2, 3, 1)` で並べ替え。
  - 座標順: `hoyo_front` の場合は **[x=Y_lat, y=-Z_up]**。
  - Yaw補正: `encode_buffer()` では **常に apply_yaw_correction=True**。

- **マスク `mask`**
  - 形状: `(B, T)`、全て `True`。
  - つまり「全フレーム有効」として扱われる（欠損や可変長は使っていない）。

- **長さ `lengths`**
  - 形状: `(B,)`、全て `T`（実フレーム長）。
  - 実質固定長入力として利用（`buffer_len=100` 前提）。

- **ラベル `y`**
  - 形状: `(B,)`、全てゼロ。推論では使わないダミー。

- **出力と正規化**
  - `motion_model(batch)` の `out["mu"]` を L2 正規化して `z_m` として使用。
  - そのため **cos類似度 = 内積**として扱える。

※ バッファが埋まる前でも `encode_buffer()` は実行されるが、報酬は `warmup_counter` でゼロ化される。

## スライド案①：MotionCLIPに入る入力テンソルの定義（何を入れてるか）

**Slide 1: 入力の最終形（MotionCLIPに渡すもの）**

- MotionCLIPが受け取る入力テンソル
  - `x: (B, 14, 2, T)`
  - **B**: 環境数（並列env）
  - **14**: HOYO skeletonの関節数
  - **2**: 2D座標（HOYO front表現）
  - **T**: 窓長（例: 100フレーム = 2秒@50Hz）

- 生成元（直前の内部状態）
  - `motion_buffer: (B, T, 14, 3)`（3Dの履歴）
  - `heading_buffer: (B, T)`（yawの履歴）

- 変換の流れ（超要約）
  - 3D履歴 → yaw補正/投影/正規化 → `x: (B, 14, 2, T)`

**Speaker note**
- 「MotionCLIPは“時系列の2Dスケルトン（14関節）”を見て潜在表現 `mu` を出す」  
  と言い切ると伝わりやすい。

## z_m がどう出てくるか（MotionCLIP encoder 内部）

MotionCLIP の実装は `MotionCLIP/src/models/architectures/transformer.py` の
`Encoder_TRANSFORMER` が担当。流れは以下。

1. **入力を1フレーム1トークン化**
   - `x: (B, 14, 2, T)` を `(T, B, 14*2)` に並べ替え。
   - `skelEmbedding`（Linear）で `latent_dim` に射影。

2. **learned token を先頭に追加**
   - `muQuery` と `sigmaQuery` の2トークンを先頭に付ける。
   - つまり系列長は `T + 2`。

3. **Transformer Encoder**
   - 位置エンコーディングを足し、Transformer で系列を処理。
   - `mask` でパディングを無視（RL側は全フレーム有効なので常に True）。

4. **mu を取り出す**
   - 出力系列 `final` の **先頭トークン** `final[0]` を `mu` として採用。
   - `logvar` も計算されるが、この経路では返していない。

5. **StyleModule で L2 正規化**
   - `encode_buffer()` で `mu` を正規化し `z_m` として使用。

要点: **z_m は「MotionCLIP Transformer encoder の learned mu token 出力」を正規化したもの**。

## 2D前処理の詳細（coord_mode別）

### hoyo_front（デフォルト）

- **センタリング**: `first_frame_com` が既定。
- **Yaw補正**: `apply_yaw_correction=True` の場合、**各フレーム**で `-yaw` 回転。
- **投影**: `(Y, -Z)` を 2Dとする（正面視点）。
- **身長正規化**: head-feetの2D距離でスケール。
- **標準化**: HOYO由来のmean/stdを適用。

## HOYO側の前処理（学習データ）

`hoyo_v1_1/models/common.py` の `HoyoInstructionDataset` が主処理。

1. **身長正規化（ロード時）**
   - pickle読込後、head-feet距離でスケール正規化。
   - 2D座標は HOYO が `[y, x]` なので `[x, y]` に入れ替え。
   - 最初フレームのCoMを原点に平行移動（移動情報を保持したまま原点化）。

2. **リサンプリング**
   - `src_fps=60` → `tgt_fps=50` を線形補間で変換（デフォルト設定）。

3. **Window Slicing（Crop）**
   - `target_len` に合わせてランダム/中心クロップ。

4. **センタリング**
   - `first_frame_com` が既定（他に pelvis 系あり）。

5. **標準化**
   - `mean/std` が設定されていれば適用。

## H1 motion側の前処理（RLランタイム）

`style_module.py` の `update_buffer` → `_prepare_centered_2d` → `encode_buffer` が主処理。

1. **H1→HOYO 14関節変換**
   - H1リンクをHOYO順にマッピングし、足りない関節はオフセット推定。

2. **バッファ更新**
   - `buffer_len=100` フレーム分を保持。
   - 50Hz想定で **約2秒** の履歴になる設計。

3. **2D投影・正規化**
   - 実体は `StyleModule._prepare_centered_2d()` の分岐。
   - **hoyo_front** のとき:
     - センタリング: `first_frame_com` が既定（または `pelvis` 指定で各フレーム骨盤）。
     - Yaw補正: `apply_yaw_correction=True` のとき **各フレームで -yaw 回転**。
     - 投影: **H1 [Y, Z] → HOYO [x, y]**（`x=+Y`, `y=-Z`）。
     - 身長正規化: **2D head–feet 距離**でスケール。
     - 標準化: HOYO由来の `mean/std` を適用。
   - 注意:
     - `apply_yaw_correction` は **hoyo_front 専用**（legacyでは常にyaw補正）。
     - バッファ未充足時は **最初の有効フレーム**を基準にセンタリング/スケール計算を行う。

## FPS / 窓長（2秒）整合性チェック

### 確認済みの前提

- **legacy_xz_yawは使用しない**（hoyo_front運用）。
- **MotionCLIPはtarget_len=100** で学習。
- **HOYOデータは前方向の動きのみ**（yaw変化が小さい前提）。

### 現状のデフォルト設定

- **HOYOデータ**: `src_fps=60` → `tgt_fps=50`（`HoyoInstructionDataset`）
- **MotionCLIP学習**: `target_len=100`（`train_motionclip_joint.py` のデフォルト）
  - 100フレーム = 2秒 @ 50Hz
- **RL側バッファ**: `buffer_len=100`（StyleModule）
- **Sim設定**: `sim.dt=0.005`（200Hz） + `decimation=4` → 50Hz

→ デフォルト値を前提にすると **HOYOとRLは2秒窓 @ 50Hzで整合** している。

### ずれの可能性（要確認）

1. **HOYOのFPS仮定に不一致あり（結論: 60fps）**
   - **HOYO元データは60fps（確認済み）**。
   - `compute_velocity_table.py` の **30fps仮定** は古い可能性が高い。
   - `visualize_hoyo_dataset.py` の **60fps前提** が妥当。

2. **target_lenの違い**
   - `visualize_joint_training.py` など一部スクリプトは `target_len=60` を使う。
   - **運用上は target_len=100 を使用する前提（確認済み）**。

3. **centering/normalizationの設定差**
   - HOYO側: `first_frame_com` が既定。
   - RL側: `coord_mode` や `apply_yaw_correction` により分布が変わる。
   - 正規化統計（mean/std）はHOYO由来なので、RL分布とずれる可能性。

## 速度差が出にくい可能性のあるポイント

1. **hoyo_front投影で前進軸(X)が捨てられる**
   - 前進速度や移動距離が2D入力に入らず、速度差は主に「歩隔の周波数」だけになる。
   - forward成分が含まれるのは `legacy_xz_yaw` 側のみ。

2. **first_frame_comセンタリングで絶対移動が弱くなる**
   - 1フレーム目のCoMを引くため、グローバルな並進は初期基準に対する変位のみ。
   - hoyo_frontではその変位のうち前進成分が投影で消える。

3. **encode_bufferは常にyaw補正を有効化**
   - コメント上は「HOYO正規化統計に合わせるためデフォルトOFF」だが、
     実際の `encode_buffer` は `apply_yaw_correction=True` 固定。
   - yaw補正+front投影により「進行方向の動き」がさらに消える。

4. **身長正規化と標準化で振幅差が圧縮される**
   - 速度差に伴う姿勢の上下動・ストライド長がスケール化されやすい。
   - HOYO統計のmean/stdがRLデータ分布に合わない場合、差が縮む可能性。

5. **固定窓長（buffer_len=100）**
   - 速度差は「窓内の歩周期回数」だけで表現される。
   - MotionCLIP側がテンポ不変な学習をしていると差がさらに薄くなる。

## 速度差が出るか確認する簡易チェック案

- **同じスタイルで速度だけ変えた入力**を作り、`z_m` の距離を比較。
- `coord_mode=legacy_xz_yaw` と `hoyo_front` で `z_m` の差がどの程度変わるか確認。
- `apply_yaw_correction=False` にした場合の差分を確認（分布変化に注意）。
